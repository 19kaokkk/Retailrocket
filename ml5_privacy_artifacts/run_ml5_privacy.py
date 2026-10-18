#!/usr/bin/env python
"""RetailRocket user-group DP-SGD benchmark; see ML5_README.md for scope.

Opacus handles flat clipping, Gaussian noise and RDP accounting. The only
custom gradient is the closed-form logistic gradient averaged over every
training session of one visitor; test_ml5_privacy.py checks it against autograd.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
import warnings
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import opacus
import pandas as pd
import torch
from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier
from opacus.optimizers import DPOptimizer
from scipy.stats import spearmanr, ttest_rel
from sklearn.metrics import (accuracy_score, average_precision_score,
    balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score)

FEATURES = ["observed_duration_sec", "observed_n_events", "observed_n_views",
    "observed_n_addtocart", "observed_n_unique_items", "has_observed_addtocart",
    "view_to_cart_ratio"]
CAPS = np.array([1800, 100, 100, 20, 100, 1, 100], dtype=np.float32)
TARGET = "label_has_purchase"
ALGORITHMS = ["Centralized", "FedAvg", "FedProx"]
METRICS = ["accuracy", "balanced_accuracy", "precision", "recall", "f1",
    "roc_auc", "pr_auc", "positive_rate_pred"]


@dataclass
class Config:
    epochs: int = 3
    rounds: int = 6
    local_epochs: int = 1
    batch_users: int = 4096
    local_batch_users: int = 2048
    lr: float = 0.1
    mu: float = 0.1
    clip: float = 100.0
    pos_weight: float = 100.0
    delta: float = 1e-7
    threshold: float = 0.5
    deterministic: bool = False


def transform(df):
    # Fixed domain bounds: no means, quantiles or class counts are fitted.
    values = df[FEATURES].to_numpy(dtype=np.float32)
    values = np.nan_to_num(values, nan=0, posinf=0, neginf=0)
    return torch.from_numpy(np.log1p(np.clip(values, 0, CAPS)) / np.log1p(CAPS))


class UserData:
    def __init__(self, frame, public_users=None):
        frame = frame.sort_values("visitorid", kind="stable")
        self.x = transform(frame)
        self.y = torch.tensor(frame[TARGET].to_numpy(dtype=np.float32))
        ids = frame.visitorid.to_numpy()
        if public_users is None:
            public_users = np.unique(ids)
        roster = pd.Index(public_users).sort_values()
        if roster.has_duplicates or not pd.Index(ids).isin(roster).all():
            raise ValueError("Public roster is duplicated or missing training visitors")
        # Retain zero-contribution slots when all sessions of a person are removed.
        self.counts = frame.groupby("visitorid").size().reindex(roster, fill_value=0).to_numpy()
        self.starts = np.cumsum(self.counts) - self.counts
        self.n_users = len(self.starts)

    def batch(self, users):
        counts = self.counts[users]
        owner = np.repeat(np.arange(len(users)), counts)
        offsets = np.repeat(np.cumsum(counts) - counts, counts)
        rows = np.repeat(self.starts[users], counts) + np.arange(counts.sum()) - offsets
        return self.x[rows], self.y[rows], torch.from_numpy(owner), torch.from_numpy(counts)


def poisson_users(rng, population, q):
    # Binomial size + uniform subset is exactly independent Bernoulli(q)
    # inclusion for each user, without scanning the entire roster every step.
    size = int(rng.binomial(population, q))
    return rng.choice(population, size=size, replace=False)


def user_gradients(model, x, y, owner, counts, pos_weight):
    """Gradient of mean(weighted BCE over a user's sessions), one row/user."""
    with torch.no_grad():
        logits = model(x).flatten()
        error = (torch.sigmoid(logits) - y) * torch.where(y == 1, pos_weight, 1.0)
        gradients = torch.cat((error[:, None] * x, error[:, None]), dim=1)
        summed = torch.zeros((len(counts), x.shape[1] + 1), dtype=x.dtype)
        summed.index_add_(0, owner, gradients)
        return summed / counts.clamp(min=1)[:, None].to(x.dtype)


def make_model(n_features):
    model = torch.nn.Linear(n_features, 1)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    return model


def schedule(n_users, batch_users, passes):
    steps_per_pass = math.ceil(n_users / batch_users)
    q = 1.0 / steps_per_pass
    return {"n_users": n_users, "q": q, "steps_per_pass": steps_per_pass,
        "steps": steps_per_pass * passes, "expected_batch_users": q * n_users}


@lru_cache(maxsize=2048)
def _epsilon_for_history(history, delta):
    accountant = RDPAccountant()
    accountant.history = list(history)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return float(accountant.get_epsilon(delta))


def accountant_epsilon(accountant, delta):
    # RDP composes additively: coalesce equal mechanisms across released runs.
    totals = {}
    for sigma, q, steps in accountant.history:
        totals[sigma, q] = totals.get((sigma, q), 0) + steps
    history = tuple((sigma, q, steps) for (sigma, q), steps in sorted(totals.items()))
    return _epsilon_for_history(history, delta)


@lru_cache(maxsize=128)
def _calibrate_envelope(epsilon, delta, q, steps):
    return float(get_noise_multiplier(target_epsilon=epsilon, target_delta=delta,
        sample_rate=q, steps=steps, accountant="rdp", epsilon_tolerance=0.005))


def calibrate(schedules, epsilon, delta):
    # Conservative envelope; exact client histories are still accounted below.
    return _calibrate_envelope(epsilon, delta, max(s["q"] for s in schedules),
                              max(s["steps"] for s in schedules))


def random_streams(config, tag):
    if config.deterministic:
        seed = int.from_bytes(hashlib.sha256(tag.encode()).digest()[:8], "little")
        rng = np.random.default_rng(seed)
        generator = torch.Generator().manual_seed(seed % (2**63 - 1))
    else:
        # Fresh independent, undisclosed PRNG seeds for each released run/client.
        rng = np.random.default_rng()
        generator = torch.Generator()
        generator.seed()
    return rng, generator


def train_steps(model, data, spec, config, mode, sigma, accountant, tag, anchor=None):
    base = torch.optim.SGD(model.parameters(), lr=config.lr)
    rng, generator = random_streams(config, tag)
    if mode == "no_dp":
        optimizer = base
    else:
        optimizer = DPOptimizer(base, noise_multiplier=sigma,
            max_grad_norm=config.clip, expected_batch_size=spec["expected_batch_users"],
            generator=generator, secure_mode=False)
        if mode == "dp":
            optimizer.attach_step_hook(accountant.get_optimizer_hook_fn(sample_rate=spec["q"]))
    for _ in range(spec["steps"]):
        optimizer.zero_grad()
        selected = poisson_users(rng, data.n_users, spec["q"])
        if len(selected):
            gradients = user_gradients(model, *data.batch(selected), config.pos_weight)
        else:
            gradients = torch.empty((0, len(FEATURES) + 1))
        if mode == "no_dp":
            mean = gradients.sum(dim=0) / spec["expected_batch_users"]
            model.weight.grad = mean[:-1].reshape_as(model.weight)
            model.bias.grad = mean[-1:].reshape_as(model.bias)
        else:
            # DPOptimizer clips the complete USER gradient, weight AND bias.
            model.weight.grad_sample = gradients[:, :-1].unsqueeze(1)
            model.bias.grad_sample = gradients[:, -1:]
            # Empty batches must also add noise and count as a privacy step.
            optimizer.pre_step()
        if anchor is not None:
            with torch.no_grad():
                for parameter, reference in zip(model.parameters(), anchor):
                    parameter.grad.add_(config.mu * (parameter - reference))
        # pre_step already clipped/noised/accounted; do not call it twice.
        base.step()


def metrics(y, scores, threshold):
    pred = scores >= threshold
    return {"accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "pr_auc": float(average_precision_score(y, scores)),
        "positive_rate_pred": float(pred.mean())}


def evaluate(model, evaluation, threshold):
    results = {}
    with torch.no_grad():
        for split, (x, y) in evaluation.items():
            scores = torch.sigmoid(model(x)).flatten().numpy()
            results.update({f"{split}_{k}": v for k, v in metrics(y, scores, threshold).items()})
    return results


def train_experiment(algorithm, mode, sigma, target, replicate, central, clients,
                     specs, config, evaluation):
    start = time.perf_counter()
    model = make_model(len(FEATURES))
    histories = []
    accounts = {key: RDPAccountant() for key in specs[algorithm]}
    tag = f"{algorithm}:{mode}:{target}:{replicate}"
    if algorithm == "Centralized":
        spec = specs[algorithm]["central"]
        train_steps(model, central, spec, config, mode, sigma, accounts["central"], tag)
    else:
        weights = {key: data.n_users / sum(d.n_users for d in clients.values())
                   for key, data in clients.items()}
        for round_id in range(config.rounds):
            anchor = [p.detach().clone() for p in model.parameters()]
            aggregated = [torch.zeros_like(p) for p in anchor]
            for client, data in clients.items():
                local = make_model(len(FEATURES))
                local.load_state_dict(model.state_dict())
                local_spec = dict(specs[algorithm][client])
                local_spec["steps"] //= config.rounds
                train_steps(local, data, local_spec, config, mode, sigma, accounts[client],
                    f"{tag}:{client}:{round_id}", anchor if algorithm == "FedProx" else None)
                with torch.no_grad():
                    for total, parameter in zip(aggregated, local.parameters()):
                        total.add_(parameter, alpha=weights[client])
            with torch.no_grad():
                for parameter, total in zip(model.parameters(), aggregated):
                    parameter.copy_(total)
            # No validation metrics are used for stopping or parameter selection.
            histories.append({"algorithm": algorithm, "mode": mode,
                "target_epsilon": target, "replicate": replicate, "round": round_id + 1})
    epsilon_values = []
    ledger = []
    for client, account in accounts.items():
        spec = specs[algorithm][client]
        actual_steps = sum(h[2] for h in account.history)
        if mode == "dp" and actual_steps != spec["steps"]:
            raise AssertionError("Accountant does not match the executed schedule")
        epsilon = accountant_epsilon(account, config.delta) if mode == "dp" else math.inf
        if mode == "dp" and epsilon > target + 0.01:
            raise AssertionError("Privacy target exceeded")
        epsilon_values.append(epsilon)
        ledger.append({"algorithm": algorithm, "mode": mode, "target_epsilon": target,
            "replicate": replicate, "client": client, "privacy_unit": "visitor_train_sessions",
            "noise_multiplier": sigma, "delta": config.delta, "epsilon": epsilon,
            "steps": spec["steps"], "q": spec["q"], "n_users": spec["n_users"],
            "expected_batch_users": spec["expected_batch_users"]})
    row = {"algorithm": algorithm, "mode": mode, "target_epsilon": target,
        "replicate": replicate, "noise_multiplier": sigma,
        "epsilon": max(epsilon_values), "delta": config.delta,
        "privacy_unit": "visitor_train_sessions", "clip_norm": config.clip,
        "elapsed_seconds": time.perf_counter() - start,
        **evaluate(model, evaluation, config.threshold)}
    parameters = {"algorithm": algorithm, "mode": mode,
        "target_epsilon": target if math.isfinite(target) else None,
        "replicate": replicate, "weight": model.weight.detach().flatten().tolist(),
        "bias": model.bias.detach().item()}
    return row, ledger, histories, accounts, parameters


def markdown_table(frame):
    def fmt(value):
        if isinstance(value, (float, np.floating)):
            return "inf" if math.isinf(value) else f"{value:.4f}"
        return str(value)
    return ("| " + " | ".join(frame.columns) + " |\n| " + " | ".join(
        "---" for _ in frame.columns) + " |\n" + "\n".join(
        "| " + " | ".join(fmt(v) for v in row) + " |" for row in frame.to_numpy()))


def aggregate_results(results):
    keys = ["algorithm", "mode", "target_epsilon"]
    grouped = results.groupby(keys, dropna=False, sort=False)
    table = grouped.agg(noise_multiplier=("noise_multiplier", "first"),
        epsilon=("epsilon", "max"), n_runs=("replicate", "count")).reset_index()
    for split in ["val", "test"]:
        for metric in METRICS:
            name = f"{split}_{metric}"
            values = grouped[name].agg(["mean", "std"]).reset_index()
            values = values.rename(columns={"mean": name + "_mean", "std": name + "_std"})
            table = table.merge(values, on=keys, validate="one_to_one")
    for algorithm in table.algorithm.unique():
        selection = table.algorithm == algorithm
        baseline = table.loc[selection & (table["mode"] == "no_dp"), "test_pr_auc_mean"].iloc[0]
        clipped = table.loc[selection & (table["mode"] == "clip_only"), "test_pr_auc_mean"].iloc[0]
        table.loc[selection, "pr_auc_retention_pct"] = 100 * table.loc[selection, "test_pr_auc_mean"] / baseline
        table.loc[selection, "pr_auc_loss_vs_no_dp"] = baseline - table.loc[selection, "test_pr_auc_mean"]
        table.loc[selection, "pr_auc_loss_vs_clip_only"] = clipped - table.loc[selection, "test_pr_auc_mean"]
    return table


def statistical_checks(results):
    rows = []
    for target in sorted(results.loc[results["mode"] == "dp", "target_epsilon"].unique()):
        selected = results[(results["mode"] == "dp") & (results.target_epsilon == target)]
        pair = selected.pivot(index="replicate", columns="algorithm", values="test_pr_auc")
        if {"FedAvg", "FedProx"}.issubset(pair.columns):
            differences = pair.FedProx - pair.FedAvg
            if len(pair) > 1 and differences.std() > 0:
                stat = ttest_rel(pair.FedProx, pair.FedAvg)
                pvalue = float(stat.pvalue)
            else:
                pvalue = np.nan
            rows.append({"analysis": "FedProx-FedAvg paired run labels",
                "target_epsilon": target, "n_pairs": len(pair),
                "pr_auc_mean_difference": differences.mean(), "p_value": pvalue})
    for algorithm in results.algorithm.unique():
        selected = results[(results.algorithm == algorithm) & (results["mode"] == "dp")]
        mean = selected.groupby("target_epsilon").test_pr_auc.mean()
        if len(mean) > 2:
            stat = spearmanr(mean.index, mean.values)
            rows.append({"analysis": f"{algorithm} Spearman epsilon/mean PR-AUC",
                "n_pairs": len(mean), "rho": float(stat.statistic), "p_value": float(stat.pvalue)})
    return pd.DataFrame(rows)


def select_privacy_configurations(table, minimum_retention=95.0):
    rows = []
    for algorithm in ALGORITHMS:
        part = table[table.algorithm == algorithm]
        baseline = part.loc[part["mode"] == "no_dp", "val_pr_auc_mean"].iloc[0]
        private = part[part["mode"] == "dp"].sort_values("target_epsilon").copy()
        private["validation_retention_pct"] = 100 * private.val_pr_auc_mean / baseline
        eligible = private[private.validation_retention_pct >= minimum_retention]
        row = {"algorithm": algorithm, "criterion": "validation PR-AUC retention",
            "minimum_retention_pct": minimum_retention,
            "status": "eligible" if len(eligible) else "no tested level qualifies"}
        if len(eligible):
            selected = eligible.iloc[0]
            row.update({"target_epsilon": selected.target_epsilon, "epsilon": selected.epsilon,
                "noise_multiplier": selected.noise_multiplier,
                "validation_retention_pct": selected.validation_retention_pct})
        rows.append(row)
    return pd.DataFrame(rows)


def plot_results(table, out, delta=1e-7):
    colors = {"Centralized": "#147d75", "FedAvg": "#345ca8", "FedProx": "#bd4363"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for algorithm, part in table.groupby("algorithm", sort=False):
        private = part[part["mode"] == "dp"].sort_values("epsilon")
        axes[0].errorbar(private.epsilon, private.test_pr_auc_mean,
            yerr=private.test_pr_auc_std.fillna(0), marker="o", capsize=3,
            color=colors[algorithm], label=algorithm)
        axes[1].plot(private.epsilon, private.pr_auc_retention_pct, marker="o",
            color=colors[algorithm], label=algorithm)
    axes[0].set(ylabel="Test average precision (PR-AUC)", xlabel=f"Accounted epsilon (delta={delta:g})")
    axes[1].set(ylabel="PR-AUC retained vs own no-DP baseline (%)",
        xlabel=f"Accounted epsilon (delta={delta:g})")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(fontsize=9)
    fig.suptitle("User-group DP-SGD: privacy and utility\nError bars: standard deviation across runs")
    fig.savefig(out / "ml5_privacy_utility.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.7), layout="constrained", sharey=True)
    for ax, algorithm in zip(axes, ALGORITHMS):
        part = table[table.algorithm == algorithm]
        labels = [r.mode if r.mode != "dp" else f"eps {r.target_epsilon:g}" for r in part.itertuples()]
        ax.bar(np.arange(len(part)), part.test_pr_auc_mean,
            yerr=part.test_pr_auc_std.fillna(0), capsize=3, color=colors[algorithm])
        ax.set_xticks(np.arange(len(part)), labels, rotation=35, ha="right")
        ax.set_title(algorithm)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Test average precision (PR-AUC)")
    fig.suptitle("No DP, clipping only, and Gaussian noise")
    fig.savefig(out / "ml5_noise_comparison.png", dpi=180)
    plt.close(fig)


def make_report(table, summary, ledger=None, statistics=None):
    private = table[table["mode"] == "dp"]
    display = table[["algorithm", "mode", "target_epsilon", "noise_multiplier", "epsilon",
        "test_pr_auc_mean", "test_pr_auc_std", "test_roc_auc_mean", "test_f1_mean",
        "pr_auc_retention_pct"]].copy()
    display["test_pr_auc_std"] = display["test_pr_auc_std"].map(lambda value: f"{value:.6f}")
    display = display.rename(columns={"algorithm": "Thuật toán", "mode": "Chế độ",
        "target_epsilon": "ε mục tiêu", "noise_multiplier": "σ", "epsilon": "ε thực tế",
        "test_pr_auc_mean": "PR-AUC test TB", "test_pr_auc_std": "SD PR-AUC",
        "test_roc_auc_mean": "ROC-AUC test TB", "test_f1_mean": "F1 test TB",
        "pr_auc_retention_pct": "Retention (%)"})

    def mean_sd(row, metric, split="test"):
        mean, std = row[f"{split}_{metric}_mean"], row[f"{split}_{metric}_std"]
        sd_text = "n/a" if pd.isna(std) else f"{std:.6f}"
        return f"{mean:.6f} ± {sd_text}"

    def metric_table(row):
        metrics = [("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"),
            ("balanced_accuracy", "Balanced Accuracy"), ("precision", "Precision"),
            ("recall", "Recall"), ("f1", "F1"), ("accuracy", "Accuracy")]
        return markdown_table(pd.DataFrame([{"Chỉ số": label,
            "Validation (TB ± SD)": mean_sd(row, metric, "val"),
            "Test (TB ± SD)": mean_sd(row, metric)} for metric, label in metrics]))

    low_target = float(private.target_epsilon.min())
    central = table[table.algorithm == "Centralized"]
    no_dp = central[central["mode"] == "no_dp"].iloc[0]
    clip_only = central[central["mode"] == "clip_only"].iloc[0]
    central_dp = central[(central["mode"] == "dp") & (central.target_epsilon == low_target)].iloc[0]
    central_schedule = summary["schedules"]["Centralized"]["central"]
    central_table = pd.DataFrame([{"Chế độ": row["mode"], "ε mục tiêu": row.target_epsilon,
        "ε thực tế": row.epsilon, "PR-AUC validation (TB ± SD)": mean_sd(row, "pr_auc", "val"),
        "PR-AUC test (TB ± SD)": mean_sd(row, "pr_auc"),
        "F1 test TB": row.test_f1_mean} for _, row in central.iterrows()])
    noise_table = pd.DataFrame([{"Thuật toán": row.algorithm, "ε mục tiêu": row.target_epsilon,
        "ε thực tế": row.epsilon, "σ": row.noise_multiplier,
        "SD nhiễu trên tổng gradient (σC)": row.noise_multiplier * summary["config"]["clip"]}
        for row in private.itertuples()])
    noise_examples = []
    for algorithm in ALGORITHMS:
        part = private[private.algorithm == algorithm].sort_values("target_epsilon")
        first, last = part.iloc[0], part.iloc[-1]
        noise_examples.append(f"- {algorithm}: ε mục tiêu {first.target_epsilon:g} dùng σ = "
            f"{first.noise_multiplier:.4f}, còn ε mục tiêu {last.target_epsilon:g} dùng σ = "
            f"{last.noise_multiplier:.4f}.")
    fed_rows = []
    for target in sorted(private.target_epsilon.unique()):
        avg = private[(private.algorithm == "FedAvg") & (private.target_epsilon == target)].iloc[0]
        prox = private[(private.algorithm == "FedProx") & (private.target_epsilon == target)].iloc[0]
        p_text = "n/a"
        if statistics is not None:
            found = statistics[(statistics.analysis == "FedProx-FedAvg paired run labels") &
                               (statistics.target_epsilon == target)]
            if len(found) and pd.notna(found.iloc[0].p_value):
                p_text = f"{found.iloc[0].p_value:.6f}"
        fed_rows.append({"ε mục tiêu": target, "ε thực tế tối đa": prox.epsilon,
            "FedProx PR-AUC val (TB ± SD)": mean_sd(prox, "pr_auc", "val"),
            "FedAvg PR-AUC test (TB ± SD)": mean_sd(avg, "pr_auc"),
            "FedProx PR-AUC test (TB ± SD)": mean_sd(prox, "pr_auc"),
            "Δ PR-AUC (Prox − Avg)": f"{prox.test_pr_auc_mean - avg.test_pr_auc_mean:+.6f}",
            "p thăm dò": p_text})
    fed_table = pd.DataFrame(fed_rows)
    prox_low = private[(private.algorithm == "FedProx") & (private.target_epsilon == low_target)].iloc[0]
    client_table = ""
    if ledger is not None:
        local = ledger[(ledger["mode"] == "dp") & (ledger.algorithm == "FedProx")]
        ranges = local.groupby("target_epsilon").agg(epsilon_min=("epsilon", "min"),
            epsilon_max=("epsilon", "max"), q_min=("q", "min"), q_max=("q", "max"),
            steps_min=("steps", "min"), steps_max=("steps", "max")).reset_index()
        ranges = ranges.rename(columns={"target_epsilon": "ε mục tiêu",
            "epsilon_min": "ε client nhỏ nhất", "epsilon_max": "ε client lớn nhất",
            "q_min": "q nhỏ nhất", "q_max": "q lớn nhất", "steps_min": "Bước nhỏ nhất",
            "steps_max": "Bước lớn nhất"})
        client_table = markdown_table(ranges)
    correlation_table = ""
    if statistics is not None:
        correlations = statistics[statistics.analysis.str.contains("Spearman", na=False)]
        if len(correlations):
            correlation_table = markdown_table(correlations[["analysis", "n_pairs", "rho", "p_value"]]
                .rename(columns={"analysis": "Phân tích", "n_pairs": "Số mức ε",
                                 "rho": "Spearman ρ", "p_value": "p thăm dò"}))
        else:
            correlation_table = "Không đủ mức ε để tính Spearman trong lần chạy này."
    sentences = []
    for algorithm in ALGORITHMS:
        part = private[private.algorithm == algorithm].sort_values("target_epsilon")
        low, high = part.iloc[0], part.iloc[-1]
        sentences.append(f"- {algorithm}: từ ε mục tiêu {low.target_epsilon:g} đến "
            f"{high.target_epsilon:g}, PR-AUC trung bình thay đổi từ "
            f"{low.test_pr_auc_mean:.4f} sang {high.test_pr_auc_mean:.4f}; "
            f"retention tương ứng là {low.pr_auc_retention_pct:.1f}% và {high.pr_auc_retention_pct:.1f}%.")
    paired = private.pivot(index="target_epsilon", columns="algorithm", values="test_pr_auc_mean")
    diff = paired.FedProx - paired.FedAvg
    maximum_change = float(private.pr_auc_loss_vs_no_dp.abs().max())
    choices = select_privacy_configurations(table)
    choice_text = "; ".join(f"{r.algorithm}: ε mục tiêu {r.target_epsilon:g}"
        if r.status == "eligible" else f"{r.algorithm}: chưa có mức đạt yêu cầu"
        for r in choices.itertuples())
    rq5 = ("Trong lần thực nghiệm này, tăng mức bảo vệ bằng cách giảm ε cần noise multiplier "
        "lớn hơn với lịch huấn luyện giữ cố định. Chất lượng dự đoán được đo trực tiếp trong "
        "bảng trên; không giả định đường cong phải giảm đơn điệu ở từng lần chạy. "
        f"Chênh lệch PR-AUC trung bình của FedProx + DP-SGD so với FedAvg + DP-SGD "
        f"nằm trong [{diff.min():+.4f}, {diff.max():+.4f}] trên các mức ε đã chạy. "
        "FedProx không mặc nhiên bù được chi phí nhiễu. Kết luận này chỉ áp dụng cho "
        "mô hình Logistic Regression dự đoán purchase theo session, 50 client hash và "
        "cấu hình hiện tại; không suy rộng sang chất lượng gợi ý SASRec/HSTU.")
    return f"""# 4.3.6. ML5 - Privacy-Utility Trade-off

## 4.3.6.1. DP-SGD

Dùng lại dữ liệu session đã chuẩn bị cho ML4: {summary['train_sessions']:,} phiên train,
{summary['val_sessions']:,} phiên validation và {summary['test_sessions']:,} phiên test.
Nhãn là `label_has_purchase`, bảy đặc trưng chỉ sử dụng phần quan sát trước giao dịch.
Đơn vị lấy mẫu và cắt gradient là một `visitorid`, gồm toàn bộ phiên TRAIN của người đó.
Tổng số người trong train là {summary['train_users']:,}. Một khách hàng thuộc đúng một client.

Với khách hàng u, loss là trung bình weighted BCE trên các phiên của u, với trọng số
purchase cố định {summary['config']['pos_weight']:g}. Gradient của tất cả phiên được
gộp thành một gradient người dùng trước khi cắt chung weight và bias tại C =
{summary['config']['clip']:g}. Opacus DPOptimizer thêm Gaussian với độ lệch chuẩn σC
vào TỔNG gradient rồi chia cho số người kỳ vọng của batch. Lấy mẫu Poisson độc lập
theo người; batch rỗng vẫn thêm nhiễu và được accountant ghi nhận.

Mô hình Logistic Regression khởi tạo từ 0. Biến đổi đặc trưng dùng các chặn miền cố
định và log1p, không fit scaler hay tính class weight trên dữ liệu nhạy cảm. Ngưỡng
phân loại cố định 0.5, không chọn checkpoint/siêu tham số theo test hoặc validation.
Mỗi cấu hình chạy {summary['replicates']} lần độc lập; báo cáo mean và sample std.
Validation/test đo trên các phiên như ML4, không gộp người khi đánh giá utility.

Đối với DP-SGD tập trung, cấu hình gồm {summary['config']['epochs']} lượt lấy mẫu
kỳ vọng, learning rate {summary['config']['lr']:g}, batch mục tiêu
{summary['config']['batch_users']:,} người. Lịch thực tế là {central_schedule['steps']}
bước, q = {central_schedule['q']:.6f}, số người kỳ vọng mỗi batch là
{central_schedule['expected_batch_users']:.2f}. Các cấu hình giữ nguyên mô hình,
loss, clipping và lịch; thay đổi mức nhiễu để đối chiếu với hai mốc tham chiếu.

Kết quả DP-SGD tập trung và các baseline, trung bình ± độ lệch chuẩn của
{summary['replicates']} lần chạy:

{markdown_table(central_table)}

Ở mức ε mục tiêu {low_target:g}, ε thực tế của một mô hình tập trung là
{central_dp.epsilon:.4f}, PR-AUC test = {mean_sd(central_dp, 'pr_auc')}.
Baseline không DP đạt {mean_sd(no_dp, 'pr_auc')}, còn chỉ cắt gradient đạt
{mean_sd(clip_only, 'pr_auc')}. Chênh lệch trung bình không DP − clip-only là
{no_dp.test_pr_auc_mean - clip_only.test_pr_auc_mean:.6f}; clip-only − DP ở mức này
là {clip_only.test_pr_auc_mean - central_dp.test_pr_auc_mean:.6f}.
Đây là phép phân rã chênh lệch quan sát, cần đọc cùng SD và số lần lặp; chưa phải
ước lượng nhân quả của clipping/nhiễu trên mọi mô hình.

Chi tiết validation/test của DP-SGD tập trung tại mức ε mục tiêu {low_target:g}:

{metric_table(central_dp)}

## 4.3.6.2. Các mức độ nhiễu

Các ε mục tiêu: {summary['epsilon_targets']}, δ = {summary['config']['delta']:g}.
Noise multiplier được hiệu chỉnh bằng RDP accountant theo q và tổng bước đã định.
Cùng ε có thể cần σ khác giữa centralized và federated vì q/lịch bước khác nhau.
Có mốc `no_dp` (không cắt, không nhiễu) và `clip_only` (cắt, không nhiễu): cả hai có
ε = ∞, không có bảo vệ DP hữu hạn. Bảng ghi ε THỰC TẾ đã tính, không chỉ ε mục tiêu.
Chế độ ngẫu nhiên: {summary['randomness_mode']}.

Bảng dưới ghi mức nhiễu được dùng thực tế. ε là ngân sách của MỘT mô hình, với
federated là max ε_client, không phải ngân sách của toàn bộ các lần chạy đã công bố.

{markdown_table(noise_table)}

{chr(10).join(noise_examples)}

Trong lịch giữ cố định của từng thuật toán, ε nhỏ hơn yêu cầu σ lớn hơn.
Cột σC là độ lệch chuẩn noise thêm vào TỔNG gradient người dùng. Sau khi chia cho
số người kỳ vọng qN, độ lệch chuẩn trên gradient trung bình là σC/(qN). Vì q/lịch
của centralized và federated khác nhau, không so mức bảo vệ chỉ bằng σ.
`no_dp` và `clip_only` đều có σ = 0 và không có ε hữu hạn, dù clip-only có cắt norm.

## 4.3.6.3. Privacy-Utility Trade-off

{markdown_table(display)}

PR-AUC ở đây là average precision của sklearn. Với purchase rất hiếm, ưu tiên PR-AUC,
ROC-AUC và F1 hơn accuracy. Retention = PR-AUC_DP / PR-AUC_no_DP cùng thuật toán.
Chi phí clipping và chi phí noise được tách trong CSV; chênh lệch âm có thể xuất hiện
do ngẫu nhiên hoặc regularization, không được tự sửa thành 0.

{chr(10).join(sentences)}

Riêng DP-SGD tập trung ở ε mục tiêu {low_target:g}, PR-AUC test giữ được
{central_dp.pr_auc_retention_pct:.4f}% baseline không DP, với chênh lệch tuyệt đối
{central_dp.pr_auc_loss_vs_no_dp:.6f}. Retention hơi vượt 100% ở một số cấu hình
federated là kết quả của lần thử hữu hạn; chưa chứng minh nhiễu luôn cải thiện mô hình.
F1 trung bình giữa các cấu hình nằm trong [{table.test_f1_mean.min():.6f},
{table.test_f1_mean.max():.6f}] tại ngưỡng phân loại 0.5. F1 có thể giữ nguyên khi
điểm dự đoán/ranking đổi nhưng nhãn ở ngưỡng này chưa đổi, nên PR-AUC vẫn cần được xem riêng.

Kiểm tra thăm dò mối liên hệ ε và PR-AUC trung bình trên các mức đã thử:

{correlation_table}

Số mức ε ít và mỗi cấu hình chỉ có {summary['replicates']} lần chạy; các giá trị này
chưa đủ để khẳng định xu hướng đơn điệu hay một điểm gập. Error bar trên hình là
sample SD giữa các lần chạy, không phải khoảng tin cậy hoặc ngân sách riêng tư.

![Privacy utility](ml5_privacy_utility.png)

![Noise comparison](ml5_noise_comparison.png)

## 4.3.6.4. FedProx + DP-SGD

Giữ nguyên {summary['num_clients']} client ML4, gộp phiên theo người trong mỗi client, {summary['config']['rounds']}
round và {summary['config']['local_epochs']} lượt lấy mẫu kỳ vọng mỗi round. Tất cả client
tham gia, không giả định khuếch đại riêng tư do chọn client. Server tổng hợp theo số
NGƯỜI của client (khác trọng số session của ML4), coi trọng số và roster là thông tin công khai.
FedProx cộng μ(w_local - w_global) vào gradient sau clipping/noise, gồm cả bias,
với μ = {summary['config']['mu']:g}. Số hạng này không đọc thêm dữ liệu.

Mỗi client có accountant duy trì xuyên suốt các round, không reset ngân sách mỗi round.
Với roster cố định và client rời nhau, ε mỗi mô hình là max ε_client; lịch thực tế từng
client nằm trong `ml5_privacy_ledger.csv`. Đã chạy thêm FedAvg + DP-SGD để đối chiếu
vai trò proximal. Các phép paired t-test theo nhãn lần chạy và Spearman chỉ là thăm dò
với số lần lặp ít, không đủ để khẳng định ưu thế thống kê chắc chắn.

Baseline trong ML5 được chạy lại cùng biến đổi, hàm loss theo người và lịch huấn luyện.
Không đối chiếu trực tiếp số ML4 để gán mọi chênh lệch cho DP vì ML4 dùng scaler,
loss và lịch tối ưu khác.

Bảng kết quả riêng của FedProx + DP-SGD và mốc FedAvg + DP-SGD cùng ε mục tiêu:

{markdown_table(fed_table)}

Ở ε mục tiêu {low_target:g}, FedProx + DP-SGD có ε tối đa thực tế
{prox_low.epsilon:.4f}, PR-AUC test = {mean_sd(prox_low, 'pr_auc')} và giữ
{prox_low.pr_auc_retention_pct:.4f}% PR-AUC của baseline FedProx không DP.
Chi tiết chất lượng dự đoán tại cấu hình này:

{metric_table(prox_low)}

Ngân sách và lịch thực tế của các client (min/max trên các client đã ghi):

{client_table}

Hiệu số PR-AUC trung bình **FedProx − FedAvg** trên các mức thử nằm trong
[{diff.min():+.6f}, {diff.max():+.6f}]. Dấu hiệu số thay đổi theo mức ε, nên chưa
quan sát được ưu thế nhất quán của proximal term. p trong bảng là kiểm tra thăm dò
ghép theo nhãn lần chạy với rất ít cặp, không chứng minh hai mô hình tương đương
hay phương pháp nào luôn vượt trội. ε trong bảng cũng chỉ mô tả cơ chế huấn luyện
dưới giả định roster/lấy mẫu đã nêu.

## 4.3.6.5. Trả lời câu hỏi RQ5

RQ5 trong báo cáo: “Việc áp dụng DP-SGD ở các mức độ bảo vệ quyền riêng tư khác nhau
ảnh hưởng như thế nào đến hiệu quả mô hình? Khi kết hợp cơ chế bảo vệ quyền riêng tư
với FedProx, sự cân bằng giữa mức độ bảo vệ dữ liệu và chất lượng mô hình thay đổi ra sao?”

{rq5}

Độ thay đổi PR-AUC tuyệt đối lớn nhất giữa một cấu hình DP và baseline không DP
cùng thuật toán là {maximum_change:.4f}. Thí nghiệm có nhiều người và mô hình chỉ
có tám tham số; đây là các yếu tố có thể giúp ảnh hưởng nhiễu nhỏ. Đây là diễn giải
cấu hình, chưa phải bằng chứng nhân quả hay kết luận DP luôn ít ảnh hưởng.
F1 tại ngưỡng 0.5 có thể trùng nhau khi các điểm dự đoán thay đổi nhưng chưa đổi
nhãn phân loại. Các mức ε hiện tại không đủ để khẳng định có một điểm gập rõ ràng.

Ví dụ lựa chọn có điều kiện: yêu cầu giữ ít nhất 95% PR-AUC trên VALIDATION so với
baseline cùng thuật toán, rồi lấy ε mục tiêu nhỏ nhất trong lưới đáp ứng tiêu chí.
Ngưỡng 95% là giả định minh họa, không tối ưu bằng test hay một yêu cầu triển khai đã xác nhận.

{markdown_table(choices)}

Theo tiêu chí minh họa giữ 95% PR-AUC validation, kết quả lựa chọn là {choice_text}.
Đây là lựa chọn có điều kiện trong
lưới thí nghiệm, chưa xác định ε tối ưu cho triển khai và chưa chứng minh có điểm gập.

Ngân sách RDP cộng dồn của TẤT CẢ mô hình DP trong lần chạy được lưu là
ε = {summary['joint_dp_models_accounted_epsilon']:.4f}, δ = {summary['joint_delta']:g}.
Giá trị này khác ngân sách từng mô hình ở bảng ε–σ. Nó vẫn chỉ áp dụng cho các cơ
chế mô hình DP theo giả định, không bao gồm việc công bố baseline và metric chưa DP.

### Phạm vi riêng tư phải ghi trong báo cáo

Code đã triển khai gộp và cắt gradient theo người. Tuy nhiên ε trong bảng là ngân sách
của CƠ CHẾ HUẤN LUYỆN theo giả định lấy mẫu/ngẫu nhiên, không chứng nhận toàn bộ
pipeline đạt user-level DP. Adjacency dùng một slot người dùng trong roster công khai:
thêm/bớt toàn bộ đóng góp train của người đó, giữ slot, q, lịch và trọng số cố định.
Không diễn giải như xóa một người rồi tính lại roster, scaler, client size và weights.

Tiền xử lý dữ liệu gốc, visitor/client IDs, kích thước client và mọi metric trên
validation/test không được privatize. Một người có thể xuất hiện ở nhiều split;
bảo vệ các phiên train không bảo vệ việc công bố thống kê từ các phiên test của họ.
Roster benchmark được trích từ train và giả định công khai. Tùy chọn `--public-roster`
nhận roster bên ngoài, giữ người không có phiên ở slot gradient bằng 0. Kiểm tra
số học xác nhận N/q không đổi khi bỏ đóng góp một người trong roster cố định.
Cutoff quan sát của dataset dùng thời điểm giao dịch đầu tiên của phiên positive,
và cuối phiên negative. Đây là thiết lập retrospective phụ thuộc cách xây nhãn;
dự đoán online ở cùng thời điểm cần thiết kế horizon/cutoff chung rồi đánh giá lại.
RNG mặc định là PRNG chưa kiểm toán CSPRNG/khả năng tấn công số thực. Chế độ
deterministic công khai seed chỉ phục vụ tái lập học thuật, không dùng để phát hành DP.
Trước khi tuyên bố bảo vệ user-level toàn hệ thống phải kiểm toán adjacency, sampler,
RNG, tiền xử lý, metrics và tất cả lần phát hành. Nếu phát hành nhiều mô hình trên
cùng người phải cộng ngân sách; CSV mean/std không biến nhiều lần chạy thành một lần.
JSON kèm ngân sách compose chỉ của các mô hình DP, riêng các baseline không DP và
metric không privatize làm cả gói kết quả không có đảm bảo DP hữu hạn.

### Nguồn phương pháp

- [Abadi et al., Deep Learning with Differential Privacy](https://arxiv.org/abs/1607.00133)
- [Opacus DPOptimizer: clipping, Gaussian, expected batch](https://opacus.ai/api/optim/dp_optimizer.html)
- [Opacus RDPAccountant](https://opacus.ai/api/_modules/opacus/accountants/rdp.html)
- [Opacus RNG và secure mode](https://opacus.ai/api/privacy_engine.html)
- [Li et al., FedProx](https://arxiv.org/abs/1812.06127)
"""


def read_data(directory):
    frames = {}
    required = FEATURES + ["visitorid", "client_id", "session_id", TARGET]
    for split in ["train", "val", "test"]:
        frame = pd.read_csv(directory / f"ml4_{split}.csv.gz", usecols=required)
        if frame.session_id.duplicated().any():
            raise ValueError(f"Duplicate sessions in {split}")
        if frame[required].isna().any().any() or not frame[TARGET].isin([0, 1]).all():
            raise ValueError(f"Missing fields or invalid labels in {split}")
        frames[split] = frame
    all_rows = pd.concat(frames.values(), ignore_index=True)
    if all_rows.session_id.duplicated().any():
        raise ValueError("Session overlap between splits")
    if all_rows.groupby("visitorid").client_id.nunique().max() != 1:
        raise ValueError("A visitor belongs to multiple clients")
    return frames


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "ml5_results")
    parser.add_argument("--public-roster", type=Path,
        help="Externally public visitorid/client_id CSV; retains absent users as zero slots")
    parser.add_argument("--epsilon-targets", nargs="+", type=float, default=[1, 3, 5, 8])
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-users", type=int, default=4096)
    parser.add_argument("--local-batch-users", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--mu", type=float, default=0.1)
    parser.add_argument("--clip", type=float, default=100)
    parser.add_argument("--pos-weight", type=float, default=100)
    parser.add_argument("--delta", type=float, default=1e-7)
    parser.add_argument("--deterministic", action="store_true",
        help="PUBLIC seeded RNG for debugging only; not a DP release")
    args = parser.parse_args()
    positive = [args.replicates, args.epochs, args.rounds, args.local_epochs,
        args.batch_users, args.local_batch_users, args.lr, args.clip, args.pos_weight]
    if any(not math.isfinite(v) or v <= 0 for v in positive):
        parser.error("Counts, lr, clip and pos-weight must be positive and finite")
    if args.mu < 0 or not math.isfinite(args.mu) or not 0 < args.delta < 1:
        parser.error("mu must be nonnegative and 0 < delta < 1")
    if any(not math.isfinite(e) or e <= 0 for e in args.epsilon_targets):
        parser.error("epsilon targets must be positive and finite")
    args.epsilon_targets = sorted(set(args.epsilon_targets))
    return args


def main():
    args = parse_args()
    config = Config(**{k: getattr(args, k) for k in Config.__dataclass_fields__ if hasattr(args, k)})
    torch.set_num_threads(1)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    print("Loading ML4 splits and checking visitor/session grouping...", flush=True)
    frames = read_data(args.data_dir)
    train = frames["train"]
    if args.public_roster:
        roster = pd.read_csv(args.public_roster, usecols=["visitorid", "client_id"])
        if roster.isna().any().any() or roster.visitorid.duplicated().any():
            raise ValueError("Public roster must contain one non-null client per visitor")
        mapping = roster.set_index("visitorid").client_id
        if not train.client_id.eq(train.visitorid.map(mapping)).all():
            raise ValueError("Training visitor/client mapping differs from public roster")
    else:
        # Benchmark convenience ONLY: this extraction is not itself private.
        roster = train[["visitorid", "client_id"]].drop_duplicates("visitorid")
    central = UserData(train, roster.visitorid)
    clients = {str(key): UserData(train[train.client_id == key], part.visitorid)
               for key, part in roster.groupby("client_id", sort=True)}
    evaluation = {split: (transform(frames[split]), frames[split][TARGET].to_numpy())
                  for split in ["val", "test"]}
    central_spec = {"central": schedule(central.n_users, config.batch_users, config.epochs)}
    fed_specs = {key: schedule(d.n_users, config.local_batch_users,
                 config.rounds * config.local_epochs) for key, d in clients.items()}
    specs = {"Centralized": central_spec, "FedAvg": fed_specs, "FedProx": fed_specs}
    sigmas = {(algorithm, target): calibrate(list(specs[algorithm].values()), target, config.delta)
              for algorithm in ALGORITHMS for target in args.epsilon_targets}
    print(f"Train: {len(frames['train']):,} sessions, {central.n_users:,} users, {len(clients)} clients.", flush=True)
    rows, ledger, history, models = [], [], [], []
    # Sequential compose across experiments; parallel compose over fixed clients.
    joint = {key: RDPAccountant() for key in clients}
    for algorithm in ALGORITHMS:
        settings = [("no_dp", math.inf, 0.0), ("clip_only", math.inf, 0.0)] + [
            ("dp", target, sigmas[algorithm, target]) for target in args.epsilon_targets]
        for mode, target, sigma in settings:
            for replicate in range(1, args.replicates + 1):
                print(f"{algorithm} {mode} target={target:g} sigma={sigma:.4f} run={replicate} ...", flush=True)
                row, entries, log, accounts, parameters = train_experiment(algorithm, mode,
                    sigma, target, replicate, central, clients, specs, config, evaluation)
                rows.append(row)
                ledger.extend(entries)
                history.extend(log)
                models.append(parameters)
                if mode == "dp":
                    for client in clients:
                        account = accounts["central"] if algorithm == "Centralized" else accounts[client]
                        joint[client].history.extend(account.history)
                pd.DataFrame(rows).to_csv(out / "ml5_results_per_run.csv", index=False)
                print(f"  PR-AUC={row['test_pr_auc']:.4f}, F1={row['test_f1']:.4f}, epsilon={row['epsilon']:.4f}, {row['elapsed_seconds']:.1f}s", flush=True)
    results = pd.DataFrame(rows)
    table = aggregate_results(results)
    table.to_csv(out / "ml5_results_summary.csv", index=False)
    pd.DataFrame(ledger).to_csv(out / "ml5_privacy_ledger.csv", index=False)
    pd.DataFrame(history).to_csv(out / "ml5_round_schedule.csv", index=False)
    checks = statistical_checks(results)
    checks.to_csv(out / "ml5_statistical_checks.csv", index=False)
    select_privacy_configurations(table).to_csv(out / "ml5_privacy_choices.csv", index=False)
    composition = max(accountant_epsilon(a, config.delta) for a in joint.values())
    summary = {"config": asdict(config), "epsilon_targets": args.epsilon_targets,
        "replicates": args.replicates, "train_sessions": len(frames["train"]),
        "val_sessions": len(frames["val"]), "test_sessions": len(frames["test"]),
        "train_users": central.n_users, "num_clients": len(clients),
        "max_train_sessions_per_user": int(central.counts.max()),
        "features": FEATURES, "feature_caps": CAPS.tolist(), "target": TARGET,
        "randomness_mode": "public deterministic seeds (NOT a DP release)" if config.deterministic
             else "independent undisclosed PRNG streams (not cryptographically audited)",
        "privacy_unit": "all TRAIN sessions per visitorid",
        "roster_source": str(args.public_roster) if args.public_roster else
            "extracted from benchmark train; assumed public, NOT privatized",
        "adjacency": "add/remove one user's contribution in a fixed public roster slot",
        "accountant": "Opacus RDP, Poisson user sampling, exact executed steps",
        "joint_dp_models_accounted_epsilon": composition,
        "joint_delta": config.delta,
        "release_scope": "model mechanisms only; whole benchmark NOT certified DP",
        "unprotected": ["raw cleaning/session construction", "roster/client sizes",
            "validation/test metrics", "no_dp and clip_only baselines"],
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "torch": torch.__version__, "opacus": opacus.__version__},
        "schedules": specs, "parameters": models}
    (out / "ml5_experiment_summary.json").write_text(json.dumps(summary, ensure_ascii=False,
        indent=2, allow_nan=False), encoding="utf-8")
    plot_results(table, out, config.delta)
    (out / "ML5_Privacy_Utility_Report.md").write_text(
        make_report(table, summary, pd.DataFrame(ledger), checks), encoding="utf-8")
    print(table[["algorithm", "mode", "target_epsilon", "epsilon", "test_pr_auc_mean",
        "test_pr_auc_std", "pr_auc_retention_pct"]].to_string(index=False), flush=True)
    print(f"Saved ML5 results: {out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
