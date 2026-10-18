#!/usr/bin/env python
"""
Run ML4 Federated Learning experiments on the prepared session-level dataset.

Experiments:
  1. Centralized Training
  2. FedAvg
  3. FedProx

The task is session-level purchase prediction using the ML4 dataset created by
prepare_ml4_ml5_datasets.py.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "observed_duration_sec",
    "observed_n_events",
    "observed_n_views",
    "observed_n_addtocart",
    "observed_n_unique_items",
    "has_observed_addtocart",
    "view_to_cart_ratio",
]

TARGET_COLUMN = "label_has_purchase"


def writable_path(path: Path) -> Path:
    if not path.exists():
        return path
    try:
        with path.open("a", encoding="utf-8"):
            return path
    except PermissionError:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return path.with_name(f"{path.stem}_{stamp}{path.suffix}")


def write_csv_safe(df: pd.DataFrame, path: Path) -> Path:
    target = writable_path(path)
    df.to_csv(target, index=False)
    return target


def write_text_safe(text: str, path: Path) -> Path:
    target = writable_path(path)
    target.write_text(text, encoding="utf-8")
    return target


@dataclass
class FLConfig:
    rounds: int = 20
    local_epochs: int = 3
    lr: float = 0.05
    fedprox_mu: float = 0.1
    threshold: float = 0.5


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -35, 35)
    return 1.0 / (1.0 + np.exp(-z))


def load_split(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def build_xy(df: pd.DataFrame, scaler: StandardScaler | None = None, fit: bool = False):
    x = df[FEATURE_COLUMNS].replace([np.inf, -np.inf], 0).fillna(0).astype("float64")
    if fit:
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)
    else:
        if scaler is None:
            raise ValueError("scaler is required when fit=False")
        x_scaled = scaler.transform(x)
    y = df[TARGET_COLUMN].astype("int8").to_numpy()
    return x_scaled.astype("float64"), y, scaler


def evaluate_scores(y_true: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> dict:
    pred = (scores >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "positive_rate_pred": float(pred.mean()),
    }


def train_centralized(x_train, y_train, x_val, y_val, x_test, y_test, threshold: float) -> tuple[dict, dict]:
    clf = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-5,
        class_weight="balanced",
        max_iter=20,
        tol=1e-4,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(x_train, y_train)
    val_scores = clf.predict_proba(x_val)[:, 1]
    test_scores = clf.predict_proba(x_test)[:, 1]
    metrics = {
        "val": evaluate_scores(y_val, val_scores, threshold),
        "test": evaluate_scores(y_test, test_scores, threshold),
    }
    params = {
        "coef": clf.coef_.ravel().tolist(),
        "intercept": float(clf.intercept_[0]),
    }
    return metrics, params


def weighted_gradient(x: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, pos_weight: float):
    p = sigmoid(x @ w + b)
    sample_weight = np.where(y == 1, pos_weight, 1.0)
    err = (p - y) * sample_weight
    denom = max(float(sample_weight.sum()), 1.0)
    grad_w = x.T @ err / denom
    grad_b = float(err.sum() / denom)
    return grad_w, grad_b


def make_client_arrays(train: pd.DataFrame, scaler: StandardScaler):
    clients = []
    for client_id, part in train.groupby("client_id", sort=True):
        x, y, _ = build_xy(part, scaler=scaler, fit=False)
        clients.append((client_id, x, y))
    return clients


def train_federated(
    clients,
    x_val,
    y_val,
    x_test,
    y_test,
    config: FLConfig,
    algorithm: str,
    pos_weight: float,
):
    n_features = x_val.shape[1]
    w_global = np.zeros(n_features, dtype="float64")
    b_global = 0.0
    history = []

    for round_idx in range(1, config.rounds + 1):
        local_updates = []
        total_examples = 0

        for client_id, x_client, y_client in clients:
            if len(y_client) == 0:
                continue
            w_local = w_global.copy()
            b_local = float(b_global)

            for _ in range(config.local_epochs):
                grad_w, grad_b = weighted_gradient(
                    x_client, y_client, w_local, b_local, pos_weight
                )
                if algorithm == "FedProx":
                    grad_w = grad_w + config.fedprox_mu * (w_local - w_global)
                w_local -= config.lr * grad_w
                b_local -= config.lr * grad_b

            n = len(y_client)
            local_updates.append((n, w_local, b_local))
            total_examples += n

        w_global = sum(n * w for n, w, _ in local_updates) / total_examples
        b_global = float(sum(n * b for n, _, b in local_updates) / total_examples)

        val_scores = sigmoid(x_val @ w_global + b_global)
        val_metrics = evaluate_scores(y_val, val_scores, config.threshold)
        history.append(
            {
                "round": round_idx,
                "algorithm": algorithm,
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
        )

    test_scores = sigmoid(x_test @ w_global + b_global)
    val_scores = sigmoid(x_val @ w_global + b_global)
    metrics = {
        "val": evaluate_scores(y_val, val_scores, config.threshold),
        "test": evaluate_scores(y_test, test_scores, config.threshold),
    }
    params = {
        "coef": w_global.tolist(),
        "intercept": b_global,
    }
    return metrics, history, params


def flatten_metrics(name: str, metrics: dict) -> dict:
    row = {"model": name}
    for split_name in ["val", "test"]:
        for metric_name, value in metrics[split_name].items():
            row[f"{split_name}_{metric_name}"] = value
    return row


def plot_results(results: pd.DataFrame, history: pd.DataFrame,
                 client_meta: pd.DataFrame, output_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    colors = ["#147d75", "#345ca8", "#bd4363"]
    labels = results["model"].replace({"Centralized Logistic Regression": "Centralized"}).tolist()
    panels = [("test_pr_auc", "PR-AUC (average precision)"), ("test_f1", "F1"),
              ("test_roc_auc", "ROC-AUC"), ("test_balanced_accuracy", "Balanced accuracy"),
              ("test_precision", "Precision"), ("test_recall", "Recall")]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), layout="constrained")
    for ax, (column, title) in zip(axes.flat, panels):
        values = results[column].to_numpy()
        ax.bar(range(len(values)), values, color=colors[:len(values)], width=0.62)
        ax.set_xticks(range(len(values)), labels, fontsize=10)
        ax.set_title(title, fontsize=12)
        ax.set_ylim(0, min(1.08, max(0.1, float(values.max()) * 1.2)))
        ax.grid(axis="y", alpha=0.18)
        ax.set_axisbelow(True)
        for index, value in enumerate(values):
            ax.annotate(f"{value:.4f}", (index, value), xytext=(0, 5),
                        textcoords="offset points", ha="center", fontsize=10)
    fig.suptitle("ML4: test results (one training run)", fontsize=16)
    comparison = writable_path(output_dir / "ml4_model_comparison.png")
    fig.savefig(comparison, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    styles = {"FedAvg": (colors[1], "-", "o"), "FedProx": (colors[2], "--", "x")}
    for algorithm, part in history.groupby("algorithm", sort=False):
        part = part.sort_values("round")
        color, line, marker = styles[algorithm]
        for ax, column, title in zip(axes, ["val_pr_auc", "val_f1"],
                                     ["Validation PR-AUC", "Validation F1"]):
            ax.plot(part["round"], part[column], color=color, linestyle=line,
                    marker=marker, markevery=4, markersize=5, label=algorithm)
            ax.set(xlabel="Communication round", ylabel=title)
            ax.grid(alpha=0.2)
            ax.legend()
    fig.suptitle("ML4: FedAvg and FedProx training history", fontsize=15)
    convergence = writable_path(output_dir / "ml4_training_curves.png")
    fig.savefig(convergence, dpi=180)
    plt.close(fig)

    part = client_meta.sort_values("client_id")
    positions = np.arange(len(part))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    axes[0].bar(positions, part["n_train_sessions"], color=colors[0])
    axes[0].set(ylabel="Training sessions", title="Training data per client", xlabel="Client index")
    axes[1].bar(positions, 100 * part["purchase_rate"], color=colors[2])
    axes[1].set(ylabel="Purchase sessions (%)", title="Purchase rate (all three splits)", xlabel="Client index")
    ticks = sorted(set(range(0, len(part), 10)) | {len(part) - 1})
    for ax in axes:
        ax.set_xticks(ticks)
        ax.grid(axis="y", alpha=0.18)
        ax.set_axisbelow(True)
    fig.suptitle(f"ML4: {len(part)} simulated clients", fontsize=15)
    clients = writable_path(output_dir / "ml4_client_distribution.png")
    fig.savefig(clients, dpi=180)
    plt.close(fig)
    return [comparison, convergence, clients]


def make_report(summary: dict, results: pd.DataFrame, client_meta: pd.DataFrame,
                config: FLConfig, history: pd.DataFrame | None = None) -> str:
    rows = results.set_index("model")
    centralized = rows.loc["Centralized Logistic Regression"]
    fedavg = rows.loc["FedAvg"]
    fedprox = rows.loc["FedProx"]

    def model_table(row) -> str:
        metrics = [("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC (average precision)"),
                   ("balanced_accuracy", "Balanced Accuracy"), ("precision", "Precision"),
                   ("recall", "Recall"), ("f1", "F1"), ("accuracy", "Accuracy")]
        return ("| Chỉ số | Validation | Test |\n| --- | ---: | ---: |\n" +
                "\n".join(f"| {label} | {row['val_' + metric]:.4f} | {row['test_' + metric]:.4f} |"
                          for metric, label in metrics))

    def progress_text(algorithm: str) -> str:
        if history is None:
            return ""
        part = history[history["algorithm"] == algorithm].sort_values("round")
        if part.empty:
            return ""
        first, last = part.iloc[0], part.iloc[-1]
        return (f"Lịch sử validation cho thấy từ round {int(first['round'])} đến round "
                f"{int(last['round'])}, PR-AUC thay đổi từ {first['val_pr_auc']:.4f} "
                f"sang {last['val_pr_auc']:.4f}, F1 từ {first['val_f1']:.4f} sang "
                f"{last['val_f1']:.4f} và Precision từ {first['val_precision']:.4f} sang "
                f"{last['val_precision']:.4f}. Recall thay đổi từ {first['val_recall']:.4f} "
                f"sang {last['val_recall']:.4f}. Các giá trị này được lấy từ "
                "`ml4_training_history.csv`, không phải lịch sử test.")

    table = results[
        [
            "model",
            "test_roc_auc",
            "test_pr_auc",
            "test_balanced_accuracy",
            "test_precision",
            "test_recall",
            "test_f1",
        ]
    ].copy()
    for col in table.columns:
        if col != "model":
            table[col] = table[col].map(lambda x: f"{x:.4f}")

    return f"""# ML4 Federated Learning Results

## 4.3.5.1. Thiết lập client

Dữ liệu ML4 được xây dựng ở mức session từ tập `user_item_event_final.csv`. Mỗi dòng biểu diễn một phiên truy cập và nhãn `label_has_purchase` cho biết session có phát sinh mua hàng hay không. Để mô phỏng Federated Learning, mỗi `visitorid` được ánh xạ cố định vào một `client_id`, nhờ đó toàn bộ session của cùng một khách hàng luôn thuộc cùng một client.

Thiết lập trong lần chạy này:

- Số client: {summary['num_clients']}
- Số session train: {summary['train_rows']:,}
- Số session validation: {summary['val_rows']:,}
- Số session test: {summary['test_rows']:,}
- Số session nhỏ nhất trên một client (tổng ba split): {int(client_meta['n_sessions'].min()):,}
- Số session lớn nhất trên một client (tổng ba split): {int(client_meta['n_sessions'].max()):,}
- Tỷ lệ purchase trung bình theo client (tổng ba split): {client_meta['purchase_rate'].mean():.4f}

![Phân bố client](ml4_client_distribution.png)

## 4.3.5.2. Centralized Training

Centralized Training dùng toàn bộ tập train để huấn luyện một mô hình Logistic Regression. Đây là baseline tập trung, dùng để so sánh với FedAvg và FedProx. Mô hình chỉ sử dụng các đặc trưng quan sát được trước nhãn mua hàng như số sự kiện quan sát được, số lượt xem, số lượt thêm giỏ hàng, số sản phẩm khác nhau, thời lượng quan sát và tỷ lệ view-to-cart.

Mô hình được huấn luyện trên {summary['train_rows']:,} phiên bằng
`SGDClassifier(loss="log_loss")`, với class weight `balanced`, regularization L2
(α = 0.00001), `max_iter = 20`, `tol = 0.0001` và `random_state = 42`.
`StandardScaler` được fit trên train rồi dùng chung cho validation/test.
Ngưỡng phân loại trong lần chạy này là {config.threshold:g}.

Kết quả của baseline trên hai tập đánh giá:

{model_table(centralized)}

Trên test, ROC-AUC đạt {centralized['test_roc_auc']:.4f} và PR-AUC đạt
{centralized['test_pr_auc']:.4f}. Recall = {centralized['test_recall']:.4f} cho biết
mô hình nhận diện khoảng {100 * centralized['test_recall']:.2f}% phiên mua hàng,
trong khi Precision = {centralized['test_precision']:.4f} cho biết chỉ khoảng
{100 * centralized['test_precision']:.2f}% phiên được dự đoán mua thực sự có mua.
Baseline phát hiện nhiều phiên positive nhưng còn nhiều dự đoán dương tính giả,
thể hiện qua F1 = {centralized['test_f1']:.4f}.

Mô hình gán nhãn mua cho {100 * centralized['test_positive_rate_pred']:.2f}% phiên
test, trong khi tỷ lệ mua thực tế chỉ {100 * summary['test_positive_rate']:.2f}%.
Accuracy = {centralized['test_accuracy']:.4f} vì vậy cần đọc cùng Balanced Accuracy
= {centralized['test_balanced_accuracy']:.4f}, PR-AUC và F1. ROC-AUC validation =
{centralized['val_roc_auc']:.4f} so với test = {centralized['test_roc_auc']:.4f};
PR-AUC validation = {centralized['val_pr_auc']:.4f} so với test =
{centralized['test_pr_auc']:.4f}. Các chỉ số này khá gần nhau trong lần chạy đã lưu.

## 4.3.5.3. FedAvg

FedAvg được mô phỏng bằng cách huấn luyện mô hình cục bộ trên từng client, sau đó server tổng hợp tham số theo trung bình có trọng số dựa trên số mẫu của mỗi client. Số vòng huấn luyện là {config.rounds}, learning rate là {config.lr}, và mỗi client chạy {config.local_epochs} local epoch trong một vòng.

Tất cả {summary['num_clients']} client tham gia mỗi round. Trong script, mỗi local
epoch là một bước gradient dùng toàn bộ dữ liệu cục bộ. Mô hình dùng cùng bảy đặc
trưng/scaler với baseline, với trọng số purchase {summary['pos_weight']:.4f}
so với trọng số 1 của lớp không mua.

Kết quả của FedAvg sau round cuối:

{model_table(fedavg)}

{progress_text('FedAvg')}

Trên test, FedAvg đạt PR-AUC = {fedavg['test_pr_auc']:.4f}, F1 =
{fedavg['test_f1']:.4f} và Precision = {fedavg['test_precision']:.4f}. Hiệu số
**FedAvg − Centralized** là {fedavg['test_pr_auc'] - centralized['test_pr_auc']:+.4f}
cho PR-AUC và {fedavg['test_f1'] - centralized['test_f1']:+.4f} cho F1 trên thang 0 đến 1.
Precision chuyển từ {centralized['test_precision']:.4f} ở centralized sang
{fedavg['test_precision']:.4f} ở FedAvg, còn Recall từ {centralized['test_recall']:.4f}
sang {fedavg['test_recall']:.4f}. Kết quả thể hiện sự đánh đổi giữa độ chính xác của
các dự đoán mua và khả năng bao phủ các phiên có mua hàng.

Phương án huấn luyện theo client đạt chất lượng dự đoán tốt trong mô phỏng này.
Tuy nhiên, centralized và FedAvg dùng lịch tối ưu khác nhau; chênh lệch quan sát
được chưa tách riêng ảnh hưởng của thuật toán FL khỏi cấu hình tối ưu.

## 4.3.5.4. FedProx

FedProx sử dụng cùng thiết lập với FedAvg nhưng bổ sung proximal term để hạn chế mô hình cục bộ lệch quá xa mô hình global. Hệ số proximal trong lần chạy này là `{config.fedprox_mu}`.

Thiết lập giữ nguyên {summary['num_clients']} client, {config.rounds} round,
{config.local_epochs} local epoch và learning rate {config.lr:g} như FedAvg.
Trong bản ML4 này, proximal term được áp dụng cho vector weight, không áp dụng
cho bias. Mọi client đều tham gia và server tổng hợp theo số phiên train.

Kết quả của FedProx sau round cuối:

{model_table(fedprox)}

{progress_text('FedProx')}

Trên test, FedProx đạt ROC-AUC = {fedprox['test_roc_auc']:.4f}, PR-AUC =
{fedprox['test_pr_auc']:.4f}, F1 = {fedprox['test_f1']:.4f} và Recall =
{fedprox['test_recall']:.4f}. Các giá trị này rất gần FedAvg. Để tránh việc làm tròn
bốn chữ số che mất chênh lệch, hiệu số **FedProx − FedAvg** trên test được ghi thêm
ở độ chính xác sáu chữ số: PR-AUC = {fedprox['test_pr_auc'] - fedavg['test_pr_auc']:+.6f},
F1 = {fedprox['test_f1'] - fedavg['test_f1']:+.6f} và ROC-AUC =
{fedprox['test_roc_auc'] - fedavg['test_roc_auc']:+.6f}.

Với μ = {config.fedprox_mu:g} và cách chia client hiện tại, chưa quan sát được lợi
thế chất lượng dự đoán rõ của FedProx so với FedAvg. Các chênh lệch rất nhỏ trong
một lần chạy chưa đủ để kết luận phương pháp nào vượt trội về thống kê. Mức độ
không đồng nhất giữa client và lựa chọn μ có thể ảnh hưởng kết quả, nhưng lần thử
này chưa quét nhiều μ hay thay đổi mức non-IID để xác định nguyên nhân.

## 4.3.5.5. So sánh kết quả

| Model | ROC-AUC | PR-AUC | Balanced Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join('| ' + ' | '.join(map(str, row)) + ' |' for row in table.to_numpy())}

Ghi chú: vì tỷ lệ session mua hàng rất thấp, Accuracy tổng thể dễ gây hiểu nhầm. Do đó phần so sánh ưu tiên ROC-AUC, PR-AUC, Recall, F1 và Balanced Accuracy.

![So sánh kết quả test](ml4_model_comparison.png)

![Lịch sử huấn luyện trên validation](ml4_training_curves.png)

Bảng và biểu đồ lấy từ một lần chạy ML4 đã lưu, không phải trung bình nhiều seed.
Đường huấn luyện dùng validation qua từng round; FedAvg và FedProx gần trùng nhau
nên dùng nét liền và nét đứt để phân biệt. Hiện không có log theo epoch của mô hình
centralized, vì vậy không vẽ đường huấn luyện giả cho baseline này.
Centralized và FL có lịch tối ưu khác nhau trong script hiện tại, nên kết quả phản
ánh các cấu hình đã chạy và chưa chứng minh FL nói chung tốt hơn huấn luyện tập trung.

## 4.3.5.6. Trả lời RQ4

Kết quả cho thấy Federated Learning có thể huấn luyện mô hình dự đoán mua hàng mà không cần gom toàn bộ dữ liệu raw event về một nơi trong quá trình mô phỏng client. FedAvg và FedProx sử dụng cùng dữ liệu đã chia theo client, chỉ tổng hợp tham số mô hình ở phía server. So với Centralized Training, hiệu quả của FL phụ thuộc vào mức độ không đồng nhất giữa client, số vòng truyền thông, learning rate và số mẫu trên từng client.

Trong thiết lập này, FedAvg và FedProx cho kết quả gần như tương đương. Chưa quan
sát được lợi thế rõ của FedProx với bộ đặc trưng session, μ và cách chia client hiện
tại. Lần chạy này chưa quét μ hoặc thay đổi mức non-IID, nên chưa thể xác định
nguyên nhân của sự tương đồng. Cả hai phương pháp FL đạt kết quả cạnh tranh với
baseline tập trung trên ROC-AUC, PR-AUC, Balanced Accuracy và F1; việc diễn giải
chênh lệch vẫn cần xét đến lịch tối ưu khác nhau của các mô hình.

Kết luận cho RQ4: Federated Learning là khả thi trong bối cảnh dữ liệu thương mại điện tử đã chia theo khách hàng, vì mô hình vẫn có thể được huấn luyện thông qua tổng hợp tham số mà không cần tập trung raw data của từng client. Tuy nhiên, chất lượng mô hình phụ thuộc vào cách chia client, mức độ non-IID, số vòng huấn luyện, learning rate và số local epoch.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ML4 Centralized/FedAvg/FedProx experiments.")
    parser.add_argument("--data-dir", default=str(Path(__file__).parent), help="Directory containing ML4 csv.gz files")
    parser.add_argument("--output-dir", default=str(Path(__file__).parent / "ml4_results"), help="Directory for ML4 result artifacts")
    parser.add_argument("--plots-only", action="store_true", help="Read saved results and regenerate figures/report without training")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--local-epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--fedprox-mu", type=float, default=0.1)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.plots_only:
        results = pd.read_csv(output_dir / "ml4_results_summary.csv")
        history = pd.read_csv(output_dir / "ml4_training_history.csv")
        summary = json.loads((output_dir / "ml4_experiment_summary.json").read_text(encoding="utf-8"))
        client_meta = pd.read_csv(data_dir / "client_partition_metadata.csv")
        config = FLConfig(**summary["config"])
        paths = plot_results(results, history, client_meta, output_dir)
        paths.append(write_text_safe(make_report(summary, results, client_meta, config, history),
                                    output_dir / "ML4_Federated_Learning_Report.md"))
        print("Updated ML4 figures and report:")
        for path in paths:
            print(path.resolve())
        return

    config = FLConfig(
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        lr=args.lr,
        fedprox_mu=args.fedprox_mu,
        threshold=args.threshold,
    )

    train = load_split(data_dir / "ml4_train.csv.gz")
    val = load_split(data_dir / "ml4_val.csv.gz")
    test = load_split(data_dir / "ml4_test.csv.gz")
    client_meta = pd.read_csv(data_dir / "client_partition_metadata.csv")

    x_train, y_train, scaler = build_xy(train, fit=True)
    x_val, y_val, _ = build_xy(val, scaler=scaler)
    x_test, y_test, _ = build_xy(test, scaler=scaler)

    positives = int(y_train.sum())
    negatives = int(len(y_train) - positives)
    pos_weight = negatives / max(positives, 1)

    centralized_metrics, centralized_params = train_centralized(
        x_train, y_train, x_val, y_val, x_test, y_test, config.threshold
    )

    clients = make_client_arrays(train, scaler)
    fedavg_metrics, fedavg_history, fedavg_params = train_federated(
        clients,
        x_val,
        y_val,
        x_test,
        y_test,
        config,
        algorithm="FedAvg",
        pos_weight=pos_weight,
    )
    fedprox_metrics, fedprox_history, fedprox_params = train_federated(
        clients,
        x_val,
        y_val,
        x_test,
        y_test,
        config,
        algorithm="FedProx",
        pos_weight=pos_weight,
    )

    result_rows = [
        flatten_metrics("Centralized Logistic Regression", centralized_metrics),
        flatten_metrics("FedAvg", fedavg_metrics),
        flatten_metrics("FedProx", fedprox_metrics),
    ]
    results = pd.DataFrame(result_rows)
    history = pd.DataFrame(fedavg_history + fedprox_history)

    results_path = write_csv_safe(results, output_dir / "ml4_results_summary.csv")
    history_path = write_csv_safe(history, output_dir / "ml4_training_history.csv")

    summary = {
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "train_rows": int(len(train)),
        "val_rows": int(len(val)),
        "test_rows": int(len(test)),
        "num_clients": int(train["client_id"].nunique()),
        "train_positive_rate": float(y_train.mean()),
        "val_positive_rate": float(y_val.mean()),
        "test_positive_rate": float(y_test.mean()),
        "pos_weight": float(pos_weight),
        "config": {
            "rounds": config.rounds,
            "local_epochs": config.local_epochs,
            "lr": config.lr,
            "fedprox_mu": config.fedprox_mu,
            "threshold": config.threshold,
        },
        "metrics": result_rows,
        "parameters": {
            "centralized": centralized_params,
            "fedavg": fedavg_params,
            "fedprox": fedprox_params,
        },
    }
    summary_path = write_text_safe(
        json.dumps(summary, indent=2, ensure_ascii=False),
        output_dir / "ml4_experiment_summary.json",
    )
    figure_paths = plot_results(results, history, client_meta, output_dir)
    report_path = write_text_safe(
        make_report(summary, results, client_meta, config, history),
        output_dir / "ML4_Federated_Learning_Report.md",
    )

    print(results.to_string(index=False))
    print("\nSaved ML4 artifacts:")
    for path in [results_path, history_path, summary_path, report_path, *figure_paths]:
        print(f"- {path.resolve()}")


if __name__ == "__main__":
    main()
