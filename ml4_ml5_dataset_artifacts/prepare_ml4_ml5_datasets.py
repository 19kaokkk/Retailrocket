#!/usr/bin/env python
"""
Prepare ML4/ML5 session-level datasets from cleaned RetailRocket events.

Input:
  user_item_event_final.csv with columns:
    visitorid, session_id, datetime, event, itemid, transactionid

Outputs:
  - session_level_ml4.csv.gz
  - ml4_train.csv.gz, ml4_val.csv.gz, ml4_test.csv.gz
  - session_split_client_index.csv.gz
  - client_partition_metadata.csv
  - dataset_summary.json
  - README_ML4_ML5_dataset.md
  - assumptions_privacy_notes.md

The dataset is designed for ML4 FedAvg/FedProx and can be reused for ML5
DP-SGD experiments. It does not implement or certify differential privacy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "visitorid",
    "session_id",
    "datetime",
    "event",
    "itemid",
    "transactionid",
}

EVENT_ORDER = {
    "view": 1,
    "addtocart": 2,
    "transaction": 3,
}


def stable_client_id(visitor_id: object, num_clients: int) -> str:
    """Assign every visitor to exactly one simulated federated client."""
    digest = hashlib.md5(str(visitor_id).encode("utf-8")).hexdigest()
    bucket = int(digest[:12], 16) % num_clients
    return f"client_{bucket:04d}"


def join_values(values: Iterable[object]) -> str:
    return "|".join(str(v) for v in values if pd.notna(v))


def build_session_rows(df: pd.DataFrame, min_observed_events: int) -> tuple[list[dict], int]:
    rows: list[dict] = []
    excluded = 0
    current_session = None
    state = None

    def flush() -> None:
        nonlocal excluded, state
        if state is None:
            return
        if state["observed_n_events"] < min_observed_events:
            excluded += 1
            return
        observed_duration = 0.0
        if state["observed_start"] is not None and state["observed_end"] is not None:
            observed_duration = max(
                0.0,
                (state["observed_end"] - state["observed_start"]).total_seconds(),
            )
        session_duration = max(
            0.0,
            (state["session_end"] - state["session_start"]).total_seconds(),
        )
        view_to_cart_ratio = (
            state["observed_n_addtocart"] / state["observed_n_views"]
            if state["observed_n_views"] > 0
            else 0.0
        )
        rows.append(
            {
                "session_id": state["session_id"],
                "visitorid": state["visitorid"],
                "session_start": state["session_start"],
                "session_end": state["session_end"],
                "observed_start": state["observed_start"],
                "observed_end": state["observed_end"],
                "session_duration_sec": session_duration,
                "observed_duration_sec": observed_duration,
                "full_n_events": state["full_n_events"],
                "observed_n_events": state["observed_n_events"],
                "observed_n_views": state["observed_n_views"],
                "observed_n_addtocart": state["observed_n_addtocart"],
                "observed_n_unique_items": len(state["observed_items"]),
                "has_observed_addtocart": int(state["observed_n_addtocart"] > 0),
                "view_to_cart_ratio": view_to_cart_ratio,
                "first_observed_itemid": state["first_observed_itemid"],
                "last_observed_itemid": state["last_observed_itemid"],
                "observed_event_sequence": "|".join(state["observed_events"]),
                "observed_item_sequence": "|".join(state["observed_itemids"]),
                "label_has_purchase": int(state["full_n_transactions"] > 0),
            }
        )

    for rec in df.itertuples(index=False):
        session_id = rec.session_id
        event = rec.event
        itemid = rec.itemid
        timestamp = rec.datetime

        if session_id != current_session:
            flush()
            current_session = session_id
            state = {
                "session_id": session_id,
                "visitorid": rec.visitorid,
                "session_start": timestamp,
                "session_end": timestamp,
                "full_n_events": 0,
                "full_n_transactions": 0,
                "seen_purchase": False,
                "observed_start": None,
                "observed_end": None,
                "observed_n_events": 0,
                "observed_n_views": 0,
                "observed_n_addtocart": 0,
                "observed_items": set(),
                "first_observed_itemid": np.nan,
                "last_observed_itemid": np.nan,
                "observed_events": [],
                "observed_itemids": [],
            }

        state["session_end"] = timestamp
        state["full_n_events"] += 1
        if event == "transaction":
            state["full_n_transactions"] += 1
            state["seen_purchase"] = True
            continue

        if not state["seen_purchase"]:
            if state["observed_start"] is None:
                state["observed_start"] = timestamp
                state["first_observed_itemid"] = itemid
            state["observed_end"] = timestamp
            state["observed_n_events"] += 1
            state["observed_n_views"] += int(event == "view")
            state["observed_n_addtocart"] += int(event == "addtocart")
            state["observed_items"].add(itemid)
            state["last_observed_itemid"] = itemid
            state["observed_events"].append(str(event))
            state["observed_itemids"].append(str(itemid))

    flush()
    return rows, excluded


def make_markdown_report(summary: dict, args: argparse.Namespace) -> str:
    split_lines = []
    for split_name in ["train", "val", "test"]:
        item = summary["splits"][split_name]
        split_lines.append(
            f"| {split_name} | {item['sessions']:,} | {item['positive_sessions']:,} | "
            f"{item['positive_rate']:.4f} | {item['unique_visitors']:,} | "
            f"{item['date_min']} | {item['date_max']} |"
        )

    return f"""# ML4 and ML5 Dataset Preparation

This folder contains a reproducible session-level dataset derived from the cleaned RetailRocket event file `user_item_event_final.csv`.

## Purpose

- **ML4 - Federated Learning:** use the train split and `client_id` assignments to simulate client-side training for FedAvg and FedProx.
- **ML5 - Privacy:** reuse the same client/session partitioning for DP-SGD or FedProx + DP-SGD experiments so privacy experiments are comparable to ML4.

## Main assumption

The prediction task is **session-level purchase prediction**: given the observed behavior in a session, predict whether the session eventually contains a purchase event.

To reduce label leakage, features are computed only from the observed prefix:

- If a session has a `transaction`, the observed prefix contains events strictly before the first transaction.
- If a session has no `transaction`, the observed prefix contains all events in the session.
- Sessions with fewer than `{args.min_observed_events}` observed event(s) are excluded from the modeling dataset.

The target column is `label_has_purchase`.

## Output files

| File | Use |
| --- | --- |
| `session_level_ml4.csv.gz` | Full modeling dataset after leakage-safe prefix filtering |
| `ml4_train.csv.gz` | Chronological train split |
| `ml4_val.csv.gz` | Chronological validation split |
| `ml4_test.csv.gz` | Chronological test split |
| `session_split_client_index.csv.gz` | Session-to-split and session-to-client mapping |
| `client_partition_metadata.csv` | Per-client counts and purchase rates for FL weighting and non-IID checks |
| `dataset_summary.json` | Machine-readable verification summary |
| `assumptions_privacy_notes.md` | Privacy and DP caveats for ML5 |

## Split policy

Splits are chronological by `session_start` with ratios:

- Train: `{args.train_frac:.2f}`
- Validation: `{args.val_frac:.2f}`
- Test: `{1 - args.train_frac - args.val_frac:.2f}`

This avoids training on future sessions before validating or testing on earlier sessions.

## Client partition policy

Each `visitorid` is assigned to exactly one `client_id` using a deterministic hash. This preserves visitor/session grouping across runs and keeps all sessions from the same visitor on the same simulated client. The default number of clients is configurable; this run used `{args.num_clients}` clients.

## Verification summary

| Split | Sessions | Positive sessions | Positive rate | Unique visitors | Min date | Max date |
| --- | ---: | ---: | ---: | ---: | --- | --- |
{chr(10).join(split_lines)}

Overall modeling rows: **{summary['modeling_rows']:,}**

Source rows read: **{summary['source_rows']:,}**

Excluded zero/short-prefix sessions: **{summary['excluded_sessions_due_to_prefix']:,}**

Sequence columns included in this run: **{summary['include_sequences']}**
"""


def make_privacy_notes(args: argparse.Namespace) -> str:
    return f"""# ML5 Privacy Notes and Required Audit Items

This preparation workflow creates data partitions that can be reused for DP-SGD and FedProx + DP-SGD experiments, but it does **not** implement differential privacy by itself.

## What this artifact does

- Preserves a deterministic mapping from `visitorid` to `client_id`.
- Keeps each `session_id` wholly inside one split and one client.
- Provides per-client metadata needed for FedAvg/FedProx weighting.
- Provides the same train/validation/test protocol for non-private and private experiments.

## What must still be implemented for ML5

Before claiming any privacy guarantee, the training code must explicitly define and audit:

- The privacy unit: event-level, session-level, visitor-level, or client-level.
- The clipping strategy and clipping norm.
- The noise multiplier or target epsilon/delta.
- The sampling procedure used by the DP accountant.
- The DP accountant implementation, for example RDP or moments accountant.
- Whether DP is applied centrally, locally on each client, or during federated aggregation.
- Whether validation/test data is used only for evaluation and never for training updates.

## Important warning

Do **not** claim user-level DP from these files alone. The partitioning is visitor-preserving, but user-level DP requires the training algorithm, sampling, clipping, noise addition, accounting, and reporting to be implemented and verified.

For this run, each visitor was assigned to one of `{args.num_clients}` simulated clients. That is useful for FL experiments, but it is not a privacy guarantee.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create ML4/ML5 session-level datasets from cleaned RetailRocket events."
    )
    parser.add_argument("--input", required=True, help="Path to user_item_event_final.csv")
    parser.add_argument("--output-dir", required=True, help="Directory for generated artifacts")
    parser.add_argument("--num-clients", type=int, default=50, help="Number of simulated FL clients")
    parser.add_argument("--train-frac", type=float, default=0.80, help="Chronological train ratio")
    parser.add_argument("--val-frac", type=float, default=0.10, help="Chronological validation ratio")
    parser.add_argument(
        "--min-observed-events",
        type=int,
        default=1,
        help="Minimum number of observed pre-label events required for modeling",
    )
    parser.add_argument(
        "--include-sequences",
        action="store_true",
        help="Also write pipe-delimited observed event/item sequences. This is slower and creates larger files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.num_clients <= 0:
        raise ValueError("--num-clients must be positive")
    if not (0 < args.train_frac < 1 and 0 < args.val_frac < 1):
        raise ValueError("--train-frac and --val-frac must be between 0 and 1")
    if args.train_frac + args.val_frac >= 1:
        raise ValueError("--train-frac + --val-frac must be less than 1")

    df = pd.read_csv(input_path, parse_dates=["datetime"])
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    source_rows = int(len(df))
    df = df[
        ["visitorid", "session_id", "datetime", "event", "itemid", "transactionid"]
    ].copy()
    df["event"] = df["event"].astype(str).str.lower().str.strip()
    df["event_rank"] = df["event"].map(EVENT_ORDER).fillna(99).astype(int)
    df = df.sort_values(["session_id", "datetime", "event_rank", "itemid"], kind="mergesort")
    df["is_transaction"] = df["event"].eq("transaction").astype("int8")
    df["is_view"] = df["event"].eq("view").astype("int8")
    df["is_addtocart"] = df["event"].eq("addtocart").astype("int8")

    first_purchase = (
        df.loc[df["event"].eq("transaction")]
        .groupby("session_id", sort=False)["datetime"]
        .min()
        .rename("first_purchase_time")
    )
    df = df.merge(first_purchase, on="session_id", how="left")
    df["is_observed_prefix"] = df["first_purchase_time"].isna() | (
        df["datetime"] < df["first_purchase_time"]
    )

    full_session = df.groupby("session_id", sort=False).agg(
        visitorid=("visitorid", "first"),
        session_start=("datetime", "min"),
        session_end=("datetime", "max"),
        full_n_events=("event", "size"),
        full_n_transactions=("is_transaction", "sum"),
    )
    full_session["label_has_purchase"] = (full_session["full_n_transactions"] > 0).astype(int)

    observed = df.loc[df["is_observed_prefix"]].copy()
    observed_session = observed.groupby("session_id", sort=False).agg(
        observed_start=("datetime", "min"),
        observed_end=("datetime", "max"),
        observed_n_events=("event", "size"),
        observed_n_views=("is_view", "sum"),
        observed_n_addtocart=("is_addtocart", "sum"),
        observed_n_unique_items=("itemid", "nunique"),
        first_observed_itemid=("itemid", "first"),
        last_observed_itemid=("itemid", "last"),
    )
    sequence_columns: list[str] = []
    if args.include_sequences:
        observed["event_for_sequence"] = observed["event"].astype(str)
        observed["item_for_sequence"] = observed["itemid"].astype(str)
        observed_sequences = observed.groupby("session_id", sort=False).agg(
            observed_event_sequence=("event_for_sequence", "|".join),
            observed_item_sequence=("item_for_sequence", "|".join),
        )
        observed_session = observed_session.join(observed_sequences, how="left")
        sequence_columns = ["observed_event_sequence", "observed_item_sequence"]

    session = full_session.join(observed_session, how="left")
    session["observed_n_events"] = session["observed_n_events"].fillna(0).astype(int)
    for col in ["observed_n_views", "observed_n_addtocart", "observed_n_unique_items"]:
        session[col] = session[col].fillna(0).astype(int)
    for col in sequence_columns:
        session[col] = session[col].fillna("")
    session["observed_duration_sec"] = (
        session["observed_end"] - session["observed_start"]
    ).dt.total_seconds().fillna(0).clip(lower=0)
    session["session_duration_sec"] = (
        session["session_end"] - session["session_start"]
    ).dt.total_seconds().fillna(0).clip(lower=0)
    session["has_observed_addtocart"] = (session["observed_n_addtocart"] > 0).astype(int)
    session["view_to_cart_ratio"] = np.where(
        session["observed_n_views"] > 0,
        session["observed_n_addtocart"] / session["observed_n_views"],
        0.0,
    )
    excluded_sessions_due_to_prefix = int(
        (session["observed_n_events"] < args.min_observed_events).sum()
    )
    session = session.loc[session["observed_n_events"] >= args.min_observed_events].copy()
    session = session.reset_index()
    session["client_id"] = session["visitorid"].map(lambda v: stable_client_id(v, args.num_clients))
    session = session.sort_values(["session_start", "session_id"], kind="mergesort")

    n = len(session)
    train_end = int(n * args.train_frac)
    val_end = int(n * (args.train_frac + args.val_frac))
    session["split"] = "test"
    session.iloc[:train_end, session.columns.get_loc("split")] = "train"
    session.iloc[train_end:val_end, session.columns.get_loc("split")] = "val"

    output_columns = [
        "visitorid",
        "client_id",
        "session_start",
        "session_end",
        "observed_start",
        "observed_end",
        "session_duration_sec",
        "observed_duration_sec",
        "full_n_events",
        "observed_n_events",
        "observed_n_views",
        "observed_n_addtocart",
        "observed_n_unique_items",
        "has_observed_addtocart",
        "view_to_cart_ratio",
        "first_observed_itemid",
        "last_observed_itemid",
        "label_has_purchase",
        "split",
    ]
    output_columns = output_columns[:-2] + sequence_columns + output_columns[-2:]
    session_out = session[["session_id"] + output_columns]

    session_out.to_csv(output_dir / "session_level_ml4.csv.gz", index=False)
    for split_name in ["train", "val", "test"]:
        session_out.loc[session_out["split"].eq(split_name)].to_csv(
            output_dir / f"ml4_{split_name}.csv.gz", index=False
        )

    index_cols = [
        "session_id",
        "visitorid",
        "client_id",
        "split",
        "session_start",
        "label_has_purchase",
        "observed_n_events",
    ]
    session_out[index_cols].to_csv(output_dir / "session_split_client_index.csv.gz", index=False)

    client_meta = (
        session_out.groupby("client_id")
        .agg(
            n_sessions=("session_id", "size"),
            n_visitors=("visitorid", "nunique"),
            n_positive_sessions=("label_has_purchase", "sum"),
            purchase_rate=("label_has_purchase", "mean"),
            n_observed_events=("observed_n_events", "sum"),
            n_observed_views=("observed_n_views", "sum"),
            n_observed_addtocart=("observed_n_addtocart", "sum"),
            min_session_start=("session_start", "min"),
            max_session_start=("session_start", "max"),
        )
        .reset_index()
    )

    split_client_counts = (
        session_out.pivot_table(
            index="client_id",
            columns="split",
            values="session_id",
            aggfunc="count",
            fill_value=0,
        )
        .rename(columns=lambda c: f"n_{c}_sessions")
        .reset_index()
    )
    client_meta = client_meta.merge(split_client_counts, on="client_id", how="left")
    for col in ["n_train_sessions", "n_val_sessions", "n_test_sessions"]:
        if col not in client_meta:
            client_meta[col] = 0
        client_meta[col] = client_meta[col].astype(int)
    client_meta.to_csv(output_dir / "client_partition_metadata.csv", index=False)

    split_summary = {}
    for split_name, part in session_out.groupby("split"):
        split_summary[split_name] = {
            "sessions": int(len(part)),
            "positive_sessions": int(part["label_has_purchase"].sum()),
            "positive_rate": float(part["label_has_purchase"].mean()),
            "unique_visitors": int(part["visitorid"].nunique()),
            "unique_clients": int(part["client_id"].nunique()),
            "date_min": str(part["session_start"].min()),
            "date_max": str(part["session_start"].max()),
        }
    for split_name in ["train", "val", "test"]:
        split_summary.setdefault(
            split_name,
            {
                "sessions": 0,
                "positive_sessions": 0,
                "positive_rate": 0.0,
                "unique_visitors": 0,
                "unique_clients": 0,
                "date_min": "",
                "date_max": "",
            },
        )

    summary = {
        "source_file": str(input_path),
        "source_rows": source_rows,
        "modeling_rows": int(len(session_out)),
        "excluded_sessions_due_to_prefix": excluded_sessions_due_to_prefix,
        "num_clients_requested": args.num_clients,
        "num_clients_observed": int(session_out["client_id"].nunique()),
        "min_observed_events": args.min_observed_events,
        "include_sequences": bool(args.include_sequences),
        "label": "label_has_purchase",
        "splits": split_summary,
        "event_values": sorted(df["event"].dropna().unique().tolist()),
        "date_min": str(session_out["session_start"].min()),
        "date_max": str(session_out["session_start"].max()),
        "notes": [
            "Features are derived from events before the first transaction for positive sessions.",
            "The output does not implement differential privacy or provide a DP guarantee.",
            "All sessions from a visitor map to one deterministic client_id.",
        ],
    }
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "README_ML4_ML5_dataset.md").write_text(
        make_markdown_report(summary, args),
        encoding="utf-8",
    )
    (output_dir / "assumptions_privacy_notes.md").write_text(
        make_privacy_notes(args),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
