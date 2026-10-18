# ML4 and ML5 Dataset Preparation

This folder contains a reproducible session-level dataset derived from the cleaned RetailRocket event file `user_item_event_final.csv`.

## Purpose

- **ML4 - Federated Learning:** use the train split and `client_id` assignments to simulate client-side training for FedAvg and FedProx.
- **ML5 - Privacy:** reuse the same client/session partitioning for DP-SGD or FedProx + DP-SGD experiments so privacy experiments are comparable to ML4.

## Main assumption

The prediction task is **session-level purchase prediction**: given the observed behavior in a session, predict whether the session eventually contains a purchase event.

To reduce label leakage, features are computed only from the observed prefix:

- If a session has a `transaction`, the observed prefix contains events strictly before the first transaction.
- If a session has no `transaction`, the observed prefix contains all events in the session.
- Sessions with fewer than `1` observed event(s) are excluded from the modeling dataset.

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

- Train: `0.80`
- Validation: `0.10`
- Test: `0.10`

This avoids training on future sessions before validating or testing on earlier sessions.

## Client partition policy

Each `visitorid` is assigned to exactly one `client_id` using a deterministic hash. This preserves visitor/session grouping across runs and keeps all sessions from the same visitor on the same simulated client. The default number of clients is configurable; this run used `50` clients.

## Verification summary

| Split | Sessions | Positive sessions | Positive rate | Unique visitors | Min date | Max date |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| train | 1,408,588 | 10,706 | 0.0076 | 1,133,850 | 2015-05-03 03:00:04.384000 | 2015-08-19 03:49:57.586000 |
| val | 176,074 | 1,383 | 0.0079 | 151,568 | 2015-08-19 03:50:05.146000 | 2015-09-03 02:49:01.447000 |
| test | 176,074 | 1,269 | 0.0072 | 152,471 | 2015-09-03 02:49:07.712000 | 2015-09-18 02:59:41.778000 |

Overall modeling rows: **1,760,736**

Source rows read: **2,755,637**

Excluded zero/short-prefix sessions: **938**

Sequence columns included in this run: **False**
