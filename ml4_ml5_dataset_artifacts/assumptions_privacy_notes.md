# ML5 Privacy Notes and Required Audit Items

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

For this run, each visitor was assigned to one of `50` simulated clients. That is useful for FL experiments, but it is not a privacy guarantee.
