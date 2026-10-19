# -*- coding: utf-8 -*-
"""
ML3 - PIPELINE CHẠY TRỰC TIẾP TRÊN MÁY (VS Code / Windows), KHÔNG CẦN GOOGLE DRIVE

So sánh 5 mô hình trên đúng benchmark của mục 4.3.4.1 - 4.3.4.4:
  Popularity, Item-CF, SASRec, SASRec đa hành vi (4.3.4.5), HSTU (4.3.4.6)

File này chỉ cần MỘT file dữ liệu đầu vào: clean_events_final.csv
Các bước (mỗi khối "# %%" chạy được riêng trong VS Code bằng "Run Cell", hoặc chạy cả file):
  1) Tái tạo benchmark chung của nhóm (lọc view, chia train/valid/test, 101 ứng viên, phá hòa)
  2) Popularity và Item-CF (cùng logic với notebook của nhóm)
  3) Dữ liệu đa hành vi căn chỉnh với benchmark + SASRec / MB-SASRec / HSTU
  4) Bảng so sánh 5 mô hình + kiểm định bootstrap

Cần: python 3.9+, numpy, pandas, scipy, torch   (pip install numpy pandas scipy torch)
RAM nên >= 8 GB (ma trận Item-CF lớn). Huấn luyện đầy đủ trên CPU rất chậm: nên dùng GPU hoặc
chạy thử với QUICK_TEST = True trước.
"""
import os

# ======================= SỬA 2 ĐƯỜNG DẪN NÀY CHO ĐÚNG MÁY BẠN =======================
CLEAN_CSV  = r"D:\NHƯ\bigdata\clean_events_final.csv"     # file dữ liệu sạch
LOCAL_ROOT = r"D:\NHƯ\bigdata\ML3_local"                  # thư mục lưu file trung gian và kết quả (tự tạo)
# =====================================================================================

QUICK_TEST      = False   # True: chạy thử nhanh (1 seed, 2 epoch, 200 batch/epoch), lưu riêng thư mục *_quick
RUN_PREPARATION = True    # đặt False sau khi bước 1-2 đã chạy xong một lần (để khỏi tính lại)

if QUICK_TEST:
    LOCAL_ROOT = LOCAL_ROOT + "_quick"
os.makedirs(LOCAL_ROOT, exist_ok=True)
assert os.path.isfile(CLEAN_CSV), f"Không thấy file dữ liệu: {CLEAN_CSV}. Sửa CLEAN_CSV ở đầu file."
print("Dữ liệu:", CLEAN_CSV)
print("Thư mục kết quả:", LOCAL_ROOT)


# %%
# ============================================================================
# BƯỚC 1a. Lọc view, khách hàng >= 3 lượt, chia train / validation / test (notebook gốc, mục 1)
# ============================================================================
if RUN_PREPARATION:
    import os
    import pandas as pd

    # 1. Đường dẫn dữ liệu chung và checkpoint mới
    clean_file = CLEAN_CSV

    checkpoint_dir = os.path.join(LOCAL_ROOT, "data_checkpoint")

    os.makedirs(checkpoint_dir, exist_ok=True)

    shared_view_file = os.path.join(
        checkpoint_dir,
        "view_events_eligible_shared.pkl"
    )

    # 2. Đọc dữ liệu theo từng phần, chỉ giữ hành vi view
    view_parts = []

    for chunk in pd.read_csv(
        clean_file,
        usecols=[
            "timestamp",
            "datetime",
            "visitorid",
            "session_id",
            "event",
            "itemid"
        ],
        chunksize=300_000
    ):
        view_chunk = chunk.loc[
            chunk["event"].eq("view")
        ].copy()

        view_parts.append(view_chunk)

    views = pd.concat(view_parts, ignore_index=True)

    print("Số lượt view trước khi lọc khách hàng:", len(views))
    print("Số khách hàng có lượt view:", views["visitorid"].nunique())
    print("Số sản phẩm được xem:", views["itemid"].nunique())

    # 3. Kiểm tra dữ liệu thiếu trước khi tiếp tục
    required_cols = ["timestamp", "visitorid", "itemid"]
    missing_counts = views[required_cols].isna().sum()

    print("\nGiá trị thiếu ở các cột bắt buộc:")
    print(missing_counts)

    if missing_counts.any():
        raise ValueError(
            "Có giá trị thiếu ở cột bắt buộc. "
            "Dừng để kiểm tra, chưa lưu checkpoint."
        )

    # 4. Sắp xếp tương tác theo khách hàng và thời gian
    views = views.sort_values(
        ["visitorid", "timestamp"],
        kind="mergesort"
    ).reset_index(drop=True)

    # 5. Chỉ giữ khách hàng có ít nhất 3 lượt view
    view_counts = views.groupby("visitorid").size()
    eligible_users = view_counts[view_counts >= 3].index

    views_eligible_shared = views[
        views["visitorid"].isin(eligible_users)
    ].copy().reset_index(drop=True)

    # 6. Lưu thành file mới, không ghi đè checkpoint cũ
    views_eligible_shared.to_pickle(shared_view_file)

    print("\n===== KẾT QUẢ CHECKPOINT MỚI =====")
    print("Số tương tác:", len(views_eligible_shared))
    print("Số khách hàng:", views_eligible_shared["visitorid"].nunique())
    print("Số sản phẩm:", views_eligible_shared["itemid"].nunique())
    print("Đường dẫn lưu:", shared_view_file)
    print(
        "Kích thước file (MB):",
        round(os.path.getsize(shared_view_file) / (1024 ** 2), 2)
    )


# %%
# ============================================================================
# BƯỚC 1b. Chia train / validation / test
# ============================================================================
if RUN_PREPARATION:
    import os
    import pandas as pd

    # Đường dẫn checkpoint mới từ dữ liệu chung
    BASE_DIR = LOCAL_ROOT
    DATA_DIR = os.path.join(BASE_DIR, "data_checkpoint")

    shared_file = os.path.join(
        DATA_DIR,
        "view_events_eligible_shared.pkl"
    )

    # Đọc dữ liệu đã lọc
    views = pd.read_pickle(shared_file)

    # Sắp xếp theo khách hàng và thời gian
    views = views.sort_values(
        ["visitorid", "timestamp"],
        kind="mergesort"
    ).reset_index(drop=True)

    # Xác định vị trí từng tương tác trong chuỗi của mỗi khách hàng
    grouped = views.groupby("visitorid", sort=False)
    position = grouped.cumcount()
    sequence_length = grouped["itemid"].transform("size")

    # Chia train / validation / test
    train_df = views[position < sequence_length - 2].copy()
    valid_df = views[position == sequence_length - 2].copy()
    test_df = views[position == sequence_length - 1].copy()

    # Chỉ lưu các cột cần thiết cho bước Recommendation tiếp theo
    train_df = train_df[["visitorid", "itemid", "timestamp"]]
    valid_df = valid_df[["visitorid", "itemid", "timestamp"]]
    test_df = test_df[["visitorid", "itemid", "timestamp"]]

    # Lưu thành file mới, không ghi đè file cũ
    train_file = os.path.join(DATA_DIR, "train_interactions_shared.pkl")
    valid_file = os.path.join(DATA_DIR, "valid_targets_shared.pkl")
    test_file = os.path.join(DATA_DIR, "test_targets_shared.pkl")

    train_df.to_pickle(train_file)
    valid_df.to_pickle(valid_file)
    test_df.to_pickle(test_file)

    # Kiểm tra độ phủ sản phẩm trong tập train
    train_catalog = set(train_df["itemid"].unique())

    valid_covered = valid_df["itemid"].isin(train_catalog)
    test_covered = test_df["itemid"].isin(train_catalog)

    print("===== KẾT QUẢ CHIA DỮ LIỆU CHUNG =====")
    print("Số khách hàng:", views["visitorid"].nunique())
    print("Số tương tác train:", len(train_df))
    print("Số khách hàng validation:", len(valid_df))
    print("Số khách hàng test:", len(test_df))
    print("Số sản phẩm trong train catalog:", len(train_catalog))

    print("\n===== ĐỘ PHỦ SẢN PHẨM =====")
    print(
        "Validation target thuộc train catalog:",
        f"{valid_covered.sum():,}/{len(valid_df):,}",
        f"({valid_covered.mean() * 100:.4f}%)"
    )
    print(
        "Test target thuộc train catalog:",
        f"{test_covered.sum():,}/{len(test_df):,}",
        f"({test_covered.mean() * 100:.4f}%)"
    )

    print("\n===== FILE ĐÃ LƯU =====")
    print(train_file)
    print(valid_file)
    print(test_file)


# %%
# ============================================================================
# BƯỚC 1c. Bộ ứng viên cố định 101 sản phẩm (seed 2026)
# ============================================================================
if RUN_PREPARATION:
    import os
    import numpy as np
    import pandas as pd

    # 1. Đường dẫn dữ liệu chung
    BASE_DIR = LOCAL_ROOT
    DATA_DIR = os.path.join(BASE_DIR, "data_checkpoint")
    EVAL_DIR = os.path.join(BASE_DIR, "evaluation")
    os.makedirs(EVAL_DIR, exist_ok=True)

    # 2. Đọc ba tập đã tạo từ dữ liệu chung
    train_df = pd.read_pickle(
        os.path.join(DATA_DIR, "train_interactions_shared.pkl")
    )
    valid_df = pd.read_pickle(
        os.path.join(DATA_DIR, "valid_targets_shared.pkl")
    )
    test_df = pd.read_pickle(
        os.path.join(DATA_DIR, "test_targets_shared.pkl")
    )

    # 3. Xác định danh mục sản phẩm trong train
    catalog = np.sort(train_df["itemid"].unique()).astype(np.int32)
    catalog_set = set(catalog.tolist())

    valid_map = dict(zip(
        valid_df["visitorid"].astype(int),
        valid_df["itemid"].astype(int)
    ))
    test_map = dict(zip(
        test_df["visitorid"].astype(int),
        test_df["itemid"].astype(int)
    ))

    # Chỉ giữ user có cả hai mục tiêu trong train catalog
    eval_users = np.array([
        uid for uid in sorted(valid_map.keys() & test_map.keys())
        if valid_map[uid] in catalog_set
        and test_map[uid] in catalog_set
    ], dtype=np.int64)

    print("Số khách hàng đủ điều kiện đánh giá:", len(eval_users))
    print("Kích thước train catalog:", len(catalog))

    # 4. Lập lịch sử sản phẩm đã xem trong train của từng user
    train_history = {}

    for uid, item in zip(
        train_df["visitorid"].to_numpy(),
        train_df["itemid"].to_numpy()
    ):
        train_history.setdefault(int(uid), set()).add(int(item))

    # 5. Hàm lấy 100 negative items không trùng nhau
    rng = np.random.default_rng(2026)
    N_NEGATIVES = 100
    N_CANDIDATES = 1 + N_NEGATIVES

    def sample_negatives(forbidden):
        negatives = []
        selected = set()

        while len(negatives) < N_NEGATIVES:
            remaining = N_NEGATIVES - len(negatives)
            batch_size = max(128, remaining * 2)

            draws = catalog[
                rng.integers(0, len(catalog), size=batch_size)
            ]

            for item in draws:
                item = int(item)

                if item in forbidden or item in selected:
                    continue

                selected.add(item)
                negatives.append(item)

                if len(negatives) == N_NEGATIVES:
                    break

        return negatives

    # 6. Tạo hai bộ ứng viên
    valid_candidates = np.empty(
        (len(eval_users), N_CANDIDATES), dtype=np.int32
    )
    test_candidates = np.empty(
        (len(eval_users), N_CANDIDATES), dtype=np.int32
    )

    for idx, uid in enumerate(eval_users):
        valid_item = valid_map[int(uid)]
        test_item = test_map[int(uid)]

        history = train_history.get(int(uid), set())

        # Validation: negative không nằm trong train history
        valid_forbidden = set(history)
        valid_forbidden.add(valid_item)

        valid_candidates[idx, 0] = valid_item
        valid_candidates[idx, 1:] = sample_negatives(valid_forbidden)

        # Test: negative không nằm trong train hoặc validation history
        test_forbidden = set(history)
        test_forbidden.add(valid_item)
        test_forbidden.add(test_item)

        test_candidates[idx, 0] = test_item
        test_candidates[idx, 1:] = sample_negatives(test_forbidden)

        if (idx + 1) % 25000 == 0:
            print(f"Đã tạo ứng viên cho {idx + 1:,} khách hàng")

    # 7. Kiểm tra kết quả trước khi lưu
    assert valid_candidates.shape == (len(eval_users), 101)
    assert test_candidates.shape == (len(eval_users), 101)

    assert np.array_equal(
        valid_candidates[:, 0],
        np.array([valid_map[int(u)] for u in eval_users], dtype=np.int32)
    )
    assert np.array_equal(
        test_candidates[:, 0],
        np.array([test_map[int(u)] for u in eval_users], dtype=np.int32)
    )

    # 8. Lưu thành file mới, không ghi đè kết quả cũ
    np.save(
        os.path.join(EVAL_DIR, "eval_users_shared.npy"),
        eval_users
    )
    np.save(
        os.path.join(EVAL_DIR, "valid_candidates_shared.npy"),
        valid_candidates
    )
    np.save(
        os.path.join(EVAL_DIR, "test_candidates_shared.npy"),
        test_candidates
    )

    print("\n===== KẾT QUẢ BỘ ỨNG VIÊN MỚI =====")
    print("Số khách hàng:", len(eval_users))
    print("Validation candidates:", valid_candidates.shape)
    print("Test candidates:", test_candidates.shape)
    print("Kiểu dữ liệu:", valid_candidates.dtype)
    print("Sản phẩm mục tiêu nằm ở cột đầu:", True)
    print("Seed lấy mẫu:", 2026)
    print("\nĐã lưu tại:", EVAL_DIR)


# %%
# ============================================================================
# BƯỚC 1d. Phá hòa dùng chung
# ============================================================================
if RUN_PREPARATION:
    import os
    import numpy as np

    BASE_DIR = LOCAL_ROOT
    EVAL_DIR = os.path.join(BASE_DIR, "evaluation")

    eval_users = np.load(
        os.path.join(EVAL_DIR, "eval_users_shared.npy")
    )

    N_CANDIDATES = 101

    def create_tie_priorities(n_users, seed):
        rng = np.random.default_rng(seed)
        priorities = np.empty(
            (n_users, N_CANDIDATES),
            dtype=np.int16
        )

        for i in range(n_users):
            # Mỗi hàng là một hoán vị duy nhất từ 0 đến 100.
            # Giá trị nhỏ hơn sẽ được ưu tiên khi điểm bằng nhau.
            priorities[i] = rng.permutation(N_CANDIDATES)

        return priorities

    valid_candidates = np.load(
        os.path.join(EVAL_DIR, "valid_candidates_shared.npy"),
        mmap_mode="r"
    )
    test_candidates = np.load(
        os.path.join(EVAL_DIR, "test_candidates_shared.npy"),
        mmap_mode="r"
    )

    assert valid_candidates.shape == (len(eval_users), 101)
    assert test_candidates.shape == (len(eval_users), 101)

    valid_priorities = create_tie_priorities(
        len(eval_users), seed=20261010
    )
    test_priorities = create_tie_priorities(
        len(eval_users), seed=20261011
    )

    np.save(
        os.path.join(EVAL_DIR, "tie_priorities_valid_shared.npy"),
        valid_priorities
    )
    np.save(
        os.path.join(EVAL_DIR, "tie_priorities_test_shared.npy"),
        test_priorities
    )

    assert np.all(
        np.sort(test_priorities, axis=1)
        == np.arange(101, dtype=np.int16)
    )

    print("Số khách hàng:", len(eval_users))
    print("Validation priorities:", valid_priorities.shape)
    print("Test priorities:", test_priorities.shape)
    print("Mỗi hàng có 101 thứ tự ưu tiên khác nhau.")
    print("Đã lưu bộ phá hòa dùng chung tại:", EVAL_DIR)


# %%
# ============================================================================
# BƯỚC 2a. Popularity
# ============================================================================
if RUN_PREPARATION:
    import os
    import json
    import numpy as np
    import pandas as pd

    BASE_DIR = LOCAL_ROOT
    DATA_DIR = os.path.join(BASE_DIR, "data_checkpoint")
    EVAL_DIR = os.path.join(BASE_DIR, "evaluation")
    RESULTS_DIR = os.path.join(BASE_DIR, "results_shared")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Đọc train và bộ ứng viên cố định
    train_df = pd.read_pickle(
        os.path.join(DATA_DIR, "train_interactions_shared.pkl")
    )

    valid_candidates = np.load(
        os.path.join(EVAL_DIR, "valid_candidates_shared.npy"),
        mmap_mode="r"
    )
    test_candidates = np.load(
        os.path.join(EVAL_DIR, "test_candidates_shared.npy"),
        mmap_mode="r"
    )

    valid_priorities = np.load(
        os.path.join(EVAL_DIR, "tie_priorities_valid_shared.npy"),
        mmap_mode="r"
    )
    test_priorities = np.load(
        os.path.join(EVAL_DIR, "tie_priorities_test_shared.npy"),
        mmap_mode="r"
    )

    assert valid_candidates.shape == valid_priorities.shape
    assert test_candidates.shape == test_priorities.shape
    assert valid_candidates.shape[1] == 101
    assert test_candidates.shape[1] == 101

    # Độ phổ biến được tính bằng số lượt view trong train
    popularity = train_df["itemid"].value_counts().to_dict()

    def evaluate_popularity_fair(candidates, priorities, split_name):
        n_users = len(candidates)
        ranks = np.empty(n_users, dtype=np.int16)
        batch_size = 5000

        for start in range(0, n_users, batch_size):
            end = min(start + batch_size, n_users)

            cand_batch = np.asarray(candidates[start:end])
            priority_batch = np.asarray(priorities[start:end])

            # Lấy điểm popularity cho từng ứng viên
            scores = np.fromiter(
                (
                    popularity.get(int(item), 0)
                    for item in cand_batch.ravel()
                ),
                dtype=np.int32,
                count=cand_batch.size
            ).reshape(cand_batch.shape)

            positive_score = scores[:, [0]]
            positive_priority = priority_batch[:, [0]]

            # Ứng viên đứng trước target nếu:
            # 1) điểm cao hơn; hoặc
            # 2) điểm bằng nhau và priority nhỏ hơn target.
            higher_score = scores[:, 1:] > positive_score
            tied_but_preferred = (
                (scores[:, 1:] == positive_score)
                & (priority_batch[:, 1:] < positive_priority)
            )

            ranks[start:end] = (
                1 + np.sum(
                    higher_score | tied_but_preferred,
                    axis=1
                )
            )

            if end % 25000 < batch_size or end == n_users:
                print(
                    f"{split_name}: đã đánh giá "
                    f"{end:,}/{n_users:,} khách hàng"
                )

        hits = ranks <= 10
        ndcg = np.where(
            hits,
            1.0 / np.log2(ranks + 1),
            0.0
        )

        metrics = {
            "model": "Popularity",
            "split": split_name,
            "tie_breaking": "shared_random_priorities",
            "num_users": int(n_users),
            "num_candidates_per_user": 101,
            "HR@10": float(hits.mean()),
            "NDCG@10": float(ndcg.mean()),
            "rank_1": int(np.sum(ranks == 1)),
            "rank_le_10": int(np.sum(ranks <= 10)),
            "misses": int(np.sum(ranks > 10)),
            "median_rank": float(np.median(ranks)),
            "mean_rank": float(np.mean(ranks)),
            "max_rank": int(np.max(ranks))
        }

        suffix = split_name.lower()

        np.save(
            os.path.join(
                RESULTS_DIR,
                f"popularity_{suffix}_positive_rank_shared_fair.npy"
            ),
            ranks
        )

        with open(
            os.path.join(
                RESULTS_DIR,
                f"popularity_{suffix}_metrics_shared_fair.json"
            ),
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        return metrics

    # Tính lại validation và test bằng cùng quy tắc phá hòa
    valid_metrics_fair = evaluate_popularity_fair(
        valid_candidates, valid_priorities, "VALID"
    )

    test_metrics_fair = evaluate_popularity_fair(
        test_candidates, test_priorities, "TEST"
    )

    print("\n===== POPULARITY — VALIDATION, PHÁ HÒA CHUNG =====")
    print(json.dumps(valid_metrics_fair, ensure_ascii=False, indent=2))

    print("\n===== POPULARITY — TEST, PHÁ HÒA CHUNG =====")
    print(json.dumps(test_metrics_fair, ensure_ascii=False, indent=2))

    print("\nKết quả mới được lưu tại:", RESULTS_DIR)


# %%
# ============================================================================
# BƯỚC 2b. Item-CF: dựng ma trận tương đồng
# ============================================================================
if RUN_PREPARATION:
    import os
    import numpy as np
    import pandas as pd
    from scipy.sparse import csr_matrix, save_npz

    # 1. Đường dẫn
    BASE_DIR = LOCAL_ROOT
    DATA_DIR = os.path.join(BASE_DIR, "data_checkpoint")
    ITEMCF_DIR = os.path.join(BASE_DIR, "itemcf_shared")
    os.makedirs(ITEMCF_DIR, exist_ok=True)

    # 2. Đọc dữ liệu train từ nguồn chung
    train_df = pd.read_pickle(
        os.path.join(DATA_DIR, "train_interactions_shared.pkl")
    )

    # 3. Tạo danh mục và danh sách khách hàng theo thứ tự cố định
    catalog = np.sort(train_df["itemid"].unique()).astype(np.int32)
    user_ids = np.sort(train_df["visitorid"].unique()).astype(np.int64)

    # Item-CF dùng ma trận nhị phân: một khách hàng có xem một sản phẩm
    # hay không; nhiều lượt xem cùng sản phẩm chỉ tính một lần.
    train_pairs = train_df[["visitorid", "itemid"]].drop_duplicates()

    rows = np.searchsorted(
        user_ids, train_pairs["visitorid"].to_numpy()
    )
    cols = np.searchsorted(
        catalog, train_pairs["itemid"].to_numpy()
    )

    X = csr_matrix(
        (
            np.ones(len(train_pairs), dtype=np.int32),
            (rows, cols)
        ),
        shape=(len(user_ids), len(catalog)),
        dtype=np.int32
    )
    X.sum_duplicates()
    X.data[:] = 1
    X.eliminate_zeros()

    print("===== MA TRẬN KHÁCH HÀNG - SẢN PHẨM =====")
    print("Kích thước:", X.shape)
    print("Số cặp user-item khác nhau:", X.nnz)

    # 4. Tính đồng xuất hiện sản phẩm: C = X.T @ X
    print("\nĐang tính ma trận đồng xuất hiện...")
    C = (X.T @ X).tocsr()
    C.setdiag(0)
    C.eliminate_zeros()

    print("Kích thước co-occurrence:", C.shape)
    print("Số phần tử khác 0:", C.nnz)

    # 5. Tính cosine similarity từ co-occurrence
    print("\nĐang tính cosine similarity...")
    item_counts = np.asarray(X.sum(axis=0)).ravel().astype(np.float64)
    norms = np.sqrt(item_counts)
    norms[norms == 0] = 1.0

    inv_norms = (1.0 / norms).astype(np.float32)

    S = C.astype(np.float32)
    S = S.multiply(inv_norms[:, None])
    S = S.multiply(inv_norms[None, :]).tocsr()
    S.eliminate_zeros()

    # 6. Kiểm tra và lưu thành file mới
    assert S.shape == (len(catalog), len(catalog))
    assert S.shape[0] == len(catalog)

    save_npz(
        os.path.join(ITEMCF_DIR, "item_similarity_shared.npz"),
        S
    )
    save_npz(
        os.path.join(ITEMCF_DIR, "user_item_binary_shared.npz"),
        X
    )
    np.save(
        os.path.join(ITEMCF_DIR, "item_catalog_shared.npy"),
        catalog
    )
    np.save(
        os.path.join(ITEMCF_DIR, "user_ids_shared.npy"),
        user_ids
    )

    print("\n===== ITEM-CF ĐÃ LƯU =====")
    print("Similarity shape:", S.shape)
    print("Similarity nnz:", S.nnz)
    print("Kiểu dữ liệu similarity:", S.dtype)
    print("Thư mục:", ITEMCF_DIR)


# %%
# ============================================================================
# BƯỚC 2c. Item-CF: đánh giá test
# ============================================================================
if RUN_PREPARATION:
    import os
    import json
    import numpy as np
    from scipy.sparse import load_npz, csr_matrix

    # 1. Đường dẫn
    BASE_DIR = LOCAL_ROOT
    DATA_DIR = os.path.join(BASE_DIR, "data_checkpoint")
    EVAL_DIR = os.path.join(BASE_DIR, "evaluation")
    ITEMCF_DIR = os.path.join(BASE_DIR, "itemcf_shared")
    RESULTS_DIR = os.path.join(BASE_DIR, "results_shared")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 2. Đọc ma trận Item-CF và dữ liệu đánh giá
    X = load_npz(
        os.path.join(ITEMCF_DIR, "user_item_binary_shared.npz")
    ).tocsr()

    S = load_npz(
        os.path.join(ITEMCF_DIR, "item_similarity_shared.npz")
    ).tocsr()

    catalog = np.load(
        os.path.join(ITEMCF_DIR, "item_catalog_shared.npy")
    )
    user_ids = np.load(
        os.path.join(ITEMCF_DIR, "user_ids_shared.npy")
    )
    eval_users = np.load(
        os.path.join(EVAL_DIR, "eval_users_shared.npy")
    )
    test_candidates = np.load(
        os.path.join(EVAL_DIR, "test_candidates_shared.npy")
    )
    test_priorities = np.load(
        os.path.join(EVAL_DIR, "tie_priorities_test_shared.npy")
    )

    assert test_candidates.shape == (len(eval_users), 101)
    assert test_priorities.shape == test_candidates.shape
    assert X.shape == (len(user_ids), len(catalog))
    assert S.shape == (len(catalog), len(catalog))

    # 3. Tạo lịch sử test = train + target validation
    train_df = __import__("pandas").read_pickle(
        os.path.join(DATA_DIR, "train_interactions_shared.pkl")
    )
    valid_df = __import__("pandas").read_pickle(
        os.path.join(DATA_DIR, "valid_targets_shared.pkl")
    )

    user_rows = np.searchsorted(user_ids, eval_users)
    assert np.array_equal(user_ids[user_rows], eval_users)

    H_train = X[user_rows].tocsr()

    valid_map = dict(zip(
        valid_df["visitorid"].astype(int),
        valid_df["itemid"].astype(int)
    ))
    valid_items = np.array(
        [valid_map[int(uid)] for uid in eval_users],
        dtype=np.int64
    )

    valid_cols = np.searchsorted(catalog, valid_items)
    assert np.array_equal(catalog[valid_cols], valid_items)

    V = csr_matrix(
        (
            np.ones(len(eval_users), dtype=np.int32),
            (np.arange(len(eval_users)), valid_cols)
        ),
        shape=H_train.shape,
        dtype=np.int32
    )

    H_test = (H_train + V).tocsr()
    H_test.data[:] = 1
    H_test.eliminate_zeros()
    H_test.sort_indices()

    # 4. Đổi các item ID trong candidate sang chỉ số cột
    candidate_cols = np.searchsorted(catalog, test_candidates)
    assert np.array_equal(catalog[candidate_cols], test_candidates)

    # 5. Đánh giá theo batch, sử dụng đúng priority chung
    ranks = np.empty(len(eval_users), dtype=np.int16)
    batch_size = 256

    for start in range(0, len(eval_users), batch_size):
        end = min(start + batch_size, len(eval_users))

        score_matrix = (H_test[start:end] @ S).tocsr()
        score_matrix.sort_indices()

        for j in range(end - start):
            row_start = score_matrix.indptr[j]
            row_end = score_matrix.indptr[j + 1]

            row_items = score_matrix.indices[row_start:row_end]
            row_scores = score_matrix.data[row_start:row_end]

            cand = candidate_cols[start + j]
            positions = np.searchsorted(row_items, cand)

            scores = np.zeros(len(cand), dtype=np.float32)
            in_range = positions < len(row_items)
            idx = np.flatnonzero(in_range)

            if len(idx):
                matched = row_items[positions[idx]] == cand[idx]
                matched_idx = idx[matched]
                scores[matched_idx] = row_scores[
                    positions[matched_idx]
                ]

            priorities = test_priorities[start + j]
            positive_score = scores[0]
            positive_priority = priorities[0]

            ahead = (
                (scores[1:] > positive_score)
                | (
                    (scores[1:] == positive_score)
                    & (priorities[1:] < positive_priority)
                )
            )

            ranks[start + j] = 1 + int(ahead.sum())

        if end % 25000 < batch_size or end == len(eval_users):
            print(
                f"Đã đánh giá {end:,}/{len(eval_users):,} khách hàng"
            )

        del score_matrix

    # 6. Tính chỉ số
    hits = ranks <= 10
    ndcg = np.where(
        hits,
        1.0 / np.log2(ranks + 1),
        0.0
    )

    metrics = {
        "model": "Item-CF",
        "split": "TEST",
        "tie_breaking": "shared_random_priorities",
        "num_users": int(len(ranks)),
        "num_candidates_per_user": 101,
        "HR@10": float(hits.mean()),
        "NDCG@10": float(ndcg.mean()),
        "rank_1": int(np.sum(ranks == 1)),
        "rank_le_10": int(np.sum(ranks <= 10)),
        "misses": int(np.sum(ranks > 10)),
        "median_rank": float(np.median(ranks)),
        "mean_rank": float(np.mean(ranks)),
        "max_rank": int(np.max(ranks))
    }

    # 7. Lưu kết quả hiệu chỉnh riêng
    np.save(
        os.path.join(
            RESULTS_DIR,
            "itemcf_test_positive_rank_shared_fair_common.npy"
        ),
        ranks
    )

    with open(
        os.path.join(
            RESULTS_DIR,
            "itemcf_test_metrics_shared_fair_common.json"
        ),
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("\n===== ITEM-CF — TEST, PHÁ HÒA CHUNG =====")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("\nKết quả đã lưu tại:", RESULTS_DIR)

    for _n in ['X', 'C', 'S', 'H_train', 'H_test', 'V', 'score_matrix', 'train_df', 'valid_df', 'views']:
        globals().pop(_n, None)
    import gc; gc.collect()


# %%
# ============================================================================
# BƯỚC 3a. Chuẩn bị dữ liệu tuần tự cho SASRec (notebook gốc, mục 6)
# ============================================================================
if RUN_PREPARATION:
    import os
    import json
    import pickle
    import numpy as np
    import pandas as pd

    # 1. Đường dẫn dữ liệu chung và thư mục SASRec mới
    BASE_DIR = LOCAL_ROOT
    DATA_DIR = os.path.join(BASE_DIR, "data_checkpoint")
    EVAL_DIR = os.path.join(BASE_DIR, "evaluation")
    SASREC_SHARED_DIR = os.path.join(BASE_DIR, "sasrec_shared")

    os.makedirs(SASREC_SHARED_DIR, exist_ok=True)

    MAXLEN = 50

    # 2. Đọc dữ liệu train và bộ ứng viên đã tạo từ nguồn chung
    train_df = pd.read_pickle(
        os.path.join(DATA_DIR, "train_interactions_shared.pkl")
    )

    eval_users = np.load(
        os.path.join(EVAL_DIR, "eval_users_shared.npy")
    )

    valid_candidates_raw = np.load(
        os.path.join(EVAL_DIR, "valid_candidates_shared.npy")
    )

    test_candidates_raw = np.load(
        os.path.join(EVAL_DIR, "test_candidates_shared.npy")
    )

    assert valid_candidates_raw.shape == (len(eval_users), 101)
    assert test_candidates_raw.shape == (len(eval_users), 101)

    # 3. Đảm bảo thứ tự tương tác đúng theo thời gian
    train_df = train_df.sort_values(
        ["visitorid", "timestamp"],
        kind="mergesort"
    ).reset_index(drop=True)

    # 4. Tạo vocabulary item; index 0 dành cho padding
    catalog = np.sort(
        train_df["itemid"].unique()
    ).astype(np.int64)

    item_to_idx = {
        int(item_id): idx + 1
        for idx, item_id in enumerate(catalog)
    }

    NUM_ITEMS = len(catalog)

    # 5. Chuyển chuỗi train sang index của SASRec
    train_sequences_encoded = {}

    for user_id, group in train_df.groupby(
        "visitorid", sort=False
    ):
        train_sequences_encoded[int(user_id)] = [
            item_to_idx[int(item)]
            for item in group["itemid"].to_numpy()
        ]

    trainable_users = np.asarray(
        [
            user_id
            for user_id, seq in train_sequences_encoded.items()
            if len(seq) > 1
        ],
        dtype=np.int64
    )

    # 6. Chuyển ứng viên từ itemid gốc sang SASRec index
    def map_candidate_matrix(candidates, name):
        positions = np.searchsorted(catalog, candidates)

        if np.any(positions >= len(catalog)):
            raise ValueError(
                f"{name}: có sản phẩm không thuộc train catalog."
            )

        if not np.array_equal(catalog[positions], candidates):
            raise ValueError(
                f"{name}: có sản phẩm không thuộc train catalog."
            )

        return (positions + 1).astype(np.int32)

    valid_candidates = map_candidate_matrix(
        valid_candidates_raw, "Validation candidates"
    )

    test_candidates = map_candidate_matrix(
        test_candidates_raw, "Test candidates"
    )

    # 7. Tạo chuỗi validation và test, left-padding đến MAXLEN
    def left_pad_sequence(sequence, maxlen=MAXLEN):
        result = np.zeros(maxlen, dtype=np.int32)
        recent = sequence[-maxlen:]

        if recent:
            result[-len(recent):] = recent

        return result

    valid_seq_input = np.zeros(
        (len(eval_users), MAXLEN), dtype=np.int32
    )
    test_seq_input = np.zeros(
        (len(eval_users), MAXLEN), dtype=np.int32
    )

    for i, user_id in enumerate(eval_users):
        user_id = int(user_id)

        if user_id not in train_sequences_encoded:
            raise ValueError(
                f"Không tìm thấy train sequence của user {user_id}"
            )

        train_seq = train_sequences_encoded[user_id]

        # Validation: chỉ dùng lịch sử train
        valid_seq_input[i] = left_pad_sequence(train_seq)

        # Test: dùng train history cộng target validation
        valid_item_raw = int(valid_candidates_raw[i, 0])
        valid_item_idx = item_to_idx[valid_item_raw]

        test_seq_input[i] = left_pad_sequence(
            train_seq + [valid_item_idx]
        )

    # 8. Kiểm tra cấu trúc dữ liệu
    assert NUM_ITEMS == 114800
    assert valid_seq_input.shape == (len(eval_users), MAXLEN)
    assert test_seq_input.shape == (len(eval_users), MAXLEN)
    assert valid_candidates.shape == (len(eval_users), 101)
    assert test_candidates.shape == (len(eval_users), 101)
    assert np.all(valid_candidates[:, 0] > 0)
    assert np.all(test_candidates[:, 0] > 0)

    # 9. Lưu file mới; không ghi đè SASRec hoặc kết quả cũ
    files_to_save = {
        "train_sequences_encoded.pkl": train_sequences_encoded,
        "item_to_idx.pkl": item_to_idx,
    }

    for filename, obj in files_to_save.items():
        with open(
            os.path.join(SASREC_SHARED_DIR, filename), "wb"
        ) as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    np.save(
        os.path.join(SASREC_SHARED_DIR, "trainable_users.npy"),
        trainable_users
    )
    np.save(
        os.path.join(SASREC_SHARED_DIR, "eval_users.npy"),
        eval_users
    )
    np.save(
        os.path.join(SASREC_SHARED_DIR, "eval_valid_seq.npy"),
        valid_seq_input
    )
    np.save(
        os.path.join(SASREC_SHARED_DIR, "eval_test_seq.npy"),
        test_seq_input
    )
    np.save(
        os.path.join(SASREC_SHARED_DIR, "eval_valid_candidates.npy"),
        valid_candidates
    )
    np.save(
        os.path.join(SASREC_SHARED_DIR, "eval_test_candidates.npy"),
        test_candidates
    )

    metadata = {
        "source": "clean_events_final.csv",
        "maxlen": MAXLEN,
        "num_items": int(NUM_ITEMS),
        "num_train_users": int(len(train_sequences_encoded)),
        "num_trainable_users": int(len(trainable_users)),
        "num_eval_users": int(len(eval_users)),
        "candidate_size": 101,
        "positive_column": 0,
        "seeds": [2026, 2027, 2028]
    }

    with open(
        os.path.join(SASREC_SHARED_DIR, "metadata.json"),
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("===== SASRec SHARED DATA PREPARATION =====")
    print("Số sản phẩm:", NUM_ITEMS)
    print("Số user có train sequence:", len(train_sequences_encoded))
    print("Số user tạo được training target:", len(trainable_users))
    print("Validation sequence:", valid_seq_input.shape)
    print("Test sequence:", test_seq_input.shape)
    print("Validation candidates:", valid_candidates.shape)
    print("Test candidates:", test_candidates.shape)
    print("Padding index: 0")
    print("Thư mục lưu:", SASREC_SHARED_DIR)
    print("Đã kiểm tra các target ứng viên thuộc train catalog.")


# %%
# ============================================================================
# BƯỚC 3b. Đường dẫn và hằng số của phần mở rộng
# ============================================================================
import os, json, time, random, pickle
import numpy as np
import pandas as pd

BASE_DIR    = LOCAL_ROOT
RAW_DIR     = os.path.dirname(CLEAN_CSV)
EVAL_DIR    = os.path.join(BASE_DIR, "evaluation")
SASREC_DIR  = os.path.join(BASE_DIR, "sasrec_shared")
RESULTS_DIR = os.path.join(BASE_DIR, "results_shared")
MB_DIR      = os.path.join(BASE_DIR, "mbsasrec_shared")     # thư mục mới, không ghi đè file cũ
for d in [MB_DIR, os.path.join(MB_DIR, "runs"), RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

MAXLEN = 50
N_CAND = 101
BEHAVIOR2ID = {"view": 1, "addtocart": 2, "transaction": 3}   # 0 = padding
print("BASE_DIR:", BASE_DIR)


# %%
# ============================================================================
# BƯỚC 3c. Dữ liệu đa hành vi căn chỉnh với benchmark chung (có kiểm tra khớp)
# ============================================================================
def left_pad_np(arr, maxlen=MAXLEN):
    out = np.zeros(maxlen, dtype=np.int32)
    recent = arr[-maxlen:]
    if len(recent):
        out[-len(recent):] = recent
    return out

# ---- 1. Nạp các file chung của notebook gốc ----
with open(os.path.join(SASREC_DIR, "item_to_idx.pkl"), "rb") as f:
    item_to_idx = pickle.load(f)
with open(os.path.join(SASREC_DIR, "train_sequences_encoded.pkl"), "rb") as f:
    train_seq_view = pickle.load(f)                      # {visitorid: [item_idx, ...]} chỉ view
trainable_view = np.load(os.path.join(SASREC_DIR, "trainable_users.npy")).astype(np.int64)
eval_users = np.load(os.path.join(EVAL_DIR, "eval_users_shared.npy"))
valid_seq_view = np.load(os.path.join(SASREC_DIR, "eval_valid_seq.npy"))
test_seq_view  = np.load(os.path.join(SASREC_DIR, "eval_test_seq.npy"))
valid_cand_idx = np.load(os.path.join(SASREC_DIR, "eval_valid_candidates.npy"))
test_cand_idx  = np.load(os.path.join(SASREC_DIR, "eval_test_candidates.npy"))
valid_priorities = np.load(os.path.join(EVAL_DIR, "tie_priorities_valid_shared.npy"), mmap_mode="r")
test_priorities  = np.load(os.path.join(EVAL_DIR, "tie_priorities_test_shared.npy"), mmap_mode="r")

NUM_ITEMS = len(item_to_idx)
assert NUM_ITEMS == 114800 and len(train_seq_view) == 190980 and len(eval_users) == 172513
assert valid_seq_view.shape == (172513, MAXLEN) and test_cand_idx.shape == (172513, N_CAND)

# ---- 2. Đọc toàn bộ sự kiện của 190.980 khách hàng, sắp xếp ổn định ----
ev = pd.read_csv(CLEAN_CSV,
                 usecols=["timestamp", "visitorid", "event", "itemid"])
ev = ev[ev["event"].isin(BEHAVIOR2ID)]
ev = ev[ev["visitorid"].isin(set(train_seq_view.keys()))]
ev = ev.sort_values(["visitorid", "timestamp"], kind="mergesort").reset_index(drop=True)
ev_idx = ev["itemid"].map(item_to_idx).fillna(0).astype(np.int64).to_numpy()   # 0 = ngoài train catalog
ev_beh = ev["event"].map(BEHAVIOR2ID).astype(np.int64).to_numpy()
users_sorted, start = np.unique(ev["visitorid"].to_numpy(), return_index=True)
bounds = np.append(start, len(ev))
print("Số sự kiện:", len(ev), "| số khách hàng:", len(users_sorted))
print("Phân bố hành vi:", ev["event"].value_counts().to_dict())

# ---- 3. Chuỗi train / test đa hành vi và kiểm tra khớp với chuỗi view của SASRec ----
eval_set = set(int(u) for u in eval_users)
valid_item_of = dict(zip(eval_users.astype(np.int64).tolist(), valid_cand_idx[:, 0].astype(np.int64).tolist()))
mb_train, mb_test_hist = {}, {}
bad_train = bad_test = 0
for k, u in enumerate(users_sorted):
    u = int(u)
    s, e = bounds[k], bounds[k + 1]
    it, bh = ev_idx[s:e], ev_beh[s:e]
    vpos = np.flatnonzero(bh == 1)
    p_val, p_test = vpos[-2], vpos[-1]

    keep = it[:p_val] > 0
    tr_i, tr_b = it[:p_val][keep].astype(np.int32), bh[:p_val][keep].astype(np.int8)
    mb_train[u] = (tr_i, tr_b)
    if not np.array_equal(tr_i[tr_b == 1], np.asarray(train_seq_view[u], dtype=np.int32)):
        bad_train += 1

    if u in eval_set:
        keep = it[:p_test] > 0
        te_i, te_b = it[:p_test][keep].astype(np.int32), bh[:p_test][keep].astype(np.int8)
        mb_test_hist[u] = (te_i, te_b)
        expected = np.asarray(list(train_seq_view[u]) + [valid_item_of[u]], dtype=np.int32)
        if not np.array_equal(te_i[te_b == 1], expected):
            bad_test += 1

print(f"Khách hàng lệch chuỗi view so với SASRec gốc: train={bad_train}, test={bad_test}")
assert bad_train == 0 and bad_test == 0, "Chuỗi view không khớp benchmark chung, cần kiểm tra lại."

# ---- 4. Mảng đầu vào đánh giá (cùng thứ tự eval_users) ----
U = len(eval_users)
mb_valid_seq = np.zeros((U, MAXLEN), np.int32); mb_valid_beh = np.zeros((U, MAXLEN), np.int32)
mb_test_seq  = np.zeros((U, MAXLEN), np.int32); mb_test_beh  = np.zeros((U, MAXLEN), np.int32)
for row, u in enumerate(eval_users):
    i_, b_ = mb_train[int(u)];      mb_valid_seq[row], mb_valid_beh[row] = left_pad_np(i_), left_pad_np(b_)
    i_, b_ = mb_test_hist[int(u)];  mb_test_seq[row],  mb_test_beh[row]  = left_pad_np(i_), left_pad_np(b_)

# ---- 5. Chuỗi view-only dạng mảng (để dùng chung hàm tạo batch) ----
view_train = {int(u): (np.asarray(s, np.int32), np.ones(len(s), np.int8)) for u, s in train_seq_view.items()}

# ---- 6. Thống kê: đa hành vi thực sự thêm được bao nhiêu thông tin? ----
has_extra_test = ((mb_test_beh > 1).any(axis=1)).mean()
n_ev_beh = np.concatenate([b for _, b in mb_train.values()])
print(f"\nChuỗi train đa hành vi: view={np.mean(n_ev_beh==1):.1%}, addtocart={np.mean(n_ev_beh==2):.1%}, "
      f"transaction={np.mean(n_ev_beh==3):.1%}")
print(f"Độ dài chuỗi train trung bình: view-only={np.mean([len(v[0]) for v in view_train.values()]):.2f}, "
      f"đa hành vi={np.mean([len(v[0]) for v in mb_train.values()]):.2f}")
print(f"Tỉ lệ khách hàng đánh giá có sự kiện ngoài view trong đầu vào test: {has_extra_test:.2%}")

# ---- 7. Lưu để dùng lại ----
np.save(os.path.join(MB_DIR, "mb_valid_seq.npy"), mb_valid_seq); np.save(os.path.join(MB_DIR, "mb_valid_beh.npy"), mb_valid_beh)
np.save(os.path.join(MB_DIR, "mb_test_seq.npy"),  mb_test_seq);  np.save(os.path.join(MB_DIR, "mb_test_beh.npy"),  mb_test_beh)
with open(os.path.join(MB_DIR, "mb_train_sequences.pkl"), "wb") as f:
    pickle.dump(mb_train, f, protocol=pickle.HIGHEST_PROTOCOL)
print("Đã lưu dữ liệu đa hành vi tại:", MB_DIR)


# %%
# ============================================================================
# BƯỚC 3d. Hàm dùng chung: tạo batch, chỉ số, bootstrap
# ============================================================================
def make_batch(seqs, user_list, batch_size, maxlen, num_items, rng):
    sel = rng.choice(user_list, size=batch_size, replace=True)
    seq = np.zeros((batch_size, maxlen), np.int64); pos = np.zeros((batch_size, maxlen), np.int64)
    neg = np.zeros((batch_size, maxlen), np.int64)
    beh = np.zeros((batch_size, maxlen), np.int64); pos_beh = np.zeros((batch_size, maxlen), np.int64)
    for r, u in enumerate(sel):
        it, bh = seqs[int(u)]
        n = len(it)
        m = min(n - 1, maxlen)
        seq[r, maxlen - m:] = it[n - 1 - m:n - 1]
        pos[r, maxlen - m:] = it[n - m:n]
        beh[r, maxlen - m:] = bh[n - 1 - m:n - 1]
        pos_beh[r, maxlen - m:] = bh[n - m:n]
        neg_row = rng.integers(1, num_items + 1, size=m)
        bad = np.isin(neg_row, it)                          # không trùng chuỗi của chính khách hàng
        while bad.any():
            neg_row[bad] = rng.integers(1, num_items + 1, size=int(bad.sum()))
            bad = np.isin(neg_row, it)
        neg[r, maxlen - m:] = neg_row
    return seq, beh, pos, pos_beh, neg


def metrics_from_ranks(ranks, model, split, seed=None, best_epoch=None):
    ranks = np.asarray(ranks)
    hits = ranks <= 10
    ndcg = np.where(hits, 1.0 / np.log2(ranks + 1), 0.0)
    out = {"model": model, "split": split, "tie_breaking": "shared_random_priorities",
           "num_users": int(len(ranks)), "num_candidates_per_user": N_CAND,
           "HR@10": float(hits.mean()), "NDCG@10": float(ndcg.mean()),
           "rank_1": int(np.sum(ranks == 1)), "rank_le_10": int(np.sum(hits)),
           "misses": int(np.sum(ranks > 10)), "median_rank": float(np.median(ranks)),
           "mean_rank": float(np.mean(ranks)), "max_rank": int(np.max(ranks))}
    if seed is not None: out["seed"] = seed
    if best_epoch is not None: out["best_epoch"] = best_epoch
    return out


def paired_bootstrap(a, b, n_boot=1000, seed=0):
    # a, b: giá trị theo từng khách hàng của hai mô hình; trả về (chênh lệch trung bình, khoảng tin cậy 95%)
    rng = np.random.default_rng(seed)
    d = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    n = len(d)
    means = np.empty(n_boot)
    for i in range(n_boot):
        means[i] = d[rng.integers(0, n, size=n)].mean()
    return float(d.mean()), (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


# %% [torch]
# ============================================================================
# BƯỚC 3e. Mô hình: SASRec, MB-SASRec (4.3.4.5), HSTU (4.3.4.6)
# ============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Thiết bị:", device, "|", torch.cuda.get_device_name(0) if torch.cuda.is_available()
      else "CPU - chậm, nên dùng GPU (Colab) cho lần chạy đầy đủ")


class SASRecBackbone(nn.Module):
    # Bản sao FaithfulSASRec của notebook gốc + tham số beh_seq tùy chọn + hàm _embed để mở rộng.
    def __init__(self, num_items, hidden_units=50, maxlen=50, num_blocks=2, num_heads=1, dropout_rate=0.5):
        super().__init__()
        assert hidden_units % num_heads == 0
        self.num_items, self.hidden_units, self.maxlen = num_items, hidden_units, maxlen
        self.num_blocks, self.num_heads = num_blocks, num_heads
        self.head_dim = hidden_units // num_heads

        self.item_emb = nn.Embedding(num_items + 1, hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen, hidden_units)
        self.input_dropout = nn.Dropout(dropout_rate)

        self.attn_norms, self.q_layers = nn.ModuleList(), nn.ModuleList()
        self.k_layers, self.v_layers = nn.ModuleList(), nn.ModuleList()
        self.attn_dropout = nn.Dropout(dropout_rate)
        self.ffn_norms, self.ffn_linear1, self.ffn_linear2 = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        self.ffn_dropout1, self.ffn_dropout2 = nn.Dropout(dropout_rate), nn.Dropout(dropout_rate)
        for _ in range(num_blocks):
            self.attn_norms.append(nn.LayerNorm(hidden_units, eps=1e-8))
            self.q_layers.append(nn.Linear(hidden_units, hidden_units))
            self.k_layers.append(nn.Linear(hidden_units, hidden_units))
            self.v_layers.append(nn.Linear(hidden_units, hidden_units))
            self.ffn_norms.append(nn.LayerNorm(hidden_units, eps=1e-8))
            self.ffn_linear1.append(nn.Linear(hidden_units, hidden_units))
            self.ffn_linear2.append(nn.Linear(hidden_units, hidden_units))
        self.final_norm = nn.LayerNorm(hidden_units, eps=1e-8)
        self.register_buffer("causal_mask",
                             torch.triu(torch.ones(maxlen, maxlen, dtype=torch.bool), diagonal=1),
                             persistent=False)
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.item_emb.weight)
        nn.init.xavier_uniform_(self.pos_emb.weight)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()
        for layers in [self.q_layers, self.k_layers, self.v_layers, self.ffn_linear1, self.ffn_linear2]:
            for layer in layers:
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        for norm in list(self.attn_norms) + list(self.ffn_norms) + [self.final_norm]:
            nn.init.ones_(norm.weight)
            nn.init.zeros_(norm.bias)

    def _split_heads(self, x):
        B, T, H = x.shape
        return x.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x):
        B, heads, T, D = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, heads * D)

    def _embed(self, input_seq, beh_seq):
        return self.item_emb(input_seq) * (self.hidden_units ** 0.5)

    def forward(self, input_seq, beh_seq=None):
        B, T = input_seq.shape
        valid_mask = input_seq != 0
        valid_mask_float = valid_mask.unsqueeze(-1).float()

        x = self._embed(input_seq, beh_seq)
        positions = torch.arange(T, device=input_seq.device).unsqueeze(0)
        x = self.input_dropout(x + self.pos_emb(positions))
        x = x * valid_mask_float

        for block in range(self.num_blocks):
            q_input = self.attn_norms[block](x)
            q = self._split_heads(self.q_layers[block](q_input))
            k = self._split_heads(self.k_layers[block](x))
            v = self._split_heads(self.v_layers[block](x))

            scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            key_padding = (~valid_mask).unsqueeze(1).unsqueeze(1)
            causal = self.causal_mask[:T, :T].unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(key_padding, -2**32 + 1)
            scores = scores.masked_fill(causal, -2**32 + 1)

            weights = self.attn_dropout(torch.softmax(scores, dim=-1))
            attn_output = self._merge_heads(torch.matmul(weights, v))
            x = (attn_output + q_input) * valid_mask_float

            ffn_input = self.ffn_norms[block](x)
            ffn_output = F.relu(self.ffn_linear1[block](ffn_input))
            ffn_output = self.ffn_dropout1(ffn_output)
            ffn_output = self.ffn_dropout2(self.ffn_linear2[block](ffn_output))
            x = (ffn_output + ffn_input) * valid_mask_float
        return self.final_norm(x) * valid_mask_float


# =========================  4.3.4.5. SASRec đa hành vi  =========================
class MultiBehaviorSASRec(SASRecBackbone):
    # x_t = Fuse( E_item(i_t), E_behavior(b_t) ) + P_t   (Fuse = cộng hoặc nối rồi chiếu tuyến tính)
    def __init__(self, num_items, fusion="add", **kw):
        super().__init__(num_items, **kw)
        assert fusion in ("add", "concat")
        self.fusion = fusion
        self.beh_emb = nn.Embedding(len(BEHAVIOR2ID) + 1, self.hidden_units, padding_idx=0)
        nn.init.xavier_uniform_(self.beh_emb.weight)
        with torch.no_grad():
            self.beh_emb.weight[0].zero_()
        if fusion == "concat":
            self.fuse = nn.Linear(2 * self.hidden_units, self.hidden_units)
            nn.init.xavier_uniform_(self.fuse.weight)
            nn.init.zeros_(self.fuse.bias)

    def _embed(self, input_seq, beh_seq):
        e_item = self.item_emb(input_seq) * (self.hidden_units ** 0.5)
        e_beh = self.beh_emb(beh_seq)
        if self.fusion == "add":
            return e_item + e_beh
        return self.fuse(torch.cat([e_item, e_beh], dim=-1))


# =========================  4.3.4.6. HSTU  =========================
class HSTULayer(nn.Module):
    def __init__(self, d, n_heads, dropout, maxlen):
        super().__init__()
        assert d % n_heads == 0
        self.h, self.dh, self.L = n_heads, d // n_heads, maxlen
        self.ln_in = nn.LayerNorm(d, eps=1e-8)
        self.f1 = nn.Linear(d, 4 * d)                 # tạo U, V, Q, K
        self.ln_av = nn.LayerNorm(d, eps=1e-8)
        self.f2 = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)
        self.rab = nn.Parameter(torch.zeros(n_heads, 2 * maxlen - 1))   # thiên lệch vị trí tương đối
        idx = torch.arange(maxlen)
        self.register_buffer("rel_idx", (idx.unsqueeze(0) - idx.unsqueeze(1)) + maxlen - 1, persistent=False)
        nn.init.xavier_uniform_(self.f1.weight); nn.init.zeros_(self.f1.bias)
        nn.init.xavier_uniform_(self.f2.weight); nn.init.zeros_(self.f2.bias)

    def forward(self, x, valid):
        B, L, d = x.shape
        # (1) Pointwise Projection
        u, v, q, k = F.silu(self.f1(self.ln_in(x))).chunk(4, dim=-1)
        sp_ = lambda t: t.view(B, L, self.h, self.dh).transpose(1, 2)
        u, v, q, k = sp_(u), sp_(v), sp_(q), sp_(k)
        # (2) Causal Spatial Aggregation: SiLU(QK^T + rab), mặt nạ nhân quả + padding, chia cho số vị trí được nhìn
        scores = torch.matmul(q, k.transpose(-1, -2)) + self.rab[:, self.rel_idx[:L, :L]].unsqueeze(0)
        attn = F.silu(scores)
        causal = torch.tril(torch.ones(L, L, device=x.device, dtype=torch.bool))
        allowed = (causal.unsqueeze(0) & valid.unsqueeze(1)).unsqueeze(1).float()
        attn = self.drop(attn * allowed / allowed.sum(-1, keepdim=True).clamp(min=1.0))
        av = torch.matmul(attn, v).transpose(1, 2).reshape(B, L, d)
        u = u.transpose(1, 2).reshape(B, L, d)
        # (3) Pointwise Transformation (cổng GLU) + phần dư
        return x + self.f2(self.drop(self.ln_av(av) * u))


class HSTURecommender(nn.Module):
    def __init__(self, num_items, hidden_units=50, maxlen=50, num_blocks=2, num_heads=1,
                 dropout_rate=0.5, use_behavior=True):
        super().__init__()
        self.hidden_units, self.use_behavior = hidden_units, use_behavior
        self.item_emb = nn.Embedding(num_items + 1, hidden_units, padding_idx=0)
        nn.init.xavier_uniform_(self.item_emb.weight)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()
        if use_behavior:
            self.beh_emb = nn.Embedding(len(BEHAVIOR2ID) + 1, hidden_units, padding_idx=0)
            nn.init.xavier_uniform_(self.beh_emb.weight)
            with torch.no_grad():
                self.beh_emb.weight[0].zero_()
        self.input_dropout = nn.Dropout(dropout_rate)
        self.layers = nn.ModuleList([HSTULayer(hidden_units, num_heads, dropout_rate, maxlen)
                                     for _ in range(num_blocks)])
        self.final_norm = nn.LayerNorm(hidden_units, eps=1e-8)

    def forward(self, input_seq, beh_seq=None):
        valid = input_seq != 0
        vf = valid.unsqueeze(-1).float()
        x = self.item_emb(input_seq) * (self.hidden_units ** 0.5)
        if self.use_behavior and beh_seq is not None:
            x = x + self.beh_emb(beh_seq)
        x = self.input_dropout(x) * vf
        for layer in self.layers:
            x = layer(x, valid) * vf
        return self.final_norm(x) * vf


# %%
# ============================================================================
# BƯỚC 3f. Huấn luyện và đánh giá test (có thể chạy lại, seed đã xong sẽ được bỏ qua)
# ============================================================================
# ---------- Siêu tham số: giống cell huấn luyện SASRec của nhóm ----------
HIDDEN_UNITS, NUM_BLOCKS, NUM_HEADS, DROPOUT_RATE = 50, 2, 1, 0.5
LEARNING_RATE, BATCH_SIZE, EPOCHS, EVAL_BATCH_SIZE = 0.001, 128, 10, 1024
NUM_BATCHES_PER_EPOCH = len(train_seq_view) // BATCH_SIZE          # 190980 // 128 = 1492
SEEDS = [2026, 2027, 2028]
if QUICK_TEST:                     # chạy thử nhanh để kiểm tra lỗi, KHÔNG dùng số liệu này cho báo cáo
    SEEDS, EPOCHS, NUM_BATCHES_PER_EPOCH = [2026], 2, 200
LOSS_VIEW_TARGETS_ONLY = False     # True: chỉ tính loss ở vị trí mà sản phẩm kế tiếp là view
HSTU_DROPOUT = DROPOUT_RATE        # đổi nếu muốn chỉnh riêng cho HSTU (chọn theo validation, không theo test)

MODEL_SPECS = {
    "SASRec":                         {"tag": "sasrec",          "kind": "sasrec",    "data": "view", "seeds": SEEDS},
    "MB-SASRec (add)":                {"tag": "mbsasrec_add",    "kind": "mb_add",    "data": "all",  "seeds": SEEDS},
    "HSTU":                           {"tag": "hstu",            "kind": "hstu",      "data": "all",  "seeds": SEEDS},
    # ---- ablation / đối chứng (bỏ comment trong RUN_MODELS để chạy) ----
    "MB-SASRec (concat)":             {"tag": "mbsasrec_concat", "kind": "mb_concat", "data": "all",  "seeds": SEEDS},
    "SASRec (all events, no beh)":    {"tag": "sasrec_allevents","kind": "sasrec",    "data": "all",  "seeds": SEEDS},
    "HSTU (view-only)":               {"tag": "hstu_viewonly",   "kind": "hstu_nob",  "data": "view", "seeds": SEEDS},
    "SASRec (control, rerun)":        {"tag": "sasrec_control",  "kind": "sasrec",    "data": "view", "seeds": [2026]},
}
RUN_MODELS = ["SASRec", "MB-SASRec (add)", "HSTU"]      # đủ 3 mô hình học sâu của so sánh 5 mô hình

DATA = {
    "view": dict(seqs=view_train, users=trainable_view,
                 valid=(valid_seq_view, None), test=(test_seq_view, None)),
    "all":  dict(seqs=mb_train,
                 users=np.asarray([u for u, (i, _) in mb_train.items() if len(i) > 1], dtype=np.int64),
                 valid=(mb_valid_seq, mb_valid_beh), test=(mb_test_seq, mb_test_beh)),
}
for k, v in DATA.items():
    print(f"Dữ liệu '{k}': {len(v['users']):,} khách hàng huấn luyện được")


def build_model(kind):
    kw = dict(hidden_units=HIDDEN_UNITS, maxlen=MAXLEN, num_blocks=NUM_BLOCKS, num_heads=NUM_HEADS,
              dropout_rate=DROPOUT_RATE)
    if kind == "sasrec":    return SASRecBackbone(NUM_ITEMS, **kw)
    if kind == "mb_add":    return MultiBehaviorSASRec(NUM_ITEMS, fusion="add", **kw)
    if kind == "mb_concat": return MultiBehaviorSASRec(NUM_ITEMS, fusion="concat", **kw)
    kw["dropout_rate"] = HSTU_DROPOUT
    if kind == "hstu":      return HSTURecommender(NUM_ITEMS, use_behavior=True, **kw)
    if kind == "hstu_nob":  return HSTURecommender(NUM_ITEMS, use_behavior=False, **kw)
    raise ValueError(kind)


@torch.inference_mode()
def rank_positive(model, seq, beh, cand, priorities=None):
    model.eval()
    out = []
    for s in range(0, len(seq), EVAL_BATCH_SIZE):
        e = min(s + EVAL_BATCH_SIZE, len(seq))
        sq = torch.from_numpy(np.array(seq[s:e], dtype=np.int64, copy=True)).to(device)
        cd = torch.from_numpy(np.array(cand[s:e], dtype=np.int64, copy=True)).to(device)
        bh = None if beh is None else torch.from_numpy(np.array(beh[s:e], dtype=np.int64, copy=True)).to(device)
        last = model(sq, bh)[:, -1, :]
        sc = torch.sum(last.unsqueeze(1) * model.item_emb(cd), dim=-1)
        better = sc[:, 1:] > sc[:, [0]]
        if priorities is not None:                               # phá hòa bằng priorities dùng chung
            pr = torch.from_numpy(np.array(priorities[s:e], dtype=np.int64, copy=True)).to(device)
            better = better | ((sc[:, 1:] == sc[:, [0]]) & (pr[:, 1:] < pr[:, [0]]))
        out.append((1 + better.sum(dim=1)).cpu().numpy().astype(np.int32))
    return np.concatenate(out)


def train_one(name, spec, seed):
    tag, d = spec["tag"], DATA[spec["data"]]
    run_dir = os.path.join(MB_DIR, "runs", tag, f"seed{seed}")
    os.makedirs(run_dir, exist_ok=True)
    metrics_path = os.path.join(RESULTS_DIR, f"{tag}_seed{seed}_test_metrics_shared_fair_common.json")
    if os.path.exists(metrics_path):
        print(f"[{name}] seed {seed} đã hoàn thành, bỏ qua.")
        with open(metrics_path, encoding="utf-8") as f:
            return json.load(f)

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    model = build_model(spec["kind"]).to(device)
    use_beh = spec["kind"] in ("mb_add", "mb_concat", "hstu")
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.98), weight_decay=0.0)
    vs, vb = d["valid"]; ts, tb = d["test"]
    if not use_beh:
        vb = tb = None
    best_ndcg, best_epoch, history, t_start = -1.0, None, [], time.time()
    print("\n" + "=" * 65 + f"\n{name} — SEED {seed}\n" + "=" * 65)

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time(); model.train(); run_loss, run_pos = 0.0, 0
        for step in range(NUM_BATCHES_PER_EPOCH):
            seq_np, beh_np, pos_np, pbeh_np, neg_np = make_batch(d["seqs"], d["users"], BATCH_SIZE,
                                                                  MAXLEN, NUM_ITEMS, rng)
            seq, pos, neg = [torch.from_numpy(a).to(device) for a in (seq_np, pos_np, neg_np)]
            beh = torch.from_numpy(beh_np).to(device) if use_beh else None
            mask = pos != 0
            if LOSS_VIEW_TARGETS_ONLY:
                mask = mask & (torch.from_numpy(pbeh_np).to(device) == 1)
            if not mask.any():
                continue
            opt.zero_grad(set_to_none=True)
            hidden = model(seq, beh)
            pos_logits = torch.sum(hidden * model.item_emb(pos), dim=-1)
            neg_logits = torch.sum(hidden * model.item_emb(neg), dim=-1)
            loss = -F.logsigmoid(pos_logits)[mask].mean() - F.logsigmoid(-neg_logits)[mask].mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f"Loss không hữu hạn: {name}, seed={seed}, epoch={epoch}, step={step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            n = int(mask.sum().item()); run_loss += float(loss.detach()) * n; run_pos += n
            if (step + 1) % 250 == 0:
                print(f"{name} | seed {seed} | epoch {epoch}/{EPOCHS} | batch {step + 1}/{NUM_BATCHES_PER_EPOCH}")

        ranks = rank_positive(model, vs, vb, valid_cand_idx)             # validation: không dùng priorities (như notebook gốc)
        hr, ndcg = float(np.mean(ranks <= 10)), float(np.mean(np.where(ranks <= 10, 1 / np.log2(ranks + 1), 0)))
        history.append({"epoch": epoch, "train_loss": run_loss / max(run_pos, 1), "valid_hr10": hr,
                        "valid_ndcg10": ndcg, "valid_median_rank": float(np.median(ranks)),
                        "epoch_time_sec": time.time() - t0})
        print(f"{name} | seed {seed} | epoch {epoch}/{EPOCHS} | loss={history[-1]['train_loss']:.6f} | "
              f"VALID HR@10={hr:.6f} NDCG@10={ndcg:.6f} | {(time.time() - t0) / 60:.2f} phút")
        if ndcg > best_ndcg:
            best_ndcg, best_epoch = ndcg, epoch
            torch.save({"epoch": epoch, "seed": seed, "model_state_dict": model.state_dict(),
                        "valid_ndcg10": ndcg}, os.path.join(run_dir, "best_checkpoint.pt"))
        with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump({"seed": seed, "best_epoch": best_epoch, "best_valid_ndcg10": best_ndcg,
                       "history": history}, f, ensure_ascii=False, indent=2)

    # ---- Đánh giá test bằng checkpoint tốt nhất, dùng test priorities chung ----
    ckpt = torch.load(os.path.join(run_dir, "best_checkpoint.pt"), map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    test_ranks = rank_positive(model, ts, tb, test_cand_idx, priorities=test_priorities)
    m = metrics_from_ranks(test_ranks, name, "TEST", seed=seed, best_epoch=int(ckpt["epoch"]))
    m["total_time_sec"] = time.time() - t_start
    np.save(os.path.join(RESULTS_DIR, f"{tag}_seed{seed}_test_positive_rank_shared_fair_common.npy"), test_ranks)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)
    print(f">>> {name} | seed {seed} | TEST HR@10={m['HR@10']:.6f} NDCG@10={m['NDCG@10']:.6f} (best epoch {m['best_epoch']})")
    del model, opt; torch.cuda.empty_cache()
    return m


def summarize_model(name, spec, per_seed):
    df = pd.DataFrame(per_seed)
    s = {"model": name, "split": "TEST", "tie_breaking": "shared_random_priorities",
         "num_seeds": len(df), "seeds": [int(x) for x in df["seed"]], "num_users": 172513,
         "num_candidates_per_user": N_CAND,
         "mean_HR@10": float(df["HR@10"].mean()), "std_HR@10_sample": float(df["HR@10"].std(ddof=1)) if len(df) > 1 else None,
         "mean_NDCG@10": float(df["NDCG@10"].mean()), "std_NDCG@10_sample": float(df["NDCG@10"].std(ddof=1)) if len(df) > 1 else None}
    with open(os.path.join(RESULTS_DIR, f"{spec['tag']}_test_summary_shared_fair_common.json"), "w", encoding="utf-8") as f:
        json.dump(s, f, ensure_ascii=False, indent=2)
    df.to_csv(os.path.join(RESULTS_DIR, f"{spec['tag']}_test_metrics_by_seed_shared_fair_common.csv"), index=False)
    return s


# ---------- Chạy ----------
for name in RUN_MODELS:
    spec = MODEL_SPECS[name]
    per_seed = [train_one(name, spec, seed) for seed in spec["seeds"]]
    s = summarize_model(name, spec, per_seed)
    print(f"\n===== {name}: TEST (trung bình {s['num_seeds']} seed) =====")
    print(f"HR@10 = {s['mean_HR@10']:.6f} | NDCG@10 = {s['mean_NDCG@10']:.6f}")


# %%
# ============================================================================
# BƯỚC 4. Bảng so sánh 5 mô hình và kiểm định bootstrap
# ============================================================================
def load_json(name):
    with open(os.path.join(RESULTS_DIR, name), encoding="utf-8") as f:
        return json.load(f)

def per_user(ranks_list):
    # trung bình theo seed của chỉ số theo từng khách hàng
    hr = np.mean([(r <= 10).astype(np.float64) for r in ranks_list], axis=0)
    nd = np.mean([np.where(r <= 10, 1.0 / np.log2(r + 1), 0.0) for r in ranks_list], axis=0)
    return hr, nd

MAIN = [  # (tên hiển thị, file tóm tắt, danh sách file rank theo từng seed)
    ("Popularity",         "popularity_test_metrics_shared_fair.json", ["popularity_test_positive_rank_shared_fair.npy"]),
    ("Item-CF",            "itemcf_test_metrics_shared_fair_common.json", ["itemcf_test_positive_rank_shared_fair_common.npy"]),
    ("SASRec",             "sasrec_test_summary_shared_fair_common.json",
        [f"sasrec_seed{s}_test_positive_rank_shared_fair_common.npy" for s in SEEDS]),
    ("MB-SASRec (add)",    "mbsasrec_add_test_summary_shared_fair_common.json",
        [f"mbsasrec_add_seed{s}_test_positive_rank_shared_fair_common.npy" for s in SEEDS]),
    ("HSTU",               "hstu_test_summary_shared_fair_common.json",
        [f"hstu_seed{s}_test_positive_rank_shared_fair_common.npy" for s in SEEDS]),
]

rows, user_hr, user_nd = [], {}, {}
for name, fsum, franks in MAIN:
    r = load_json(fsum)
    assert r["num_users"] == 172513 and r["num_candidates_per_user"] == 101, f"{name}: khác benchmark"
    assert r["tie_breaking"] == "shared_random_priorities", f"{name}: khác cách phá hòa"
    ranks = [np.load(os.path.join(RESULTS_DIR, f)) for f in franks]
    user_hr[name], user_nd[name] = per_user(ranks)
    multi = "mean_HR@10" in r
    rows.append({"Model": name + (f" (mean {r['num_seeds']} seeds)" if multi else ""),
                 "HR@10": r["mean_HR@10"] if multi else r["HR@10"],
                 "HR@10 std": r.get("std_HR@10_sample"),
                 "NDCG@10": r["mean_NDCG@10"] if multi else r["NDCG@10"],
                 "NDCG@10 std": r.get("std_NDCG@10_sample"), "key": name})

table = pd.DataFrame(rows)
table["HR@10 rank"] = table["HR@10"].rank(ascending=False, method="min").astype(int)
table["NDCG@10 rank"] = table["NDCG@10"].rank(ascending=False, method="min").astype(int)
print("===== SO SÁNH 5 MÔ HÌNH — TEST, 172.513 khách hàng, 101 ứng viên, phá hòa chung =====")
print(table.drop(columns="key").round(6).to_string(index=False))

best = table.sort_values("NDCG@10", ascending=False).iloc[0]["key"]
print(f"\nMô hình tốt nhất theo NDCG@10: {best}")
print(f"Mô hình tốt nhất theo HR@10 : {table.sort_values('HR@10', ascending=False).iloc[0]['key']}")

print(f"\n--- Kiểm định paired bootstrap: {best} so với từng mô hình còn lại (chênh lệch = {best} − mô hình kia) ---")
tests = []
for other in table["key"]:
    if other == best:
        continue
    d_nd, ci_nd = paired_bootstrap(user_nd[best], user_nd[other])
    d_hr, ci_hr = paired_bootstrap(user_hr[best], user_hr[other])
    sig = "khác biệt có ý nghĩa" if (ci_nd[0] > 0 or ci_nd[1] < 0) else "chưa phân biệt được"
    tests.append({"so với": other, "Δ NDCG@10": d_nd, "CI95 NDCG@10": f"[{ci_nd[0]:+.4f}, {ci_nd[1]:+.4f}]",
                  "Δ HR@10": d_hr, "CI95 HR@10": f"[{ci_hr[0]:+.4f}, {ci_hr[1]:+.4f}]", "kết luận (NDCG@10)": sig})
tests_df = pd.DataFrame(tests)
print(tests_df.round(4).to_string(index=False))

table.drop(columns="key").to_csv(os.path.join(RESULTS_DIR, "ml3_five_model_comparison.csv"), index=False, encoding="utf-8-sig")
tests_df.to_csv(os.path.join(RESULTS_DIR, "ml3_five_model_bootstrap_vs_best.csv"), index=False, encoding="utf-8-sig")
print("\nĐã lưu: ml3_five_model_comparison.csv, ml3_five_model_bootstrap_vs_best.csv tại", RESULTS_DIR)

# ---- Bảng phụ: các biến thể/ablation đã chạy (nếu có) ----
extra = []
for name, spec in MODEL_SPECS.items():
    p = os.path.join(RESULTS_DIR, f"{spec['tag']}_test_summary_shared_fair_common.json")
    if spec["tag"] not in ("mbsasrec_add", "hstu") and os.path.exists(p):
        r = load_json(p)
        extra.append({"Model": name, "seeds": r["num_seeds"], "HR@10": r["mean_HR@10"], "NDCG@10": r["mean_NDCG@10"]})
if extra:
    print("\n===== BIẾN THỂ / ABLATION ĐÃ CHẠY =====")
    print(pd.DataFrame(extra).round(6).to_string(index=False))