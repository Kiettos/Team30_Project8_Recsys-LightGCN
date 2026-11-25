"""preprocess_ml1m.py - Tiền xử lý MovieLens-1M cho MF và LightGCN.

Chạy từ thư mục gốc project:
    python src/data/preprocess_ml1m.py                     # mặc định: implicit, k-core 5, split TEMPORAL
    python src/data/preprocess_ml1m.py --split random      # (không khuyến nghị) split ngẫu nhiên
    python src/data/preprocess_ml1m.py --feedback explicit # giữ rating 1-5 cho MF dự đoán rating

Đầu vào : data/ratings.dat  (UserID::MovieID::Rating::Timestamp)
          data/users.dat, data/movies.dat (side information, tuỳ chọn)
Đầu ra  : processed/   (mặc định, nằm cạnh data/)
    mf/train.csv, val.csv, test.csv        -> cột: user,item,rating,timestamp
    lightgcn/train.txt, val.txt, test.txt  -> mỗi dòng: "user item1 item2 ..."
    lightgcn/adj_norm.npz                  -> D^-1/2 A D^-1/2 (scipy sparse, (U+I)x(U+I))
    side_info/users.csv                    -> gender, age_group, occupation, zip_id (từ users.dat)
    side_info/items.csv                    -> title, year, genres + multi-hot (từ movies.dat)
    user2id.json, item2id.json             -> mapping ID gốc -> ID mới (0..N-1)
    stats.json                             -> n_users, n_items, số tương tác, chế độ split...

Ghi chú về split:
    temporal (mặc định): trong mỗi user, sắp theo timestamp tăng dần và lấy
        10% cuối làm test, 10% ngay trước đó làm val, phần còn lại làm train.
        Cách này mô phỏng đúng bài toán thực tế (dự đoán tương lai từ quá khứ).
    random: xáo trộn ngẫu nhiên trong từng user. Nhanh nhưng để lộ thông tin
        tương lai (một item tương lai có thể nằm trong train), chỉ nên dùng để
        so sánh phụ và phải ghi rõ trong báo cáo.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp

# Console Windows mặc định cp1252/cp1258 -> tránh crash khi in tiếng Việt
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Thư mục gốc project (…/Team30_Project8_Recsys-LightGCN)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def resolve(path):
    """Đường dẫn tương đối: ưu tiên CWD, nếu không có thì thử thư mục gốc project."""
    if os.path.isabs(path):
        return path
    return path if os.path.exists(path) else os.path.join(ROOT, path)


def load_ratings(data_dir):
    path = os.path.join(data_dir, "ratings.dat")
    return pd.read_csv(
        path, sep="::", engine="python",
        names=["user", "item", "rating", "timestamp"],
    )


def k_core(df, k):
    """Lọc lặp đến khi mọi user và item đều có >= k tương tác."""
    while True:
        uc = df["user"].value_counts()
        ic = df["item"].value_counts()
        keep = df["user"].map(uc).ge(k) & df["item"].map(ic).ge(k)
        if keep.all():
            return df
        df = df[keep]


def split_per_user(df, val_ratio, test_ratio, mode, seed):
    """Chia train/val/test cho từng user.
    mode='temporal': tương tác cũ nhất -> train, mới nhất -> test (sort theo timestamp)
    mode='random'  : xáo trộn ngẫu nhiên
    """
    rng = np.random.RandomState(seed)
    train, val, test = [], [], []
    for _, g in df.groupby("user", sort=True):
        if mode == "temporal":
            g = g.sort_values("timestamp", kind="mergesort")
        else:
            g = g.iloc[rng.permutation(len(g))]
        n = len(g)
        n_test = max(1, int(round(n * test_ratio)))
        n_val = max(1, int(round(n * val_ratio)))
        n_train = n - n_val - n_test
        if n_train <= 0:                      # user quá ít tương tác: bỏ
            continue
        train.append(g.iloc[:n_train])
        val.append(g.iloc[n_train:n_train + n_val])
        test.append(g.iloc[n_train + n_val:])
    if not train:
        raise SystemExit("Không còn user nào sau khi chia split.")
    return pd.concat(train), pd.concat(val), pd.concat(test)


def write_adj_list(df, path):
    """Định dạng LightGCN gốc: 'user item1 item2 ...'"""
    with open(path, "w") as f:
        for u, g in df.groupby("user", sort=True):
            f.write(str(u) + " " + " ".join(map(str, g["item"].tolist())) + "\n")


def build_norm_adj(train_df, n_users, n_items):
    """A = [[0, R], [R^T, 0]],  A_hat = D^-1/2 A D^-1/2"""
    R = sp.coo_matrix(
        (np.ones(len(train_df), dtype=np.float32),
         (train_df["user"].values, train_df["item"].values)),
        shape=(n_users, n_items),
    )
    top = sp.hstack([sp.csr_matrix((n_users, n_users)), R])
    bottom = sp.hstack([R.T, sp.csr_matrix((n_items, n_items))])
    A = sp.vstack([top, bottom]).tocsr()
    deg = np.asarray(A.sum(axis=1)).flatten()
    d_inv_sqrt = np.zeros_like(deg)
    np.power(deg, -0.5, out=d_inv_sqrt, where=deg > 0)
    D = sp.diags(d_inv_sqrt)
    return (D @ A @ D).tocoo().astype(np.float32)


def build_user_features(data_dir, user2id):
    """users.dat: UserID::Gender::Age::Occupation::Zip-code -> bảng đặc trưng đã encode."""
    users = pd.read_csv(
        os.path.join(data_dir, "users.dat"), sep="::", engine="python",
        names=["orig_user", "gender", "age", "occupation", "zip"], encoding="latin-1",
    )
    users = users[users["orig_user"].isin(user2id)].copy()
    users["user"] = users["orig_user"].map(user2id)
    users["gender"] = (users["gender"] == "M").astype(int)           # F=0, M=1
    age_map = {a: i for i, a in enumerate(sorted(users["age"].unique()))}
    users["age_group"] = users["age"].map(age_map)                    # 7 nhóm tuổi -> 0..6
    users["zip_id"] = users["zip"].astype(str).astype("category").cat.codes
    users = users.rename(columns={"age": "age_raw"})
    cols = ["user", "orig_user", "gender", "age_raw", "age_group", "occupation", "zip", "zip_id"]
    return users[cols].sort_values("user").reset_index(drop=True)


def build_item_features(data_dir, item2id):
    """movies.dat: MovieID::Title::Genres -> title, year, genres + multi-hot."""
    movies = pd.read_csv(
        os.path.join(data_dir, "movies.dat"), sep="::", engine="python",
        names=["orig_item", "title", "genres"], encoding="latin-1",
    )
    movies = movies[movies["orig_item"].isin(item2id)].copy()
    movies["item"] = movies["orig_item"].map(item2id)
    movies["year"] = movies["title"].str.extract(r"\((\d{4})\)\s*$")[0].astype("float").astype("Int64")
    genre_onehot = movies["genres"].str.get_dummies(sep="|").add_prefix("genre_")
    movies = pd.concat([movies, genre_onehot], axis=1)
    cols = ["item", "orig_item", "title", "year", "genres"] + list(genre_onehot.columns)
    return movies[cols].sort_values("item").reset_index(drop=True)


def check_contiguous(df, col, n):
    """Đảm bảo ID mới là 0..n-1 liên tục (điều kiện để dùng làm index embedding)."""
    mx = int(df[col].max()) if len(df) else -1
    if mx + 1 != n:
        raise SystemExit(f"Lỗi remap ID: cột {col} có max={mx} nhưng n={n} -> ID không liên tục.")
    return n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="./data",
                   help="thư mục chứa ratings.dat/users.dat/movies.dat (mặc định ./data)")
    p.add_argument("--out_dir", default=None,
                   help="mặc định: <thư mục cha của data_dir>/processed, vd ./processed")
    p.add_argument("--feedback", choices=["implicit", "explicit"], default="implicit",
                   help="implicit: giữ rating>=threshold làm tương tác dương (khuyến nghị cho LightGCN); "
                        "explicit: giữ toàn bộ rating 1-5 (cho MF dự đoán rating)")
    p.add_argument("--threshold", type=float, default=4.0)
    p.add_argument("--core", type=int, default=5, help="k-core filtering")
    p.add_argument("--split", choices=["temporal", "random"], default="temporal",
                   help="temporal (mặc định): theo thời gian, không rò rỉ tương lai; "
                        "random: xáo trộn ngẫu nhiên (chỉ dùng để so sánh phụ)")
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--test_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=2024)
    args = p.parse_args()

    data_dir = resolve(args.data_dir)
    if args.out_dir is None:
        args.out_dir = os.path.join(os.path.dirname(os.path.abspath(data_dir)), "processed")
    out_dir = resolve(args.out_dir)

    df = load_ratings(data_dir)
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    raw = dict(n_ratings=len(df), n_users=int(df.user.nunique()), n_items=int(df.item.nunique()))
    print(f"Raw: {raw['n_ratings']} ratings, {raw['n_users']} users, {raw['n_items']} items")

    if args.feedback == "implicit":
        df = df[df["rating"] >= args.threshold].copy()
        print(f"Implicit (rating >= {args.threshold}): còn {len(df)} tương tác")
    df = df.drop_duplicates(["user", "item"], keep="last")
    df = k_core(df, args.core).copy()

    # Remap ID về 0..N-1 liên tục
    user2id = {int(u): i for i, u in enumerate(sorted(df["user"].unique()))}
    item2id = {int(m): i for i, m in enumerate(sorted(df["item"].unique()))}
    df["user"] = df["user"].map(user2id)
    df["item"] = df["item"].map(item2id)
    n_users, n_items = len(user2id), len(item2id)
    check_contiguous(df, "user", n_users)
    check_contiguous(df, "item", n_items)
    density = len(df) / (n_users * n_items)
    print(f"Sau lọc: {len(df)} tương tác, {n_users} users, {n_items} items, density={density:.4%}")

    train, val, test = split_per_user(df, args.val_ratio, args.test_ratio,
                                      args.split, args.seed)
    print(f"Split={args.split} | Train/Val/Test = {len(train)}/{len(val)}/{len(test)}")

    # Kiểm tra chống rò rỉ: mỗi cặp (user,item) chỉ thuộc đúng 1 tập
    keys = [set(map(tuple, d[["user", "item"]].values)) for d in (train, val, test)]
    if keys[0] & keys[1] or keys[0] & keys[2] or keys[1] & keys[2]:
        raise SystemExit("Phát hiện cặp (user,item) xuất hiện ở nhiều tập -> split bị lỗi.")
    if args.split == "temporal":
        for name, d in (("train", train), ("val", val), ("test", test)):
            print(f"  {name}: timestamp {int(d.timestamp.min())}..{int(d.timestamp.max())}")
        # Kiểm tra theo từng user: train cũ hơn val, val cũ hơn test
        tmax = train.groupby("user")["timestamp"].max()
        vmin = val.groupby("user")["timestamp"].min()
        vmax = val.groupby("user")["timestamp"].max()
        tmin = test.groupby("user")["timestamp"].min()
        if (vmin < tmax).any():
            raise SystemExit("Split temporal bị lỗi: có item val cũ hơn item train của cùng user.")
        if (tmin < vmax).any():
            raise SystemExit("Split temporal bị lỗi: có item test cũ hơn item val của cùng user.")

    # ---- MF ----
    mf_dir = os.path.join(out_dir, "mf")
    os.makedirs(mf_dir, exist_ok=True)
    for name, d in [("train", train), ("val", val), ("test", test)]:
        d.sort_values(["user", "item"]).to_csv(
            os.path.join(mf_dir, f"{name}.csv"), index=False)

    # ---- LightGCN ----
    gcn_dir = os.path.join(out_dir, "lightgcn")
    os.makedirs(gcn_dir, exist_ok=True)
    for name, d in [("train", train), ("val", val), ("test", test)]:
        write_adj_list(d.sort_values(["user", "item"]), os.path.join(gcn_dir, f"{name}.txt"))
    adj = build_norm_adj(train, n_users, n_items)
    sp.save_npz(os.path.join(gcn_dir, "adj_norm.npz"), adj)
    print(f"Adjacency: {adj.shape[0]}x{adj.shape[1]}, nnz={adj.nnz}")

    # ---- Users / Items (side information) + mapping ID gốc -> ID mới ----
    side_dir = os.path.join(out_dir, "side_info")
    os.makedirs(side_dir, exist_ok=True)
    build_user_features(data_dir, user2id).to_csv(
        os.path.join(side_dir, "users.csv"), index=False)
    build_item_features(data_dir, item2id).to_csv(
        os.path.join(side_dir, "items.csv"), index=False)
    with open(os.path.join(out_dir, "user2id.json"), "w") as f:
        json.dump(user2id, f)
    with open(os.path.join(out_dir, "item2id.json"), "w") as f:
        json.dump(item2id, f)

    stats = dict(
        n_users=int(n_users), n_items=int(n_items),
        n_train=len(train), n_val=len(val), n_test=len(test),
        n_interactions=int(len(df)), density=float(density),
        feedback=args.feedback, threshold=args.threshold,
        core=args.core, split=args.split, seed=args.seed,
        val_ratio=args.val_ratio, test_ratio=args.test_ratio,
        raw=raw,
    )
    with open(os.path.join(out_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print("Xong ->", out_dir)


if __name__ == "__main__":
    main()
