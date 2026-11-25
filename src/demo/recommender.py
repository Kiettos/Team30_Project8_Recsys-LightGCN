"""src/demo/recommender.py - Nạp dữ liệu + checkpoint, sinh gợi ý và dữ liệu đồ thị cho demo.

Hai chế độ:
  1. User có sẵn  : dùng embedding đã học của user (có thể trộn thêm phim vừa xem).
  2. User mới     : cold-start - hồ sơ user = trung bình CÓ TRỌNG SỐ THỜI GIAN của
                    embedding các phim vừa xem (phim xem sau có trọng số cao hơn).
                    Mỗi lần thêm/bớt phim, hồ sơ và danh sách gợi ý được tính lại ngay,
                    không cần train lại.

Embedding item của LightGCN là embedding SAU lan truyền (trung bình các tầng), đúng
như lúc đánh giá trong Train_lightgcn.py. Với MF thì dùng embedding gốc.
"""
import json
import os

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
K = 20
CKPT_DIR = os.path.join(ROOT, "checkpoints")


def list_checkpoints(root=ROOT):
    """Danh sách checkpoint có sẵn, ưu tiên lightgcn_s0.pth rồi mf_s0.pth."""
    d = os.path.join(root, "checkpoints")
    if not os.path.isdir(d):
        return []
    files = [f for f in os.listdir(d) if f.endswith(".pth")]
    order = {"lightgcn_s0.pth": 0, "mf_s0.pth": 1}
    return [os.path.join(d, f) for f in sorted(files, key=lambda f: (order.get(f, 9), f))]


class DemoModel:
    def __init__(self, root=ROOT, ckpt=None):
        self.root = root
        with open(os.path.join(root, "processed", "stats.json"), encoding="utf-8") as f:
            self.stats = json.load(f)
        self.nu = int(self.stats["n_users"])
        self.ni = int(self.stats["n_items"])

        items = pd.read_csv(os.path.join(root, "processed", "side_info", "items.csv"))
        items = items.set_index("item")
        self.items = items
        self.titles = [self._label(i) for i in range(self.ni)]

        with open(os.path.join(root, "processed", "item2id.json"), encoding="utf-8") as f:
            self.item2id = {int(k): int(v) for k, v in json.load(f).items()}
        self.id2item = {v: k for k, v in self.item2id.items()}
        with open(os.path.join(root, "processed", "user2id.json"), encoding="utf-8") as f:
            self.user2id = {int(k): int(v) for k, v in json.load(f).items()}
        self.id2user = {v: k for k, v in self.user2id.items()}

        tr = pd.read_csv(os.path.join(root, "processed", "mf", "train.csv"),
                         usecols=["user", "item"])
        self.train_pairs = tr
        self.users_of_item = {int(i): set(g["user"].tolist())
                              for i, g in tr.groupby("item")}
        self.pop = tr["item"].value_counts()
        self.item_ids_by_pop = [int(i) for i in self.pop.index]

        self.ckpt_path = ckpt or (list_checkpoints(root) or [None])[0]
        if not self.ckpt_path or not os.path.exists(self.ckpt_path):
            raise FileNotFoundError(
                "Chưa có checkpoint trong checkpoints/. Hãy chạy:\n"
                "  venv\\Scripts\\python.exe src\\model\\Train_lightgcn.py --final\n"
                "  venv\\Scripts\\python.exe src\\model\\Train_mf.py --final"
            )
        self.kind, self.user_emb, self.item_emb = self._load(self.ckpt_path)

    # ------------------------------ dữ liệu ------------------------------
    def _label(self, idx):
        row = self.items.loc[idx]
        title = str(row["title"]).strip()
        year = row.get("year")
        try:
            if pd.notna(year):
                return f"{title} ({int(year)})"
        except (TypeError, ValueError):
            pass
        return title

    def title(self, idx, max_len=46):
        t = self.titles[int(idx)]
        return t if len(t) <= max_len else t[: max_len - 1] + "…"

    def genres(self, idx):
        return str(self.items.loc[int(idx)]["genres"]).replace("|", ", ")

    def year(self, idx):
        y = self.items.loc[int(idx)]["year"]
        return "" if pd.isna(y) else str(int(y))

    def raw_item_id(self, idx):
        return self.id2item.get(int(idx), "?")

    def raw_user_id(self, uid):
        return self.id2user.get(int(uid), "?")

    def watched_by(self, uid):
        """Các item user đã tương tác trong train, theo thứ tự file (thời gian tăng dần)."""
        return self.train_pairs.loc[self.train_pairs["user"] == uid, "item"].tolist()

    def popularity_rank(self, k=K, exclude=()):
        ex = set(exclude)
        return [i for i in self.item_ids_by_pop if i not in ex][:k]

    def cooccur(self, i, j):
        s = self.users_of_item.get(int(i), set()) & self.users_of_item.get(int(j), set())
        return len(s)

    # ------------------------------ model ------------------------------
    def _graph(self):
        u = torch.as_tensor(self.train_pairs["user"].to_numpy(np.int64, copy=True))
        i = torch.as_tensor(self.train_pairs["item"].to_numpy(np.int64, copy=True))
        du = torch.bincount(u, minlength=self.nu).float()
        di = torch.bincount(i, minlength=self.ni).float()
        w = (du[u] * di[i]).rsqrt()
        R = torch.sparse_coo_tensor(torch.stack([u, i]), w, (self.nu, self.ni)).coalesce()
        Rt = torch.sparse_coo_tensor(torch.stack([i, u]), w, (self.ni, self.nu)).coalesce()
        return R, Rt

    def _load(self, path):
        ck = torch.load(path, map_location="cpu", weights_only=True)
        sd = ck.get("state_dict", ck)
        cfg = ck.get("cfg", {}) or {}
        eu = sd["user.weight"].float()
        ei = sd["item.weight"].float()
        name = os.path.basename(path).lower()
        layers = int(cfg.get("layers", 0) or 0)
        kind = "LightGCN" if (layers > 0 or "lightgcn" in name) else "MF"
        if kind == "LightGCN":
            layers = layers or 2
            R, Rt = self._graph()
            us, its = [eu], [ei]
            for _ in range(layers):
                eu, ei = torch.sparse.mm(R, ei), torch.sparse.mm(Rt, eu)
                us.append(eu)
                its.append(ei)
            eu = torch.stack(us).mean(0)
            ei = torch.stack(its).mean(0)
        return kind, eu.numpy(), ei.numpy()

    # ------------------------------ hồ sơ user ------------------------------
    def profile_from_items(self, item_ids, decay=0.85):
        """Trung bình có trọng số: phim xem sau (cuối danh sách) quan trọng hơn.

        decay = 1.0 -> mọi phim như nhau; decay < 1 -> càng về sau càng quan trọng.
        """
        ids = [int(i) for i in item_ids]
        if not ids:
            return None
        w = np.power(decay, np.arange(len(ids) - 1, -1, -1, dtype=np.float64))
        vec = (self.item_emb[ids] * w[:, None]).sum(0) / w.sum()
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def recommend(self, vec, k=K, exclude=()):
        if vec is None:                                  # chưa có phim nào -> phổ biến
            return self.popularity_rank(k, exclude)
        scores = self.item_emb @ vec
        ex = np.asarray(sorted(set(int(x) for x in exclude)), dtype=np.int64)
        if ex.size:
            scores[ex] = -np.inf
        k = min(k, self.ni)
        top = np.argpartition(-scores, k - 1)[:k]
        return top[np.argsort(-scores[top])].tolist()

    def recommend_user(self, uid, k=K, new_items=(), decay=0.85, alpha=0.5):
        """User có sẵn; nếu vừa xem thêm phim thì trộn thêm hồ sơ tạm thời.

        alpha = trọng số của 'phim vừa xem' so với sở thích đã học (0 = bỏ qua).
        """
        base = self.user_emb[int(uid)]
        base = base / (np.linalg.norm(base) + 1e-12)
        prof = self.profile_from_items(new_items, decay) if new_items else None
        if prof is not None and alpha > 0:
            vec = base + alpha * prof
            vec = vec / (np.linalg.norm(vec) + 1e-12)
        else:
            vec = base
        seen = self.watched_by(uid)
        return self.recommend(vec, k, exclude=seen + list(new_items))

    def recommend_coldstart(self, watched, k=K, decay=0.85):
        """User mới: hồ sơ dựng từ các phim đã xem trong phiên hiện tại."""
        prof = self.profile_from_items(watched, decay)
        return self.recommend(prof, k, exclude=list(watched))

    # ------------------------------ đồ thị ------------------------------
    def build_graph(self, seen_ids, rec_ids, support=True, max_support_edges=25):
        """Đồ thị 2 phía user-item cho demo.

        Node:  ("u", "user")  và  ("i", idx)
        Cạnh:  user - item đã xem (xám), user - item gợi ý (xanh),
               và cạnh 'ủng hộ' giữa item gợi ý với item đã xem nếu nhiều user cùng xem cả hai.
        """
        import networkx as nx

        G = nx.Graph()
        G.add_node(("u", "user"), kind="user", label="Bạn")
        for i in seen_ids:
            n = ("i", int(i))
            G.add_node(n, kind="seen", label=self.title(i, 24))
            G.add_edge(("u", "user"), n, kind="seen")
        for rank, i in enumerate(rec_ids):
            n = ("i", int(i))
            G.add_node(n, kind="rec", label=self.title(i, 24), rank=rank)
            G.add_edge(("u", "user"), n, kind="rec")

        if support:
            pairs = []
            for i in rec_ids:
                for j in seen_ids:
                    c = self.cooccur(i, j)
                    if c >= 3:
                        pairs.append((c, ("i", int(i)), ("i", int(j))))
            for c, a, b in sorted(pairs, reverse=True)[:max_support_edges]:
                G.add_edge(a, b, kind="support", weight=c)
        return G


def load_model(root=ROOT, ckpt=None):
    """Hàm tiện dụng cho app (được cache ở phía Streamlit)."""
    return DemoModel(root=root, ckpt=ckpt)
