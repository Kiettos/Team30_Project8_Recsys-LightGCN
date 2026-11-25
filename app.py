"""app.py - Demo Streamlit: gợi ý phim + đồ thị user-item + cold-start cho user mới.

Chạy:
    venv\\Scripts\\python.exe -m streamlit run app.py

Ý tưởng demo:
  - Tab "User có sẵn"  : chọn user trong MovieLens, xem top-20 gợi ý và đồ thị user-item.
  - Tab "User mới"     : user chưa từng có trong dữ liệu. Người dùng bấm xem phim,
                         hồ sơ sở thích được dựng lại ngay từ embedding các phim vừa xem
                         (trung bình có trọng số thời gian) -> gợi ý CẬP NHẬT TỨC THÌ,
                         không cần train lại.
"""
import os
import sys

import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from demo import recommender as rec  # noqa: E402

st.set_page_config(page_title="Demo RecSys - LightGCN vs MF", page_icon="🎬", layout="wide")


def draw_graph(G, height=6.5):
    """Vẽ đồ thị 2 phía user-item: user đỏ, phim đã xem xám, phim gợi ý xanh."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx

    pos = nx.spring_layout(G, seed=42, k=0.95, iterations=200)
    colors, sizes = [], []
    for _, d in G.nodes(data=True):
        kind = d.get("kind")
        if kind == "user":
            colors.append("#e63946")
            sizes.append(2600)
        elif kind == "seen":
            colors.append("#adb5bd")
            sizes.append(1200)
        else:
            colors.append("#2a9d8f")
            sizes.append(max(900, 1500 - 40 * d.get("rank", 0)))
    normal = [(u, v) for u, v, d in G.edges(data=True) if d.get("kind") != "support"]
    support = [(u, v) for u, v, d in G.edges(data=True) if d.get("kind") == "support"]
    labels = {n: d.get("label", str(n)) for n, d in G.nodes(data=True)}

    fig, ax = plt.subplots(figsize=(12, height))
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=sizes, ax=ax,
                           edgecolors="white", linewidths=1.5)
    nx.draw_networkx_edges(G, pos, edgelist=normal, edge_color="#ced4da", width=1.4, ax=ax)
    if support:
        nx.draw_networkx_edges(G, pos, edgelist=support, edge_color="#f4a261",
                               style="dashed", width=1.0, ax=ax)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, ax=ax)
    ax.set_axis_off()
    st.pyplot(fig)
    plt.close(fig)


def support_text(model, rec_id, seen_ids, topn=2):
    scores = sorted(((model.cooccur(rec_id, s), s) for s in seen_ids), reverse=True)
    parts = [f"{n} người cũng xem “{model.title(s, 34)}”" for n, s in scores[:topn] if n >= 2]
    return " · ".join(parts)


# ------------------------------- nạp model -------------------------------
ckpts = rec.list_checkpoints(ROOT)
if not ckpts:
    st.error(
        "Chưa có checkpoint trong `checkpoints/`.\n\n"
        "Chạy trước:\n```\n"
        "venv\\Scripts\\python.exe src\\model\\Train_lightgcn.py --final\n"
        "venv\\Scripts\\python.exe src\\model\\Train_mf.py --final\n```"
    )
    st.stop()


@st.cache_resource(show_spinner="Loading data and generating embeddings...")
def get_model(ckpt_path):
    return rec.load_model(ROOT, ckpt_path)


with st.sidebar:
    st.header("⚙️ Cấu hình")
    ckpt = st.selectbox("Checkpoint", ckpts, format_func=os.path.basename)
    k = st.slider("Số gợi ý (K)", 5, 20, 10)
    seen_shown = st.slider("Số phim đã xem hiển thị trên đồ thị", 3, 10, 6)
    decay = st.slider("Trọng số thời gian (decay)", 0.5, 1.0, 0.85, 0.05,
                      help="1.0 = mọi phim quan trọng như nhau; nhỏ hơn = phim xem sau quan trọng hơn")

model = get_model(ckpt)
st.title("🎬 Demo gợi ý phim — LightGCN vs Matrix Factorization")
st.caption(
    f"Model: **{model.kind}** · `{os.path.basename(ckpt)}` · "
    f"{model.nu:,} user · {model.ni:,} phim · dữ liệu MovieLens-1M (implicit, rating ≥ 4)"
)

tab_user, tab_new, tab_about = st.tabs(
    ["👤 User có sẵn", "🆕 User mới (cold-start)", "ℹ️ Về mô hình & cách đánh giá"]
)

# ------------------------------- tab 1: user có sẵn -------------------------------
with tab_user:
    c1, c2 = st.columns([1, 2])
    with c1:
        uid = st.number_input(f"User index (0 – {model.nu - 1})", 0, model.nu - 1, 0, 1)
        raw = model.raw_user_id(uid)
        watched = model.watched_by(uid)
        st.metric("Phim đã tương tác (train)", len(watched))
        st.caption(f"UserID gốc trong MovieLens: **{raw}**")
    with c2:
        new_items = st.multiselect(
            "Phim người này VỪA xem thêm (tuỳ chọn) — gợi ý sẽ cập nhật ngay",
            options=model.item_ids_by_pop,
            format_func=lambda i: model.title(i, 55),
            max_selections=8,
        )
        alpha = st.slider("Ảnh hưởng của phim vừa xem", 0.0, 2.0, 0.6, 0.1,
                          help="0 = bỏ qua phim mới, chỉ dùng sở thích đã học; "
                               "lớn = ưu tiên phim vừa xem (giống cold-start)")

    rec_ids = model.recommend_user(uid, k=k, new_items=new_items, decay=decay, alpha=alpha)

    left, right = st.columns([1.15, 1])
    with left:
        st.subheader(f"Top {k} gợi ý cho user {uid}")
        rows = [{"Hạng": r + 1, "Phim": model.title(i, 60), "Năm": model.year(i),
                 "Thể loại": model.genres(i)} for r, i in enumerate(rec_ids)]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        with st.expander("Vì sao gợi ý những phim này?"):
            for i in rec_ids[:5]:
                why = support_text(model, i, watched[-20:])
                st.write(f"- **{model.title(i)}**" + (f" — {why}" if why else ""))
    with right:
        st.subheader("🕸️ Đồ thị user–item")
        seen_for_graph = watched[-seen_shown:]
        G = model.build_graph(seen_for_graph, rec_ids[:6])
        draw_graph(G)
        st.caption("Đỏ: user · Xám: phim đã xem · Xanh: phim gợi ý (to hơn = hạng cao hơn) · "
                   "Nét đứt: quan hệ 'cùng được xem'")

# ------------------------------- tab 2: user mới -------------------------------
with tab_new:
    st.markdown(
        "User **chưa có** trong dữ liệu train. Hồ sơ sở thích được dựng từ chính các phim "
        "người này bấm xem — **mỗi lần thêm/bớt phim, danh sách gợi ý và đồ thị cập nhật ngay**, "
        "không cần train lại."
    )
    if "watched_new" not in st.session_state:
        st.session_state["watched_new"] = []

    picked = st.multiselect(
        "Chọn phim để thêm vào phiên xem (gõ để tìm theo tên)",
        options=model.item_ids_by_pop,
        format_func=lambda i: model.title(i, 55),
        max_selections=10,
        key=f"picked_{st.session_state.get('pick_round', 0)}",
    )
    b1, b2, _ = st.columns([1, 1, 3])
    if b1.button("➕ Xem các phim đã chọn", type="primary") and picked:
        for i in picked:
            if i not in st.session_state["watched_new"]:
                st.session_state["watched_new"].append(int(i))
        st.session_state["pick_round"] = st.session_state.get("pick_round", 0) + 1
        st.rerun()
    if b2.button("🗑️ Xoá tất cả"):
        st.session_state["watched_new"] = []
        st.rerun()

    watched_new = st.session_state["watched_new"]
    st.divider()
    if watched_new:
        st.subheader(f"Phiên xem hiện tại — {len(watched_new)} phim")
        cols = st.columns(min(5, len(watched_new)))
        for pos, i in enumerate(list(watched_new)):
            with cols[pos % len(cols)]:
                st.caption(model.title(i, 40))
                if st.button("✖ bỏ", key=f"rm_{i}"):
                    st.session_state["watched_new"].remove(i)
                    st.rerun()
    else:
        st.info("Chưa xem phim nào — thêm 1–3 phim để thấy gợi ý thay đổi.")

    rec_new = model.recommend_coldstart(watched_new, k=k, decay=decay)
    left, right = st.columns([1.15, 1])
    with left:
        st.subheader("Gợi ý cho user mới" + ("" if watched_new else " (tạm: phim phổ biến)"))
        rows = [{"Hạng": r + 1, "Phim": model.title(i, 60), "Năm": model.year(i),
                 "Thể loại": model.genres(i)} for r, i in enumerate(rec_new)]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        if watched_new:
            with st.expander("Vì sao gợi ý những phim này?"):
                for i in rec_new[:5]:
                    why = support_text(model, i, watched_new)
                    st.write(f"- **{model.title(i)}**" + (f" — {why}" if why else ""))
    with right:
        st.subheader("🕸️ Đồ thị user–item (cập nhật theo phiên)")
        G = model.build_graph(watched_new[-seen_shown:], rec_new[:6])
        draw_graph(G)

# ------------------------------- tab 3: thông tin -------------------------------
with tab_about:
    st.subheader("Mô hình")
    st.markdown(
        f"""
- **{model.kind}** · checkpoint `{os.path.basename(ckpt)}`
- Số user (train): **{model.nu:,}** · số phim: **{model.ni:,}**
- Dữ liệu: MovieLens-1M, chỉ giữ rating ≥ 4 (implicit), k-core 5, split **theo thời gian**
  (10% cuối làm test, 10% trước đó làm validation)
- Trọng số lan truyền của LightGCN: `1/√(d_u·d_i)`, trung bình embedding các tầng
- Metric: Recall@20 / NDCG@20, full-ranking, đã che phim user đã biết
"""
    )
    st.subheader("Cold-start (user mới) hoạt động thế nào?")
    st.markdown(
        """
Không có embedding học sẵn cho user mới, nên hồ sơ được dựng từ embedding của các phim
người đó vừa xem:

$$u_{new} = \\frac{\\sum_j w_j \\, e_j}{\\sum_j w_j}, \\qquad w_j = decay^{\\,|S|-1-j}$$

với `S` là danh sách phim đã xem theo thứ tự thời gian (phim xem sau có `w` lớn hơn), `e_j`
là embedding item **sau lan truyền** (LightGCN) hoặc embedding gốc (MF). Điểm gợi ý là
`u_new · e_i`. Vì vậy mỗi lần thêm một phim, hồ sơ đổi → bảng gợi ý và đồ thị đổi ngay,
tốn chưa tới 1 ms, không cần lan truyền lại toàn đồ thị.
"""
    )
    cmp_path = os.path.join(ROOT, "results", "comparison.md")
    if os.path.exists(cmp_path):
        st.subheader("Kết quả thực nghiệm")
        with open(cmp_path, encoding="utf-8") as f:
            st.markdown(f.read())
    else:
        st.info("Chưa có `results/comparison.md` — chạy `--final` cho cả hai mô hình rồi "
                "`python src/summarize_results.py` để hiện bảng so sánh ở đây.")
