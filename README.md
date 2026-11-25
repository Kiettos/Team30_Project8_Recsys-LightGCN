# Team 30 — Project 8: LightGCN vs Matrix Factorization (MovieLens-1M)

So sánh **LightGCN** (Graph Convolutional Network cho lọc cộng tác) với baseline **BPR-MF**
(Matrix Factorization) trên MovieLens-1M, khảo sát ảnh hưởng của **kích thước embedding** và
**số tầng lan truyền**, kèm **demo trực quan hoá đồ thị user–item + gợi ý cho user mới (cold-start)**.

| | |
|---|---|
| Dataset | MovieLens-1M → 6.034 user, 3.125 phim, 574.376 tương tác |
| Metric | Recall@20, NDCG@20 (full-ranking, đã che item đã biết) |
| Split | **Theo thời gian** (10% cuối = test, 10% trước đó = validation) |
| Model chính | LightGCN (BPR loss, trung bình embedding các tầng) |
| Baseline | BPR-MF (0 tầng lan truyền) |
| Kết quả TEST (3 seed) | LightGCN **0.1226 ± 0.0018** / **0.0781 ± 0.0012** · MF **0.1194 ± 0.0004** / **0.0764 ± 0.0006** |

---

## 1. Thành viên

| Thành viên | MSSV | Phụ trách |
|---|---|---|
| Lương Chí Kiệt | *(điền)* | Tiền xử lý, LightGCN, HPO, đánh giá, demo, báo cáo |
| *(thành viên 2)* | *(điền)* | *(điền)* |

## 2. Bài toán và dữ liệu

**Bài toán.** Với mỗi user, xếp hạng toàn bộ phim chưa xem và lấy top-20. Dữ liệu được
dùng ở dạng **implicit feedback**: mọi rating ≥ 4 được coi là tương tác dương (rating thấp
bị loại bỏ vì không thể hiện sở thích).

**Tiền xử lý** (`src/data/preprocess_ml1m.py`):

| Bước | Kết quả |
|---|---|
| Đọc `data/ratings.dat` | 1.000.209 rating · 6.040 user · 3.706 phim |
| Lọc `rating ≥ 4` (implicit) | 575.281 tương tác |
| Bỏ cặp trùng + k-core 5 (mọi user và phim ≥ 5 tương tác) | **574.376 tương tác · 6.034 user · 3.125 phim · density 3,05%** |
| Chia theo thời gian trong từng user (10% test, 10% val) | **459.436 train / 57.470 val / 57.470 test** |
| Đồ thị 2 phía cho LightGCN | 9.159 × 9.159, 918.872 phần tử khác 0 |

**Vì sao split theo thời gian?** Split ngẫu nhiên để lộ thông tin tương lai (một phim của
"tương lai" có thể nằm trong train), làm điểm cao giả tạo. Cách làm ở đây: trong mỗi user,
sắp theo timestamp, 10% cuối → test, 10% ngay trước → validation, phần còn lại → train.
Script còn kiểm tra chéo: không có cặp `(user, item)` nào nằm ở hai tập, và mọi user đều
thoả `max(ts train) ≤ min(ts val) ≤ max(ts val) ≤ min(ts test)`.

## 3. Phương pháp

### 3.1 Baseline: BPR-MF

`ŝ(u,i) = p_u · q_i`, tối ưu BPR:

$$\mathcal{L} = -\log \sigma(\hat s_{u,i^+} - \hat s_{u,i^-}) + \lambda \left(\|p_u\|^2 + \|q_{i^+}\|^2 + \|q_{i^-}\|^2\right)$$

Âm bản lấy đều (uniform) và bốc lại nếu trúng phim user đã xem. MF chính là LightGCN với
**0 tầng lan truyền**, nên hai mô hình dùng chung dữ liệu, chung hàm đánh giá và chung
không gian siêu tham số → so sánh công bằng.

### 3.2 Model chính: LightGCN

Lan truyền trên đồ thị 2 phía user–item, chỉ giữ cạnh train:

$$e_u^{(k+1)} = \sum_{i \in \mathcal{N}(u)} \frac{1}{\sqrt{d_u\, d_i}}\, e_i^{(k)},
\qquad e_i^{(k+1)} = \sum_{u \in \mathcal{N}(i)} \frac{1}{\sqrt{d_i\, d_u}}\, e_u^{(k)}$$

Embedding cuối = **trung bình các tầng** (kể cả tầng 0), điểm `ŝ(u,i) = e_u · e_i`, cùng loss BPR
và L2 trên embedding gốc. Không dùng feature transformation / dropout phi tuyến — đúng
tinh thần LightGCN. Cài đặt dùng sparse tensor nên không dựng ma trận 9.159² dày.

### 3.3 Giao thức đánh giá

- **Full ranking** trên toàn bộ 3.125 phim, che các phim user đã biết (train + val khi tính test).
- Recall@20 = `|top20 ∩ target| / |target|`; NDCG@20 với IDCG cắt tại `min(|target|, 20)`.
- **HPO chỉ chọn theo validation.** Test chỉ được chạy **một lần** trong `--final`, với **3 seed**
  (0, 1, 2), báo cáo **mean ± std** — không chọn số đẹp nhất trong 3 lần.

## 4. Thiết lập thực nghiệm

| | MF | LightGCN |
|---|---|---|
| Số cấu hình HPO | 12 | 11 |
| Quét | dim {32, 64, 128, 256}; lr × reg | layers {1,2,3,4}; dim {32,64,128}; lr × reg |
| Embedding dim | 32–256 | 32–128 |
| Batch / epoch tối đa / patience | 2048 / 300 / 10 | 2048 / 300 / 10 |
| Early stopping | theo VAL NDCG@20 (min_delta 1e-4) | như MF |
| Final | 3 seed × cấu hình tốt nhất | như MF |

Phần cứng: **RTX 3050 Laptop 6 GB**, PyTorch 2.5.1 + CUDA 12.1, Python 3.11.
Thời gian đo được: MF ~2,2 s/epoch (dim 256), LightGCN ~9 s/epoch (dim 128, 2 tầng).
Tổng HPO: MF ~23 phút, LightGCN ~1 giờ 55 phút; `--final` 3 seed: MF ~7 phút, LightGCN ~55 phút.

## 5. Kết quả

### 5.1 HPO — Matrix Factorization (chọn theo validation)

| Cấu hình | dim | lr | reg | Best epoch | VAL Recall@20 | VAL NDCG@20 |
|---|---|---|---|---|---|---|
| A_mf_d32 | 32 | 0,001 | 0,0001 | 52 | 0,1393 | 0,0909 |
| A_mf_d64 | 64 | 0,001 | 0,0001 | 56 | 0,1424 | 0,0908 |
| A_mf_d128 | 128 | 0,001 | 0,0001 | 50 | 0,1451 | 0,0929 |
| A_mf_d256 | 256 | 0,001 | 0,0001 | 48 | 0,1448 | 0,0929 |
| B_mf_lr0.001_reg0.0003 | 64 | 0,001 | 0,0003 | 56 | 0,1436 | 0,0917 |
| B_mf_lr0.001_reg0.001 | 64 | 0,001 | 0,001 | 56 | 0,1454 | 0,0932 |
| B_mf_lr0.003_reg0.0001 | 64 | 0,003 | 0,0001 | 24 | 0,1368 | 0,0880 |
| B_mf_lr0.003_reg0.0003 | 64 | 0,003 | 0,0003 | 24 | 0,1372 | 0,0888 |
| B_mf_lr0.003_reg0.001 | 64 | 0,003 | 0,001 | 49 | 0,1466 | 0,0938 |
| C_mf_d128_lr0.001_reg0.001 | 128 | 0,001 | 0,001 | 50 | 0,1492 | 0,0963 |
| **C_mf_d256_lr0.001_reg0.001** | **256** | **0,001** | **0,001** | **48** | **0,1522** | **0,0981** |
| C_mf_d128_lr0.003_reg0.001 | 128 | 0,003 | 0,001 | 12 | 0,1376 | 0,0903 |

Nhận xét: dim tăng 32 → 128 thì tốt lên rõ (NDCG 0,0909 → 0,0929) rồi **bão hoà** ở 256.
`reg = 1e-3` quan trọng hơn việc tăng dim: stage C (dim 256 + reg 1e-3) đạt **0,0981**, cao hơn
hẳn 0,0929 của grid ban đầu. `lr = 3e-3` chỉ tốt khi đi kèm reg lớn, nếu không thì phân kỳ nhẹ
(0,0880). Đây là lý do cần stage C: grid 2 stage ban đầu chỉ quét reg tại dim 64.

### 5.2 HPO — LightGCN (chọn theo validation)

| Cấu hình | layers | dim | lr | reg | Best epoch | VAL Recall@20 | VAL NDCG@20 |
|---|---|---|---|---|---|---|---|
| A_lgcn_L1_d64 | 1 | 64 | 0,001 | 0,0001 | 54 | 0,1477 | 0,0939 |
| A_lgcn_L2_d64 | 2 | 64 | 0,001 | 0,0001 | 50 | 0,1410 | 0,0916 |
| A_lgcn_L3_d64 | 3 | 64 | 0,001 | 0,0001 | 64 | 0,1400 | 0,0911 |
| A_lgcn_L4_d64 | 4 | 64 | 0,001 | 0,0001 | 79 | 0,1379 | 0,0907 |
| A_lgcn_L2_d32 | 2 | 32 | 0,001 | 0,0001 | 69 | 0,1382 | 0,0909 |
| **A_lgcn_L2_d128** | **2** | **128** | **0,001** | **0,0001** | **100** | **0,1612** | **0,0999** |
| B_lgcn_lr0.001_reg0.0003 | 2 | 64 | 0,001 | 0,0003 | 52 | 0,1428 | 0,0921 |
| B_lgcn_lr0.001_reg0.001 | 2 | 64 | 0,001 | 0,001 | 70 | 0,1417 | 0,0921 |
| B_lgcn_lr0.003_reg0.0001 | 2 | 64 | 0,003 | 0,0001 | 50 | 0,1552 | 0,0975 |
| B_lgcn_lr0.003_reg0.0003 | 2 | 64 | 0,003 | 0,0003 | 52 | 0,1566 | 0,0984 |
| B_lgcn_lr0.003_reg0.001 | 2 | 64 | 0,003 | 0,001 | 76 | 0,1564 | 0,0991 |

Nhận xét:
- Tại dim 64, **nhiều tầng lại kém hơn**: L1 0,0939 > L2 0,0916 > L3 0,0911 > L4 0,0907.
  Nguyên nhân hợp lý: propagation tạo ra "hiệu ứng làm mịn" (oversmoothing) và ở đồ thị
  MovieLens khá dày (3,05%) thì 1–2 tầng đã đủ; tăng tầng chỉ làm embedding các node gần nhau
  trở nên giống nhau, mất tín hiệu phân biệt. Đây là điểm **khác với kỳ vọng trong paper**
  (paper báo 2–3 tầng tốt nhất) và là một phát hiện đáng thảo luận của đồ án.
- Nhưng **dim lại ảnh hưởng mạnh hơn**: ở 2 tầng, dim 128 đạt 0,0999 (so với 64: 0,0916;
  32: 0,0909) → dim và layers có tương tác, không nên kết luận chỉ từ lưới 1 chiều.
- `lr = 3e-3` giúp LightGCN rõ rệt (0,0975–0,0991 so với 0,0916).

### 5.3 Kết quả cuối trên TEST (3 seed, `--final`)

| Model | Cấu hình tốt nhất (theo VAL) | Seed 0 | Seed 1 | Seed 2 | **TB ± SD** |
|---|---|---|---|---|---|
| MF | `C_mf_d256_lr0.001_reg0.001` | 0,1198 / 0,0771 | 0,1190 / 0,0760 | 0,1195 / 0,0762 | **0,1194 ± 0,0004** / **0,0764 ± 0,0006** |
| LightGCN | `A_lgcn_L2_d128` | 0,1232 / 0,0781 | 0,1240 / 0,0793 | 0,1206 / 0,0770 | **0,1226 ± 0,0018** / **0,0781 ± 0,0012** |
| MostPop (mốc) | top-20 phim phổ biến | — | — | — | 0,0778 / 0,0557 |

*(mỗi ô: Recall@20 / NDCG@20)*

Bảng tự sinh: `results/comparison.md` (chạy `python src/summarize_results.py`).

**Đọc kết quả cho trung thực:**
- Cả hai mô hình đều vượt xa MostPop (Recall 0,12 so với 0,078) → mô hình thực sự học được
  tín hiệu cá nhân hoá, không chỉ học độ phổ biến.
- LightGCN hơn MF **+0,0032 Recall@20** (~1,8 lần độ lệch chuẩn của LightGCN) và
  **+0,0017 NDCG@20** (~1,4 lần SD). Nghĩa là **nhỉnh hơn, nhưng chênh lệch nhỏ so với
  dao động giữa các seed** — không thể khẳng định "vượt trội" mạnh mẽ trên dataset này.
- MF **ổn định hơn** (SD 0,0004 so với 0,0018; best epoch 48–53 so với 74–129).
- Val → test giảm ~0,021 NDCG ở **cả hai** mô hình (MF 0,0981 → 0,0764; LightGCN 0,0999 → 0,0781):
  mức giảm gần như giống nhau, cho thấy nguyên nhân là **dịch chuyển thời gian** (test nằm xa
  hơn trong tương lai so với val) chứ không phải overfit của riêng model nào.

## 6. Phân tích lỗi

Số liệu do `python src/error_analysis.py` sinh ra (`results/error_analysis.md`, dùng checkpoint seed 0).

| Model | Recall@20 | User không có hit nào trong top-20 | Tỉ lệ cặp test vào top-20 |
|---|---|---|---|
| MF | 0,1197 | 3.373 / 6.034 (**55,9%**) | 7,50% |
| LightGCN | 0,1232 | 3.407 / 6.034 (**56,5%**) | 7,62% |

**1. Hơn một nửa user không có bất kỳ gợi ý đúng nào trong top-20.** Vì test là tương lai xa
nhất của mỗi user, đây là bài toán khó; 20 ô gợi ý là quá ít so với 3.125 phim. Điều này lý giải
vì sao Recall@20 chỉ ~0,12 dù mô hình đã tốt hơn hẳn baseline phổ biến.

**2. Lỗi tập trung ở phim hiếm (long-tail).**

| Nhóm phim (theo độ phổ biến) | Độ phổ biến TB | MF: cặp test vào top-20 | LightGCN |
|---|---|---|---|
| 1 — hiếm nhất | 53,9 user | 0,51% | 0,65% |
| 2 | 181,5 | 2,20% | 2,56% |
| 3 | 393,3 | 5,93% | 6,26% |
| 4 — phổ biến nhất | 1.013,5 | **21,43%** | **21,08%** |

Chênh lệch **~40 lần** giữa nhóm phim phổ biến nhất và hiếm nhất. Cả hai mô hình gần như
"bó tay" với phim tail — phù hợp với quan sát bên dưới.

**3. Thiên lệch popularity rất mạnh.** Điểm phổ biến trung bình của phim được gợi ý là
**0,935 (MF) / 0,930 (LightGCN)** trên thang 0–1, và **0%** gợi ý đến từ nhóm 25% phim hiếm nhất.
Nguyên nhân: BPR với âm bản uniform + chuẩn hoá bậc `1/√(d_u d_i)` của LightGCN đều đẩy
điểm về phía item bậc cao; phim hiếm ít cạnh nên embedding ít được cập nhật.
Độ phủ catalog vẫn đạt **53–54%** (tức ~1.660/3.125 phim từng xuất hiện trong top-20 của ai đó),
nhưng đó là nhờ tổng hợp qua 6.034 user, không phải vì mỗi user được gợi ý đa dạng.

**4. User càng hoạt động thì dự đoán càng khó.**

| Nhóm user (số tương tác train) | MF: tỉ lệ cặp test đúng | LightGCN |
|---|---|---|
| 1 (3–21) | 12,67% | 12,91% |
| 2 (22–46) | 7,17% | 7,29% |
| 3 (47–99) | 5,58% | 5,44% |
| 4 (100–1.133) | 4,50% | 4,76% |

User ít tương tác thường xem phim đại chúng → mô hình dễ "trúng" nhờ thiên lệch popularity.
User nhiều tương tác có khẩu vị rộng và phim test đa dạng hơn → khó hơn. Lưu ý: bảng
**Recall trung bình theo user** trong `results/error_analysis.md` còn bị méo hơn, vì recall mỗi
user bị chia cho số item test của user đó — user ít tương tác (1 item test) chỉ cần 1 hit là
được 1,0, nên phải đọc bảng "tỉ lệ cặp" này mới công bằng.

**Hạn chế của phân tích:** mới dùng seed 0, chưa tách lỗi do *dịch chuyển sở thích* khỏi lỗi do
*phim mới ra mắt* (phim trong test có thể chưa từng xuất hiện trong train nên không có embedding tốt).

## 7. Demo (Streamlit)

```powershell
venv\Scripts\python.exe -m streamlit run app.py
```

Ba tab:

1. **User có sẵn** — nhập user index, xem top-K, đồ thị user–item, giải thích "vì sao gợi ý"
   (bao nhiêu người cùng xem cặp phim), và thêm phim "vừa xem" để thấy gợi ý trôi theo.
2. **User mới (cold-start)** — người dùng bấm xem phim, hồ sơ sở thích dựng lại **ngay**:
   `u_new = Σ w_j·e_j / Σ w_j` với `w_j = decay^(|S|-1-j)` (phim xem sau nặng hơn),
   `e_j` là embedding item sau lan truyền của LightGCN. Thêm/bớt phim → bảng gợi ý và đồ thị
   **cập nhật tức thì**, không train lại, không lan truyền lại toàn đồ thị (~1 ms).
3. **Về mô hình** — thông tin checkpoint, protocol đánh giá và bảng kết quả (`results/comparison.md`).

Đồ thị: user đỏ, phim đã xem xám, phim gợi ý xanh (node to = hạng cao), nét đứt nối phim gợi ý
với phim đã xem khi ≥ 3 user cùng xem cả hai. Demo chạy trên CPU nên mở song song lúc train được.

## 8. Cài đặt và chạy

```powershell
# 0) môi trường (Windows, Python 3.11)
powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1        # PyTorch GPU (CUDA 12.1)
# hoặc: ... setup_env.ps1 -Cpu   (chỉ chạy thử)

# 1) dữ liệu: data/ratings.dat, users.dat, movies.dat -> processed/
venv\Scripts\python.exe src\data\preprocess_ml1m.py

# 2) mốc tham chiếu
venv\Scripts\python.exe src\baseline_pop.py

# 3) HPO (chọn theo validation)
venv\Scripts\python.exe src\model\Train_mf.py
venv\Scripts\python.exe src\model\Train_lightgcn.py

# 4) final: cấu hình tốt nhất × 3 seed -> TEST
venv\Scripts\python.exe src\model\Train_mf.py --final
venv\Scripts\python.exe src\model\Train_lightgcn.py --final

# 5) bảng so sánh + phân tích lỗi
venv\Scripts\python.exe src\summarize_results.py
venv\Scripts\python.exe src\error_analysis.py

# demo
venv\Scripts\python.exe -m streamlit run app.py
```

Runbook chi tiết (thời gian ước tính, cách theo dõi, xử lý khi bị ngắt):
[docs/command to train.txt](docs/command%20to%20train.txt).

**Tái lập:** kết quả không bit-identical giữa các lần chạy (khởi tạo + âm bản ngẫu nhiên, phép
toán atomics trên GPU). Vì vậy mọi kết quả đều báo **mean ± std trên 3 seed**, không chọn seed đẹp.

## 9. Cấu trúc thư mục

```
Team30_Project8_Recsys-LightGCN/
├── app.py                        # demo Streamlit (user có sẵn + cold-start + đồ thị)
├── data/                         # MovieLens-1M raw (ratings/users/movies.dat)
├── processed/                    # sinh bởi preprocess_ml1m.py (không commit)
├── configs/
│   ├── hpo_config_mf.json        # 12 thí nghiệm MF (stage A/B/C)
│   └── hpo_config_lgcn.json      # 11 thí nghiệm LightGCN
├── scripts/setup_env.ps1         # tạo/sửa venv, cài PyTorch đúng bản
├── src/
│   ├── data/preprocess_ml1m.py   # temporal split, kiểm tra rò rỉ, adjacency
│   ├── model/Train_mf.py         # baseline BPR-MF
│   ├── model/Train_lightgcn.py   # LightGCN
│   ├── demo/recommender.py       # nạp model + gợi ý + dựng đồ thị
│   ├── utils.py                  # đọc dữ liệu, metric, vòng train, CSV/checkpoint dùng chung
│   ├── baseline_pop.py           # mốc MostPop
│   ├── summarize_results.py      # bảng so sánh
│   └── error_analysis.py         # phân tích lỗi
├── results/                      # hpo_*.csv, final_*.csv, comparison.md, error_analysis.md
├── checkpoints/                  # *_s{0,1,2}.pth của lần chạy --final (không commit)
├── docs/                         # runbook, tài liệu lý thuyết
└── requirements.txt
```

## 10. Hạn chế và hướng phát triển

- **Chỉ MovieLens-1M**, chưa kiểm chứng trên Amazon/Other; kết luận về số tầng có thể khác
  với đồ thị thưa hơn.
- **Ngưỡng implicit 4.0** loại ~42% rating; thử ngưỡng 3.5 hoặc dùng rating làm trọng số
  (confidence) có thể cải thiện.
- **Thiên lệch popularity chưa được xử lý**: có thể thêm inverse propensity weighting, hoặc
  regularize để tăng novelty/coverage — và đánh giá thêm novelty/diversity chứ không chỉ Recall/NDCG.
- **Side information chưa dùng**: `processed/side_info/{users,items}.csv` (giới tính, tuổi,
  nghề nghiệp, thể loại phim) đã được dựng sẵn nhưng chưa đưa vào mô hình. Hướng tiếp theo
  là Knowledge-aware model (KGCN/KGAT) hoặc thêm feature cho cold-start.
- **Cold-start hiện là heuristic** (trung bình embedding phim đã xem), chưa học được mapping
  từ vài hành vi sang sở thích.
- **HPO còn mỏng**: grid 1 chiều + stage C thủ công; chưa quét batch size, số epoch, kiến trúc
  khác (PinSage, NGCF), chưa chạy nhiều seed cho từng cấu hình HPO (chỉ 1 seed ở HPO).

## 11. Tài liệu tham khảo

1. X. He, K. Deng, X. Wang, Y. Li, Y. Zhang, M. Wang. *LightGCN: Simplifying and Powering
   Graph Convolution Network for Recommendation.* SIGIR 2020.
2. S. Rendle, C. Freudenthaler, Z. Gantner, L. Schmidt-Thieme. *BPR: Bayesian Personalized
   Ranking from Implicit Feedback.* UAI 2009.
3. F. M. Harper, J. A. Konstan. *The MovieLens Datasets: History and Context.* ACM TiiS 2015.
4. D. P. Kingma, J. Ba. *Adam: A Method for Stochastic Optimization.* ICLR 2015.

## 12. Ghi chú dữ liệu

MovieLens-1M thuộc GroupLens Research (Đại học Minnesota), chỉ dùng cho mục đích nghiên cứu
và học tập; không kèm bảo đảm nào từ nhà cung cấp. Xem `data/README` để biết điều khoản gốc.
