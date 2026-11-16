# So sánh cấu hình — split `dev`

| Cấu hình | Thiết lập | Hit@1 query gốc | EvHit@3 query gốc | Hit@1 query agent | EvHit@3 query agent | Context Evidence Rate | Grounded 0/1/2 · TB | Citation hợp lệ | Abstention đúng | Từ chối nhầm | Lượt search TB {phân bố} | Trễ trung vị (s) | Token | Chi phí | Lỗi |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dense-B-k3 | dense, index B, top-3, filter on, budget 6000 chars, ≤4 searches | 18/20 (90%) | 20/21 (95%) | 15/20 (75%) | 20/21 (95%) | 20/21 (95%) | 1/1/19 · 1.86 | 28/28 (100%) | 2/3 (67%) | 0/21 (0%) | 1.25 {'1': 20, '2': 2, '3': 2} | 2.44 | 51,450 | $0.0231 | 0 |
| bm25-B-k3 | bm25, index B, top-3, filter on, budget 6000 chars, ≤4 searches | 18/20 (90%) | 19/21 (90%) | 19/20 (95%) | 19/21 (90%) | 19/21 (90%) | 0/3/18 · 1.86 | 30/33 (91%) | 2/3 (67%) | 0/21 (0%) | 1.29 {'1': 20, '2': 1, '3': 3} | 2.23 | 52,544 | $0.0242 | 0 |
| hybrid-B-k3 | hybrid, index B, top-3, filter on, budget 6000 chars, ≤4 searches | 19/20 (95%) | 19/21 (90%) | 18/20 (90%) | 19/21 (90%) | 21/21 (100%) | 0/1/20 · 1.95 | 34/34 (100%) | 1/3 (33%) | 0/21 (0%) | 1.17 {'1': 21, '2': 2, '3': 1} | 2.46 | 47,216 | $0.0226 | 0 |

Hit@1 tính trên câu một evidence; EvHit@3 và Context Evidence Rate tính trên mọi câu có đáp án (multi-hop cần đủ mọi nhóm). Grounded, citation và abstention lấy từ nhãn thủ công (`labels/`), tổng hợp bằng `starter/agent_metrics.py`. Chi phí tính theo token × đơn giá trong config (Gemini 2.5 Flash $0.30/1M input, $2.50/1M output), không phải hóa đơn.
