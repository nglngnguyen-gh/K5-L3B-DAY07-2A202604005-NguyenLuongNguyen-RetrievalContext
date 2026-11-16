# Code Day 7

Chạy từ thư mục gốc bằng `.venv\Scripts\python.exe -m src.<module>`. Trên Windows, đặt `PYTHONUTF8=1` khi chạy unittest.

| Module | Công dụng |
|---|---|
| `export_day6.py` | Xuất chunk A/B và text trang từ vector store Day 6 sang `starter/data/day6/` (thư mục này bị gitignore) |
| `verify_benchmark.py` | Kiểm chứng benchmark với corpus, khóa SHA-256 và sinh `cases.json` |
| `retrieval.py` | Ba phương pháp `dense`, `bm25`, `hybrid` (RRF) và lớp `RetrievalStack` để mở DB, embedder và các phương pháp |
| `metrics.py` | Định nghĩa hit: khớp evidence không phân biệt khoảng trắng; câu multi-hop phải đủ mọi nhóm |
| `eval_retrieval.py` | Benchmark retrieval theo phương pháp × index × filter. Chỉ chạy `dev`; muốn chạy `test` phải có `starter/results/selection.json` |

## Phương pháp retrieval

| Phương pháp | Cách hoạt động | Filter | Score |
|---|---|---|---|
| `dense` | Embedding câu hỏi bằng `gemini-embedding-001` (`RETRIEVAL_QUERY`, 768 chiều, chuẩn hóa L2, giống Day 6), rồi tìm cosine exact trong pgvector Day 6 | Lọc bằng SQL `WHERE` trước khi xếp hạng | cosine similarity |
| `bm25` | Okapi BM25 (k1=1.5, b=0.75) trên cùng tập chunk. Tokenizer NFKC và lowercase, giữ nguyên số (`50,177`, `13.9`) và thêm dạng bỏ dấu phân cách (`50177`), bỏ stopword tiếng Anh | Lọc trong bộ nhớ trước khi xếp hạng | BM25 |
| `hybrid` | Reciprocal Rank Fusion: lấy 20 candidate từ dense và 20 từ bm25, score = Σ 1/(60 + rank). Chỉ dùng rank nên không cộng trực tiếp cosine với BM25 | Truyền xuống cả hai retriever | RRF, kèm `component_ranks` |

Index `A` là cửa sổ cố định 800/100 ký tự (276 chunk); index `B` là chunk theo đoạn văn, tối đa 800 ký tự (331 chunk). Strategy C của Day 6 tương đương `B` cộng filter section.

Lỗi API, lỗi DB, filter không hợp lệ, hoặc vector không tương thích (khác model hay số chiều) đều raise `RetrievalError`; code không trả kết quả giả. Vector của query được cache trong bảng `embedding_cache` của Day 6.

```powershell
.venv\Scripts\python.exe -m src.eval_retrieval --split dev                     # 12 cấu hình
.venv\Scripts\python.exe -m src.eval_retrieval --split dev --methods hybrid --indexes B --filters on
```
