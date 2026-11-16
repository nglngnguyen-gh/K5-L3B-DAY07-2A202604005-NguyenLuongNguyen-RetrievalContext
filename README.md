# Day 7 — Agentic RAG

## Bài nộp — Nguyễn Lương Nguyên (2A202604005)

**Báo cáo:** [report/individual/2A202604005_NguyenLuongNguyen.md](report/individual/2A202604005_NguyenLuongNguyen.md)

- **Corpus:** World Bank Group Annual Report 2026 (CC BY-NC-ND 3.0 IGO), dùng chunk và vector store của Day 6.
- **Retrieval:** PostgreSQL 16 + pgvector 0.8.7; `gemini-embedding-001` 768 chiều; ba phương pháp dense, BM25 và hybrid RRF.
- **Agent:** `gemini-2.5-flash` có function calling.
- **Kết quả test:** cấu hình chọn là `dense-B-k3`. Context Evidence Rate 9/9, grounded 1,89/2, citation hợp lệ 13/13, abstention 2/3, từ chối nhầm 0/9.

| Thư mục / file | Nội dung |
|---|---|
| [src/](src/README.md) | Export Day 6, kiểm chứng benchmark, retrieval, agent, runner, review sheet, bảng so sánh |
| [configs/](configs/) | 6 cấu hình agent (JSON) |
| [data/benchmark/](data/benchmark/README.md) | Benchmark dev/test (30 câu có đáp án, 6 câu thiếu bằng chứng), hash và kết quả kiểm chứng |
| [labels/](labels/README.md) | Nhãn thủ công dev/test (đáp án, context, citation, abstention) |
| [results/](results/) | Tiêu chí chọn, bảng so sánh dev/test, phân tích lỗi, kết quả test từng câu |
| `starter/results/selection.json` | Cấu hình đã khóa trước khi chạy test |

### Cách chạy lại

Yêu cầu: Python 3.11+ (đã dùng 3.14.5), Docker và repo Day 6 với container pgvector đã nạp dữ liệu, cùng một Gemini API key.

```powershell
py -3.14 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env          # điền GEMINI_API_KEY; DATABASE_URL trỏ tới DB Day 6 (cổng 55433)

# 1. Xuất chunk và text trang từ Day 6 (ra starter/data/day6/, bị gitignore vì giấy phép ND)
.venv\Scripts\python.exe -m src.export_day6 --day6 "<đường dẫn repo Day 6>"
# 2. Kiểm chứng benchmark (hash phải khớp data/benchmark/verification.json)
.venv\Scripts\python.exe -m src.verify_benchmark
# 3. Benchmark retrieval với query gốc
.venv\Scripts\python.exe -m src.eval_retrieval --split dev
# 4. Chạy agent (test chỉ chạy được khi đã có starter/results/selection.json)
.venv\Scripts\python.exe -m src.run_agent --config configs/dense-B-k3.json --split dev
# 5. Tổng hợp nhãn thủ công và lập bảng so sánh
.venv\Scripts\python.exe starter/agent_metrics.py --cases data/benchmark/cases.json --traces starter/results/agent/dense-B-k3/dev/traces.json --reviews labels/dev/dense-B-k3.reviews.json --split dev --output starter/results/agent/dense-B-k3/dev/agent_metrics.json
.venv\Scripts\python.exe -m src.compare --split dev --configs dense-B-k3 bm25-B-k3 hybrid-B-k3
# Unit test offline (Windows cần UTF-8)
$env:PYTHONUTF8 = '1'; .venv\Scripts\python.exe -m unittest discover -s tests
```

Trace (`runs.jsonl`) chứa text tài liệu nên không được commit. Chạy lại bước 4 sẽ tạo trace mới; vì model có tính ngẫu nhiên, câu trả lời có thể khác đôi chút so với nhãn đã chấm. Chi phí ước tính cho một lượt chạy dev (24 câu) khoảng $0,02–0,05.

---

**Bài cá nhân · Python 3.11+ · tiếp nối Day 6**

Dùng embedding và vector store từ Day 6 để xây một agent biết tìm bằng chứng, trả lời có nguồn và nhận biết khi không đủ thông tin. Bạn tự chọn kiến trúc, số tool và cách tổ chức retrieval.

> **Tùy chọn:** [Gợi ý dùng AI để lập trình](AGENT_SETUP.md) giúp bạn làm lab; phần này không bắt buộc và không tính điểm.

## Bài cần làm

![Agent tìm tài liệu trong kho Day 6 và trả lời có nguồn](assets/retrieval-pipeline.svg)

1. Kết nối dữ liệu, embedding và vector store Day 6.
2. Xây agent dùng model thật và ít nhất một khả năng truy xuất dữ liệu. Agent có thể dùng nhiều tool và nhiều lượt gọi.
3. Thử **ít nhất 3 phương pháp retrieval khác nhau**, trong đó có dense retrieval dùng vector store Day 6. Tự chọn các phương pháp còn lại; xem [gợi ý](GUIDE.md#2-thử-các-phương-pháp-retrieval).
4. Thử nhiều cấu hình, benchmark, phân tích lỗi và chọn cấu hình tốt trên **dev**; sau đó đánh giá trên **test**.

Benchmark tối thiểu có **24 câu**: dev gồm 15 câu có đáp án và 2 câu thiếu bằng chứng; test gồm 5 câu có đáp án và 2 câu thiếu bằng chứng. Xem [benchmark](BENCHMARK.md) và [rubric](RUBRIC.md).

## Thử thách nâng cao · +15 điểm

Index một tài liệu thực tế từ **100 trang** có bảng, ảnh và biểu đồ; xây agent truy xuất và trả lời các câu hỏi về nội dung đó. Hoàn thành [thử thách nâng cao](ADVANCED.md) và đạt **ít nhất 90% benchmark test** để nhận 15 điểm bonus. Phần cơ bản tối đa 85 điểm; toàn bài tối đa 100 điểm.

## Bắt đầu

1. Đọc [hướng dẫn](GUIDE.md), [rubric](RUBRIC.md) và [quy tắc](RULES.md).
2. Tạo repo cá nhân theo [cách nộp bài](SUBMISSION.md), chuẩn bị đầu ra Day 6 và model hỗ trợ tool calling.
3. Dùng hoặc thay thế [starter](starter/README.md); bạn cần kết nối model và vector store thật.
4. Chạy `python -m unittest discover -s tests -v`. Test mock chỉ kiểm tra code, không phải benchmark agent thật.

![Các bước thử nghiệm trên dev và đánh giá trên test](assets/experiment-loop.svg)

Nộp code, cách chạy, cấu hình, benchmark, trace và [báo cáo](report/individual/MSSV_HoTen.md). Thời gian ở [checkpoints](CHECKPOINTS.md) là gợi ý, có thể điều chỉnh theo lớp và phạm vi thử nghiệm.
