# Nhãn thủ công

`labels/<split>/<config>.reviews.json` là đầu vào `--reviews` của `starter/agent_metrics.py`. Mỗi câu có một bản ghi:

| Trường | Ý nghĩa |
|---|---|
| `answer_grade` | Chỉ cho câu có đáp án. **0** = sai, thiếu evidence, hoặc từ chối; **1** = đúng một phần (thiếu điều kiện hoặc một vế); **2** = đúng và đủ, đúng đơn vị và tổ chức |
| `citation_support` | Một giá trị `true`/`false` cho mỗi citation. Chỉ `true` khi chunk nằm trong context **và** text của nó hỗ trợ đúng khẳng định mà citation gắn vào |
| `abstention_correct` | Chỉ cho câu thiếu bằng chứng. `true` khi agent **thực sự từ chối** (status `abstain`) và không đưa ra khẳng định vô căn cứ |
| `abstention_content_ok` | Thông tin phụ: nội dung câu trả lời có từ chối đúng không, kể cả khi model đặt cờ `abstain=false` |
| `reason` | Lý do chấm |
| `labeler` / `reviewed_by` | Người chấm nháp và người duyệt lại |

Cách chấm:
- Đọc toàn văn từng chunk được cite (`python -m src.review_sheet --run <thư mục run>`) và đối chiếu với evidence gold trong `data/benchmark/`.
- Không chấp nhận citation chỉ vì ID hợp lệ: chunk phải thật sự chứa thông tin của khẳng định.
- Các nhãn này do Claude chấm nháp. Học viên cần duyệt lại theo PDF rồi điền `reviewed_by`; nếu sửa nhãn nào thì ghi lý do vào `reason`.
