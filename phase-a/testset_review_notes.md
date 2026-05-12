# Testset Review Notes — phase-a/testset_v1.csv

**Reviewer:** Nguyen Duc Hoang Phuc (2A202600150)  
**Date:** 2026-05-12  
**Tool:** RAGAS 0.4.3 · gpt-4o-mini · 15 document sections from Day18 corpus

---

## 1. Distribution Verification

| evolution_type | Count | % Actual | % Target | Status |
|----------------|-------|----------|----------|--------|
| simple         | 26    | 49%      | 50%      | OK     |
| multi_context  | 14    | 26%      | 25%      | OK     |
| reasoning      | 13    | 25%      | 25%      | OK     |
| **Total**      | **53**| 100%     | —        | **PASS (>= 50)** |

Distribution is within ±1% of target for all three types. Total row count 53 >= 50. **PASS.**

---

## 2. Review of First 10 Questions

### Q1 (row 1 · simple)
**Question:** Công ty Cổ phần DHA Surfaces nộp tờ khai thuế gì và khi nào?  
**Ground truth:** Công ty Cổ phần DHA Surfaces nộp tờ khai thuế giá trị gia tăng mẫu số 01/GTGT, kỳ tính thuế quý 4 năm 2024. Tờ khai được ký điện tử ngày 24 tháng 01 năm 2025.  
**Context:** Section "Thông tin chung" — 1 chunk, relevant.  
**Assessment:** Tốt. Câu hỏi rõ ràng, kết hợp hai thông tin (loại tờ khai + thời điểm). Ground truth chính xác và đủ thông tin. Suitable for faithfulness / relevancy evaluation.

---

### Q2 (row 2 · simple)
**Question:** Tờ khai thuế giá trị gia tăng mẫu số 01/GTGT được ký điện tử vào ngày 24 tháng 01 năm 2025 bởi ai?  
**Ground truth:** ...bởi Công ty Cổ phần DHA Surfaces; người đại diện ghi trên tờ khai là Trịnh Thị Sang.  
**Context:** Section "Thông tin chung" — correct.  
**Assessment:** Chấp nhận được. Tuy nhiên câu hỏi quá dài, lặp lại ngày tháng trong phần điều kiện làm câu hỏi mang tính factoid thấp (reader đã biết ngày). Có thể rút ngắn thành "Ai ký điện tử tờ khai 01/GTGT quý 4/2024?" — nhưng không sửa vì câu hỏi vẫn grammatically correct.

---

### Q3 (row 3 · simple)
**Question:** Giá trị hàng hóa trong kỳ là bao nhiêu và thuế GTGT của hàng hóa đó là bao nhiêu?  
**Ground truth:** Giá trị hàng hóa, dịch vụ mua vào trong kỳ là 2.405.743.241 đồng. Thuế GTGT của hàng hóa, dịch vụ mua vào trong kỳ là 215.163.767 đồng.  
**Context:** Section "Chỉ tiêu thuế GTGT đầu vào" — correct.  
**Assessment:** Tốt. Câu hỏi 2-in-1 (giá trị + thuế), ground truth trả lời đầy đủ cả hai. Phù hợp để đo context_precision.

---

### Q4 (row 4 · simple) — **[TÔI ĐÃ SỬA]**

**ORIGINAL (lỗi):** `thuế giá trị gia tang la gi?`  
**EDITED:** `Thuế GTGT của hàng hóa, dịch vụ mua vào trong kỳ là bao nhiêu?`  
**Ground truth (giữ nguyên):** Thuế giá trị gia tăng của hàng hóa, dịch vụ mua vào trong kỳ là 215.163.767 đồng.  
**Lý do sửa:**  
1. Câu gốc thiếu dấu tiếng Việt ("tang" thay vì "tăng", "la gi" thay vì "là gì") — lỗi encoding do persona của RAGAS sinh ra text không dấu.  
2. Câu gốc hỏi về định nghĩa ("là gì") nhưng ground truth trả lời bằng một con số cụ thể — câu hỏi và câu trả lời không nhất quán về loại thông tin.  
3. Câu đã sửa phù hợp đúng với ground truth hiện có.

---

### Q5 (row 5 · simple)
**Question:** Hàng hóa dịch vụ bán ra là gì và thuế suất của nó là bao nhiêu?  
**Ground truth:** Hàng hóa, dịch vụ bán ra chịu thuế suất 10% có giá trị 3.703.685.610 đồng và thuế GTGT tương ứng là 344.675.400 đồng...  
**Assessment:** Chấp nhận được. Phần "là gì" trong câu hỏi không được trả lời trực tiếp (ground truth bắt đầu với số tiền, không định nghĩa). Tuy nhiên câu hỏi vẫn truyền đạt được intent chính là "thuế suất bán ra". Không ảnh hưởng nghiêm trọng đến evaluation.

---

### Q6 (row 6 · simple)
**Question:** Giá trị hàng hóa dịch vụ bán ra chịu thuế suất 10% là bao nhiêu?  
**Ground truth:** Hàng hóa, dịch vụ bán ra chịu thuế suất 10% có giá trị 3.703.685.610 đồng.  
**Assessment:** Rất tốt. Câu hỏi rõ ràng, ground truth chính xác và concise. Đây là câu mẫu tốt nhất trong batch đầu.

---

### Q7 (row 7 · simple)
**Question:** Hàng hóa là gì và tổng doanh thu của nó là bao nhiêu?  
**Ground truth:** Tổng doanh thu của hàng hóa, dịch vụ bán ra trong kỳ là 3.703.685.610 đồng.  
**Assessment:** Chất lượng trung bình. Phần "Hàng hóa là gì" không được trả lời trong GT (GT chỉ nêu tổng doanh thu). Tạo ra mismatch giữa question scope và answer scope. Acceptable vì phần quan trọng hơn (tổng doanh thu) vẫn được trả lời.

---

### Q8 (row 8 · simple)
**Question:** Tổng doanh thu từ dịch vụ là bao nhiêu?  
**Ground truth:** Tổng doanh thu của hàng hóa, dịch vụ bán ra trong kỳ là 3.703.685.610 đồng.  
**Assessment:** Có vấn đề nhỏ: câu hỏi hỏi riêng "dịch vụ" nhưng ground truth trả lời tổng hợp "hàng hóa + dịch vụ". Người dùng thực có thể hiểu lầm rằng chỉ doanh thu dịch vụ là 3.7 tỷ. Trong tài liệu gốc, doanh thu hàng hóa và dịch vụ không được tách riêng, nên GT là chính xác nhưng câu hỏi tạo ra ambiguity.

---

### Q9 (row 9 · simple)
**Question:** Thuế giá trị gia tăng là bao nhiêu?  
**Ground truth:** Thuế giá trị gia tăng phải nộp của hoạt động sản xuất kinh doanh trong kỳ là 52.133.830 đồng.  
**Assessment:** Câu hỏi quá chung chung — "thuế GTGT là bao nhiêu" có thể có nhiều đáp án (thuế đầu vào, đầu ra, phát sinh, phải nộp đều là những con số khác nhau). Ground truth chọn "phải nộp" không có context rõ ràng tại sao. Câu này có thể gây nhiễu khi đánh giá context_recall vì retriever có thể trả về nhiều đoạn GTGT khác nhau.

---

### Q10 (row 10 · simple)
**Question:** Thuế GTGT còn phải nộp là bao nhiêu nếu bù trừ với 0 đồng?  
**Ground truth:** Thuế giá trị gia tăng còn phải nộp trong kỳ là 52.133.830 đồng, mặc dù thuế giá trị gia tăng của dự án đầu tư được bù trừ với thuế GTGT còn phải nộp là 0 đồng.  
**Context:** Section "Xác định nghĩa vụ thuế" — correct.  
**Assessment:** Tốt. Câu hỏi có thêm điều kiện "nếu bù trừ với 0 đồng" giúp xác định rõ scenario. Ground truth phản ánh đúng bản chất của bài toán bù trừ và kết quả cuối cùng.

---

## 3. Tổng kết chất lượng

| Tiêu chí | Đánh giá |
|----------|----------|
| Câu hỏi rõ ràng, grammatically correct | 9/10 (Q4 lỗi dấu đã sửa) |
| Ground truth khớp với câu hỏi | 8/10 (Q5, Q7, Q8, Q9 có partial mismatch) |
| Context phù hợp | 10/10 |
| Duplicate/near-duplicate | 1 cặp (Q26+Q27 gần giống nhau về nội dung chuyển dữ liệu ra nước ngoài) |

**Tổng thể:** Chất lượng **chấp nhận được** cho mục đích đánh giá RAG. Các câu simple phần lớn rõ ràng. Câu reasoning và multi_context có chất lượng cao hơn vì câu hỏi phức tạp hơn và context đa nguồn.

---

## 4. Bản ghi chỉnh sửa (MANDATORY EDIT LOG)

| Row | Trường | Nội dung cũ | Nội dung mới | Lý do |
|-----|--------|-------------|--------------|-------|
| 4   | question | `thuế giá trị gia tang la gi?` | `Thuế GTGT của hàng hóa, dịch vụ mua vào trong kỳ là bao nhiêu?` | Lỗi thiếu dấu tiếng Việt; hỏi "định nghĩa" nhưng GT trả lời bằng số tiền — không nhất quán |
