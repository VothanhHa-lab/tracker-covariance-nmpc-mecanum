# Nhật ký thiết kế nghiên cứu mô phỏng (chốt 2026-10-29 07:15)

Mọi quyết định dưới đây được đưa ra trên hạt phát triển 0–9 (`sim_campaign --campaign dev`), TRƯỚC khi chạy
hạt báo cáo (main 1000–1099, robust 2000–2049, ablation 1000–1099). Không chỉnh tham số theo kết quả báo cáo.

1. Hành lang H2 nới từ 2.0 m lên 2.6 m; H3 dùng hành lang 3.0 m với chỗ thắt 1.6 m. Lý do: với người không né,
   hành lang 2.0 m không đủ chỗ cho hai bên tránh nhau (va chạm không phụ thuộc bộ điều khiển).
2. Phạt vi phạm ràng buộc tường × 100 so với ràng buộc người (áp dụng cho cả ba phương pháp).
3. q của mô hình người = 0.022 m²/s³, ước lượng hợp lý cực đại trên 300 quỹ đạo người của hạt 0–49 (thay giá trị đặt tay 0.5).
4. Va chạm được phân loại chủ động/thụ động; kiểm tra chạm tường theo khung chữ nhật 0.60 × 0.55 m.
5. Người danh định né robot theo lực xã hội (A = 3 m/s², B = 0.5 m, λ = 0.35, giả định); người không né giữ làm điều kiện xấu nhất.
   Lý do: với người không né, mọi va chạm ở pilot đều là thụ động (robot đã dừng), không phản ánh bộ điều khiển.
6. Điểm tĩnh: một điểm gần nhất trong mỗi cung góc, 16 cung (8 cung để lọt góc lõm của chỗ thắt H3).
7. Bù trễ chấp hành trong NMPC (input_delay = trễ chấp hành giả định 0.05 s), áp dụng cho cả ba phương pháp.

Kết quả pilot cuối (hạt 0–9, 150 lượt): 150/150 thành công cho cả ba phương pháp. Số liệu pilot KHÔNG báo cáo.

Mã băm: code=ca4e993fe8c95636 simstudy.yaml=44783b41c653ecbd controller.yaml=5827ce9c3969c872 tracker.yaml=12753bf8fef76ff8

## Bản sửa sau phản biện (29/10/2026)
Các bổ sung này được quyết định sau khi đã xem kết quả của chiến dịch báo cáo, nên bài báo ghi rõ là phân tích bổ sung hậu kiểm (post hoc). Chúng được đánh giá trên các hạt mới 3000–3049, là những hạt chưa từng được chạy.
1. **Phóng hiệp phương sai γ.** Quy tắc được chốt trước khi chạy chiến dịch revision: chọn γ nhỏ nhất trong lưới (1; 1,25; 1,5; 1,75; 2; 2,5; 3; 4) sao cho độ phủ gộp đạt ≥ 95 % ở cả ba tầm dự báo 0; 0,5 và 1 s. Việc chọn chỉ dùng hạt dev 0–9 (chiến dịch devcal, phương pháp cov). Kết quả: γ = 1,5 (xem `results/calibration.json`).
2. **Quét lề cố định.** Thêm các mức đệm 0,45; 0,60 và 0,75 m bên cạnh mức gốc 0,30 m, để so sánh trên đường Pareto giữa khoảng cách và thời gian.
3. **Đo thời gian giải.** Chiến dịch timing chạy 1 tiến trình trên máy không có chiến dịch nào khác (hạt 3000–3009).
4. **Thay đổi mã.** Thêm `MethodParams.cov_inflate` (mặc định 1, nên không đổi kết quả của các chiến dịch trước) và các bộ đếm độ phủ theo γ trong `simstudy.py`.
5. **Đối chứng thứ tư "ccconst"** (người dùng yêu cầu ngày 29/10/2026).
   - Nội dung: ràng buộc xác suất giống hệt phương pháp đề xuất, nhưng hiệp phương sai người là hằng số tính trước. Đó là hiệp phương sai hậu nghiệm dừng của bộ lọc CV khi cập nhật vị trí ở 10 Hz, với nhiễu đẳng hướng của LiDAR ở 3 m cộng sai số định vị. Kết quả: σ_p ≈ 0,039 m, σ_v ≈ 0,105 m/s.
   - Các tham số được chốt trước khi chạy và không được tinh chỉnh.
   - Chạy trên hạt chính 1000–1099 (altmain), hạt mới 3000–3049 (altrev), và đo thời gian riêng với 1 tiến trình (alttiming).
6. **Đo lại thời gian.** Trong lần đo đầu của chiến dịch timing, lượt H5_side / hạt 3002 / cov chạy hết 348 s thời gian thực, trong khi các lượt khác chỉ mất khoảng 4–19 s. Trong lượt này có một lần giải dài 8224 ms. Hiện tượng phù hợp với việc tiến trình bị tạm dừng, chẳng hạn máy ngủ. Toàn bộ chiến dịch timing được đo lại sau chiến dịch alt, với chế độ giữ máy thức. Lần đo đầu được lưu ở `sim_timing_run1.*`.
7. **Kiểm chứng trên DROW v2 (29/10/2026).**
   - Kết quả: tỉ lệ phát hiện chỉ đạt giả định 0,90 tới khoảng 3 m; ở 6–8 m chỉ còn khoảng 62 %.
   - Quyết định của tác giả liên hệ: **giữ nguyên mô phỏng**, chỉ báo cáo kết quả DROW (mục 5.7 của bài).
   - Không chạy thêm điều kiện mô phỏng nào dựa trên số đo của DROW.
