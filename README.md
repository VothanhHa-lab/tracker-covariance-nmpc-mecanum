# pu_nmpc: NMPC nhận biết bất định cảm biến cho robot Mecanum (bài *Sensors*)

Thư mục này chứa mã nguồn thu dữ liệu và phân tích cho bài báo trong `../paper_sensors`. Mã thay thế hoàn toàn bản CA-NMPC cũ; bản cũ không có log gốc nên **không dùng lại số liệu nào của nó**.

| Thư mục | Nội dung |
|---|---|
| `pu_nmpc_core/` | Thư viện Python thuần, kiểm thử được mà không cần ROS: mô hình (`models`), bộ bám hợp nhất bất đồng bộ (`tracker`), phát hiện người (`detect`), ràng buộc theo hiệp phương sai (`constraint`), NMPC CasADi/IPOPT (`nmpc`), bộ điều khiển và 3 phương pháp (`controller`), nhận dạng (`identification`), chỉ số và thống kê (`metrics`), mô phỏng kiểm thử phần mềm (`sim`) |
| `ros2_ws/src/pu_nmpc_msgs` | Message: phát hiện, vệt người, chẩn đoán bộ điều khiển |
| `ros2_ws/src/pu_nmpc_ros` | Node: `lidar_people_detector`, `camera_people_detector`, `fusion_tracker`, `nmpc_controller`, `scan_stop_guard`, `gt_tag_tracker`, `trial_manager`, `step_test`, `mark_event`; launch cho từng pha |
| `config/` | YAML tham số. Mục `TO_MEASURE` / `TO_IDENTIFY` phải thay bằng giá trị đo trước khi thí nghiệm |
| `tools/` | `make_schedule` → `bag_to_csv` → `analyze_gt_validation`, `analyze_identification`, `analyze_sensor_noise`, `analyze_tracking`, `analyze_closed_loop` → `make_macros`, `make_figures` |
| `protocol/GIAO_THUC_THUC_NGHIEM.md` | Giao thức P0–P7, an toàn, đạo đức, phiếu lượt, nhật ký chốt tham số |
| `tests/` | `test_core.py` (14 kiểm thử), `test_bag.py` (ghi và đọc lại bag), `make_synthetic_sessions.py` (dữ liệu **tổng hợp** chỉ để thử đường ống) |
| `results/` | Kết quả từ dữ liệu thật (hiện trống) |

## Bảo đảm trung thực số liệu (cài sẵn trong mã)

- **Nguồn gốc kết quả.** Mỗi tệp trong `results/` ghi mã băm của dữ liệu đầu vào, mã băm của mã nguồn và cờ `synthetic`.
- **Chặn dữ liệu tổng hợp.** `make_macros.py --paper` và `make_figures.py --paper` **từ chối** chạy nếu bất kỳ kết quả nào là dữ liệu tổng hợp. Mục nào chưa có kết quả thì in ra `\TBD{}`.
- **Tham số thiết kế cũng qua macro.** Các giá trị như d0, ε, N, Δt, v_ref được đọc từ `config/`, nên bài báo và mã luôn khớp nhau.
- **Liệt kê đủ lượt thử.** `analyze_closed_loop.py` liệt kê **mọi** lượt trong lịch, kể cả lượt không chạy hay bị hủy.

## Lệnh nhanh

```bash
pip install -e .[analysis]            # numpy, scipy, casadi, pyyaml, pandas, matplotlib, rosbags
python3 tests/test_core.py && python3 tests/test_bag.py
python3 tools/make_schedule.py --phase R --participants 5 --reps 4 --seed <hạt> --out protocol/schedule_R.csv
python3 tools/bag_to_csv.py ~/pu_nmpc_data/R001/bag
python3 tools/analyze_closed_loop.py --phase R ~/pu_nmpc_data
python3 tools/make_macros.py --paper ../paper_sensors && python3 tools/make_figures.py --paper ../paper_sensors
../paper_sensors/build_preview.sh
```

## Đã kiểm chứng và chưa kiểm chứng

**Đã kiểm chứng trên máy này:**
- 14 kiểm thử phần lõi.
- Đọc và ghi bag với message tự định nghĩa.
- Toàn bộ đường ống phân tích trên dữ liệu tổng hợp: khôi phục đúng τ, độ trễ và nhiễu đã đặt.
- Chặn dữ liệu tổng hợp khi sinh macro và vẽ hình cho bài.

**Chưa kiểm chứng**, vì máy này không có ROS 2:
- Các node ROS 2 mới chỉ được kiểm tra cú pháp (`py_compile`).
- Phải build bằng colcon và chạy thử trên robot ở pha P0 và P5.
- Thời gian giải trên Raspberry Pi 4 chưa được đo. Trên Apple M3, mô phỏng cho p50 khoảng 10–20 ms. Đây chỉ là con số tham khảo, không đưa vào bài.

**Lỗi đã tìm và sửa trong kiểm thử phần mềm:**
- Phương pháp hiệp phương sai quá dè dặt: robot không vượt được người đi ngược chiều. Đã sửa bằng cách giữ nguyên σ sau T_f, và bài báo nêu rõ phạm vi bảo đảm.
- Hợp nhất hai cảm biến sinh vệt trùng. Đã sửa bằng cách chặn sinh vệt mới trong cổng và gộp vệt trùng; đã có kiểm thử hồi quy.

## Nghiên cứu mô phỏng (bài Sensors)
- `pu_nmpc_core/simstudy.py`: môi trường có tường và che khuất, người đi bộ theo OU và lực xã hội, mô hình cảm biến theo datasheet, 5 kịch bản H1–H5.
- `config/simstudy.yaml`: mọi tham số kèm nguồn (Spec / DS / Assumed).
- `tools/sim_campaign.py`: chiến dịch dev (hạt 0–9, chỉ dùng phát triển), main, robust, ablation. `run_sim_campaigns.sh` chạy cả ba chiến dịch báo cáo.
- `tools/analyze_sim.py` → `results/sim_summary.json`; `tools/make_sim_macros.py` → macro LaTeX; `tools/make_sim_figures.py` → hình.
- `results/sim_design_log.md`: 7 thay đổi rút ra từ pilot trên hạt dev, cùng mã băm cấu hình đã chốt trước khi chạy hạt báo cáo.
- Thời gian giải được đo khi 7 lượt chạy song song trên Apple M3, nên chỉ có giá trị tham khảo. Trong chiến dịch robust có lần giải lên tới vài giây (thời gian thực tế, vượt giới hạn CPU 80 ms); nguyên nhân chưa được cô lập.


## Kiểm chứng một phần trên dữ liệu thật công khai (DROW v2)
- **Dữ liệu:** không đóng gói kèm (khoảng 1,1 GB sau khi giải nén). Tải về bằng các lệnh sau:
  ```
  mkdir -p external_data/DROWv2 && cd external_data/DROWv2
  curl -L -o DROWv2-data.zip https://github.com/VisualComputingInstitute/DROW/releases/download/v2/DROWv2-data.zip
  unzip DROWv2-data.zip
  ```
  SHA-256 của file zip: `1537422ab02e14de0cbd60dfde60ca9efb354e9d48e4d7bcca5ecfcdbe0df227`. Giấy phép MIT; riêng tập con `reha` là CC-BY-NC-SA 3.0.
- **Phân tích:** `python3 -I tools/analyze_drow.py external_data/DROWv2/DROWv2-data` ghi kết quả ra `results/drow_validation.json`. Sau đó `make_sim_macros.py` sinh các macro `Dr*` và `make_sim_figures.py` vẽ `drow.pdf`.
- **Lưu ý:** cảm biến là SICK S300, khác với RPLIDAR A2 của robot. Nhãn là vị trí do người gán tay, không phải ground truth độc lập. Dữ liệu không có bản đồ tĩnh, nên số phát hiện giả là cận trên.
