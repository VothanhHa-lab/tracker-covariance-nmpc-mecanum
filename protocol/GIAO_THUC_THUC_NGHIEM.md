# Giao thức thực nghiệm: NMPC nhận biết bất định cảm biến cho robot Mecanum

Phiên bản 1.0, ngày 28/10/2026. Giao thức này phải được **chốt trước** khi thu dữ liệu pha báo cáo (pha R).
Mọi thay đổi sau đó ghi vào mục 11 (nhật ký sửa đổi), kèm ngày và lý do.

Nguyên tắc chung:
- Mọi số liệu trong bài báo lấy từ rosbag2 ghi trong các pha dưới đây, qua các script `tools/analyze_*.py` và `tools/make_macros.py`.
- Không gõ số tay vào bài. Không loại lượt thử nào sau khi đã xem kết quả.
- Dữ liệu và thông số của bản thảo CA-NMPC cũ **không được dùng**, vì không có log gốc.

---

## 0. Chuẩn bị phần mềm

1. **Trên máy chạy bộ điều khiển** (Ubuntu 22.04, ROS 2 Humble):
   ```bash
   pip install -e pu_nmpc            # thư viện lõi
   pip install ultralytics            # YOLO cho camera
   ```
   Sau đó sao chép `ros2_ws/src/*` vào workspace, chạy `colcon build`, rồi `source install/setup.bash`.
2. **Kiểm thử phần lõi** (không cần ROS), phải PASS hết:
   ```bash
   python3 tests/test_core.py
   python3 tests/test_bag.py
   ```
3. **Driver** (bringup riêng của robot): `rplidar_ros` (RPLIDAR A2), `realsense2_camera` với `align_depth.enable:=true`, cầu nối STM32 nhận `/cmd_vel` và phát `/odom`, AMCL hoặc `robot_localization` phát TF `map→odom→base_link` và `/amcl_pose`.
4. **Đồng bộ đồng hồ** giữa máy robot và máy ground truth: dùng `chrony`, một máy làm server.
   - Ghi độ lệch (`chronyc tracking`) đầu và cuối mỗi buổi.
   - **Chỉ chấp nhận buổi có |offset| < 2 ms.**
5. **Ghi phiên bản:**
   - mã băm SHA-256 của `pu_nmpc_core/*.py` (hàm `tools/common.code_hash`) và của các tệp `config/*.yaml`, ghi vào sổ thí nghiệm mỗi buổi;
   - ghi thêm model CPU/RAM của máy chạy NMPC.

## 1. P1: Dựng hiện trường và kiểm chứng ground truth

1. **Bản đồ và vùng thử.** Dựng bản đồ (SLAM Toolbox), cố định khung `map`. Đánh dấu vùng thử trên sàn: hành lang và sảnh, kích thước đo bằng thước laser.
2. **Camera ground truth.** Gắn camera trần cố định, nhìn bao trọn vùng thử. Hiệu chỉnh nội tham số bằng bàn cờ (lưu tệp `camera_info`).
3. **Thẻ mốc trên sàn.** Dán ≥ 5 thẻ AprilTag 36h11 lên sàn. Đo tọa độ của chúng trong `map` và ghi vào `config/ground_truth.yaml`.
4. **Kiểm chứng ground truth.** Đặt thẻ mục tiêu tại ≥ 9 điểm đã đo, ở hai chiều cao (0,8 m và 1,45 m). Ghi 10 s cho mỗi điểm (phiên `G01`…).
   - Khai báo các điểm này trong `config/gt_validation_points.yaml`.
   - Chạy `tools/analyze_gt_validation.py` → `results/gt_validation.json`.
5. **Tiêu chí chấp nhận:** RMSE ground truth ≤ 3 cm. Nếu không đạt thì chỉnh lại hệ ground truth; không được tiếp tục sang các pha sau.

## 2. P2: Nhiễu cảm biến, tỉ lệ phát hiện, trễ và chuyển động người

Robot **đứng yên**, tắt động cơ. Mỗi người tham gia đội thẻ trên mũ, đo chiều cao thẻ cho từng người.

- **Phiên `N01`…`N06`** (mỗi phiên khoảng 3 phút, `sensor_noise.launch.py`):
  - người đi theo các đường vẽ trên sàn, phủ khoảng cách 0,5–7 m và góc ±180°, với tốc độ chậm, bình thường và nhanh;
  - một phiên có hai người đi cắt nhau và che khuất nhau.
- **Đứng tại chỗ:** ít nhất 2 phiên người đứng yên tại các mốc 1, 2, …, 7 m, để tách nhiễu khỏi sai số do chuyển động.
- **Phân tích:**
  ```bash
  python3 tools/analyze_sensor_noise.py --gt-std <RMSE ground truth của P1> data/N0*/csv
  ```
  Đầu ra:
  - mô hình nhiễu σ_r = a + b·r, σ_b;
  - tỉ lệ phát hiện theo khoảng cách;
  - số phát hiện giả mỗi lần quét;
  - trễ hiệu dụng;
  - hệ số q của mô hình CV cho người đi bộ.
- **Cập nhật cấu hình:** kiểm tra `config/sensors_identified.yaml`, rồi sao chép sang `config/sensors.yaml`. Đặt `q_acc = q_acc_pred =` giá trị `pedestrian_q`. Đặt `lag` ≥ trễ lớn nhất quan sát được.

## 3. P3: Nhận dạng khâu vận tốc

**Đo cơ khí.** Đo bán kính bánh, lx, ly và bán kính bao của robot. Ghi vào `config/robot.yaml` và đổi `source` sang `measured`.

**Thí nghiệm bước** (`identification.launch.py`, phiên `I01`–`I03`, trên sàn trống có người giữ e-stop):
- Bật cổng an toàn: `ros2 topic pub /trial/enable std_msgs/Bool "data: true"`.
- Chuỗi bước theo vx, vy và ω ở nhiều biên độ.

**Phân tích:** `tools/analyze_identification.py data/I0*/csv` → τ, K, trễ cho từng trục, cùng NRMSE trên 1/3 cuối mỗi phiên.
- **Chấp nhận:** NRMSE ≤ 10 %. Nếu không đạt, ghi lại như một kết quả bất lợi và cân nhắc mô hình bậc cao hơn.
- Cập nhật `tau` trong `config/robot.yaml`.

**Bánh và gia tốc.** Xác định `wheel_speed_max` (80 % mức bão hòa đo được) và `body_acc_max` (mức không trượt bánh, kiểm bằng ground truth).

## 4. P4: Đánh giá bộ bám

Chạy offline trên mọi phiên có ground truth (P2 và cả các lượt vòng kín):
```bash
python3 tools/analyze_tracking.py --q-pred <pedestrian_q> data/N0*/csv data/R*/csv
```

Các cấu hình so sánh:
- fusion, chỉ LiDAR, chỉ camera;
- mỗi cấu hình đều có và không bù trễ.

Chỉ số:
- RMSE vị trí, OSPA (c = 1 m), tỉ lệ phát hiện, số lần đổi ID;
- **độ phủ elip 95 % ở tầm 0, 0,5 và 1,0 s**.

**Chỉ tiêu then chốt:** độ phủ ở tầm 1 s của cấu hình fusion có bù trễ phải ≥ 90 %. Nếu thấp hơn, hiệp phương sai quá lạc quan. Khi đó, chỉ được chỉnh q **trên dữ liệu P2** (không dùng dữ liệu pha R), sau đó chốt lại. Phải báo cáo trung thực cả hai lần thử.

## 5. P5: Pilot (pha D)

- Lập lịch pilot:
  ```bash
  python3 tools/make_schedule.py --phase D --participants 2 --reps 1 --seed <hạt D>
  ```
- Mục đích: kiểm tra an toàn, thời gian tính trên máy thật, lỗi phần mềm, và độ khó của kịch bản.
- **Được phép** chỉnh: v_ref, timeout, vị trí xuất phát của người, và trọng số NMPC **chung cho mọi phương pháp**.
- **Không được** chỉnh riêng cho phương pháp đề xuất.
- Dữ liệu pha D **không được báo cáo** làm kết quả chính. Có thể nêu trong phần phương pháp: "pilot gồm n lượt dùng để chỉnh…".
- **Chốt tham số.** Sau pilot, đóng băng `config/*.yaml`. Ghi mã băm vào mục 11. Từ đây không sửa nữa.

## 6. P6: Chiến dịch báo cáo (pha R)

### 6.1 Thiết kế
- **Phương pháp:** cov (đề xuất), fixed và heuristic (tái hiện CA-NMPC). Ba phương pháp dùng chung NMPC, chung chi phí và chung giới hạn.
- **Kịch bản:**
  - H1: cắt ngang
  - H2: đối đầu trong hành lang
  - H3: lối hẹp có hai người
  - H4: người bước ra từ sau vật che
  - H5: tiếp cận ngoài vùng nhìn của camera
- **Cỡ mẫu:** 5 người × 4 lần lặp = **20 lượt mỗi ô**, tổng cộng 300 lượt.
  - Với 0 va chạm trong 20 lượt, cận trên 95 % vẫn là 13,9 %. Vì vậy bài **không** được tuyên bố xác suất va chạm thấp. Mục tiêu chính là đo khoảng hở, thời gian xâm phạm vùng cá nhân, tỉ lệ thành công và thời gian.
  - Nếu có thêm người tham gia, tăng số người, không tăng số lần lặp của mỗi người.
- **Ngẫu nhiên hóa:**
  ```bash
  python3 tools/make_schedule.py --phase R --participants 5 --reps 4 --seed <hạt R>
  ```
  Thứ tự kịch bản được xáo trộn cho từng người; thứ tự phương pháp xoay vòng theo hình vuông Latin.
- **Làm mù:** người tham gia không biết phương pháp đang chạy. Người vận hành biết, nên không được hướng dẫn người tham gia khác nhau giữa các phương pháp.

### 6.2 Quy trình một lượt
1. Kiểm tra đèn ground truth, chrony và pin. Điền phiếu lượt (mục 9).
2. Khởi chạy:
   ```bash
   ros2 launch pu_nmpc_ros closed_loop.launch.py trial_id:=R017 method:=<theo lịch> schedule:=protocol/schedule_R.csv
   ```
   Node ground truth chạy trên máy camera trần, cùng ROS_DOMAIN_ID.
3. Người tham gia đứng ở vị trí xuất phát của kịch bản. Người vận hành mở `mark_event` và nhấn `s`. Người tham gia bắt đầu đi theo tín hiệu đèn hoặc vạch quy định.
4. Lượt kết thúc khi robot tới đích, hết giờ, hoặc người vận hành nhấn `i` (can thiệp: e-stop hoặc dừng tay) hay `a` (hủy vì lỗi kỹ thuật).
5. Mọi va chạm hay tiếp xúc ghi bằng `n` → `note:contact`, đồng thời ghi trên phiếu.

### 6.3 Xử lý lượt hỏng
- Lượt hủy vì lỗi kỹ thuật không do bộ điều khiển (mất ground truth, rơi bag, pin): **chạy lại một lần** ở cuối buổi với cùng trial_id thêm hậu tố `b`, và ghi lý do.
- Lượt gốc vẫn được giữ và báo cáo trong bảng "lượt hủy".
- Lượt can thiệp **không** được chạy lại, và tính là thất bại.

## 7. Kế hoạch phân tích (chốt trước)

- **Bước 1:** trích dữ liệu: `tools/bag_to_csv.py data/<id>/bag` cho mọi phiên.
- **Bước 2:** tính chỉ số:
  ```bash
  python3 tools/analyze_closed_loop.py --phase R data
  ```
  Bước này liệt kê **mọi** lượt trong lịch, kể cả lượt không chạy.
- **Bước 3:** sinh macro cho bài:
  ```bash
  python3 tools/make_macros.py --paper ../paper_sensors
  ```

Định nghĩa các chỉ số:
- **Kết cục chính:** khoảng hở nhỏ nhất (mép robot tới mép người, lấy từ ground truth) và thời gian ở trong vùng cá nhân 0,5 m.
- **Kết cục phụ:** tỉ lệ thành công, thời gian tới đích, độ giật, thời gian tính (p50, p95, max).
- **Tiếp xúc:** khoảng cách tâm nhỏ hơn r_robot + 0,25 m, hoặc người vận hành ghi `note:contact`.

Thống kê:
- **So sánh ghép cặp** theo (người, kịch bản, lần lặp): cov − fixed và cov − heuristic. Dùng khoảng tin cậy bootstrap 95 % với 10 000 mẫu.
- **Tỉ lệ:** khoảng Clopper–Pearson hai phía. Khi quan sát 0 sự kiện, ghi cận trên một phía.
- **Không kiểm định đa giả thuyết** cho các kịch bản riêng lẻ. Kết quả theo kịch bản chỉ mang tính mô tả.

Kết quả bất lợi:
- Báo cáo đầy đủ, kể cả khi cov thua fixed về thời gian hoặc tỉ lệ thành công. Bản chất phương pháp là thận trọng hơn khi bất định lớn.
- Báo cáo cả các lượt dừng vì `stale_stop`, `watchdog_brake` hay `solver_fail_brake`.

## 8. An toàn và đạo đức

- Phải có quyết định của Hội đồng đạo đức hoặc khoa học của trường **trước P2** (P2 đã có người tham gia), hoặc văn bản miễn xét có lý do.
- Người tham gia là người trưởng thành, tự nguyện, ký phiếu đồng thuận (mục 10) và được quyền rút lui bất cứ lúc nào.
- Dữ liệu được ẩn danh bằng mã P01…; không lưu tên trong bag.
  - Ảnh màu chỉ dùng cho bộ phát hiện, **không ghi vào bag**: topic ảnh không có trong danh sách ghi.
  - Nếu cần ghi ảnh để minh họa, phải xin đồng ý riêng và làm mờ mặt.
- **Giới hạn an toàn:**
  - v_ref ≤ 0,6 m/s;
  - người vận hành cầm e-stop không dây trong suốt lượt;
  - mỗi lượt đều có vùng dừng cứng 0,3 m trên quét thô ở cầu nối động cơ, độc lập với NMPC (kiểm tra trước buổi);
  - người tham gia mang giày kín và được hướng dẫn không chạy.
- **Trước mỗi buổi:** kiểm tra e-stop 3 lần, và kiểm tra watchdog bằng cách tắt `/odom`: robot phải dừng.

## 9. Phiếu lượt thử (in hoặc bảng tính)

Mã lượt | Ngày giờ | Người (Pxx) | Kịch bản | Phương pháp | Lần lặp | Offset chrony (ms) | Pin (%) | Kết cục (đích / hết giờ / can thiệp / hủy) | Tiếp xúc (có/không) | Ghi chú | Người vận hành

## 10. Mẫu phiếu đồng thuận (tóm tắt nội dung bắt buộc)

- Tên đề tài, cơ quan, người chịu trách nhiệm, số điện thoại liên hệ.
- Mục đích, việc người tham gia phải làm (đi bộ theo đường quy định gần robot, khoảng 1–2 giờ mỗi buổi), và rủi ro (va chạm nhẹ ở tốc độ ≤ 0,6 m/s), cùng các biện pháp an toàn.
- Dữ liệu thu thập: quỹ đạo, quét LiDAR, ảnh độ sâu. Ảnh màu không được lưu. Thông tin về ẩn danh, thời gian lưu và nơi lưu.
- Quyền rút lui bất cứ lúc nào, không cần lý do.
- Chữ ký và ngày của người tham gia và của người nghiên cứu.

## 11. Nhật ký sửa đổi và chốt tham số

| Ngày | Nội dung | Mã băm code / config | Người ghi |
|---|---|---|---|
| 28/10/2026 | Phiên bản 1.0 | — | — |
