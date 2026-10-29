uint32 id
float64[4] state        # x, y, vx, vy trong khung map
float64[16] covariance  # ma trận 4x4 theo hàng
float64 age             # s, từ lần cập nhật cuối
bool camera_confirmed
