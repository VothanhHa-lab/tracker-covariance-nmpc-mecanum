uint32 id
float64[4] state        # x, y, vx, vy trong khung map
float64[16] covariance  # 4x4 matrix, row-major
float64 age             # s, since the last update
bool camera_confirmed
