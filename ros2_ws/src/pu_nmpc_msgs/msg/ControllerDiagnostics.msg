std_msgs/Header header
string trial_id
string method
string mode            # nmpc | stale_stop | solver_fail_brake | disabled | watchdog_brake
string solver_status
float64 solve_ms
int32 iters
float64 max_slack
float64 speed_scale
uint32 n_people
float64[] sigma_n      # σ theo pháp tuyến lớn nhất của từng người (phương pháp cov)
float64[] rho          # chỉ số rủi ro (phương pháp heuristic)
float64[3] cmd         # vx, vy, omega đã gửi
