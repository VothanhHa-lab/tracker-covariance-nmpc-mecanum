#!/bin/bash
# Đo lại thời gian của 3 bộ cũ: lần đo đầu có một lượt (H5_side, hạt 3002, cov) kéo dài 348 s thời gian thực
# thay vì ~6 s, dấu hiệu tiến trình bị treo/máy ngủ. Lần đo đầu được giữ lại ở sim_timing_run1.* và nêu trong bài.
cd "$(dirname "$0")"
until grep -q "ALL DONE" results/sim_alt_log.txt 2>/dev/null; do sleep 20; done
mv results/sim_timing.csv results/sim_timing_run1.csv
mv results/sim_timing.meta.json results/sim_timing_run1.meta.json
python3 tools/sim_campaign.py --campaign timing --workers 1 >> results/sim_timing2_log.txt 2>&1
echo "$(date '+%F %T') ALL DONE" >> results/sim_timing2_log.txt
