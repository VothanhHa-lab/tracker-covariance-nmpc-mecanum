#!/bin/bash
# Repeat the timing of the original controllers: in the first attempt one run (H5_side, seed 3002, cov) took 348 s of wall-clock
# time instead of ~6 s, indicating a suspended process / machine sleep. The first attempt is kept as sim_timing_run1.* and reported.
cd "$(dirname "$0")"
until grep -q "ALL DONE" results/sim_alt_log.txt 2>/dev/null; do sleep 20; done
mv results/sim_timing.csv results/sim_timing_run1.csv
mv results/sim_timing.meta.json results/sim_timing_run1.meta.json
python3 tools/sim_campaign.py --campaign timing --workers 1 >> results/sim_timing2_log.txt 2>&1
echo "$(date '+%F %T') ALL DONE" >> results/sim_timing2_log.txt
