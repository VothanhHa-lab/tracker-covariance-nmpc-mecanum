#!/bin/bash
# Đối chứng ccconst: chờ run_revision.sh xong (kể cả đo thời gian), rồi chạy altmain + altrev, sau đó alttiming 1 tiến trình.
cd "$(dirname "$0")"
until grep -q "ALL DONE" results/sim_revision_log.txt 2>/dev/null; do sleep 20; done
L=results/sim_alt_log.txt
python3 tools/sim_campaign.py --campaign altmain >> $L 2>&1
python3 tools/sim_campaign.py --campaign altrev >> $L 2>&1
python3 tools/sim_campaign.py --campaign alttiming --workers 1 >> $L 2>&1
echo "$(date '+%F %T') ALL DONE" >> $L
