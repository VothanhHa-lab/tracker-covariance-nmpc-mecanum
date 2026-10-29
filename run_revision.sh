#!/bin/bash
# Chiến dịch bổ sung sau phản biện: revision (7 nhánh × 250 lượt, 7 tiến trình), rồi timing (1 tiến trình, máy rảnh).
cd "$(dirname "$0")"
L=results/sim_revision_log.txt
python3 tools/sim_campaign.py --campaign revision >> $L 2>&1
python3 tools/sim_campaign.py --campaign timing --workers 1 >> $L 2>&1
echo "$(date '+%F %T') ALL DONE" >> $L
