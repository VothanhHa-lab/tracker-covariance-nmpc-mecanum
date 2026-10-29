#!/bin/bash
# Post hoc campaign: revision (7 arms x 250 runs, 7 processes), then timing (1 process, idle machine).
cd "$(dirname "$0")"
L=results/sim_revision_log.txt
python3 tools/sim_campaign.py --campaign revision >> $L 2>&1
python3 tools/sim_campaign.py --campaign timing --workers 1 >> $L 2>&1
echo "$(date '+%F %T') ALL DONE" >> $L
