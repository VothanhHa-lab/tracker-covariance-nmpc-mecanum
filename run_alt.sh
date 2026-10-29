#!/bin/bash
# CC-const baseline: wait until run_revision.sh has finished (including timing), then run altmain + altrev, then alttiming with 1 process.
cd "$(dirname "$0")"
until grep -q "ALL DONE" results/sim_revision_log.txt 2>/dev/null; do sleep 20; done
L=results/sim_alt_log.txt
python3 tools/sim_campaign.py --campaign altmain >> $L 2>&1
python3 tools/sim_campaign.py --campaign altrev >> $L 2>&1
python3 tools/sim_campaign.py --campaign alttiming --workers 1 >> $L 2>&1
echo "$(date '+%F %T') ALL DONE" >> $L
