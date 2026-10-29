#!/bin/bash
cd "$(dirname "$0")"
LOG=results/sim_campaigns_log.txt
for c in main robust ablation; do
  echo "$(date '+%F %T') START $c" >> $LOG
  python3 tools/sim_campaign.py --campaign $c >> $LOG 2>&1
done
echo "$(date '+%F %T') ALL DONE" >> $LOG
