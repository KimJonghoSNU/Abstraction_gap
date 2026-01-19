#!/bin/bash

# # Create log file with timestamp
# mkdir -p ../logs
# LOG_FILE="../logs/run_all_experiments_$(date '+%Y_%m_%d').log"

# # Function to log with timestamp
# log() {
#     echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
# }

# log "Starting run_all_experiments.sh script"

SCRIPTS=(
    "bash/run_exp1_qe_iter.sh"
    "bash/run_exp2_preflat_branch_noiter.sh"
    "bash/run_exp2_preflat_leaf_noiter.sh"
    "bash/run_exp2_preflat_all_noiter.sh"
    "bash/run_exp3_preflat_branch_iter.sh"
    # "bash/run_leaf_rank.sh"
)

for script in "${SCRIPTS[@]}"; do
echo "Running $script ..."
bash "$script"
done
