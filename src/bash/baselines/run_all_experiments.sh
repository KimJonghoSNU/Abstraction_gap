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
    # "bash/baselines/run_baseline1_tree_only.sh"
    "bash/baselines/run_baseline1_tree_iter_rewrite.sh"
    # "bash/baselines/run_baseline2_qe_noctx.sh"
    "bash/baselines/run_baseline2_qe_agent_executor.sh"
    "bash/baselines/run_baseline3_leaf_only_loop.sh"
)

for script in "${SCRIPTS[@]}"; do
    echo "Running $script ..."
    bash "$script"
done
