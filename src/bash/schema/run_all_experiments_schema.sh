#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$SCRIPT_DIR/run_exp1_qe_iter_schema.sh"
bash "$SCRIPT_DIR/run_exp2_preflat_all_noiter_schema.sh"
bash "$SCRIPT_DIR/run_exp2_preflat_leaf_noiter_schema.sh"
bash "$SCRIPT_DIR/run_exp3_preflat_branch_iter_schema.sh"
bash "$SCRIPT_DIR/run_exp3_preflat_leaf_iter_schema.sh"
