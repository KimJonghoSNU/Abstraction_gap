#!/bin/bash
# Script to start vLLM servers in data parallel or tensor parallel mode
#
# Usage:
#   ./start_vllm_cluster_branch_policy_dpo.sh [MODEL] [MODE] [TP_PER_SERVER]
#
# Examples:
#   ./start_vllm_cluster_branch_policy_dpo.sh
#   ./start_vllm_cluster_branch_policy_dpo.sh "/data2/Qwen3-30B-A3B-Instruct-2507" data 2
#   ./start_vllm_cluster_branch_policy_dpo.sh "/data2/Qwen3-30B-A3B-Instruct-2507" tensor 4
#
# Modes:
#   data   - Run multiple servers; each server may use multiple GPUs via tensor parallel
#   tensor - Run a single server with model split across GPUs

set -e

# Resolve script and repository paths for robust relative-path handling.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Configuration
# Intent: default to your DPO output path so you can launch without retyping a long checkpoint path.
MODEL="${1:-${REPO_ROOT}/results/train/branch_policy_dpo_1500_sample_20260227_095158/merged}"
MODE="${2:-data}"  # "data" or "tensor"
TP_PER_SERVER_ARG="${3:-1}" # auto
BASE_PORT=8000
GPU_MEM_UTIL=0.95
# Specify GPU IDs to use
GPU_IDS=(4 5 6 7)
MAX_MODEL_LEN=128000

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

_join_by_comma() {
    local IFS=","
    echo "$*"
}

_is_positive_integer() {
    [[ "$1" =~ ^[0-9]+$ ]] && [ "$1" -ge 1 ]
}

_auto_tp_per_server() {
    local model_name
    model_name="$(echo "$1" | tr '[:upper:]' '[:lower:]')"
    # Intent: large models (e.g., 30B/70B) default to >=2 GPUs per server to avoid OOM in data mode.
    if echo "$model_name" | grep -Eq "(30b|32b|34b|40b|70b|72b|a3b|a14b)"; then
        echo 2
    else
        echo 1
    fi
}

_is_large_model() {
    local model_name
    model_name="$(echo "$1" | tr '[:upper:]' '[:lower:]')"
    if echo "$model_name" | grep -Eq "(30b|32b|34b|40b|70b|72b|a3b|a14b)"; then
        return 0
    fi
    return 1
}

# Resolve relative model paths against cwd first, then repository root.
if [[ "$MODEL" != /* ]]; then
    if [ -d "$MODEL" ]; then
        MODEL="$(cd "$MODEL" && pwd)"
    elif [ -d "${REPO_ROOT}/${MODEL}" ]; then
        MODEL="${REPO_ROOT}/${MODEL}"
    fi
fi

# Guardrail: adapter-only checkpoints cannot be served as a base model path directly.
if [ -d "$MODEL" ] && [ -f "$MODEL/adapter_model.safetensors" ] && [ ! -f "$MODEL/config.json" ]; then
    echo -e "${RED}Error: ${MODEL} looks like a PEFT adapter-only checkpoint.${NC}"
    echo -e "${YELLOW}Action:${NC} merge adapter -> base model first, then pass merged model path as MODEL."
    exit 1
fi

# Validate mode
if [[ "$MODE" != "data" && "$MODE" != "tensor" ]]; then
    echo -e "${RED}Error: MODE must be 'data' or 'tensor', got: $MODE${NC}"
    echo "Usage: $0 [MODEL] [MODE] [TP_PER_SERVER]"
    exit 1
fi

TOTAL_GPUS=${#GPU_IDS[@]}
if [ "$TOTAL_GPUS" -lt 1 ]; then
    echo -e "${RED}Error: GPU_IDS is empty. Set at least one GPU id.${NC}"
    exit 1
fi

# Create logs directory
mkdir -p logs
rm -f logs/cluster_mode.txt logs/cluster_num_servers.txt logs/cluster_ports.txt logs/cluster_gpus_per_server.txt logs/vllm_base_url.txt

# Save mode to file for stop/check scripts
echo "$MODE" > logs/cluster_mode.txt

if [[ "$MODE" == "tensor" ]]; then
    # ============================================
    # TENSOR PARALLEL MODE
    # ============================================
    if [[ "$TP_PER_SERVER_ARG" == "auto" ]]; then
        TP_SIZE=$TOTAL_GPUS
    else
        TP_SIZE=$TP_PER_SERVER_ARG
    fi
    if ! _is_positive_integer "$TP_SIZE"; then
        echo -e "${RED}Error: TP_PER_SERVER must be a positive integer or 'auto', got: ${TP_PER_SERVER_ARG}${NC}"
        exit 1
    fi
    if [ "$TP_SIZE" -gt "$TOTAL_GPUS" ]; then
        echo -e "${RED}Error: tensor parallel size (${TP_SIZE}) exceeds available GPUs (${TOTAL_GPUS})${NC}"
        exit 1
    fi

    TENSOR_GPU_IDS=("${GPU_IDS[@]:0:$TP_SIZE}")
    GPU_LIST=$(_join_by_comma "${TENSOR_GPU_IDS[@]}")
    LOG_FILE="logs/vllm_tensor_port${BASE_PORT}.log"

    echo -e "${BLUE}================================================${NC}"
    echo -e "${BLUE}Starting vLLM Tensor Parallel Server${NC}"
    echo -e "${BLUE}================================================${NC}"
    echo -e "Model: ${MODEL}"
    echo -e "GPUs: ${TENSOR_GPU_IDS[*]}"
    echo -e "Tensor Parallel Size: ${TP_SIZE}"
    echo -e "Port: ${BASE_PORT}"
    echo -e "${BLUE}================================================${NC}\n"

    echo -e "${GREEN}Starting tensor parallel server on GPUs: $GPU_LIST${NC}"

    CUDA_VISIBLE_DEVICES=$GPU_LIST nohup python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --port "$BASE_PORT" \
        --host 0.0.0.0 \
        --tensor-parallel-size "$TP_SIZE" \
        --gpu-memory-utilization "$GPU_MEM_UTIL" \
        --disable-log-requests \
        --max-model-len "$MAX_MODEL_LEN" \
        --trust-remote-code \
        > "$LOG_FILE" 2>&1 &

    PID=$!
    echo "$PID" > logs/vllm_tensor.pid
    echo 1 > logs/cluster_num_servers.txt
    echo "$BASE_PORT" > logs/cluster_ports.txt
    echo "$TP_SIZE" > logs/cluster_gpus_per_server.txt
    echo "http://localhost:${BASE_PORT}/v1" > logs/vllm_base_url.txt
    echo -e "${GREEN}Server started with PID: ${PID}${NC}"
    echo -e "Log file: ${LOG_FILE}\n"

    echo -e "\nWaiting for server to initialize..."
    sleep 15

    echo -e "\n${BLUE}Checking server health...${NC}"
    if curl -s "http://localhost:${BASE_PORT}/health" > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} Tensor parallel server (port ${BASE_PORT}): HEALTHY"
    else
        echo -e "${YELLOW}⚠${NC} Tensor parallel server (port ${BASE_PORT}): Still initializing..."
    fi

    echo -e "\n${BLUE}================================================${NC}"
    echo -e "${GREEN}Tensor Parallel Server Ready!${NC}"
    echo -e "${BLUE}================================================${NC}"
    echo -e "\nUsage in Python:"
    echo -e "  export VLLM_BASE_URL=\"http://localhost:${BASE_PORT}/v1\""
    echo -e "  # or read from logs/vllm_base_url.txt"
    echo -e "\nNote: Tensor mode maximizes model fit but provides lower throughput than multi-server data mode.\n"

else
    # ============================================
    # DATA PARALLEL MODE (each worker may use TP>1)
    # ============================================
    if [[ "$TP_PER_SERVER_ARG" == "auto" ]]; then
        TP_PER_SERVER=$(_auto_tp_per_server "$MODEL")
    else
        TP_PER_SERVER=$TP_PER_SERVER_ARG
    fi
    if ! _is_positive_integer "$TP_PER_SERVER"; then
        echo -e "${RED}Error: TP_PER_SERVER must be a positive integer or 'auto', got: ${TP_PER_SERVER_ARG}${NC}"
        exit 1
    fi
    if [ "$TP_PER_SERVER" -gt "$TOTAL_GPUS" ]; then
        echo -e "${RED}Error: TP_PER_SERVER (${TP_PER_SERVER}) exceeds available GPUs (${TOTAL_GPUS})${NC}"
        exit 1
    fi
    if [ $((TOTAL_GPUS % TP_PER_SERVER)) -ne 0 ]; then
        echo -e "${RED}Error: total GPUs (${TOTAL_GPUS}) must be divisible by TP_PER_SERVER (${TP_PER_SERVER}).${NC}"
        echo -e "${YELLOW}Tip:${NC} adjust GPU_IDS or pass a different TP_PER_SERVER."
        exit 1
    fi

    DATA_GPU_MEM_UTIL="${VLLM_DATA_GPU_MEM_UTIL:-$GPU_MEM_UTIL}"
    DATA_MAX_MODEL_LEN="${VLLM_DATA_MAX_MODEL_LEN:-$MAX_MODEL_LEN}"
    DATA_MAX_NUM_SEQS="${VLLM_DATA_MAX_NUM_SEQS:-128}"
    ENFORCE_EAGER_FLAG=""
    if _is_large_model "$MODEL" && [ "$TP_PER_SERVER" -ge 2 ]; then
        # Intent: for large-model data mode, use conservative defaults to avoid CUDA-graph startup OOM.
        if [ -z "${VLLM_DATA_GPU_MEM_UTIL+x}" ]; then
            DATA_GPU_MEM_UTIL="0.85"
        fi
        if [ -z "${VLLM_DATA_MAX_MODEL_LEN+x}" ]; then
            DATA_MAX_MODEL_LEN="32768"
        fi
        if [ -z "${VLLM_DATA_MAX_NUM_SEQS+x}" ]; then
            DATA_MAX_NUM_SEQS="16"
        fi
        # Intent: for large-model data mode, force eager to reduce CUDA-graph capture OOM risk.
        # ENFORCE_EAGER_FLAG="--enforce-eager"
    fi
    if ! _is_positive_integer "$DATA_MAX_MODEL_LEN"; then
        echo -e "${RED}Error: VLLM_DATA_MAX_MODEL_LEN must be a positive integer, got: ${DATA_MAX_MODEL_LEN}${NC}"
        exit 1
    fi
    if ! _is_positive_integer "$DATA_MAX_NUM_SEQS"; then
        echo -e "${RED}Error: VLLM_DATA_MAX_NUM_SEQS must be a positive integer, got: ${DATA_MAX_NUM_SEQS}${NC}"
        exit 1
    fi

    NUM_SERVERS=$((TOTAL_GPUS / TP_PER_SERVER))
    PORTS=()
    BASE_URLS=()

    echo -e "${BLUE}================================================${NC}"
    echo -e "${BLUE}Starting vLLM Data Parallel Cluster${NC}"
    echo -e "${BLUE}================================================${NC}"
    echo -e "Model: ${MODEL}"
    echo -e "GPUs: ${GPU_IDS[*]}"
    echo -e "GPUs per server (TP): ${TP_PER_SERVER}"
    echo -e "Data mode max model len: ${DATA_MAX_MODEL_LEN}"
    echo -e "Data mode max num seqs: ${DATA_MAX_NUM_SEQS}"
    echo -e "Data mode gpu memory util: ${DATA_GPU_MEM_UTIL}"
    if [ -n "$ENFORCE_EAGER_FLAG" ]; then
        echo -e "Data mode enforce eager: enabled (policy)"
    else
        echo -e "Data mode enforce eager: disabled"
    fi
    echo -e "Server count: ${NUM_SERVERS}"
    echo -e "Ports: ${BASE_PORT}-$((BASE_PORT + NUM_SERVERS - 1))"
    echo -e "${BLUE}================================================${NC}\n"

    for i in $(seq 0 $((NUM_SERVERS - 1))); do
        START_IDX=$((i * TP_PER_SERVER))
        WORKER_GPU_IDS=("${GPU_IDS[@]:$START_IDX:$TP_PER_SERVER}")
        GPU_LIST=$(_join_by_comma "${WORKER_GPU_IDS[@]}")
        PORT=$((BASE_PORT + i))
        LOG_FILE="logs/vllm_gpu${i}_port${PORT}.log"
        PORTS+=("$PORT")
        BASE_URLS+=("http://localhost:${PORT}/v1")

        echo -e "${GREEN}[Worker ${i}]${NC} Starting vLLM server on GPUs ${WORKER_GPU_IDS[*]} (port ${PORT}, tp=${TP_PER_SERVER})..."

        CUDA_VISIBLE_DEVICES=$GPU_LIST nohup python -m vllm.entrypoints.openai.api_server \
            --model "$MODEL" \
            --port "$PORT" \
            --host 0.0.0.0 \
            --tensor-parallel-size "$TP_PER_SERVER" \
            --gpu-memory-utilization "$DATA_GPU_MEM_UTIL" \
            --disable-log-requests \
            --max-model-len "$DATA_MAX_MODEL_LEN" \
            --trust-remote-code \
            --max_num_seqs "$DATA_MAX_NUM_SEQS" \
            $ENFORCE_EAGER_FLAG \
            > "$LOG_FILE" 2>&1 &

        PID=$!
        echo "$PID" > "logs/vllm_gpu${i}.pid"
        echo -e "${GREEN}[Worker ${i}]${NC} Server started with PID: ${PID}"
        echo -e "             Log file: ${LOG_FILE}\n"
        sleep 3
    done

    PORT_LIST=$(_join_by_comma "${PORTS[@]}")
    BASE_URL_LIST=$(_join_by_comma "${BASE_URLS[@]}")
    echo "$NUM_SERVERS" > logs/cluster_num_servers.txt
    echo "$PORT_LIST" > logs/cluster_ports.txt
    echo "$TP_PER_SERVER" > logs/cluster_gpus_per_server.txt
    echo "$BASE_URL_LIST" > logs/vllm_base_url.txt

    echo -e "\n${BLUE}================================================${NC}"
    echo -e "${GREEN}All vLLM servers started successfully!${NC}"
    echo -e "${BLUE}================================================${NC}"
    echo -e "URLs:"
    for i in $(seq 0 $((NUM_SERVERS - 1))); do
        PORT=$((BASE_PORT + i))
        echo -e "  Worker ${i}: http://localhost:${PORT}/v1"
    done

    echo -e "\nWaiting for servers to fully initialize..."
    sleep 10

    echo -e "\n${BLUE}Checking server health...${NC}"
    for i in $(seq 0 $((NUM_SERVERS - 1))); do
        PORT=$((BASE_PORT + i))
        if curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
            echo -e "${GREEN}✓${NC} Worker ${i} (port ${PORT}): HEALTHY"
        else
            echo -e "${YELLOW}⚠${NC} Worker ${i} (port ${PORT}): Still initializing..."
        fi
    done

    echo -e "\n${BLUE}================================================${NC}"
    echo -e "${GREEN}Data Parallel Cluster Ready!${NC}"
    echo -e "${BLUE}================================================${NC}"
    echo -e "\nUsage in Python:"
    echo -e "  export VLLM_BASE_URL=\"${BASE_URL_LIST}\""
    echo -e "  # or: export VLLM_BASE_URL=\"\$(cat logs/vllm_base_url.txt)\""
fi

echo -e "\nTo stop: ./stop_vllm_cluster.sh"
echo -e "To check: ./check_vllm_cluster.sh"
echo -e "View logs: tail -f logs/vllm*.log\n"
