#!/bin/bash


export PYTHONPATH=$(pwd)
cuda_devices="4,5"

# model_name_or_path=../../models/Qwen2.5-0.5B-Instruct
# model_name_or_path=../../models/DeepSeek-R1-Distill-Qwen-14B
# model_name_or_path=/data2/Qwen3-30B-A3B-Instruct-2507
model_name_or_path=/data2/da02/models/Qwen3-4B-Instruct-2507
embedding_name_or_path=/data4/jaeyoung/models/Diver-Retriever-4B

max_dlen=512
max_tokens=16000
# prompt_modes=("stepback_json") # "thinkqe" 
prompt_modes=("inf-x-retriever")
# prompt_modes=("thinkqe")
n_round=5
keep_doc_nums=5
# categories=("economics") #"biology" 
categories=(
    "biology"
    # "economics"
    # "pony"
    # "leetcode"
    # "earth_science"
    "psychology"
    # "robotics"
    # "stackoverflow"
    # "sustainable_living"
    # "aops"
    # "theoremqa_questions"
    # "theoremqa_theorems"
)

# for prompt_mode in ${prompt_modes[@]}; do
#     echo "Generating queries with prompt mode: ${prompt_mode}"
#     # for category in  economics; do # biology 
#     for category in ${categories[@]}; do
#         CUDA_VISIBLE_DEVICES=${cuda_devices} \
#         python -u -m scripts.run \
#             --task generate_query \
#             --batch_size 8 \
#             --llm_dtype bfloat16 \
#             --max_dlen ${max_dlen} \
#             --max_tokens ${max_tokens} \
#             --prompt_mode ${prompt_mode} \
#             --embedding_name_or_path ${embedding_name_or_path} \
#             --model_name_or_path ${model_name_or_path} \
#             --gpu_memory_utilization 0.8 \
#             --qa_path ../data/QA_Datasets/bright/${category}.json \
#             --corpus_path ./data/bright/${category}/corpus.jsonl \
#             --cache_dir ./data/bright/cache/cache_diver-retriever \
#             --n_round ${n_round} \
#             --keep_doc_nums ${keep_doc_nums} \
#             --vllm \
#             --agent_mode route_exec
#     done
# done


for prompt_mode in ${prompt_modes[@]}; do
    echo "Generating queries with prompt mode: ${prompt_mode}"
    # for category in  economics; do # biology 
    for category in ${categories[@]}; do
        CUDA_VISIBLE_DEVICES=${cuda_devices} \
        python -u -m scripts.run \
            --task generate_query \
            --batch_size 8 \
            --llm_dtype bfloat16 \
            --max_dlen ${max_dlen} \
            --max_tokens ${max_tokens} \
            --prompt_mode ${prompt_mode} \
            --embedding_name_or_path ${embedding_name_or_path} \
            --model_name_or_path ${model_name_or_path} \
            --gpu_memory_utilization 0.8 \
            --qa_path ../data/QA_Datasets/bright/${category}.json \
            --corpus_path ./data/bright/${category}/corpus.jsonl \
            --cache_dir ./data/bright/cache/cache_diver-retriever \
            --n_round ${n_round} \
            --keep_doc_nums ${keep_doc_nums} \
            --vllm
    done
done


# model_name_or_path=/data2/Qwen3-30B-A3B-Instruct-2507
# n_round=5
# for prompt_mode in ${prompt_modes[@]}; do
#     echo "Generating queries with prompt mode: ${prompt_mode}"
#     # for category in  economics; do # biology 
#     for category in ${categories[@]}; do
#         CUDA_VISIBLE_DEVICES=${cuda_devices} \
#         python -u -m scripts.run \
#             --task generate_query \
#             --batch_size 8 \
#             --llm_dtype bfloat16 \
#             --max_dlen ${max_dlen} \
#             --max_tokens ${max_tokens} \
#             --prompt_mode ${prompt_mode} \
#             --embedding_name_or_path ${embedding_name_or_path} \
#             --model_name_or_path ${model_name_or_path} \
#             --gpu_memory_utilization 0.8 \
#             --qa_path ../data/QA_Datasets/bright/${category}.json \
#             --corpus_path ./data/bright/${category}/corpus.jsonl \
#             --cache_dir ./data/bright/cache/cache_diver-retriever \
#             --n_round ${n_round} \
#             --keep_doc_nums ${keep_doc_nums} \
#             --vllm
#     done
# done