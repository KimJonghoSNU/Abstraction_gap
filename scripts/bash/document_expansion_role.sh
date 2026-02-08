export CUDA_VISIBLE_DEVICES=0,1

subsets=(
    biology
    earth_science
    economics
    psychology
    robotics
    sustainable_living
    pony
    stackoverflow
)

for subset in "${subsets[@]}"; do
    python scripts/document_expansion_role.py \
        --dataset BRIGHT \
        --subset $subset \
        --llm  /data2/Qwen3-30B-A3B-Instruct-2507 \
        --hybrid_depths 1,2,3 \
        --batch_size 24 \
        --max_tokens 128 \
        --prompt_token_margin 32 \
        --max_nonleaf_desc_words 4096 \
        --max_desc_words 4096
done
