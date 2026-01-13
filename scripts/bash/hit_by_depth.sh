# results/BRIGHT/biology/all_eval_sample_dicts-S=biology-TV=bottom-up-TPV=5-RInTP=-1-NumLC=10-PlTau=5-RCF=0-LlmApiB=vllm-Llm=Qwen3-4B-Instruct-2507-NumI=10-NumES=1000-MaxBS=2-S=flat_gate-FTT=True-FT=100-GBT=10-QeCP=biology_converted_qe_woplan.pkl

# results/BRIGHT/biology/all_eval_sample_dicts-S=biology-TV=bottom-up-TPV=5-RInTP=-1-NumLC=10-PlTau=5.0-RCF=0.5-LlmApiB=vllm-Llm=__data2__da02__models__Qwen3-4B-Instruct-2507-NumI=20-NumES=1000-MaxBS=2-S=mse_calib_leaf_think-1.pkl


python scripts/ancestor_hit_by_depth.py \
    --tree_pkl trees/BRIGHT/biology/tree-bottom-up.pkl \
    --eval_samples_pkl results/BRIGHT/biology/all_eval_sample_dicts-S=biology-TV=bottom-up-TPV=5-RInTP=-1-NumLC=10-PlTau=5.0-RCF=0.5-LlmApiB=vllm-Llm=Qwen3-4B-Instruct-2507-NumI=20-NumES=1000-MaxBS=2-S=flat_gate-FTT=True-FT=100-GBT=10.pkl \
    --topk 10 \
    --out_csv results/BRIGHT/biology/ancestor_hit_by_depth_.csv
