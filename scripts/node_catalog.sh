categories=("biology" "psychology")
# categories=("leetcode" "earth_science")
# "biology" "earth_science" "psychology" "leetcode"
# categories=("economics" "robotics" "stackoverflow" "sustainable_living" "pony" "aops" "theoremqa_questions" "theoremqa_theorems")

export CUDA_VISIBLE_DEVICES=3

for category in "${categories[@]}"; do
  python scripts/export_node_catalog.py \
    --tree_pkl "trees/BRIGHT/${category}/tree-bottom-up.pkl" \
    --out_jsonl "trees/BRIGHT/${category}/node_catalog.jsonl"

  python scripts/embed_node_catalog.py \
    --node_catalog_jsonl "trees/BRIGHT/${category}/node_catalog.jsonl" \
    --model_path /data4/jaeyoung/models/Diver-Retriever-4B \
    --out_npy "trees/BRIGHT/${category}/node_embs.diver.npy"
done

# bash python scripts/export_node_catalog.py \
#   --tree_pkl trees/BRIGHT/biology/tree-bottom-up.pkl \
#   --out_jsonl trees/BRIGHT/biology/node_catalog.jsonl

# bash python scripts/embed_node_catalog.py \
#   --node_catalog_jsonl trees/BRIGHT/biology/node_catalog.jsonl \
#   --model_path /data4/jaeyoung/models/Diver-Retriever-4B \
#   --out_npy trees/BRIGHT/biology/node_embs.diver.npy