export CUDA_VISIBLE_DEVICES=6,7

python scripts/embed_node_catalog.py \
  --node_catalog_jsonl trees/BRIGHT/biology/node_catalog.jsonl \
  --model_path /data4/jaeyoung/models/Diver-Retriever-4B \
  --out_npy trees/BRIGHT/biology/node_embs.diver.npy