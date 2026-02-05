from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="hanhainebula/reason-embed-data",
    repo_type="dataset",
    local_dir="/data4/jongho/lattice/reasonembed/data",
    local_dir_use_symlinks=False,
)