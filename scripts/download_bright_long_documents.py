from huggingface_hub import snapshot_download


def main() -> None:
    # Intent: fetch both long_documents and documents so subset fallback works without global cache dependency.
    snapshot_download(
        repo_id="xlangai/BRIGHT",
        repo_type="dataset",
        allow_patterns=["long_documents/*", "documents/*", "README.md"],
        local_dir="data/BRIGHT",
    )
    print("Downloaded BRIGHT long_documents and documents to data/BRIGHT")


if __name__ == "__main__":
    main()
