from huggingface_hub import snapshot_download

REPO_ID = "Soughing/MLRA"

def download_dataset():
    local_root = snapshot_download(
        repo_id=REPO_ID,
        repo_type="model",
        local_dir=".",
        local_dir_use_symlinks=False,
        allow_patterns=["eval_datasets/*"],
    )

if __name__ == "__main__":
    download_dataset()
