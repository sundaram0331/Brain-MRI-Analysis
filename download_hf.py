from huggingface_hub import snapshot_download

print("Downloading dataset with high concurrency...")
snapshot_download(
    repo_id='BJyotibrat/Masoud-Nickparvar-Brain-Tumor-MRI-Dataset',
    repo_type='dataset',
    local_dir='dataset',
    max_workers=20
)
print("Done!")
