from huggingface_hub import snapshot_download
import os

# Define the model ID and target directory
model_id = 'Qwen/CodeQwen1.5-7B'
target_dir = "checkpoints/CodeQwen1.5-7B"
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# Download the model
snapshot_download(repo_id=model_id, local_dir=target_dir)