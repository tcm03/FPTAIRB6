from huggingface_hub import snapshot_download
import os

# Define the model ID and target directory
# model_id = 'Qwen/CodeQwen1.5-7B'
# target_dir = "checkpoints/CodeQwen1.5-7B"
# model_id = "Qwen/Qwen2.5-Coder-14B"
# target_dir = "checkpoints/Qwen2.5-Coder-14B"
# model_id = "Qwen/Qwen2.5-Coder-7B"
# target_dir = "checkpoints/Qwen2.5-Coder-7B"
# model_id = "Qwen/Qwen2.5-Coder-7B-Instruct"
# target_dir = "checkpoints/Qwen2.5-Coder-7B-Instruct"
model_id = "Qwen/Qwen2.5-Coder-14B-Instruct"
target_dir = "checkpoints/Qwen2.5-Coder-14B-Instruct"
if not os.path.exists(target_dir):
    os.makedirs(target_dir)


# Download the model
snapshot_download(repo_id=model_id, local_dir=target_dir)