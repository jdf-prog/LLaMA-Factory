uv sync --extra torch --extra metrics
uv pip install deepspeed==0.16.9
uv pip install flash-attn==2.7.2.post1 --no-build-isolation
uv pip install wandb