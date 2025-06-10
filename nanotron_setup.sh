sudo snap install astral-uv --classic
uv venv nanotron --python 3.11 && source nanotron/bin/activate && uv pip install --upgrade pip
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
uv pip install -e .
uv pip install datasets transformers datatrove[io] numba wandb
uv pip install ninja triton "flash-attn>=2.5.0" --no-build-isolation
huggingface-cli login
wandb login

torchrun --nproc_per_node=4 /datadrive/pavan/CurLL/nanotron/run_train.py --config-file /datadrive/pavan/CurLL/nanotron/examples/config_tinystories_33M.yaml
