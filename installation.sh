conda create -n nanotron python=3.11
conda activate nanotron
conda install -c nvidia cuda-nvcc
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e .
pip install datasets transformers datatrove[io] numba wandb
pip install ninja triton "flash-attn>=2.5.0" --no-build-isolation
huggingface-cli login
wandb login
git-lfs --version