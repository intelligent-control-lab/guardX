# guardX
In the enhanced GUARD environment, RL training benefits from the power of GPU parallelization, enabling the training of RL agents in a matter of minutes.

# Installation
conda create -n guardX

pip install -r requirements.txt

pip install -U "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

cd safe_rl_envs
pip install -e.