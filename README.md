# guardX
In the enhanced GUARD environment, RL training benefits from the power of GPU parallelization, enabling the training of RL agents in a matter of minutes.

# Installation
clone this repo

conda create -n guardX

pip install -r requirements.txt

pip install -U "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

cd safe_rl_envs
pip install -e.

# Following two steps are required for environments of IsaacGym

# Install Isaac Gym
cd a path for Issac Gym(can be different from the path of guardX)
Download and install Isaac Gym Preview 4 from https://developer.nvidia.com/isaac-gym
`cd isaacgym/python && pip install -e .`
Try running an example `cd examples && python3 1080_balls_of_solitude.py`

# Install IsaacGymEnvs
cd back to the guardX repo
cd IsaacGymEnvs
pip install -e.

# Example for IsaacGym environments KukaTwoArms
cd safe_rl_libX/trpo
python trpo.py --task KukaTwoArms --env_num 2000