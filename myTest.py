from safe_rl_envs.envs.engine import Engine
from safe_rl_envs.envs.mjx_device import device_put
import numpy as np

import torch
import time
import mediapy as media
config = {'num_envs':1}
env = Engine(config)
obs = env.reset()

t = time.time()
print("start")
images = []
model_path = '/home/yifan/guardX/guardX/safe_rl_libX/trpo_guardX/logs/Goal_Point_8Hazards_trpo_kl0.02_epochs10_step400000/Goal_Point_8Hazards_trpo_kl0.02_epochs10_step400000_s0/pyt_save/model.pt'
ac = torch.load(model_path)
for i in range(1000):
    # act = np.random.uniform(-1,1,env.action_space.shape)
    # act = torch.from_numpy(act).reshape(1,-1)
    act, v, logp, _, _ = ac.step(obs)
    obs, reward, done, info = env.step(act)
    
    images.append(env.render())
print("finish ", time.time() - t)
print(obs.shape)
# print(len(images), images[0].shape)
# path = '/home/yifan/guardX/guardX/video.mp4'
# media.write_video('/home/yifan/guardX/guardX/video.mp4', images, fps=60.0)

# import jax
# from jax import numpy as jp
# import mujoco
# from mujoco import mjx
