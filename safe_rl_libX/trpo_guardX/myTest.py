from safe_rl_envs.envs.engine import Engine
from safe_rl_envs.envs.mjx_device import device_put
import numpy as np

import torch
import time
import mediapy as media
from jax import numpy as jp
num_envs = 2048
config = {'num_envs':num_envs}
env = Engine(config)
obs = env.reset()
env.done = jp.ones(num_envs)
env.reset_done()
t = time.time()
print("start")
images = []
model_path = '/home/yifan/guardX/guardX/safe_rl_libX/trpo_guardX/logs/Goal_Point_8Hazards_trpo_kl0.02_epochs10_step400000/Goal_Point_8Hazards_trpo_kl0.02_epochs10_step400000_s0/pyt_save/model.pt'
ac = torch.load(model_path)
reward_list = []

for i in range(1000):
    # print(i)
    act = np.random.uniform(-1,1,(num_envs, env.action_space.shape[0]))
    # act = np.zeros(env.action_space.shape)
    act = torch.from_numpy(act).reshape(num_envs,-1)
    # act, v, logp, _, _ = ac.step(obs)
    obs, reward, done, info = env.step(act)
    # reward_list.append(reward)
    
    
    # env.render()
    if done.cpu().numpy().any() >  0:
        # reward_list = torch.tensor(reward_list)
        # print(torch.sum(reward_list))
        # print("###############  ", np.sum(done.cpu().numpy()))
        obs = env.reset_done()
        # import ipdb;ipdb.set_trace()
        # reward_list = []
    # if i%100 == 0:
    #     obs = env.reset()
    # images.append(env.render())
print("finish ", time.time() - t)
# print(obs.shape)
# print(len(images), images[0].shape)
# path = '/home/yifan/guardX/guardX/video.mp4'
# media.write_video('/home/yifan/guardX/guardX/video.mp4', images, fps=60.0)

# import jax
# from jax import numpy as jp
# import mujoco
# from mujoco import mjx
# # path = 'safe_rl_envs/safe_rl_envs/xmls/barkour_v0/assets/barkour_v0_mjx.xml'
# path = 'safe_rl_envs/safe_rl_envs/xmls/barkour_v0/assets/barkour.xml'
# mj_model = mujoco.MjModel.from_xml_path(path)
# import ipdb;ipdb.set_trace()