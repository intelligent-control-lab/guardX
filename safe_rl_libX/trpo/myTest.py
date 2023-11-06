from safe_rl_envs.envs.engine import Engine
from safe_rl_envs.envs.mjx_device import device_put
import numpy as np

import torch
import time
import mediapy as media
num_envs = 1
config = {'num_envs':num_envs}
env = Engine(config)

# layout, success = env._sample_layout(env.key)
obs = env.reset()
# env.render()
# import ipdb;ipdb.set_trace()
t = time.time()

images = []
model_path = '/home/yifan/guardX/guardX/safe_rl_libX/trpo_guardX/logs/Goal_Point_8Hazards_trpo_kl0.02_epochs10_step400000/Goal_Point_8Hazards_trpo_kl0.02_epochs10_step400000_s0/pyt_save/model.pt'
ac = torch.load(model_path)
total_reward = 0
print("start")
for i in range(1000):
    print(i)
    act = np.random.uniform(-1,1,(num_envs, env.action_space.shape[0]))
    # act = np.zeros(env.action_space.shape)
    
    act = torch.from_numpy(act).reshape(num_envs,-1)
    # act, v, logp, _, _ = ac.step(obs)
    # import ipdb;ipdb.set_trace()
    obs, reward, done, info = env.step(act)
    total_reward += reward
    env.render()
    # import ipdb;ipdb.set_trace()
    # print(env._data.xpos[1,14,:])
    # print(reward)
    # print(info['cost'])
    if done.cpu().numpy().any() > 0:
        print("#######")
        obs = env.reset_done()
        # print(total_reward)
        # total_reward = 0
        import ipdb;ipdb.set_trace()
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