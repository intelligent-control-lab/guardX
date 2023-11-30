from safe_rl_envs.envs.engine import Engine
from safe_rl_envs.envs.mjx_device import device_put
import numpy as np

import torch
import time
import mediapy as media
env_num = 1
config = {'env_num':env_num,
          'num_steps':200,
          '_seed':0}

env = Engine(config)

# layout, success = env._sample_layout(env.key)

# layout, success = env._sample_layout(env.key)
obs = env.reset()
# env.render()
# import ipdb;ipdb.set_trace()# env.render()
# import ipdb;ipdb.set_trace()
t = time.time()


images = []
model_path = '/home/yifan/guardX/guardX/safe_rl_libX/trpo_guardX/logs/Goal_Point_8Hazards_trpo_kl0.02_epochs10_step400000/Goal_Point_8Hazards_trpo_kl0.02_epochs10_step400000_s0/pyt_save/model.pt'
# ac = torch.load(model_path)
rs = np.random.RandomState(0)
for i in range(2000):
    
    act = rs.uniform(-1,1,(env_num, env.action_space.shape[0]))
    t = i % 200
    # if t < 50:
    #     act = np.array([0.0, 1.0, 0.0])
    # elif t < 100:
    #     act = np.array([0.0, 0.0, 0.0])
    # elif t < 150:
    #     # import ipdb;ipdb.set_trace()
    #     act = np.array([0.0, -1.0, 0.0])
    # else:
    #     act = np.array([0.0, 0.0, 0.0])
    # import ipdb;ipdb.set_trace()
    print(i,env._data.xmat[0,1,0,:2])
    # act = np.zeros(env.action_space.shape)
    act = torch.from_numpy(act).reshape(env_num,-1)
    # act, v, logp, _, _ = ac.step(obs)
    # import ipdb;ipdb.set_trace()
    # import ipdb;ipdb.set_trace()
    obs, reward, done, info = env.step(act)
    total_reward += reward
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