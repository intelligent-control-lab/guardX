from safe_rl_envs.envs.engine import Engine
from safe_rl_envs.envs.mjx_device import device_put
import numpy as np

import torch
import time
import mediapy as media
model_path = './model.pt'
ac = torch.load(model_path)
config = {'num_envs':1}
env = Engine(config)
obs = env.reset()

t = time.time()
print("start")
images = []
for i in range(1000):
# while 1:
    # act = np.random.uniform(-1.0,1.0,(env.action_space.shape))
    # act = 2 * (torch.rand(1, env.action_space.shape) - 0.5)
    # import ipdb;ipdb.set_trace()    
    act, v, logp, _, _ = ac.step(obs)
    # act = np.zeros(act.shape)
    # if i < 100:
    #     act = np.array([1.,0.])
    # else:
    #     act = np.array([0.,0.])
    # act = torch.from_numpy(act.reshape(1,2))
    obs, reward, done, info = env.step(act)
    goal_vec = env.data.xpos[0,1,:] - env.data.xpos[0,2,:]
    last_goal_vec = env.last_data.xpos[0,1,:] - env.last_data.xpos[0,2,:]
    goal_dist = np.sqrt(np.sum(np.square(goal_vec)))
    last_goal_dist = np.sqrt(np.sum(np.square(last_goal_vec)))
    print(reward)
    env.render()
    images.append(env.render())
print("finish ", time.time() - t)
print(obs.shape)
print(len(images), images[0].shape)
path = '/home/yifan/guardX/guardX/video.mp4'
media.write_video('/home/yifan/guardX/guardX/video.mp4', images, fps=60.0)

# import jax
# from jax import numpy as jp
# import mujoco
# from mujoco import mjx


# device_id = 6
# path = './safe_rl_envs/xmls/point.xml'
# mj_model = mujoco.MjModel.from_xml_path(path) # Load Mujoco model from xml
# mj_data = mujoco.MjData(mj_model) # Genearte Mujoco data from Mujoco model
# mjx_model = device_put(mj_model, device_id) # Convert Mujoco model to MJX model for device acceleration
# mjx_data = device_put(mj_data, device_id) # Convert Mujoco data to MJX data for device acceleration
# print(mjx_model.device())