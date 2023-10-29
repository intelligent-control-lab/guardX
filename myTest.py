import jax
import os
from jax import numpy as jp
# jax.config.update("jax_default_device", jax.devices()[6])
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.10'
# ja = jp.ones(1)
# key = jax.random.PRNGKey(0)
# import ipdb;ipdb.set_trace()

from safe_rl_envs.envs.engine import Engine
from safe_rl_envs.envs.mjx_device import device_put
import numpy as np

import torch
import time
import mediapy as media

config = {
            'num_envs':10,
            'device_id':7
        }
# env = Engine(config)
# obs = env.reset()

t = time.time()
print("start")
images = []
for i in range(600):
# while 1:
# test feihan push
    act = np.random.uniform(-1,1,(env.action_space.shape))
    # act = 2 * (torch.rand(env.action_space.shape) - 0.5)
    act = torch.from_numpy(act)
    obs, reward, done, info = env.step(act)
    # images.append(env.render())
print("finish ", time.time() - t)
print(obs.shape)
# print(len(images), images[0].shape)
# path = '/home/yifan/guardX/guardX/video.mp4'
# media.write_video('/home/yifan/guardX/guardX/video.mp4', images, fps=60.0)

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