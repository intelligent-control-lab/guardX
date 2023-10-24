from safe_rl_envs.engine import Engine
from safe_rl_envs.utils.mjx_device import device_put

env = Engine()
obs = env.reset()

#test
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