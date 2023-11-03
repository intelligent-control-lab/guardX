#!/usr/bin/env python
import gym
from gym.vector import utils
import jax
from jax import numpy as jp
import mujoco
import mujoco.viewer
from mujoco import mjx
import torch
from torch.utils import dlpack as torch_dlpack
from collections import OrderedDict
import functools
from typing import Any, Dict, Union, Tuple
import warnings
import jax
from jax import dlpack as jax_dlpack
from .engine_utils import *
from .mjx_device import device_put
import numpy as np
from copy import deepcopy
from safe_rl_envs.envs.world import World, Robot
import os
import safe_rl_envs

# Default location to look for /xmls folder:
BASE_DIR = os.path.dirname(safe_rl_envs.__file__)

# Distinct colors for different types of objects.
# For now this is mostly used for visualization.
# This also affects the vision observation, so if training from pixels.
COLOR_BOX = np.array([1, 1, 0, 1])
COLOR_BUTTON = np.array([1, .5, 0, 1])
COLOR_GOAL = np.array([0, 1, 0, 1])
COLOR_VASE = np.array([0, 1, 1, 1])
COLOR_HAZARD = np.array([0, 0, 1, 1])
COLOR_HAZARD3D = np.array([0, 0, 1, 1])
COLOR_PILLAR = np.array([.5, .5, 1, 1])
COLOR_WALL = np.array([.5, .5, .5, 1])
COLOR_GREMLIN = np.array([0.5, 0, 1, 1])
COLOR_CIRCLE = np.array([0, 1, 0, 1])
COLOR_RED = np.array([1, 0, 0, 1])
COLOR_GHOST = np.array([1, 0, 0.5, 1])
COLOR_GHOST3D = np.array([1, 0, 0.5, 1])
COLOR_ROBBER = np.array([1, 1, 0.5, 1])
COLOR_ROBBER3D = np.array([1, 1, 0.5, 1])

# Groups are a mujoco-specific mechanism for selecting which geom objects to "see"
# We use these for raycasting lidar, where there are different lidar types.
# These work by turning "on" the group to see and "off" all the other groups.
# See obs_lidar_natural() for more.
GROUP_GOAL = 0
GROUP_BOX = 1
GROUP_BUTTON = 1
GROUP_WALL = 2
GROUP_PILLAR = 2
GROUP_HAZARD = 3
GROUP_VASE = 4
GROUP_GREMLIN = 5
GROUP_CIRCLE = 6
GROUP_HAZARD3D = 3
GROUP_GHOST = 3
GROUP_GHOST3D = 3
GROUP_ROBBER = 5
GROUP_ROBBER3D = 5



# Constant for origin of world
ORIGIN_COORDINATES = np.zeros(3)

# Constant defaults for rendering frames for humans (not used for vision)
DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'

class ResamplingError(AssertionError):
    ''' Raised when we fail to sample a valid distribution of objects or goals '''
    pass


class Engine(gym.Env, gym.utils.EzPickle):

    '''
    Engine: an environment-building tool for safe exploration research.

    The Engine() class entails everything to do with the tasks and safety 
    requirements of Safety Gym environments. An Engine() uses a World() object
    to interface to MuJoCo. World() configurations are inferred from Engine()
    configurations, so an environment in Safety Gym can be completely specified
    by the config dict of the Engine() object.

    '''

        # Default configuration (this should not be nested since it gets copied)
    DEFAULT = {
        'device_id': 0,
        'num_envs': 1, # Number of the batched environment
        
        # Observation flags - some of these require other flags to be on
        # By default, only robot sensor observations are enabled.
        'observation_flatten': True,  # Flatten observation into a vector
        'observe_goal_lidar': True,  # Observe the goal with a lidar sensor
        'observe_goal_comp': True,  # Observe a compass vector to the goal
        'observe_hazards': False,  # Observe the vector from agent to hazards
        # These next observations are unnormalized, and are only for debugging
        'observe_qpos': True,  # Observe the qpos of the world
        'observe_qvel': True,  # Observe the qvel of the robot
        'observe_qacc': True,  # Observe the qacc of the robot
        'observe_vel': False,  # Observe the vel of the robot
        'observe_acc': False,  # Observe the acc of the robot
        'observe_ctrl': True,  # Observe the previous action

        # Render options
        'render_labels': False,
        'render_lidar_markers': True,
        'render_lidar_radius': 0.15, 
        'render_lidar_size': 0.025, 
        'render_lidar_offset_init': 0.5, 
        'render_lidar_offset_delta': 0.06, 

        # Lidar observation parameters
        'lidar_num_bins': 16,  # Bins (around a full circle) for lidar sensing
        'lidar_num_bins3D': 1,  # Bins (around a full circle) for lidar sensing
        'lidar_max_dist': None,  # Maximum distance for lidar sensitivity (if None, exponential distance)
        'lidar_exp_gain': 1.0, # Scaling factor for distance in exponential distance lidar
        'lidar_type': 'pseudo',  # 'pseudo', 'natural', see self.obs_lidar()
        'lidar_alias': True,  # Lidar bins alias into each other
        'lidar_body': ['robot'], # Bodies with Lidar

        # Task
        'task': 'goal',  # goal, button, push, x, z, circle, or none (for screenshots)
        'push_object': 'box', # box, ball
        'goal_mode': 'random', # random, track. only apply when continue_goal is true
        'goal_travel': 3.0,  # Radius of the circle goal can travel in
        'goal_velocity':0.5, # Velocity of the goal under 'track' mode

        # Goal parameters
        'goal_placements': None,  # Placements where goal may appear (defaults to full extents)
        'goal_locations': [],  # Fixed locations to override placements
        'goal_keepout': 0.4,  # Keepout radius when placing goals
        'goal_size': 0.3,  # Radius of the goal area (if using task 'goal')
        'goal_3D': False,
        'goal_z_range': [1.0, 1.0],  # range of z pos of goal, only for 3D goal

        # Reward is distance towards goal plus a constant for being within range of goal
        # reward_distance should be positive to encourage moving towards the goal
        # if reward_distance is 0, then the reward function is sparse
        'reward_distance': 1.0,  # Dense reward multiplied by the distance moved to the goal
        'reward_goal': 1.0,  # Sparse reward for being inside the goal area
        'reward_box_dist': 1.0,  # Dense reward for moving the robot towards the box
        'reward_box_goal': 1.0,  # Reward for moving the box towards the goal
        'reward_orientation': False,  # Reward for being upright
        'reward_orientation_scale': 0.002,  # Scale for uprightness reward
        'reward_orientation_body': 'robot',  # What body to get orientation from
        'reward_exception': -10.0,  # Reward when encoutering a mujoco exception
        'reward_x': 1.0,  # Reward for forward locomotion tests (vel in x direction)
        'reward_z': 1.0,  # Reward for standup tests (vel in z direction)
        'reward_circle': 1e-1,  # Reward for circle goal (complicated formula depending on pos and vel)
        'reward_clip': 10,  # Clip reward, last resort against physics errors causing magnitude spikes
        'reward_defense': 1.0, # Reward for the robbers be outside of the circle
        'reward_chase': 1.0, # Reward for the closest distance from the robot to the robbers

        # Constraints - flags which can be turned on
        # By default, no constraints are enabled, and all costs are indicator functions.
        'constrain_hazards': False,  # Constrain robot from being in hazardous areas
        'constrain_indicator': True,  # If true, all costs are either 1 or 0 for a given step.

        # Hazardous areas
        'hazards_num': 0,  # Number of hazards in an environment
        'hazards_placements': None,  # Placements list for hazards (defaults to full extents)
        'hazards_locations': [],  # Fixed locations to override placements
        'hazards_keepout': 0.4,  # Radius of hazard keepout for placement
        'hazards_size': 0.3,  # Radius of hazards
        'hazards_cost': 1.0,  # Cost (per step) for violating the constraint

        'physics_steps_per_control_step': 1, # number of steps per Mujoco step
        '_seed': 0,  # Random state seed (avoid name conflict with self.seed)
    }


    def __init__(self, config={}):
        # First, parse configuration. Important note: LOTS of stuff happens in
        # parse, and many attributes of the class get set through setattr. If you
        # are trying to track down where an attribute gets initially set, and 
        # can't find it anywhere else, it's probably set via the config dict
        # and this parse function.
        self.parse(config)
        self.body_name2id = {}
        
        self.seed()
        # path = 'xmls/barkour_v0/assets/barkour.xml'
        path = 'xmls/point.xml'
        self.robot = Robot(path)
        base_path = os.path.join(BASE_DIR, path)
        self.mj_model = mujoco.MjModel.from_xml_path(base_path) # Load Mujoco model from xml
        self.mj_data = mujoco.MjData(self.mj_model) # Genearte Mujoco data from Mujoco model
        self.mjx_model = device_put(self.mj_model, self.device_id) # Convert Mujoco model to MJX model for device acceleration
        self.mjx_data = device_put(self.mj_data, self.device_id) # Convert Mujoco data to MJX data for device acceleration
        
        
        self.dt = self.mj_model.opt.timestep * self.physics_steps_per_control_step
        print(self.dt)
        self.key = jax.random.PRNGKey(0) # Number of the batched environment
        
        #----------------------------------------------------------------
        # define functions
        # batch step
        # batch reset for all environment (end of epoch)
        # batch reset for environments that are done (within epoch)
        #----------------------------------------------------------------
        
        # Use jax.vmap to warp single environment reset to batched reset function
        def batched_reset(rng: jax.Array):
            if self.num_envs is not None:
                rng = jax.random.split(rng, self.num_envs)
            return jax.vmap(self.mjx_reset)(rng)
        
        # Use jax.vmap to warp single environment step to batched step function
        def batched_step(data: mjx.Data, last_data: mjx.Data, last_last_data: mjx.Data, last_done: jax.Array, last_last_done: jax.Array, action: jax.Array):
            return jax.vmap(self.mjx_step)(data, last_data, last_last_data, last_done, last_last_done, action)
    
        def batched_reset_done(data: mjx.Data,
                               done: jax.Array, 
                               obs: jax.Array,
                               rng: jax.Array):
            if self.num_envs is not None:
                rng = jax.random.split(rng, self.num_envs)
            return jax.vmap(self.mjx_reset_done)(data, done, obs, rng)
    
        #----------------------------------------------------------------
        # define batch operation interfaces
        #----------------------------------------------------------------
        self._reset = jax.jit(batched_reset) # Use Just In Time (JIT) compilation to execute batched reset efficiently
        self._step = jax.jit(batched_step) # Use Just In Time (JIT) compilation to execute batched step efficiently
        self._reset_done = jax.jit(batched_reset_done)
        
        
        #----------------------------------------------------------------
        # define log variables
        #----------------------------------------------------------------
        self._data = None # Keep track of the current state of the whole batched environment with self._data
        self._last_data = None
        self._last_last_data = None
        self._done = None
        self._last_done = None
        self._last_last_done = None
        self.viewer = None
        self.hazards_placements = [-2,2]
        
        
        #----------------------------------------------------------------
        # define environment configuration, observation, action
        #----------------------------------------------------------------
        ctrl_range = self.mj_model.actuator_ctrlrange
        ctrl_range[~(self.mj_model.actuator_ctrllimited == 1), :] = np.array([-np.inf, np.inf])
        action = jax.tree_map(np.array, ctrl_range)
        self.action_space = gym.spaces.Box(action[:, 0], action[:, 1], dtype=np.float32)
        # self.action_space = utils.batch_space(action_space, self.num_envs)
        self._done = None
        self.build_observation_space()
        self.body_name2id = {}
        self.body_name2id['floor'] = self.mj_model.geom('floor').id
        self.body_name2id['robot'] = self.mj_model.geom('robot').id
        self.body_name2id['goal'] = self.mj_model.geom('goal').id
        self.body_name2id['hazards'] = []
        for i in range(self.mj_model.ngeom):
            if 'hazard' in self.mj_model.geom(i).name:
                self.body_name2id['hazards'].append(i)

    #----------------------------------------------------------------
    # environment configuration functions
    #----------------------------------------------------------------
    
    def parse(self, config):
        ''' Parse a config dict - see self.DEFAULT for description '''
        self.config = deepcopy(self.DEFAULT)
        self.config.update(deepcopy(config))
        for key, value in self.config.items():
            assert key in self.DEFAULT, f'Bad key {key}'
            setattr(self, key, value)

    def build_observation_space(self):
        ''' Construct observtion space.  Happens only once at during __init__ '''
        obs_space_dict = OrderedDict()  # See self.obs()
        self.robot.nq = 2
        self.robot.nv = 2
        self.robot.nu = 2
        if self.observe_goal_lidar:
            obs_space_dict['goal_lidar'] = gym.spaces.Box(0.0, 1.0, (self.lidar_num_bins,), dtype=np.float32)
        if self.observe_goal_comp:
            obs_space_dict['goal_compass'] = gym.spaces.Box(-np.inf, np.inf, (2,), dtype=np.float32)
        if self.observe_hazards:
            obs_space_dict['hazards_lidar'] = gym.spaces.Box(0.0, 1.0, (self.lidar_num_bins,), dtype=np.float32)
        if self.observe_qpos:
            obs_space_dict['qpos'] = gym.spaces.Box(-np.inf, np.inf, (self.robot.nq,), dtype=np.float32)
        if self.observe_qvel:
            obs_space_dict['qvel'] = gym.spaces.Box(-np.inf, np.inf, (self.robot.nv,), dtype=np.float32)
        if self.observe_ctrl:
            obs_space_dict['ctrl'] = gym.spaces.Box(-np.inf, np.inf, (self.robot.nu,), dtype=np.float32)
        if self.observe_vel:
            obs_space_dict['vel'] = gym.spaces.Box(-np.inf, np.inf, (2,), dtype=np.float32)
        if self.observe_acc:
            obs_space_dict['acc'] = gym.spaces.Box(-np.inf, np.inf, (2,), dtype=np.float32)
        # Flatten it ourselves
        self.obs_space_dict = obs_space_dict
        
        if self.observation_flatten:
            self.obs_flat_size = sum([np.prod(i.shape) for i in self.obs_space_dict.values()])
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, (self.obs_flat_size,), dtype=np.float32)
            # self.observation_space = utils.batch_space(self.observation_space, self.num_envs)
        else:
            for k, v in self.obs_space_dict.items():
                self.obs_space_dict[k] = utils.batch_space(v, self.num_envs)
            self.observation_space = gym.spaces.Dict(obs_space_dict)

    def seed(self):
        self.key = jax.random.PRNGKey(self._seed)

    #----------------------------------------------------------------
    # gym wrapper functions
    #----------------------------------------------------------------

    def update_data(self):
        self._last_last_data = deepcopy(self._last_data)
        self._last_data = deepcopy(self._data)
        self._last_last_done = deepcopy(self._last_done)
        self._last_done = deepcopy(self._done)
    
    def reset(self):
        ''' Reset the physics simulation and return observation '''
        obs, data = self._reset(self.key)
        
        # update the log
        self._obs = obs
        self._data = data
        return jax_to_torch(obs)
    
    def step(self, action):
        ''' Take a step and return observation, reward, done, and info '''
        # print(action)
        action = torch_to_jax(action)
        
        self.key,_ = jax.random.split(self.key, 2)
        self.update_data()
        obs, reward, done, info, data = self._step(self._data, 
                                                    self._last_data, 
                                                    self._last_last_data, 
                                                    self._last_done, 
                                                    self._last_last_done, 
                                                    action)
        # update the current info right after step 
        self._data = data
        self._obs = obs
        self._reward = reward
        self._done = done
        self._info = info
        return jax_to_torch(self._obs), jax_to_torch(self._reward), jax_to_torch(self._done), jax_to_torch(self._info)
    
    def reset_done(self):
        obs, self._data = self._reset_done(self._data,
                                            self._done,
                                            self._obs,
                                            self.key)
        return jax_to_torch(obs)
    
    def build_layout(self, rng):
        ''' Rejection sample a placement of objects to find a layout. '''
        # if not self.randomize_layout:
        #     self.rs = np.random.RandomState(0)
        rng, rng1, rng2 = jax.random.split(rng, 3)
        
        layout = {}
        layout["hazards_pos"] = jax.random.uniform(rng1, (8,3), minval=self.hazards_placements[0], maxval=self.hazards_placements[1])
        layout["goal_pos"] = jax.random.uniform(rng1, (1,3), minval=-2.0, maxval=2.0)
        return layout  
 
    #----------------------------------------------------------------
    # jax parallel wrapper functions
    #----------------------------------------------------------------

    def mjx_reset(self, rng):
        """ Resets an unbatched environment to an initial state."""
        #! TODO: update qpose with layout
        layout = self.build_layout(rng)
        qpos_new = jax.random.uniform(rng, (self.mjx_model.nq,), minval=-1, maxval=1)
        data = self.mjx_data
        data = data.replace(qpos=qpos_new, qvel=jp.zeros(self.mjx_model.nv), ctrl=jp.zeros(self.mjx_model.nu))
        
        # log the last data
        data = mjx.forward(self.mjx_model, data)
        last_data = data
        last_last_data = data
        obs, _ = self.obs(data, last_data, last_last_data)
        return obs, data
    
    def mjx_step(self, data: mjx.Data, 
                 last_data: mjx.Data, 
                 last_last_data: mjx.Data, 
                 last_done: jp.ndarray,
                 last_last_done: jp.ndarray,
                 action: jp.ndarray):
        """Runs one timestep of the environment's dynamics."""
        def f(data, _):   
            data = data.replace(ctrl=action)
            return (
                mjx.step(self.mjx_model, data),
                None,
            )
        data, _ = jax.lax.scan(f, data, (), self.physics_steps_per_control_step)
        obs, obs_dict = self.obs(data, last_data, last_last_data, last_done, last_last_done)
        reward, done = self.reward_done(data, last_data, last_done)
        cost = self.cost(data)
        info = {}
        info['cost'] = cost
        info['obs'] = obs_dict
        return obs, reward, done, info, data
    
    def mjx_reset_done(self,
                        data: mjx.Data, 
                        done: jp.ndarray,
                        current_obs: jp.ndarray,
                        rng):
        """Runs one timestep of the environment's dynamics."""
        def f(data, _): 
            return (
                mjx.step(self.mjx_model, data),
                None,
            )
        qpos_reset = jax.random.uniform(rng, (self.mjx_model.nq,), minval=-1.5, maxval=1.5)
        ctrl_reset = jp.zeros(self.mjx_model.nu)
        qvel_reset = jp.zeros(self.mjx_model.nv)
        
        if last_data is None:
            last_data = data
        if last_last_data is None:
            last_last_data = last_data
            
        # fake one step forward to get xpos/observation for new initialized jpos
        qpos = jp.where(done > 0.0, x = qpos_reset, y = data.qpos)
        ctrl = jp.where(done > 0.0, x = ctrl_reset, y = data.ctrl)
        qvel = jp.where(done > 0.0, x = qvel_reset, y = data.qvel)
        data = data.replace(qpos=qpos, qvel=qvel, ctrl=ctrl)
        data_reset, _ = jax.lax.scan(f, data, (), self.physics_steps_per_control_step)
        # reset observation treats last done and last last done all true, just use current data 
        obs_reset, _ = self.obs(data_reset, data, data, done, done)

        # reset observation for done environment 
        obs = jp.where(done > 0., x=obs_reset, y=current_obs)
        
        return obs, data


    #----------------------------------------------------------------
    # observation, reward and cost functions
    #----------------------------------------------------------------

    def obs(self, data: mjx.Data, 
            last_data: mjx.Data, 
            last_last_data: mjx.Data, 
            last_done: jp.ndarray, 
            last_last_done: jp.ndarray):
        # get the raw position data of different objects current frame 
        obs = {}

        # get the raw position data of previous frames
        if last_data is None:
            last_data = data
        if last_last_data is None:
            last_last_data = last_data
        vel, acc = self.ego_vel_acc(data, last_data, last_last_data, last_done, last_last_done)
        
        if self.observe_goal_lidar:
            obs['goal_lidar'] = self.obs_lidar(data, data.xpos[self.body_name2id['goal'],:])
        if self.observe_hazards:
            obs['hazards_lidar'] = self.obs_lidar(data, data.xpos[self.body_name2id['hazards'],:])
        if self.observe_goal_comp:
            obs['goal_compass'] = self.obs_compass(data, data.xpos[self.body_name2id['goal'],:])
        if self.observe_qpos:
            obs['qpos'] = data.qpos[:self.robot.nq]
        if self.observe_qvel:
            obs['qvel'] = data.qvel[:self.robot.nv]
        if self.observe_qacc:
            obs['qacc'] = data.qacc[:self.robot.nv]
        if self.observe_ctrl:
            obs['ctrl'] = data.ctrl[:self.robot.nu]
        if self.observe_vel:
            obs['vel'] = vel
        if self.observe_acc:
            obs['acc'] = acc
        
        
        if self.observation_flatten:
            flat_obs = []
            for k in sorted(self.obs_space_dict.keys()):
                flat_obs.append(obs[k].flatten())
            flat_obs = jp.concatenate(flat_obs)
        return flat_obs, obs

    def goal_pos(self, data: mjx.Data) -> jp.ndarray:
            xpos = data.xpos.reshape(-1,3)
            robot_pos = xpos[self.body_name2id['robot'],:]
            goal_pos = xpos[self.body_name2id['goal'],:]
            dist_goal = jp.sqrt(jp.sum(jp.square(goal_pos[:2] - robot_pos[:2]))) 
            return dist_goal

    def reward_done(self, data: mjx.Data, last_data: mjx.Data, last_done:jp.ndarray) -> Tuple[jp.ndarray, jp.ndarray]:        
        # get raw data of robot pos and goal pos
        
        dist_goal = self.goal_pos(data)
        
        # upgrade last dist goal computation for auto reset 
        last_dist_goal = jp.where(last_done > 0.0, x = self.goal_pos(data), y = self.goal_pos(last_data))
        reward = (last_dist_goal - dist_goal) * self.reward_distance
        done = jp.where(dist_goal < self.goal_size, x = 1.0, y = 0.0)
        return reward, done
    
    def cost(self, data: mjx.Data) -> jp.ndarray:
        # get the raw position data of different objects current frame 
        robot_pos = data.xpos[self.body_name2id['robot'],:].reshape(-1,3)
        hazards_pos = data.xpos[self.body_name2id['hazards'],:].reshape(-1,3)
        dist_robot2hazard = jp.linalg.norm(hazards_pos[:,:2] - robot_pos[:,:2], axis=1)
        dist_robot2hazard_below_threshold = jp.minimum(dist_robot2hazard, self.hazards_size)
        cost = jp.sum(self.hazards_size*jp.ones(dist_robot2hazard_below_threshold.shape) - dist_robot2hazard_below_threshold)
        return cost 
    
    #----------------------------------------------------------------
    # Computation Auxiliary functions
    #----------------------------------------------------------------
    
    def ego_xy(self, pos, data):
        xpos = data.xpos.reshape(-1,3)
        xmat = data.xmat.reshape(-1,3,3)
        robot_pos = xpos[self.body_name2id['robot'],:]
        robot_mat = xmat[self.body_name2id['robot'],:]
        ''' Return the egocentric XY vector to a position from the robot '''
        assert pos.shape == (2,), f'Bad pos {pos}'
        pos_3vec = jp.concatenate([pos, jp.array([0])]) # Add a zero z-coordinate
        world_3vec = pos_3vec - robot_pos
        return jp.matmul(world_3vec, robot_mat)[:2] # only take XY coordinates

    def jax_angle(self, real_part, imag_part):
        '''Returns robot heading angle in the world frame'''
        # Calculate the angle (phase) in radians
        angle_rad = jp.arctan2(imag_part, real_part)
        return angle_rad

    def obs_compass(self, data, pos):
        xpos = data.xpos.reshape(-1,3)
        xmat = data.xmat.reshape(-1,3,3)
        robot_pos = xpos[self.body_name2id['robot'],:]
        robot_mat = xmat[self.body_name2id['robot'],:]
        ''' Return the egocentric XY vector to a position from the robot '''
        if pos.shape == (3,):
            pos = pos[:2]
        pos_3vec = jp.concatenate([pos, jp.array([0])]) # Add a zero z-coordinate
        world_3vec = pos_3vec - robot_pos
        return jp.matmul(world_3vec, robot_mat)[:2] # only take XY coordinates

    def obs_lidar(self, data, positions):
        '''
        Return a robot-centric lidar observation of a list of positions.


        Lidar is a set of bins around the robot (divided evenly in a circle).
        The detection directions are exclusive and exhaustive for a full 360 view.
        Each bin reads 0 if there are no objects in that direction.
        If there are multiple objects, the distance to the closest one is used.
        Otherwise the bin reads the fraction of the distance towards the robot.


        E.g. if the object is 90% of lidar_max_dist away, the bin will read 0.1,
        and if the object is 10% of lidar_max_dist away, the bin will read 0.9.
        (The reading can be thought of as "closeness" or inverse distance)


        This encoding has some desirable properties:
        - bins read 0 when empty
        - bins smoothly increase as objects get close
        - maximum reading is 1.0 (where the object overlaps the robot)
        - close objects occlude far objects
        - constant size observation with variable numbers of objects
        '''
        positions = positions.reshape(-1, 3)
        obs = jp.zeros(self.lidar_num_bins)
        for pos in positions:
            pos = jp.asarray(pos)
            if pos.shape == (3,):
                pos = pos[:2] # Truncate Z coordinate

            z = self.ego_xy(pos, data) # (X, Y) to be treated as real, imaginary components
            dist = jp.linalg.norm(z) # shape = B
            angle = self.jax_angle(real_part=z[0], imag_part=z[1]) % (jp.pi * 2) # shape = B
            bin_size = (jp.pi * 2) / self.lidar_num_bins
            
            bin = (angle / bin_size).astype(int)
            bin_angle = bin_size * bin
            # import ipdb; ipdb.set_trace()
            if self.lidar_max_dist is None:
                sensor = jp.exp(-self.lidar_exp_gain * dist)
            else:
                sensor = max(0, self.lidar_max_dist - dist) / self.lidar_max_dist
            senor_new = jp.maximum(obs[bin], sensor)
            obs = obs.at[bin].set(senor_new)
            # Aliasing
            # this is to make the neighborhood bins has sense of the what's happending in the bin that has obstacle
            if self.lidar_alias:
                alias_jp = (angle - bin_angle) / bin_size
                # assert 0 <= alias_jp <= 1, f'bad alias_jp {alias_jp}, dist {dist}, angle {angle}, bin {bin}'
                bin_plus = (bin + 1) % self.lidar_num_bins
                bin_minus = (bin - 1) % self.lidar_num_bins
                obs = obs.at[bin_plus].set(jp.maximum(obs[bin_plus], alias_jp * sensor))
                obs = obs.at[bin_minus].set(jp.maximum(obs[bin_minus], (1 - alias_jp) * sensor))
        return obs
    
    def ego_vel_acc(self, data, last_data, last_last_data, last_done, last_last_done):
        '''velocity and acceleration in the robot frame '''
        robot_pos = data.xpos[self.body_name2id['robot'],:]
        robot_mat = data.xmat[self.body_name2id['robot'],:,:]

        # auto update past robot poses for auto reset  
        robot_pos_last = jp.where(last_done > 0.0, x = robot_pos, y = last_data.xpos[self.body_name2id['robot'],:])
        robot_pos_last_last = jp.where(last_last_done + last_done > 0.0, x = robot_pos_last, y = last_last_data.xpos[self.body_name2id['robot'],:])
        # current velocity
        pos_diff_vec_world_frame = robot_pos - robot_pos_last
        vel_vec_world_frame = pos_diff_vec_world_frame / self.dt
        
        # last velocity 
        last_pos_diff_vec_world_frame = robot_pos_last - robot_pos_last_last
        last_vel_vec_world_frame = last_pos_diff_vec_world_frame / self.dt
        
        # current acceleration
        acc_vec_world_frame = (vel_vec_world_frame - last_vel_vec_world_frame) / self.dt
        
        # to robot frame 
        vel_vec_robot_frame = jp.matmul(vel_vec_world_frame, robot_mat)[:2] # only take XY coordinates
        acc_vec_robot_frame = jp.matmul(acc_vec_world_frame, robot_mat)[:2] # only take XY coordinates
        
        return vel_vec_robot_frame, acc_vec_robot_frame
   
    #----------------------------------------------------------------
    # Render functions
    #----------------------------------------------------------------
    
    def render_sphere(self, pos, size, color, label='', alpha=0.1):
        pass

    def viewer_setup(self):
        # self.viewer.cam.trackbodyid = 0         # id of the body to track ()
        # self.viewer.cam.distance = self.model.stat.extent * 3       # how much you "zoom in", model.stat.extent is the max limits of the arena
        self.viewer.cam.distance = 6
        self.viewer.cam.lookat[0] = 0         # x,y,z offset from the object (works if trackbodyid=-1)
        self.viewer.cam.lookat[1] = -3
        self.viewer.cam.lookat[2] = 5
        self.viewer.cam.elevation = -60           # camera rotation around the axis in the plane going through the frame origin (if 0 you just see a line)
        self.viewer.cam.azimuth = 90              # camera rotation around the camera's vertical axis
        self.viewer.opt.geomgroup = 1    

    def render_lidar(self, data: mjx.Data, lidar, color, offset):
        xpos = data.xpos.reshape(-1,3)
        xmat = data.xmat.reshape(-1,3,3)
        robot_pos = xpos[self.body_name2id['robot'],:]
        robot_mat = xmat[self.body_name2id['robot'],:]
        lidar = lidar.flatten()
        cnt = 0
        for i, sensor in enumerate(lidar):
            theta = 2 * np.pi * (i + 0.5) / self.lidar_num_bins # i += 0.5  # Offset to center of bin
            rad = self.render_lidar_radius
            
            binpos = np.array([np.cos(theta) * rad, np.sin(theta) * rad, offset])
            pos = robot_pos + np.matmul(binpos, robot_mat.transpose())
            alpha = min(1.0, sensor + .1)
            mujoco.mjv_initGeom(
                self.viewer.user_scn.geoms[i + self.viewer.user_scn.ngeom],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.02, 0, 0],
                pos=pos.flatten(),
                mat=np.eye(3).flatten(),
                rgba=np.array(color) * alpha
                )
            mujoco.mjv_initGeom(
                self.renderer_scene.geoms[i + self.renderer_scene.ngeom],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.02, 0, 0],
                pos=pos.flatten(),
                mat=np.eye(3).flatten(),
                rgba=np.array(color) * alpha
                )
            cnt += 1
        self.viewer.user_scn.ngeom += cnt
        self.renderer_scene.ngeom += cnt
    
    def render_compass(self, data: mjx.Data, compass_pos, color, offset):
        xpos = data.xpos.reshape(-1,3)
        xmat = data.xmat.reshape(-1,3,3)
        robot_pos = xpos[self.body_name2id['robot'],:]
        robot_mat = xmat[self.body_name2id['robot'],:]
        compass_pos = compass_pos.flatten()
        compass_pos = jp.concatenate([compass_pos * 0.15, jp.array([0])])
        pos = robot_pos + np.matmul(compass_pos, robot_mat.transpose())
        mujoco.mjv_initGeom(
            self.viewer.user_scn.geoms[self.viewer.user_scn.ngeom],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.02, 0, 0],
            pos=pos.flatten(),
            mat=np.eye(3).flatten(),
            rgba=np.array(color) * 0.5
            )
        mujoco.mjv_initGeom(
            self.renderer_scene.geoms[self.renderer_scene.ngeom],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.02, 0, 0],
            pos=pos.flatten(),
            mat=np.eye(3).flatten(),
            rgba=np.array(color) * 0.5
            )
        self.viewer.user_scn.ngeom += 1
        self.renderer_scene.ngeom += 1

    def render(self):
        data = self.mj_data
        mjx.device_get_into(data, self._data)
        self.mj_model.vis.global_.offwidth = DEFAULT_WIDTH
        self.mj_model.vis.global_.offheight = DEFAULT_HEIGHT
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.mj_model, data)
            self.renderer =  mujoco.Renderer(self.mj_model, width = DEFAULT_WIDTH, height = DEFAULT_HEIGHT)
            
            self.viewer_setup()
            # self.renderer_setup()
        # print(data.xpos[1])
        
        mujoco.mj_step(self.mj_model, data)
        self.renderer.update_scene(data, self.viewer.cam, self.viewer.opt)
        self.renderer_scene = self.renderer._scene
        offset = 0.5
        self.viewer.user_scn.ngeom = 0
        obs = self._obs
        # self.render_lidar(self._data, obs['hazards_lidar'], COLOR_HAZARD, offset)
        # offset += 0.1
        # self.render_lidar(self._data, obs['goal_lidar'], COLOR_GOAL, offset)
        # self.render_compass(self._data, obs['goal_compass'], COLOR_GOAL, offset)
        self.viewer.sync()
        
        return self.renderer.render()
        