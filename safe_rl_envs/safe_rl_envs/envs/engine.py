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
from functools import partial

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

JOINT_SIZE = [7,4,1,1]

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
        'num_steps': 1000,  # Maximum number of environment steps in an episode
        'device_id': 0,
        'env_num': 1, # Number of the batched environment
        
        'placements_extents': [-2, -2, 2, 2],  # Placement limits (min X, min Y, max X, max Y)
        'placements_margin': 0.0,  # Additional margin added to keepout when placing objects

        # Floor
        'floor_display_mode': False,  # In display mode, the visible part of the floor is cropped

        # Robot
        'robot_placements': None,  # Robot placements list (defaults to full extents)
        'robot_locations': [],  # Explicitly place robot XY coordinate
        'robot_keepout': 0.4,  # Needs to be set to match the robot XML used
        'robot_base': 'xmls/point.xml',  # Which robot XML to use as the base
        'robot_rot': None,  # Override robot starting angle
        
        # Observation flags - some of these require other flags to be on
        # By default, only robot sensor observations are enabled.
        'observation_flatten': True,  # Flatten observation into a vector
        'observe_goal_lidar': True,  # Observe the goal with a lidar sensor
        'observe_goal_comp': True,  # Observe a compass vector to the goal
        'observe_hazards': True,  # Observe the vector from agent to hazards
        # These next observations are unnormalized, and are only for debugging
        'observe_qpos': True,  # Observe the qpos of the world
        'observe_qvel': True,  # Observe the qvel of the robot
        'observe_qacc': True,  # Observe the qacc of the robot
        'observe_vel': False,  # Observe the vel of the robot
        'observe_acc': False,  # Observe the acc of the robot
        'observe_ctrl': True,  # Observe the previous action
        'observe_vision': False,  # Observe vision from the robot

        # Render options
        'render_labels': False,
        'render_lidar_markers': True,
        'render_lidar_radius': 0.15, 
        'render_lidar_size': 0.025, 
        'render_lidar_offset_init': 0.5, 
        'render_lidar_offset_delta': 0.06, 

        # Sensor observations
        # Specify which sensors to add to observation space
        'sensors_obs': ['accelerometer', 'velocimeter', 'gyro', 'magnetometer'],
        'sensors_hinge_joints': True,  # Observe named joint position / velocity sensors
        'sensors_ball_joints': True,  # Observe named balljoint position / velocity sensors
        'sensors_angle_components': True,  # Observe sin/cos theta instead of theta

        
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
        'goal_keepout': 0.5,  # Keepout radius when placing goals
        'goal_size': 0.5,  # Radius of the goal area (if using task 'goal')
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
        'hazards_num': 8,  # Number of hazards in an environment
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
        self.body_name2xpos_id = {}
        self.key = jax.random.PRNGKey(self._seed) 

        # path = 'xmls/barkour_v0/assets/barkour.xml'
        path = 'xmls/ant.xml'
        self.robot = Robot(self.robot_base)
        base_path = os.path.join(BASE_DIR, path)
        self.world_config_dict = self.build_world_config()
        self.world = World(self.world_config_dict)
        self.world.build()
        
        # self.mj_model = mujoco.MjModel.from_xml_path(base_path) # Load Mujoco model from xml
        self.mj_model = self.world.model
        self.mj_model = mujoco.MjModel.from_xml_string(self.world.xml_string)
        self.mj_data = mujoco.MjData(self.mj_model) # Genearte Mujoco data from Mujoco model
        
        self.mjx_model = device_put(self.mj_model, self.device_id) # Convert Mujoco model to MJX model for device acceleration
        self.mjx_data = device_put(self.mj_data, self.device_id) # Convert Mujoco data to MJX data for device acceleration
        
        
        self.dt = self.mj_model.opt.timestep * self.physics_steps_per_control_step
        print(self.dt)
        # Number of the batched environment
        
        #----------------------------------------------------------------
        # define functions
        # batch step
        # batch reset for all environment (end of epoch)
        # batch reset for environments that are done (within epoch)
        #----------------------------------------------------------------
        
        # Use jax.vmap to warp single environment reset to batched reset function
        def batched_reset(layout_valid):
            return jax.vmap(self.mjx_reset)(layout_valid)
        
        # Use jax.vmap to warp single environment step to batched step function
        def batched_step(data: mjx.Data, last_data: mjx.Data, last_last_data: mjx.Data, last_done: jax.Array, last_last_done: jax.Array, action: jax.Array):
            return jax.vmap(self.mjx_step)(data, last_data, last_last_data, last_done, last_last_done, action)
    
        def batched_reset_done(data: mjx.Data,
                               done: jax.Array, 
                               obs: jax.Array,
                               layout):
            return jax.vmap(self.mjx_reset_done)(data, done, obs, layout)
        
        # Use jax.vmap to warp single environment reset to batched reset function
        def batched_sample_layout(rng: jax.Array):
            if self.env_num is not None:
                rng = jax.random.split(rng, int(1e6))
            return jax.vmap(self.sample_layout)(rng)
    
        #----------------------------------------------------------------
        # define batch operation interfaces
        #----------------------------------------------------------------
        self._reset = jax.jit(batched_reset) # Use Just In Time (JIT) compilation to execute batched reset efficiently
        self._step = jax.jit(batched_step) # Use Just In Time (JIT) compilation to execute batched step efficiently
        self._reset_done = jax.jit(batched_reset_done)
        self._sample_layout = jax.jit(batched_sample_layout)
        
        
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
        # self.hazards_placements = [-2,2]
        
        
        #----------------------------------------------------------------
        # define environment configuration, observation, action
        #----------------------------------------------------------------
        ctrl_range = self.mj_model.actuator_ctrlrange
        ctrl_range[~(self.mj_model.actuator_ctrllimited == 1), :] = np.array([-np.inf, np.inf])
        if 'point' in self.robot_base:
            ctrl_range = ctrl_range[:2,:]
        action = jax.tree_map(np.array, ctrl_range)
        self.action_space = gym.spaces.Box(action[:, 0], action[:, 1], dtype=np.float32)
        self.action_space = gym.spaces.Box(action[:, 0], action[:, 1], dtype=np.float32)
        # self.action_space = utils.batch_space(action_space, self.env_num)
        self._done = None
        self.build_observation_space()
        self.build_placements_dict()
        self.body_name2xpos_id = {}
        self.body_name2xpos_id['robot'] = self.mj_model.body('robot').id
        self.body_name2xpos_id['goal'] = self.mj_model.body('goal').id
        self.body_name2xpos_id['hazards'] = []
        for i in range(self.mj_model.nbody):
            if 'hazard' in self.mj_model.body(i).name:
                self.body_name2xpos_id['hazards'].append(i)

        self.joint_name2qpos_id = {}
        idx = 0
        for i in range(self.mj_model.njnt):
            name = self.mj_model.jnt(i).name
            type = self.mj_model.jnt(i).type[0]
            self.joint_name2qpos_id[name] = idx
            idx += JOINT_SIZE[type]

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

    def random_rot(self):
        ''' Use internal random state to get a random rotation in radians '''
        # return float(jax.random.uniform(self.key, minval=0, maxval=2 * jp.pi))
        return 0.0

    def build_world_config(self):
        ''' Create a world_config from our own config '''
        # TODO: parse into only the pieces we want/need
        world_config = {}

        world_config['robot_base'] = self.robot_base
        world_config['robot_xy'] = [0.0, 0.0]
        if self.robot_rot is None:
            world_config['robot_rot'] = self.random_rot()
        else:
            world_config['robot_rot'] = float(self.robot_rot)

        if self.floor_display_mode:
            floor_size = max(self.placements_extents)
            world_config['floor_size'] = [floor_size + .1, floor_size + .1, 1]

        #if not self.observe_vision:
        #    world_config['render_context'] = -1  # Hijack this so we don't create context
        world_config['observe_vision'] = self.observe_vision
        # Extra objects to add to the scene
        world_config['objects'] = {}
        # Extra geoms (immovable objects) to add to the scene
        world_config['geoms'] = {}
        if self.task in ['goal', 'push']:
            goal_pos = np.r_[0.0, 0.0, 0.0]
            geom = {'name': 'goal',
                'size': [self.goal_size],
                'pos': goal_pos,
                'rot': self.random_rot(),
                'type': 'sphere',
                'contype': 0,
                'conaffinity': 0,
                'group': GROUP_GOAL,
                'rgba': COLOR_GOAL* [1, 1, 1, 0.25]}  # transparent * [1, 1, 1, 0.25]
            world_config['geoms']['goal'] = geom
        if self.hazards_num:
            for i in range(self.hazards_num):
                name = f'hazard{i}'
                geom = {'name': name,
                        'size': [self.hazards_size, 1e-2],#self.hazards_size / 2],
                        'pos': np.r_[0.0, 0.0, 2e-2],#self.hazards_size / 2 + 1e-2],
                        'rot': self.random_rot(),
                        'type': 'cylinder',
                        'contype': 0,
                        'conaffinity': 0,
                        'group': GROUP_HAZARD,
                        'rgba': COLOR_HAZARD * [1, 1, 1, 0.25]} #0.1]}  # transparent
                world_config['geoms'][name] = geom

            return world_config

    def build_observation_space(self):
        ''' Construct observtion space.  Happens only once at during __init__ '''
        obs_space_dict = OrderedDict()  # See self.obs()
        # self.robot.nq = 2
        # self.robot.nv = 2
        # self.robot.nu = 2
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
            # self.observation_space = utils.batch_space(self.observation_space, self.env_num)
        else:
            for k, v in self.obs_space_dict.items():
                self.obs_space_dict[k] = utils.batch_space(v, self.env_num)
            self.observation_space = gym.spaces.Dict(obs_space_dict)



    #----------------------------------------------------------------
    # gym wrapper functions
    #----------------------------------------------------------------

    def update_data(self):
        self._last_last_data = deepcopy(self._last_data)
        self._last_data = deepcopy(self._data)
        self._last_last_done = deepcopy(self._last_done)
        self._last_done = deepcopy(self._done)
        self.key,_ = jax.random.split(self.key, 2)

    def reset_layout(self):
        layout, success = self._sample_layout(self.key)
        
        idx = jp.where(success > 0.)[0]
        self.layout = {}
        for key in layout.keys():
            self.layout[key] = layout[key][idx]
            
        self.layout
        self.layout_size = len(idx)
        print("number of valid layout is: ", self.layout_size)
        assert self.layout_size > self.env_num
        
    def get_layout(self):
        idx = jax.random.randint(self.key, (self.env_num,), minval=0, maxval=self.layout_size)
        layout_valid = {}
        for key in self.layout.keys():
            layout_valid[key] = self.layout[key][idx]
            
        return layout_valid
    
    def reset(self):
        ''' Reset the physics simulation and return observation '''
        
        self.reset_layout()
        layout = self.get_layout()
        # import ipdb;ipdb.set_trace()
        obs, data = self._reset(layout)
        # import ipdb;ipdb.set_trace()
        # update the log
        self._steps = jp.zeros(self.env_num)
        self._obs = obs
        self._data = data
        
        return jax_to_torch(obs)
    
    def step(self, action):
        ''' Take a step and return observation, reward, done, and info '''
        # print(action)
        action = torch_to_jax(action)
        
        
        self.update_data()
        # import ipdb;ipdb.set_trace()
        obs, reward, done, info, data = self._step(self._data, 
                                                    self._last_data, 
                                                    self._last_last_data, 
                                                    self._last_done, 
                                                    self._last_last_done, 
                                                    action)
        # import ipdb;ipdb.set_trace()
        # assert self._data.xpos[1,14,0] == data.xpos[1,14,0]
        # update the current info right after step 
        self._data = data
        self._obs = obs
        self._reward = reward
        self._done = done
        self._info = info
        
        self._done  = jp.where(self._steps > self.num_steps, x = 1.0, y = self._done)
        self._steps = jp.where(self._done > 0.0, x = 0, y = self._steps + 1)
        
        return jax_to_torch(self._obs), jax_to_torch(self._reward), jax_to_torch(self._done), jax_to_torch(self._info)
    
    def reset_done(self):
        # self.reset_layout()
        # import ipdb;ipdb.set_trace()
        layout = self.get_layout()
        obs, self._data = self._reset_done(self._data,
                                            self._done,
                                            self._obs,
                                            layout)
        return jax_to_torch(obs)
    
    def placements_dict_from_object(self, object_name):
        ''' Get the placements dict subset just for a given object name '''
        placements_dict = {}
        if hasattr(self, object_name + 's_num'):  # Objects with multiplicity
            plural_name = object_name + 's'
            object_fmt = object_name + '{i}'
            object_num = getattr(self, plural_name + '_num', None)
            object_locations = getattr(self, plural_name + '_locations', [])
            object_placements = getattr(self, plural_name + '_placements', None)
            object_keepout = getattr(self, plural_name + '_keepout')
        else:  # Unique objects
            object_fmt = object_name
            object_num = 1
            object_locations = getattr(self, object_name + '_locations', [])
            object_placements = getattr(self, object_name + '_placements', None)
            object_keepout = getattr(self, object_name + '_keepout')
        for i in range(object_num):
            if i < len(object_locations):
                x, y = object_locations[i]
                k = object_keepout + 1e-9  # Epsilon to account for numerical issues
                placements = [(x - k, y - k, x + k, y + k)]
            else:
                placements = object_placements
            placements_dict[object_fmt.format(i=i)] = (placements, object_keepout)
        return placements_dict

    def build_placements_dict(self):
        ''' Build a dict of placements.  Happens once during __init__. '''
        # Dictionary is map from object name -> tuple of (placements list, keepout)
        placements = {}

        placements.update(self.placements_dict_from_object('robot'))

        if self.task in ['goal']:
            placements.update(self.placements_dict_from_object('goal'))
        if self.hazards_num: #self.constrain_hazards:
            placements.update(self.placements_dict_from_object('hazard'))

        self.placements = placements

    def sample_layout(self, rng):
        ''' Sample a single layout, returning True if successful, else False. '''
        
        def placement_is_valid(xy, layout):
            flag = 1.
            for other_name, other_xy in layout.items():
                other_keepout = self.placements[other_name][1]
                dist = jp.sqrt(jp.sum(jp.square(xy - other_xy)))
                flag = jp.where(dist < other_keepout + self.placements_margin + keepout, x=0., y=flag)
            return flag

        layout = {}
        success = 1
        for name, (placements, keepout) in self.placements.items():
            conflicted = 1.
            xy = jp.array([-jp.inf, -jp.inf])
            for _ in range(10):
                rng, rng1 = jax.random.split(rng, 2)
                cur_xy = self.draw_placement(placements, keepout, rng1)
                flag = placement_is_valid(cur_xy, layout)
                xy = jp.where(flag > 0., x=cur_xy, y=xy)
                conflicted = jp.where(flag > 0., x=0., y=conflicted)
            layout[name] = xy
            success = jp.where(conflicted > 0., x=0., y=success) 
        return layout, success

    def constrain_placement(self, placement, keepout):
        ''' Helper function to constrain a single placement by the keepout radius '''
        xmin, ymin, xmax, ymax = placement
        return (xmin + keepout, ymin + keepout, xmax - keepout, ymax - keepout)
 
    def draw_placement(self, placements, keepout, rng):
        ''' 
        Sample an (x,y) location, based on potential placement areas.

        Summary of behavior: 

        'placements' is a list of (xmin, xmax, ymin, ymax) tuples that specify 
        rectangles in the XY-plane where an object could be placed. 

        'keepout' describes how much space an object is required to have
        around it, where that keepout space overlaps with the placement rectangle.

        To sample an (x,y) pair, first randomly select which placement rectangle
        to sample from, where the probability of a rectangle is weighted by its
        area. If the rectangles are disjoint, there's an equal chance the (x,y) 
        location will wind up anywhere in the placement space. If they overlap, then
        overlap areas are double-counted and will have higher density. This allows
        the user some flexibility in building placement distributions. Finally, 
        randomly draw a uniform point within the selected rectangle.

        '''
        if placements is None:
            choice = self.constrain_placement(self.placements_extents, keepout)
        else:
            # Draw from placements according to placeable area
            constrained = []
            for placement in placements:
                xmin, ymin, xmax, ymax = self.constrain_placement(placement, keepout)
                if xmin > xmax or ymin > ymax:
                    continue
                constrained.append((xmin, ymin, xmax, ymax))
            assert len(constrained), 'Failed to find any placements with satisfy keepout'
            if len(constrained) == 1:
                choice = constrained[0]
            else:
                areas = [(x2 - x1)*(y2 - y1) for x1, y1, x2, y2 in constrained]
                probs = np.array(areas) / np.sum(areas)
                choice = constrained[self.rs.choice(len(constrained), p=probs)]
        xmin, ymin, xmax, ymax = choice
        rng1, rng2 = jax.random.split(rng, 2)
        x_rand = jax.random.uniform(rng1, minval=xmin, maxval=xmax)
        y_rand = jax.random.uniform(rng2, minval=ymin, maxval=ymax)
        return jp.array([x_rand, y_rand])

    def layout2qpos(self, layout):
        qpos_reset = jp.zeros(self.mjx_model.nq)
        for key in self.placements.keys():
            if key == 'robot' and key in self.joint_name2qpos_id.keys():
                x_idx = self.joint_name2qpos_id['robot']
                y_idx = self.joint_name2qpos_id['robot'] + 1
                qpos_reset = qpos_reset.at[x_idx].set(layout[key][0])
                qpos_reset = qpos_reset.at[y_idx].set(layout[key][1])
                qpos_reset = qpos_reset.at[x_idx + 2].set(self.robot.z_height)
                qpos_reset = qpos_reset.at[x_idx + 3].set(1.0)
            else:
                x_idx = self.joint_name2qpos_id[key + '_x']
                y_idx = self.joint_name2qpos_id[key + '_y']
                qpos_reset = qpos_reset.at[x_idx].set(layout[key][0])
                qpos_reset = qpos_reset.at[y_idx].set(layout[key][1])
            # import ipdb;ipdb.set_trace()
        return qpos_reset
    #----------------------------------------------------------------
    # jax parallel wrapper functions
    #----------------------------------------------------------------

    def mjx_reset(self, layout):
        """ Resets an unbatched environment to an initial state."""
        #! TODO: update qpose with layout
        # layout = self.build_layout(rng)
        # qpos_reset = jax.random.uniform(rng, (self.mjx_model.nq,), minval=-1.5, maxval=1.5)
        # import ipdb;ipdb.set_trace()
        qpos_reset = self.layout2qpos(layout)
        data = self.mjx_data
        data = data.replace(qpos=qpos_reset, qvel=jp.zeros(self.mjx_model.nv), ctrl=jp.zeros(self.mjx_model.nu))
        
        # log the last data
        data = mjx.forward(self.mjx_model, data)
        obs, _ = self.obs(data, None, None, None, None)
        return obs, data
    
    def mjx_step(self, data: mjx.Data, 
                 last_data: mjx.Data, 
                 last_last_data: mjx.Data, 
                 last_done: jp.ndarray,
                 last_last_done: jp.ndarray,
                 action: jp.ndarray):
        """Runs one timestep of the environment's dynamics."""
        def f(data, _):   
            return (
                mjx.step(self.mjx_model, data),
                None,
            )
        
        def convert_action(action):
            mjx_action = action
            if 'point' in self.robot_base:
                mjx_action = jp.zeros(3)
                xmat = data.xmat.reshape(-1,3,3)
                robot_mat = xmat[self.body_name2xpos_id['robot'],:]
                robot_action = jp.array([action[0],0.,0.])
                world_action = jp.matmul(robot_action, robot_mat.transpose())[:2]
                world_action = jp.matmul(robot_mat, robot_action)[:2]
                mjx_action = mjx_action.at[0].set(world_action[0])
                mjx_action = mjx_action.at[1].set(world_action[1])
                mjx_action = mjx_action.at[2].set(action[1])
                # import ipdb;ipdb.set_trace()
            return mjx_action
        
        mjx_action = convert_action(action)
        data = data.replace(ctrl=mjx_action)
        data, _ = jax.lax.scan(f, data, (), self.physics_steps_per_control_step)
        obs, obs_dict = self.obs(data, last_data, last_last_data, last_done, last_last_done)
        reward, done = self.reward_done(data, last_data, last_done)
        cost = self.cost(data)
        info = {}
        info['cost'] = cost
        info['obs'] = obs_dict
        reward = jp.where(jp.isnan(obs).any() > 0, x = 0, y = reward)
        done = jp.where(jp.isnan(obs).any() > 0, x = 1, y = done)
        reward = jp.where(jp.isinf(obs).any() > 0, x = 0, y = reward)
        done = jp.where(jp.isinf(obs).any() > 0, x = 1, y = done)
        return obs, reward, done, info, data
    
    def mjx_reset_done(self,
                        data: mjx.Data, 
                        done: jp.ndarray,
                        obs: jp.ndarray,
                        layout):
        """Runs one timestep of the environment's dynamics."""
        # qpos_reset = jax.random.uniform(rng, (self.mjx_model.nq,), minval=-1.5, maxval=1.5)
        # import ipdb;ipdb.set_trace()
        qpos_reset = self.layout2qpos(layout)
        ctrl_reset = jp.zeros(self.mjx_model.nu)
        qvel_reset = jp.zeros(self.mjx_model.nv)
        if done is not None:
            # fake one step forward to get xpos/observation for new initialized jpos
            qpos = jp.where(done > 0.0, x = qpos_reset, y = data.qpos)
            ctrl = jp.where(done > 0.0, x = ctrl_reset, y = data.ctrl)
            qvel = jp.where(done > 0.0, x = qvel_reset, y = data.qvel)
            data = data.replace(qpos=qpos, qvel=qvel, ctrl=ctrl)
            def f(data, _):   
                return (
                        mjx.step(self.mjx_model, data),
                        None,
                )
            data_reset, _ = jax.lax.scan(f, data, (), self.physics_steps_per_control_step)
            # reset observation treats last done and last last done all true, just use current data 
            obs_reset, _ = self.obs(data_reset, None, None, None, None)

            # reset observation for done environment 
            obs = jp.where(done > 0., x=obs_reset, y=obs)
        
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
            obs['goal_lidar'] = self.obs_lidar(data, data.xpos[self.body_name2xpos_id['goal'],:])
        if self.observe_hazards:
            obs['hazards_lidar'] = self.obs_lidar(data, data.xpos[self.body_name2xpos_id['hazards'],:])
        if self.observe_goal_comp:
            obs['goal_compass'] = self.obs_compass(data, data.xpos[self.body_name2xpos_id['goal'],:])
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
            robot_pos = xpos[self.body_name2xpos_id['robot'],:]
            goal_pos = xpos[self.body_name2xpos_id['goal'],:]
            dist_goal = jp.sqrt(jp.sum(jp.square(goal_pos[:2] - robot_pos[:2]))) 
            return dist_goal

    def reward_done(self, data: mjx.Data, last_data: mjx.Data, last_done:jp.ndarray) -> Tuple[jp.ndarray, jp.ndarray]:        
        # get raw data of robot pos and goal pos
        
        dist_goal = self.goal_pos(data)
        
        # upgrade last dist goal computation for auto reset 
        last_dist_goal = dist_goal
        if last_done is not None:
            last_dist_goal = jp.where(last_done > 0.0, x = self.goal_pos(data), y = self.goal_pos(last_data))
        d_dist = last_dist_goal - dist_goal
        reward = d_dist * self.reward_distance
        done = jp.where(dist_goal < self.goal_size, x = 1.0, y = 0.0)
        # filter the invalid simulation of the robot
        done = jp.where(abs(d_dist) > 1.0, x = 1.0, y = done)
        reward = jp.where(abs(d_dist) > 1.0, x = 0, y = reward)
        return reward, done
    
    def cost(self, data: mjx.Data) -> jp.ndarray:
        # get the raw position data of different objects current frame 
        robot_pos = data.xpos[self.body_name2xpos_id['robot'],:].reshape(-1,3)
        hazards_pos = data.xpos[self.body_name2xpos_id['hazards'],:].reshape(-1,3)
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
        robot_pos = xpos[self.body_name2xpos_id['robot'],:]
        robot_mat = xmat[self.body_name2xpos_id['robot'],:]
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
        robot_pos = xpos[self.body_name2xpos_id['robot'],:]
        robot_mat = xmat[self.body_name2xpos_id['robot'],:]
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
        robot_pos = data.xpos[self.body_name2xpos_id['robot'],:]
        robot_mat = data.xmat[self.body_name2xpos_id['robot'],:,:]

        # auto update past robot poses for auto reset  
        robot_pos_last = robot_pos
        robot_pos_last_last = robot_pos
        if last_done is not None:
            robot_pos_last = jp.where(last_done > 0.0, x = robot_pos, y = last_data.xpos[self.body_name2xpos_id['robot'],:])
            if last_last_done is not None:
                robot_pos_last_last = jp.where(last_last_done + last_done > 0.0, x = robot_pos_last, y = last_last_data.xpos[self.body_name2xpos_id['robot'],:])
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
        robot_pos = xpos[self.body_name2xpos_id['robot'],:]
        robot_mat = xmat[self.body_name2xpos_id['robot'],:]
        lidar = lidar.flatten()
        cnt = 0
        for i, sensor in enumerate(lidar):
            theta = 2 * np.pi * (i + 0.5) / self.lidar_num_bins # i += 0.5  # Offset to center of bin
            rad = self.render_lidar_radius
            
            binpos = np.array([np.cos(theta) * rad, np.sin(theta) * rad, offset])
            pos = robot_pos + np.matmul(binpos, robot_mat.transpose())
            alpha = min(1.0, sensor + .1)
            size=[0.02, 0, 0]
            self.render_sphere(pos, size, color, alpha)
        #     mujoco.mjv_initGeom(
        #         self.viewer.user_scn.geoms[i + self.viewer.user_scn.ngeom],
        #         type=mujoco.mjtGeom.mjGEOM_SPHERE,
        #         size=[0.02, 0, 0],
        #         pos=pos.flatten(),
        #         mat=np.eye(3).flatten(),
        #         rgba=np.array(color) * alpha
        #         )
        #     mujoco.mjv_initGeom(
        #         self.renderer_scene.geoms[i + self.renderer_scene.ngeom],
        #         type=mujoco.mjtGeom.mjGEOM_SPHERE,
        #         size=[0.02, 0, 0],
        #         pos=pos.flatten(),
        #         mat=np.eye(3).flatten(),
        #         rgba=np.array(color) * alpha
        #         )
        #     cnt += 1
        # self.viewer.user_scn.ngeom += cnt
        # self.renderer_scene.ngeom += cnt
    
    def render_compass(self, data: mjx.Data, compass_pos, color, offset):
        xpos = data.xpos.reshape(-1,3)
        xmat = data.xmat.reshape(-1,3,3)
        robot_pos = xpos[self.body_name2xpos_id['robot'],:]
        robot_mat = xmat[self.body_name2xpos_id['robot'],:]
        compass_pos = compass_pos.flatten()
        compass_pos = jp.concatenate([compass_pos * 0.15, jp.array([0])])
        pos = robot_pos + np.matmul(compass_pos, robot_mat.transpose())
        # mujoco.mjv_initGeom(
        #     self.viewer.user_scn.geoms[self.viewer.user_scn.ngeom],
        #     type=mujoco.mjtGeom.mjGEOM_SPHERE,
        #     size=[0.02, 0, 0],
        #     pos=pos.flatten(),
        #     mat=np.eye(3).flatten(),
        #     rgba=np.array(color) * 0.5
        #     )
        # mujoco.mjv_initGeom(
        #     self.renderer_scene.geoms[self.renderer_scene.ngeom],
        #     type=mujoco.mjtGeom.mjGEOM_SPHERE,
        #     size=[0.02, 0, 0],
        #     pos=pos.flatten(),
        #     mat=np.eye(3).flatten(),
        #     rgba=np.array(color) * 0.5
        #     )
        # self.viewer.user_scn.ngeom += 1
        # self.renderer_scene.ngeom += 1
        size=[0.05, 0, 0]
        alpha = 0.5
        self.render_sphere(pos, size, color, alpha)

    def render_sphere(self, pos, size, color, alpha=0.1):
        ''' Render a radial area in the environment '''
        pos = np.asarray(pos)
        if pos.shape == (2,):
            pos = np.r_[pos, 0]  # Z coordinate 0
        mujoco.mjv_initGeom(
            self.viewer.user_scn.geoms[self.viewer.user_scn.ngeom],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=size * np.ones(3),
            pos=pos.flatten(),
            mat=np.eye(3).flatten(),
            rgba=np.array(color) * alpha,
            )
        mujoco.mjv_initGeom(
            self.renderer_scene.geoms[self.renderer_scene.ngeom],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=size * np.ones(3),
            pos=pos.flatten(),
            mat=np.eye(3).flatten(),
            rgba=np.array(color) * alpha,
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
        robot_pos = data.xpos[self.body_name2xpos_id['robot'],:].reshape(-1,3)
        mujoco.mj_step(self.mj_model, data)
        self.renderer.update_scene(data, self.viewer.cam, self.viewer.opt)
        self.renderer_scene = self.renderer._scene
        offset = 0.5
        self.viewer.user_scn.ngeom = 0
        obs = self._info['obs']
        cost = self._info['cost']
        if cost > 0:
            self.render_sphere(robot_pos, 0.5, COLOR_RED, alpha=.5)
        # if self.observe_hazards:
        #     self.render_lidar(self._data, obs['hazards_lidar'], COLOR_HAZARD, offset)
        #     offset += 0.1
        # if self.observe_goal_lidar:
        #     self.render_lidar(self._data, obs['goal_lidar'], COLOR_GOAL, offset)
        #     offset += 0.1
        # if self.observe_goal_comp:
        #     self.render_compass(self._data, obs['goal_compass'], COLOR_GOAL, offset)
        #     offset += 0.1
        
        self.viewer.sync()
        
        return self.renderer.render()
        