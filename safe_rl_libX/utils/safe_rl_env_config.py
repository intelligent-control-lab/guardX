import isaacgym
import isaacgymenvs
import gym
import numpy as np
import torch
from safe_rl_envs.envs.engine import Engine as  safe_rl_envs_Engine

def configuration_list(task):
    """
    Configuration for customized environment for safety gym 
    
    ================== Special Intructions for SCPO Related Experiment ==================
    Use configuration for non-continuous environment, meaning episodic tasks.
    Task options should be followed with 'noconti' 
    Refer to the set of tested suites in State-wise Constrained Policy Optimization paper
    Note:   task named "TASK_noconti" should be used for training, the corresponding task named "TASK" 
            should be used for video visualization. For example: 'Goal_Point_8Hazards_noconti' for training
            and 'Goal_Point_8Hazards' for video visulization.
    """
    ################ Goal Tasks #################  
    
    if task == "Goal_Point_8Hazards_nosensor":
        config = {
            # robot setting
            'robot_base': 'xmls/point.xml',  

            # task setting
            'task': 'goal',
            'goal_size': 0.5,

            # observation setting
            'observe_goal_comp': True,  # Observe the goal with a lidar sensor
            'observe_hazards': True,  # Observe the vector from agent to hazards
            
            # constraint setting
            'constrain_hazards': True,  # Constrain robot from being in hazardous areas
            'constrain_indicator': False,  # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

            # lidar setting
            'lidar_num_bins': 16,
            
            # object setting
            'hazards_num': 8,
            'hazards_size': 0.3,
            
            # sensor 
            'sensors_obs': [],
            'sensors_hinge_joints': False,
            'sensors_ball_joints': False,
            'sensors_angle_components': False,
            'observation_flatten': True,
            'observe_sensors': False,
            'observe_qpos': True,
            'observe_qvel': True,
            'observe_ctrl': True
        }


    
          
    if task == "Goal_Point_8Hazards":
        config = {
            # robot setting
            'robot_base': 'xmls/point.xml',  

            # task setting
            'task': 'goal',
            'goal_size': 0.5,

            # observation setting
            'observe_goal_comp': True,  # Observe the goal with a lidar sensor
            'observe_hazards': True,  # Observe the vector from agent to hazards
            
            # constraint setting
            'constrain_hazards': True,  # Constrain robot from being in hazardous areas
            'constrain_indicator': False,  # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

            # lidar setting
            'lidar_num_bins': 16,
            
            # object setting
            'hazards_num': 8,
            'hazards_size': 0.3,
        }

    if task == "Goal_Point_8Ghosts":
        config = {
            # robot setting
            'robot_base': 'xmls/point.xml',  

            # task setting
            'task': 'goal',
            'goal_size': 0.5,

            # observation setting
            'observe_goal_comp': True,  # Observe the goal with a lidar sensor
            'observe_ghosts': True,  # Observe the vector from agent to hazards
            
            # constraint setting
            'constrain_ghosts': True,  # Constrain robot from being in hazardous areas
            'constrain_indicator': False,  # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

            # lidar setting
            'lidar_num_bins': 16,
            
            # object setting
            'ghosts_num': 8,
            'ghosts_size': 0.3,
            'ghosts_travel':2.5,
            'ghosts_safe_dist': 1.5,
        }

    if task == "Goal_Swimmer_8Hazards":
        config = {
            # robot setting
            'robot_base': 'xmls/swimmer.xml',  

            # task setting
            'task': 'goal',
            'goal_size': 0.5,

            # observation setting
            'observe_goal_comp': True,  # Observe the goal with a lidar sensor
            'observe_hazards': True,  # Observe the vector from agent to hazards
            'sensors_obs': ['accelerometer', 'velocimeter', 'gyro', 'magnetometer',
                            'touch_point1', 'touch_point2', 'touch_point3', 'touch_point4'],

            # constraint setting
            'constrain_hazards': True,  # Constrain robot from being in hazardous areas
            'constrain_indicator': False,  # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

            # lidar setting
            'lidar_num_bins': 16,
            
            # object setting
            'hazards_num': 8,
            'hazards_size': 0.3,
        }

    if task == "Goal_Swimmer_8Ghosts":
        config = {
            # robot setting
            'robot_base': 'xmls/swimmer.xml',  

            # task setting
            'task': 'goal',
            'goal_size': 0.5,

            # observation setting
            'observe_goal_comp': True,  # Observe the goal with a lidar sensor
            'observe_ghosts': True,  # Observe the vector from agent to hazards
            'sensors_obs': ['accelerometer', 'velocimeter', 'gyro', 'magnetometer',
                            'touch_point1', 'touch_point2', 'touch_point3', 'touch_point4'],
            
            # constraint setting
            'constrain_ghosts': True,  # Constrain robot from being in hazardous areas
            'constrain_indicator': False,  # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

            # lidar setting
            'lidar_num_bins': 16,
            
            # object setting
            'ghosts_num': 8,
            'ghosts_size': 0.3,
            'ghosts_travel':2.5,
            'ghosts_safe_dist': 1.5,
        }

    if task == "Goal_Doggo_8Hazards":
        config = {
            # robot setting
            'robot_base': 'xmls/doggo.xml',  

            # task setting
            'task': 'goal',
            'goal_size': 0.5,

            # observation setting
            'observe_goal_comp': True,  # Observe the goal with a lidar sensor
            'observe_hazards': True,  # Observe the vector from agent to hazards
            'sensors_obs': ['accelerometer', 'velocimeter', 'gyro', 'magnetometer',
                            'touch_ankle_1a', 'touch_ankle_2a', 'touch_ankle_3a', 'touch_ankle_4a',
                            'touch_ankle_1b', 'touch_ankle_2b', 'touch_ankle_3b', 'touch_ankle_4b'],

            # constraint setting
            'constrain_hazards': True,  # Constrain robot from being in hazardous areas
            'constrain_indicator': False,  # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

            # lidar setting
            'lidar_num_bins': 16,
            
            # object setting
            'hazards_num': 8,
            'hazards_size': 0.3,
        }

    if task == "Goal_Ant_8Hazards":
        config = {
            # robot setting
            'robot_base': 'xmls/ant.xml',  

            # task setting
            'task': 'goal',
            'goal_size': 0.5,

            # observation setting
            'observe_goal_comp': True,  # Observe the goal with a lidar sensor
            'observe_hazards': True,  # Observe the vector from agent to hazards
            'sensors_obs': ['accelerometer', 'velocimeter', 'gyro', 'magnetometer',
                            'touch_ankle_1a', 'touch_ankle_2a', 'touch_ankle_3a', 'touch_ankle_4a',
                            'touch_ankle_1b', 'touch_ankle_2b', 'touch_ankle_3b', 'touch_ankle_4b'],

            # constraint setting
            'constrain_hazards': True,  # Constrain robot from being in hazardous areas
            'constrain_indicator': False,  # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

            # lidar setting
            'lidar_num_bins': 16,
            
            # object setting
            'hazards_num': 8,
            'hazards_size': 0.3,
        }

    if task == "Goal_Ant_8Ghosts":
        config = {
            # robot setting
            'robot_base': 'xmls/ant.xml',  

            # task setting
            'task': 'goal',
            'goal_size': 0.5,

            # observation setting
            'observe_goal_comp': True,  # Observe the goal with a lidar sensor
            'observe_ghosts': True,  # Observe the vector from agent to hazards
            'sensors_obs': ['accelerometer', 'velocimeter', 'gyro', 'magnetometer',
                            'touch_ankle_1a', 'touch_ankle_2a', 'touch_ankle_3a', 'touch_ankle_4a',
                            'touch_ankle_1b', 'touch_ankle_2b', 'touch_ankle_3b', 'touch_ankle_4b'],

            
            # constraint setting
            'constrain_ghosts': True,  # Constrain robot from being in hazardous areas
            'constrain_indicator': False,  # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

            # lidar setting
            'lidar_num_bins': 16,
            
            # object setting
            'ghosts_num': 8,
            'ghosts_size': 0.3,
            'ghosts_travel':2.5,
            'ghosts_safe_dist': 1.5,
        }
    
    if task == "Goal_Walker_8Hazards":
        config = {
            # robot setting
            'robot_base': 'xmls/walker.xml',  

            # task setting
            'task': 'goal',
            'goal_size': 0.5,

            # observation setting
            'observe_goal_comp': True,  # Observe the goal with a lidar sensor
            'observe_hazards': True,  # Observe the vector from agent to hazards
            'sensors_obs': ['accelerometer', 'velocimeter', 'gyro', 'magnetometer',
                            'touch_right_foot', 'touch_left_foot'],
            
            # constraint setting
            'constrain_hazards': True,  # Constrain robot from being in hazardous areas
            'constrain_indicator': False,  # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

            # lidar setting
            'lidar_num_bins': 16,
            
            # object setting
            'hazards_num': 8,
            'hazards_size': 0.3,
        }

    if task == "Goal_Walker_8Ghosts":
        config = {
            # robot setting
            'robot_base': 'xmls/walker.xml',  

            # task setting
            'task': 'goal',
            'goal_size': 0.5,

            # observation setting
            'observe_goal_comp': True,  # Observe the goal with a lidar sensor
            'observe_ghosts': True,  # Observe the vector from agent to hazards
            'sensors_obs': ['accelerometer', 'velocimeter', 'gyro', 'magnetometer',
                            'touch_right_foot', 'touch_left_foot'],
            
            # constraint setting
            'constrain_ghosts': True,  # Constrain robot from being in hazardous areas
            'constrain_indicator': False,  # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

            # lidar setting
            'lidar_num_bins': 16,
            
            # object setting
            'ghosts_num': 8,
            'ghosts_size': 0.3,
            'ghosts_travel':2.5,
            'ghosts_safe_dist': 1.5,
        }

    if task == "Goal_Humanoid_8Hazards":
        config = {
            # robot setting
            'robot_base': 'xmls/humanoid.xml',  

            # task setting
            'task': 'goal',
            'goal_size': 0.5,

            # observation setting
            'observe_goal_comp': True,  # Observe the goal with a lidar sensor
            'observe_hazards': True,  # Observe the vector from agent to hazards
            'sensors_obs': ['accelerometer', 'velocimeter', 'gyro', 'magnetometer',
                            'touch_right_foot', 'touch_left_foot'],
            
            # constraint setting
            'constrain_hazards': True,  # Constrain robot from being in hazardous areas
            'constrain_indicator': False,  # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

            # lidar setting
            'lidar_num_bins': 16,
            
            # object setting
            'hazards_num': 8,
            'hazards_size': 0.3,
        }

    if task == "Goal_Humanoid_8Ghosts":
        config = {
            # robot setting
            'robot_base': 'xmls/humanoid.xml',  

            # task setting
            'task': 'goal',
            'goal_size': 0.5,

            # observation setting
            'observe_goal_comp': True,  # Observe the goal with a lidar sensor
            'observe_ghosts': True,  # Observe the vector from agent to hazards
            'sensors_obs': ['accelerometer', 'velocimeter', 'gyro', 'magnetometer',
                            'touch_right_foot', 'touch_left_foot'],
            
            # constraint setting
            'constrain_ghosts': True,  # Constrain robot from being in hazardous areas
            'constrain_indicator': False,  # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

            # lidar setting
            'lidar_num_bins': 16,
            
            # object setting
            'ghosts_num': 8,
            'ghosts_size': 0.3,
            'ghosts_travel':2.5,
            'ghosts_safe_dist': 1.5,
        }

    if task == "Goal_Hopper_8Hazards":
        config = {
            # robot setting
            'robot_base': 'xmls/hopper.xml',  

            # task setting
            'task': 'goal',
            'goal_size': 0.5,

            # observation setting
            'observe_goal_comp': True,  # Observe the goal with a lidar sensor
            'observe_hazards': True,  # Observe the vector from agent to hazards
            'sensors_obs': ['accelerometer', 'velocimeter', 'gyro', 'magnetometer',
                            'touch_foot'],
            
            # constraint setting
            'constrain_hazards': True,  # Constrain robot from being in hazardous areas
            'constrain_indicator': False,  # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

            # lidar setting
            'lidar_num_bins': 16,
            
            # object setting
            'hazards_num': 8,
            'hazards_size': 0.3,
        }

    if task == "Goal_Hopper_8Ghosts":
        config = {
            # robot setting
            'robot_base': 'xmls/hopper.xml',  

            # task setting
            'task': 'goal',
            'goal_size': 0.5,

            # observation setting
            'observe_goal_comp': True,  # Observe the goal with a lidar sensor
            'observe_ghosts': True,  # Observe the vector from agent to hazards
            'sensors_obs': ['accelerometer', 'velocimeter', 'gyro', 'magnetometer',
                            'touch_foot'],
            
            # constraint setting
            'constrain_ghosts': True,  # Constrain robot from being in hazardous areas
            'constrain_indicator': False,  # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

            # lidar setting
            'lidar_num_bins': 16,
            
            # object setting
            'ghosts_num': 8,
            'ghosts_size': 0.3,
            'ghosts_travel':2.5,
            'ghosts_safe_dist': 1.5,
        }

    if task == "Goal_Arm3_8Hazards":
        config = {
            # robot setting
            'robot_base': 'xmls/arm_3.xml', 
            'robot_locations':[(0.0,0.0)],

            # task setting
            'task': 'goal',
            'goal_3D': True,
            'goal_z_range': [0.5,1.5],
            'goal_size': 0.5,

            # observation setting
            'observe_goal_comp': True,  # Observe the goal with a lidar sensor
            'observe_hazard3Ds': True,  # Observe the vector from agent to hazards
            'compass_shape': 3,
            'sensors_obs': ['accelerometer_link_1', 'velocimeter_link_1', 'gyro_link_1', 'magnetometer_link_1',
                            'accelerometer_link_2', 'velocimeter_link_2', 'gyro_link_2', 'magnetometer_link_2',
                            'accelerometer_link_3', 'velocimeter_link_3', 'gyro_link_3', 'magnetometer_link_3',
                            'accelerometer_link_4', 'velocimeter_link_4', 'gyro_link_4', 'magnetometer_link_4',
                            'accelerometer_link_5', 'velocimeter_link_5', 'gyro_link_5', 'magnetometer_link_5',
                            'touch_end_effector'],
            
            # constraint setting
            'constrain_hazard3Ds': True,  # Constrain robot from being in hazardous areas
            'constrain_indicator': False,  # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

            # lidar setting
            'lidar_num_bins': 10,
            'lidar_num_bins3D': 6,
            'lidar_body': ['link_1', 'link_3', 'link_5'],
            
            # object setting
            'hazard3Ds_num': 8,
            'hazard3Ds_size': 0.3,
            'hazard3Ds_z_range': [0.5, 1.5],
        }

    if task == "Goal_Arm3_8Ghosts":
        config = {
            # robot setting
            'robot_base': 'xmls/arm_3.xml',  
            'robot_locations':[(0.0,0.0)],

            # task setting
            'task': 'goal',
            'goal_3D': True,
            'goal_z_range': [0.5,1.5],
            'goal_size': 0.5,

            # observation setting
            'observe_goal_comp': True,  # Observe the goal with a lidar sensor
            'observe_ghost3Ds': True,  # Observe the vector from agent to hazards
            'compass_shape': 3,
            'sensors_obs': ['accelerometer_link_1', 'velocimeter_link_1', 'gyro_link_1', 'magnetometer_link_1',
                            'accelerometer_link_2', 'velocimeter_link_2', 'gyro_link_2', 'magnetometer_link_2',
                            'accelerometer_link_3', 'velocimeter_link_3', 'gyro_link_3', 'magnetometer_link_3',
                            'accelerometer_link_4', 'velocimeter_link_4', 'gyro_link_4', 'magnetometer_link_4',
                            'accelerometer_link_5', 'velocimeter_link_5', 'gyro_link_5', 'magnetometer_link_5',
                            'touch_end_effector'],
            
            # constraint setting
            'constrain_ghost3Ds': True,  # Constrain robot from being in hazardous areas
            'constrain_indicator': False,  # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

            # lidar setting
            'lidar_num_bins': 10,
            'lidar_num_bins3D': 6,
            'lidar_body': ['link_1', 'link_3', 'link_5'],
            
            # object setting
            'ghost3Ds_num': 8,
            'ghost3Ds_size': 0.3,
            'ghost3Ds_travel':2.5,
            'ghost3Ds_safe_dist': 1.5,
            'ghost3Ds_z_range': [0.5, 1.5],
        }

    if task == "Goal_Arm6_8Hazards":
        config = {
            # robot setting
            'robot_base': 'xmls/arm_6.xml', 
            'robot_locations':[(0.0,0.0)],

            # task setting
            'task': 'goal',
            'goal_3D': True,
            'goal_z_range': [0.5,1.5],
            'goal_size': 0.5,

            # observation setting
            'observe_goal_comp': True,  # Observe the goal with a lidar sensor
            'observe_hazard3Ds': True,  # Observe the vector from agent to hazards
            'compass_shape': 3,
            'sensors_obs': ['accelerometer_link_1', 'velocimeter_link_1', 'gyro_link_1', 'magnetometer_link_1',
                            'accelerometer_link_2', 'velocimeter_link_2', 'gyro_link_2', 'magnetometer_link_2',
                            'accelerometer_link_3', 'velocimeter_link_3', 'gyro_link_3', 'magnetometer_link_3',
                            'accelerometer_link_4', 'velocimeter_link_4', 'gyro_link_4', 'magnetometer_link_4',
                            'accelerometer_link_5', 'velocimeter_link_5', 'gyro_link_5', 'magnetometer_link_5',
                            'accelerometer_link_6', 'velocimeter_link_6', 'gyro_link_6', 'magnetometer_link_6',
                            'accelerometer_link_7', 'velocimeter_link_7', 'gyro_link_7', 'magnetometer_link_7',
                            'touch_end_effector'],
            
            # constraint setting
            'constrain_hazard3Ds': True,  # Constrain robot from being in hazardous areas
            'constrain_indicator': False,  # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

            # lidar setting
            'lidar_num_bins': 10,
            'lidar_num_bins3D': 6,
            'lidar_body': ['link_1', 'link_3', 'link_5', 'link_7'],
            
            # object setting
            'hazard3Ds_num': 8,
            'hazard3Ds_size': 0.3,
            'hazard3Ds_z_range': [0.5, 1.5],
        }

    if task == "Goal_Arm6_8Ghosts":
        config = {
            # robot setting
            'robot_base': 'xmls/arm_6.xml',  
            'robot_locations':[(0.0,0.0)],

            # task setting
            'task': 'goal',
            'goal_3D': True,
            'goal_z_range': [0.5,1.5],
            'goal_size': 0.5,

            # observation setting
            'observe_goal_comp': True,  # Observe the goal with a lidar sensor
            'observe_ghost3Ds': True,  # Observe the vector from agent to hazards
            'compass_shape': 3,
            'sensors_obs': ['accelerometer_link_1', 'velocimeter_link_1', 'gyro_link_1', 'magnetometer_link_1',
                            'accelerometer_link_2', 'velocimeter_link_2', 'gyro_link_2', 'magnetometer_link_2',
                            'accelerometer_link_3', 'velocimeter_link_3', 'gyro_link_3', 'magnetometer_link_3',
                            'accelerometer_link_4', 'velocimeter_link_4', 'gyro_link_4', 'magnetometer_link_4',
                            'accelerometer_link_5', 'velocimeter_link_5', 'gyro_link_5', 'magnetometer_link_5',
                            'accelerometer_link_6', 'velocimeter_link_6', 'gyro_link_6', 'magnetometer_link_6',
                            'accelerometer_link_7', 'velocimeter_link_7', 'gyro_link_7', 'magnetometer_link_7',
                            'touch_end_effector'],
            
            # constraint setting
            'constrain_ghost3Ds': True,  # Constrain robot from being in hazardous areas
            'constrain_indicator': False,  # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

            # lidar setting
            'lidar_num_bins': 10,
            'lidar_num_bins3D': 6,
            'lidar_body': ['link_1', 'link_3', 'link_5', 'link_7'],
            
            # object setting
            'ghost3Ds_num': 8,
            'ghost3Ds_size': 0.3,
            'ghost3Ds_travel':2.5,
            'ghost3Ds_safe_dist': 1.5,
            'ghost3Ds_z_range': [0.5, 1.5],
        }
    
    if task == "Goal_Drone_8Hazards":
        config = {
            # robot setting
            'robot_base': 'xmls/drone.xml', 

            # task setting
            'task': 'goal',
            'goal_3D': True,
            'goal_z_range': [0.5,1.5],
            'goal_size': 0.5,

            # observation setting
            'observe_goal_comp': True,  # Observe the goal with a lidar sensor
            'observe_hazard3Ds': True,  # Observe the vector from agent to hazards
            'compass_shape': 3,
            'sensors_obs': ['accelerometer', 'velocimeter', 'gyro', 'magnetometer',
                            'touch_p1a', 'touch_p1b', 'touch_p2a', 'touch_p2b',
                            'touch_p3a', 'touch_p3b', 'touch_p4a', 'touch_p4b'],
            
            # constraint setting
            'constrain_hazard3Ds': True,  # Constrain robot from being in hazardous areas
            'constrain_indicator': False,  # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

            # lidar setting
            'lidar_num_bins': 10,
            'lidar_num_bins3D': 6,
            
            # object setting
            'hazard3Ds_num': 8,
            'hazard3Ds_size': 0.3,
            'hazard3Ds_z_range': [0.5, 1.5],
        }

    if task == "Goal_Drone_8Ghosts":
        config = {
            # robot setting
            'robot_base': 'xmls/drone.xml',  

            # task setting
            'task': 'goal',
            'goal_3D': True,
            'goal_z_range': [0.5,1.5],
            'goal_size': 0.5,

            # observation setting
            'observe_goal_comp': True,  # Observe the goal with a lidar sensor
            'observe_ghost3Ds': True,  # Observe the vector from agent to hazards
            'compass_shape': 3,
            'sensors_obs': ['accelerometer', 'velocimeter', 'gyro', 'magnetometer',
                            'touch_p1a', 'touch_p1b', 'touch_p2a', 'touch_p2b',
                            'touch_p3a', 'touch_p3b', 'touch_p4a', 'touch_p4b'],
            
            # constraint setting
            'constrain_ghost3Ds': True,  # Constrain robot from being in hazardous areas
            'constrain_indicator': False,  # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

            # lidar setting
            'lidar_num_bins': 10,
            'lidar_num_bins3D': 6,
            
            # object setting
            'ghost3Ds_num': 8,
            'ghost3Ds_size': 0.3,
            'ghost3Ds_travel':2.5,
            'ghost3Ds_safe_dist': 1.5,
            'ghost3Ds_z_range': [0.5, 1.5],
        }

    ################ Push Tasks #################

    if task == "Push_Point_8Hazards":
        config = {
            # robot setting
            'robot_base': 'xmls/point.xml',  

            # task setting
            'task': 'push',
            'push_object': 'ball',
            'goal_size': 0.5,

            # observation setting
            'observe_goal_comp': True,  # Observe the goal with a lidar sensor
            'observe_box_comp': True,   # Observe the box with a lidar sensor
            'observe_hazards': True,  # Observe the vector from agent to hazards
            
            # constraint setting
            'constrain_hazards': True,  # Constrain robot from being in hazardous areas
            'constrain_indicator': False,  # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

            # lidar setting
            'lidar_num_bins': 16,
            
            # object setting
            'hazards_num': 8,
            'hazards_size': 0.3,
        }

    if task == "Push_Point_8Ghosts":
        config = {
            # robot setting
            'robot_base': 'xmls/point.xml',  

            # task setting
            'task': 'push',
            'push_object': 'ball',
            'goal_size': 0.5,

            # observation setting
            'observe_goal_comp': True,  # Observe the goal with a lidar sensor
            'observe_box_comp': True,   # Observe the box with a lidar sensor
            'observe_ghosts': True,  # Observe the vector from agent to hazards
            
            # constraint setting
            'constrain_ghosts': True,  # Constrain robot from being in hazardous areas
            'constrain_indicator': False,  # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

            # lidar setting
            'lidar_num_bins': 16,
            
            # object setting
            'ghosts_num': 8,
            'ghosts_size': 0.3,
            'ghosts_travel':2.5,
            'ghosts_safe_dist': 1.5,
        }

    if task == "Push_Swimmer_8Hazards":
        config = {
            # robot setting
            'robot_base': 'xmls/swimmer.xml',  

            # task setting
            'task': 'push',
            'push_object': 'ball',
            'goal_size': 0.5,

            # observation setting
            'observe_goal_comp': True,  # Observe the goal with a lidar sensor
            'observe_box_comp': True,   # Observe the box with a lidar sensor
            'observe_hazards': True,  # Observe the vector from agent to hazards
            'sensors_obs': ['accelerometer', 'velocimeter', 'gyro', 'magnetometer',
                            'touch_point1', 'touch_point2', 'touch_point3', 'touch_point4'],

            # constraint setting
            'constrain_hazards': True,  # Constrain robot from being in hazardous areas
            'constrain_indicator': False,  # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

            # lidar setting
            'lidar_num_bins': 16,
            
            # object setting
            'hazards_num': 8,
            'hazards_size': 0.3,
        }

    if task == "Push_Swimmer_8Ghosts":
        config = {
            # robot setting
            'robot_base': 'xmls/swimmer.xml',  

            # task setting
            'task': 'push',
            'push_object': 'ball',
            'goal_size': 0.5,

            # observation setting
            'observe_goal_comp': True,  # Observe the goal with a lidar sensor
            'observe_box_comp': True,   # Observe the box with a lidar sensor
            'observe_ghosts': True,  # Observe the vector from agent to hazards
            'sensors_obs': ['accelerometer', 'velocimeter', 'gyro', 'magnetometer',
                            'touch_point1', 'touch_point2', 'touch_point3', 'touch_point4'],
            
            # constraint setting
            'constrain_ghosts': True,  # Constrain robot from being in hazardous areas
            'constrain_indicator': False,  # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

            # lidar setting
            'lidar_num_bins': 16,
            
            # object setting
            'ghosts_num': 8,
            'ghosts_size': 0.3,
            'ghosts_travel':2.5,
            'ghosts_safe_dist': 1.5,
        }

    if task == "Push_Ant_8Hazards":
        config = {
            # robot setting
            'robot_base': 'xmls/ant.xml',  

            # task setting
            'task': 'push',
            'push_object': 'ball',
            'goal_size': 0.5,

            # observation setting
            'observe_goal_comp': True,  # Observe the goal with a lidar sensor
            'observe_box_comp': True,   # Observe the box with a lidar sensor
            'observe_hazards': True,  # Observe the vector from agent to hazards
            'sensors_obs': ['accelerometer', 'velocimeter', 'gyro', 'magnetometer',
                            'touch_ankle_1a', 'touch_ankle_2a', 'touch_ankle_3a', 'touch_ankle_4a',
                            'touch_ankle_1b', 'touch_ankle_2b', 'touch_ankle_3b', 'touch_ankle_4b'],

            # constraint setting
            'constrain_hazards': True,  # Constrain robot from being in hazardous areas
            'constrain_indicator': False,  # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

            # lidar setting
            'lidar_num_bins': 16,
            
            # object setting
            'hazards_num': 8,
            'hazards_size': 0.3,
        }

    if task == "Push_Ant_8Ghosts":
        config = {
            # robot setting
            'robot_base': 'xmls/ant.xml',  

            # task setting
            'task': 'push',
            'push_object': 'ball',
            'goal_size': 0.5,

            # observation setting
            'observe_goal_comp': True,  # Observe the goal with a lidar sensor
            'observe_box_comp': True,   # Observe the box with a lidar sensor
            'observe_ghosts': True,  # Observe the vector from agent to hazards
            'sensors_obs': ['accelerometer', 'velocimeter', 'gyro', 'magnetometer',
                            'touch_ankle_1a', 'touch_ankle_2a', 'touch_ankle_3a', 'touch_ankle_4a',
                            'touch_ankle_1b', 'touch_ankle_2b', 'touch_ankle_3b', 'touch_ankle_4b'],

            
            # constraint setting
            'constrain_ghosts': True,  # Constrain robot from being in hazardous areas
            'constrain_indicator': False,  # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

            # lidar setting
            'lidar_num_bins': 16,
            
            # object setting
            'ghosts_num': 8,
            'ghosts_size': 0.3,
            'ghosts_travel':2.5,
            'ghosts_safe_dist': 1.5,
        }
    
    if task == "Push_Walker_8Hazards":
        config = {
            # robot setting
            'robot_base': 'xmls/walker.xml',  

            # task setting
            'task': 'push',
            'push_object': 'ball',
            'goal_size': 0.5,

            # observation setting
            'observe_goal_comp': True,  # Observe the goal with a lidar sensor
            'observe_box_comp': True,   # Observe the box with a lidar sensor
            'observe_hazards': True,  # Observe the vector from agent to hazards
            'sensors_obs': ['accelerometer', 'velocimeter', 'gyro', 'magnetometer',
                            'touch_right_foot', 'touch_left_foot'],
            
            # constraint setting
            'constrain_hazards': True,  # Constrain robot from being in hazardous areas
            'constrain_indicator': False,  # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

            # lidar setting
            'lidar_num_bins': 16,
            
            # object setting
            'hazards_num': 8,
            'hazards_size': 0.3,
        }

    if task == "Push_Walker_8Ghosts":
        config = {
            # robot setting
            'robot_base': 'xmls/walker.xml',  

            # task setting
            'task': 'push',
            'push_object': 'ball',
            'goal_size': 0.5,

            # observation setting
            'observe_goal_comp': True,  # Observe the goal with a lidar sensor
            'observe_box_comp': True,   # Observe the box with a lidar sensor
            'observe_ghosts': True,  # Observe the vector from agent to hazards
            'sensors_obs': ['accelerometer', 'velocimeter', 'gyro', 'magnetometer',
                            'touch_right_foot', 'touch_left_foot'],
            
            # constraint setting
            'constrain_ghosts': True,  # Constrain robot from being in hazardous areas
            'constrain_indicator': False,  # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

            # lidar setting
            'lidar_num_bins': 16,
            
            # object setting
            'ghosts_num': 8,
            'ghosts_size': 0.3,
            'ghosts_travel':2.5,
            'ghosts_safe_dist': 1.5,
        }

    if task == "Push_Humanoid_8Hazards":
        config = {
            # robot setting
            'robot_base': 'xmls/humanoid.xml',  

            # task setting
            'task': 'push',
            'push_object': 'ball',
            'goal_size': 0.5,

            # observation setting
            'observe_goal_comp': True,  # Observe the goal with a lidar sensor
            'observe_box_comp': True,   # Observe the box with a lidar sensor
            'observe_hazards': True,  # Observe the vector from agent to hazards
            'sensors_obs': ['accelerometer', 'velocimeter', 'gyro', 'magnetometer',
                            'touch_right_foot', 'touch_left_foot'],
            
            # constraint setting
            'constrain_hazards': True,  # Constrain robot from being in hazardous areas
            'constrain_indicator': False,  # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

            # lidar setting
            'lidar_num_bins': 16,
            
            # object setting
            'hazards_num': 8,
            'hazards_size': 0.3,
        }

    if task == "Push_Humanoid_8Ghosts":
        config = {
            # robot setting
            'robot_base': 'xmls/humanoid.xml',  

            # task setting
            'task': 'push',
            'push_object': 'ball',
            'goal_size': 0.5,

            # observation setting
            'observe_goal_comp': True,  # Observe the goal with a lidar sensor
            'observe_box_comp': True,   # Observe the box with a lidar sensor
            'observe_ghosts': True,  # Observe the vector from agent to hazards
            'sensors_obs': ['accelerometer', 'velocimeter', 'gyro', 'magnetometer',
                            'touch_right_foot', 'touch_left_foot'],
            
            # constraint setting
            'constrain_ghosts': True,  # Constrain robot from being in hazardous areas
            'constrain_indicator': False,  # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

            # lidar setting
            'lidar_num_bins': 16,
            
            # object setting
            'ghosts_num': 8,
            'ghosts_size': 0.3,
            'ghosts_travel':2.5,
            'ghosts_safe_dist': 1.5,
        }

    if task == "Push_Hopper_8Hazards":
        config = {
            # robot setting
            'robot_base': 'xmls/hopper.xml',  

            # task setting
            'task': 'push',
            'push_object': 'ball',
            'goal_size': 0.5,

            # observation setting
            'observe_goal_comp': True,  # Observe the goal with a lidar sensor
            'observe_box_comp': True,   # Observe the box with a lidar sensor
            'observe_hazards': True,  # Observe the vector from agent to hazards
            'sensors_obs': ['accelerometer', 'velocimeter', 'gyro', 'magnetometer',
                            'touch_foot'],
            
            # constraint setting
            'constrain_hazards': True,  # Constrain robot from being in hazardous areas
            'constrain_indicator': False,  # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

            # lidar setting
            'lidar_num_bins': 16,
            
            # object setting
            'hazards_num': 8,
            'hazards_size': 0.3,
        }

    if task == "Push_Hopper_8Ghosts":
        config = {
            # robot setting
            'robot_base': 'xmls/hopper.xml',  

            # task setting
            'task': 'push',
            'push_object': 'ball',
            'goal_size': 0.5,

            # observation setting
            'observe_goal_comp': True,  # Observe the goal with a lidar sensor
            'observe_box_comp': True,   # Observe the box with a lidar sensor
            'observe_ghosts': True,  # Observe the vector from agent to hazards
            'sensors_obs': ['accelerometer', 'velocimeter', 'gyro', 'magnetometer',
                            'touch_foot'],
            
            # constraint setting
            'constrain_ghosts': True,  # Constrain robot from being in hazardous areas
            'constrain_indicator': False,  # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

            # lidar setting
            'lidar_num_bins': 16,
            
            # object setting
            'ghosts_num': 8,
            'ghosts_size': 0.3,
            'ghosts_travel':2.5,
            'ghosts_safe_dist': 1.5,
        }

    if task == "Push_Arm3_8Hazards":
        config = {
            # robot setting
            'robot_base': 'xmls/arm_3.xml', 
            'robot_locations':[(0.0,0.0)],

            # task setting
            'task': 'push',
            'push_object': 'ball',
            'goal_z_range': [0.5,1.5],
            'goal_size': 0.5,

            # observation setting
            'observe_goal_comp': True,  # Observe the goal with a lidar sensor
            'observe_box_comp': True,   # Observe the box with a lidar sensor
            'observe_hazard3Ds': True,  # Observe the vector from agent to hazards
            'compass_shape': 3,
            'sensors_obs': ['accelerometer_link_1', 'velocimeter_link_1', 'gyro_link_1', 'magnetometer_link_1',
                            'accelerometer_link_2', 'velocimeter_link_2', 'gyro_link_2', 'magnetometer_link_2',
                            'accelerometer_link_3', 'velocimeter_link_3', 'gyro_link_3', 'magnetometer_link_3',
                            'accelerometer_link_4', 'velocimeter_link_4', 'gyro_link_4', 'magnetometer_link_4',
                            'accelerometer_link_5', 'velocimeter_link_5', 'gyro_link_5', 'magnetometer_link_5',
                            'touch_end_effector'],
            
            # constraint setting
            'constrain_hazard3Ds': True,  # Constrain robot from being in hazardous areas
            'constrain_indicator': False,  # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

            # lidar setting
            'lidar_num_bins': 10,
            'lidar_num_bins3D': 6,
            'lidar_body': ['link_1', 'link_3', 'link_5'],
            
            # object setting
            'hazard3Ds_num': 8,
            'hazard3Ds_size': 0.3,
            'hazard3Ds_z_range': [0.5, 1.5],
        }

    if task == "Push_Arm3_8Ghosts":
        config = {
            # robot setting
            'robot_base': 'xmls/arm_3.xml',  
            'robot_locations':[(0.0,0.0)],

            # task setting
            'task': 'push',
            'push_object': 'ball',
            'goal_z_range': [0.5,1.5],
            'goal_size': 0.5,

            # observation setting
            'observe_goal_comp': True,  # Observe the goal with a lidar sensor
            'observe_box_comp': True,   # Observe the box with a lidar sensor
            'observe_ghost3Ds': True,  # Observe the vector from agent to hazards
            'compass_shape': 3,
            'sensors_obs': ['accelerometer_link_1', 'velocimeter_link_1', 'gyro_link_1', 'magnetometer_link_1',
                            'accelerometer_link_2', 'velocimeter_link_2', 'gyro_link_2', 'magnetometer_link_2',
                            'accelerometer_link_3', 'velocimeter_link_3', 'gyro_link_3', 'magnetometer_link_3',
                            'accelerometer_link_4', 'velocimeter_link_4', 'gyro_link_4', 'magnetometer_link_4',
                            'accelerometer_link_5', 'velocimeter_link_5', 'gyro_link_5', 'magnetometer_link_5',
                            'touch_end_effector'],
            
            # constraint setting
            'constrain_ghost3Ds': True,  # Constrain robot from being in hazardous areas
            'constrain_indicator': False,  # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

            # lidar setting
            'lidar_num_bins': 10,
            'lidar_num_bins3D': 6,
            'lidar_body': ['link_1', 'link_3', 'link_5'],
            
            # object setting
            'ghost3Ds_num': 8,
            'ghost3Ds_size': 0.3,
            'ghost3Ds_travel':2.5,
            'ghost3Ds_safe_dist': 1.5,
            'ghost3Ds_z_range': [0.5, 1.5],
        }

    if task == "Push_Arm6_8Hazards":
        config = {
            # robot setting
            'robot_base': 'xmls/arm_6.xml', 
            'robot_locations':[(0.0,0.0)],

            # task setting
            'task': 'push',
            'push_object': 'ball',
            'goal_z_range': [0.5,1.5],
            'goal_size': 0.5,

            # observation setting
            'observe_goal_comp': True,  # Observe the goal with a lidar sensor
            'observe_box_comp': True,   # Observe the box with a lidar sensor
            'observe_hazard3Ds': True,  # Observe the vector from agent to hazards
            'compass_shape': 3,
            'sensors_obs': ['accelerometer_link_1', 'velocimeter_link_1', 'gyro_link_1', 'magnetometer_link_1',
                            'accelerometer_link_2', 'velocimeter_link_2', 'gyro_link_2', 'magnetometer_link_2',
                            'accelerometer_link_3', 'velocimeter_link_3', 'gyro_link_3', 'magnetometer_link_3',
                            'accelerometer_link_4', 'velocimeter_link_4', 'gyro_link_4', 'magnetometer_link_4',
                            'accelerometer_link_5', 'velocimeter_link_5', 'gyro_link_5', 'magnetometer_link_5',
                            'accelerometer_link_6', 'velocimeter_link_6', 'gyro_link_6', 'magnetometer_link_6',
                            'accelerometer_link_7', 'velocimeter_link_7', 'gyro_link_7', 'magnetometer_link_7',
                            'touch_end_effector'],
            
            # constraint setting
            'constrain_hazard3Ds': True,  # Constrain robot from being in hazardous areas
            'constrain_indicator': False,  # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

            # lidar setting
            'lidar_num_bins': 10,
            'lidar_num_bins3D': 6,
            'lidar_body': ['link_1', 'link_3', 'link_5', 'link_7'],
            
            # object setting
            'hazard3Ds_num': 8,
            'hazard3Ds_size': 0.3,
            'hazard3Ds_z_range': [0.5, 1.5],
        }

    if task == "Push_Arm6_8Ghosts":
        config = {
            # robot setting
            'robot_base': 'xmls/arm_6.xml',  
            'robot_locations':[(0.0,0.0)],

            # task setting
            'task': 'push',
            'push_object': 'ball',
            'goal_z_range': [0.5,1.5],
            'goal_size': 0.5,

            # observation setting
            'observe_goal_comp': True,  # Observe the goal with a lidar sensor
            'observe_box_comp': True,   # Observe the box with a lidar sensor
            'observe_ghost3Ds': True,  # Observe the vector from agent to hazards
            'compass_shape': 3,
            'sensors_obs': ['accelerometer_link_1', 'velocimeter_link_1', 'gyro_link_1', 'magnetometer_link_1',
                            'accelerometer_link_2', 'velocimeter_link_2', 'gyro_link_2', 'magnetometer_link_2',
                            'accelerometer_link_3', 'velocimeter_link_3', 'gyro_link_3', 'magnetometer_link_3',
                            'accelerometer_link_4', 'velocimeter_link_4', 'gyro_link_4', 'magnetometer_link_4',
                            'accelerometer_link_5', 'velocimeter_link_5', 'gyro_link_5', 'magnetometer_link_5',
                            'accelerometer_link_6', 'velocimeter_link_6', 'gyro_link_6', 'magnetometer_link_6',
                            'accelerometer_link_7', 'velocimeter_link_7', 'gyro_link_7', 'magnetometer_link_7',
                            'touch_end_effector'],
            
            # constraint setting
            'constrain_ghost3Ds': True,  # Constrain robot from being in hazardous areas
            'constrain_indicator': False,  # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

            # lidar setting
            'lidar_num_bins': 10,
            'lidar_num_bins3D': 6,
            'lidar_body': ['link_1', 'link_3', 'link_5', 'link_7'],
            
            # object setting
            'ghost3Ds_num': 8,
            'ghost3Ds_size': 0.3,
            'ghost3Ds_travel':2.5,
            'ghost3Ds_velocity':0.0001,
            'ghost3Ds_velocity':0.0001,
            'ghost3Ds_safe_dist': 1.5,
            'ghost3Ds_z_range': [0.5, 1.5],
        }
    
    if task == "Push_Drone_8Hazards":
        config = {
            # robot setting
            'robot_base': 'xmls/drone.xml', 

            # task setting
            'task': 'push',
            'push_object': 'ball',
            'goal_z_range': [0.5,1.5],
            'goal_size': 0.5,

            # observation setting
            'observe_goal_comp': True,  # Observe the goal with a lidar sensor
            'observe_box_comp': True,   # Observe the box with a lidar sensor
            'observe_hazard3Ds': True,  # Observe the vector from agent to hazards
            'compass_shape': 3,
            'sensors_obs': ['accelerometer', 'velocimeter', 'gyro', 'magnetometer',
                            'touch_p1a', 'touch_p1b', 'touch_p2a', 'touch_p2b',
                            'touch_p3a', 'touch_p3b', 'touch_p4a', 'touch_p4b'],
            
            # constraint setting
            'constrain_hazard3Ds': True,  # Constrain robot from being in hazardous areas
            'constrain_indicator': False,  # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

            # lidar setting
            'lidar_num_bins': 10,
            'lidar_num_bins3D': 6,
            
            # object setting
            'hazard3Ds_num': 8,
            'hazard3Ds_size': 0.3,
            'hazard3Ds_z_range': [0.5, 1.5],
        }

    if task == "Push_Drone_8Ghosts":
        config = {
            # robot setting
            'robot_base': 'xmls/drone.xml',  

            # task setting
            'task': 'push',
            'push_object': 'ball',
            'goal_z_range': [0.5,1.5],
            'goal_size': 0.5,

            # observation setting
            'observe_goal_comp': True,  # Observe the goal with a lidar sensor
            'observe_box_comp': True,   # Observe the box with a lidar sensor
            'observe_ghost3Ds': True,  # Observe the vector from agent to hazards
            'compass_shape': 3,
            'sensors_obs': ['accelerometer', 'velocimeter', 'gyro', 'magnetometer',
                            'touch_p1a', 'touch_p1b', 'touch_p2a', 'touch_p2b',
                            'touch_p3a', 'touch_p3b', 'touch_p4a', 'touch_p4b'],
            
            # constraint setting
            'constrain_ghost3Ds': True,  # Constrain robot from being in hazardous areas
            'constrain_indicator': False,  # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

            # lidar setting
            'lidar_num_bins': 10,
            'lidar_num_bins3D': 6,
            
            # object setting
            'ghost3Ds_num': 8,
            'ghost3Ds_size': 0.3,
            'ghost3Ds_travel':2.5,
            'ghost3Ds_safe_dist': 1.5,
            'ghost3Ds_z_range': [0.5, 1.5],
        }
 ################ Chase Tasks #################

    if task == "Chase_Point_8Hazards":
        config = {
            # robot setting
            'robot_base': 'xmls/point.xml',  

            # task setting
            'task': 'chase',
            'goal_size': 0.5,

            # observation setting
            'observe_robbers': True,  # Observe the goal with a lidar sensor
            'observe_hazards': True,  # Observe the vector from agent to hazards
            
            # constraint setting
            'constrain_hazards': True,  # Constrain robot from being in hazardous areas
            'constrain_indicator': False,  # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

            # lidar setting
            'lidar_num_bins': 16,
            
            # object setting
            'hazards_num': 8,
            'hazards_size': 0.3,
            'robbers_num': 2,
            'robbers_size': 0.3,
        }

    if task == "Chase_Point_8Ghosts":
        config = {
            # robot setting
            'robot_base': 'xmls/point.xml',  

            # task setting
            'task': 'chase',
            'goal_size': 0.5,

            # observation setting
            'observe_robbers': True,  # Observe the goal with a lidar sensor
            'observe_ghosts': True,  # Observe the vector from agent to hazards
            
            # constraint setting
            'constrain_ghosts': True,  # Constrain robot from being in hazardous areas
            'constrain_indicator': False,  # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

            # lidar setting
            'lidar_num_bins': 16,
            
            # object setting
            'ghosts_num': 8,
            'ghosts_size': 0.3,
            'ghosts_travel':2.5,
            'ghosts_safe_dist': 1.5,
            'robbers_num': 2,
            'robbers_size': 0.3,
        }

    if task == "Chase_Swimmer_8Hazards":
        config = {
            # robot setting
            'robot_base': 'xmls/swimmer.xml',  

            # task setting
            'task': 'chase',
            'goal_size': 0.5,

            # observation setting
            'observe_robbers': True,  # Observe the goal with a lidar sensor
            'observe_hazards': True,  # Observe the vector from agent to hazards
            'sensors_obs': ['accelerometer', 'velocimeter', 'gyro', 'magnetometer',
                            'touch_point1', 'touch_point2', 'touch_point3', 'touch_point4'],

            # constraint setting
            'constrain_hazards': True,  # Constrain robot from being in hazardous areas
            'constrain_indicator': False,  # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

            # lidar setting
            'lidar_num_bins': 16,
            
            # object setting
            'hazards_num': 8,
            'hazards_size': 0.3,
            'robbers_num': 2,
            'robbers_size': 0.3,
        }

    if task == "Chase_Swimmer_8Ghosts":
        config = {
            # robot setting
            'robot_base': 'xmls/swimmer.xml',  

            # task setting
            'task': 'chase',
            'goal_size': 0.5,

            # observation setting
            'observe_robbers': True,  # Observe the goal with a lidar sensor
            'observe_ghosts': True,  # Observe the vector from agent to hazards
            'sensors_obs': ['accelerometer', 'velocimeter', 'gyro', 'magnetometer',
                            'touch_point1', 'touch_point2', 'touch_point3', 'touch_point4'],
            
            # constraint setting
            'constrain_ghosts': True,  # Constrain robot from being in hazardous areas
            'constrain_indicator': False,  # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

            # lidar setting
            'lidar_num_bins': 16,
            
            # object setting
            'ghosts_num': 8,
            'ghosts_size': 0.3,
            'ghosts_travel':2.5,
            'ghosts_safe_dist': 1.5,
            'robbers_num': 2,
            'robbers_size': 0.3,
        }

    if task == "Chase_Ant_8Hazards":
        config = {
            # robot setting
            'robot_base': 'xmls/ant.xml',  

            # task setting
            'task': 'chase',
            'goal_size': 0.5,

            # observation setting
            'observe_robbers': True,  # Observe the goal with a lidar sensor
            'observe_hazards': True,  # Observe the vector from agent to hazards
            'sensors_obs': ['accelerometer', 'velocimeter', 'gyro', 'magnetometer',
                            'touch_ankle_1a', 'touch_ankle_2a', 'touch_ankle_3a', 'touch_ankle_4a',
                            'touch_ankle_1b', 'touch_ankle_2b', 'touch_ankle_3b', 'touch_ankle_4b'],

            # constraint setting
            'constrain_hazards': True,  # Constrain robot from being in hazardous areas
            'constrain_indicator': False,  # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

            # lidar setting
            'lidar_num_bins': 16,
            
            # object setting
            'hazards_num': 8,
            'hazards_size': 0.3,
            'robbers_num': 2,
            'robbers_size': 0.3,
        }

    if task == "Chase_Ant_8Ghosts":
        config = {
            # robot setting
            'robot_base': 'xmls/ant.xml',  

            # task setting
            'task': 'chase',
            'goal_size': 0.5,

            # observation setting
            'observe_robbers': True,  # Observe the goal with a lidar sensor
            'observe_ghosts': True,  # Observe the vector from agent to hazards
            'sensors_obs': ['accelerometer', 'velocimeter', 'gyro', 'magnetometer',
                            'touch_ankle_1a', 'touch_ankle_2a', 'touch_ankle_3a', 'touch_ankle_4a',
                            'touch_ankle_1b', 'touch_ankle_2b', 'touch_ankle_3b', 'touch_ankle_4b'],

            
            # constraint setting
            'constrain_ghosts': True,  # Constrain robot from being in hazardous areas
            'constrain_indicator': False,  # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

            # lidar setting
            'lidar_num_bins': 16,
            
            # object setting
            'ghosts_num': 8,
            'ghosts_size': 0.3,
            'ghosts_travel':2.5,
            'ghosts_safe_dist': 1.5,
            'robbers_num': 2,
            'robbers_size': 0.3,
        }
    
    if task == "Chase_Walker_8Hazards":
        config = {
            # robot setting
            'robot_base': 'xmls/walker.xml',  

            # task setting
            'task': 'chase',
            'goal_size': 0.5,

            # observation setting
            'observe_robbers': True,  # Observe the goal with a lidar sensor
            'observe_hazards': True,  # Observe the vector from agent to hazards
            'sensors_obs': ['accelerometer', 'velocimeter', 'gyro', 'magnetometer',
                            'touch_right_foot', 'touch_left_foot'],
            
            # constraint setting
            'constrain_hazards': True,  # Constrain robot from being in hazardous areas
            'constrain_indicator': False,  # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

            # lidar setting
            'lidar_num_bins': 16,
            
            # object setting
            'hazards_num': 8,
            'hazards_size': 0.3,
            'robbers_num': 2,
            'robbers_size': 0.3,
        }

    if task == "Chase_Walker_8Ghosts":
        config = {
            # robot setting
            'robot_base': 'xmls/walker.xml',  

            # task setting
            'task': 'chase',
            'goal_size': 0.5,

            # observation setting
            'observe_robbers': True,  # Observe the goal with a lidar sensor
            'observe_ghosts': True,  # Observe the vector from agent to hazards
            'sensors_obs': ['accelerometer', 'velocimeter', 'gyro', 'magnetometer',
                            'touch_right_foot', 'touch_left_foot'],
            
            # constraint setting
            'constrain_ghosts': True,  # Constrain robot from being in hazardous areas
            'constrain_indicator': False,  # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

            # lidar setting
            'lidar_num_bins': 16,
            
            # object setting
            'ghosts_num': 8,
            'ghosts_size': 0.3,
            'ghosts_travel':2.5,
            'ghosts_safe_dist': 1.5,
            'robbers_num': 2,
            'robbers_size': 0.3,
        }

    if task == "Chase_Humanoid_8Hazards":
        config = {
            # robot setting
            'robot_base': 'xmls/humanoid.xml',  

            # task setting
            'task': 'chase',
            'goal_size': 0.5,

            # observation setting
            'observe_robbers': True,  # Observe the goal with a lidar sensor
            'observe_hazards': True,  # Observe the vector from agent to hazards
            'sensors_obs': ['accelerometer', 'velocimeter', 'gyro', 'magnetometer',
                            'touch_right_foot', 'touch_left_foot'],
            
            # constraint setting
            'constrain_hazards': True,  # Constrain robot from being in hazardous areas
            'constrain_indicator': False,  # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

            # lidar setting
            'lidar_num_bins': 16,
            
            # object setting
            'hazards_num': 8,
            'hazards_size': 0.3,
            'robbers_num': 2,
            'robbers_size': 0.3,
        }

    if task == "Chase_Humanoid_8Ghosts":
        config = {
            # robot setting
            'robot_base': 'xmls/humanoid.xml',  

            # task setting
            'task': 'chase',
            'goal_size': 0.5,

            # observation setting
            'observe_robbers': True,  # Observe the goal with a lidar sensor
            'observe_ghosts': True,  # Observe the vector from agent to hazards
            'sensors_obs': ['accelerometer', 'velocimeter', 'gyro', 'magnetometer',
                            'touch_right_foot', 'touch_left_foot'],
            
            # constraint setting
            'constrain_ghosts': True,  # Constrain robot from being in hazardous areas
            'constrain_indicator': False,  # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

            # lidar setting
            'lidar_num_bins': 16,
            
            # object setting
            'ghosts_num': 8,
            'ghosts_size': 0.3,
            'ghosts_travel':2.5,
            'ghosts_safe_dist': 1.5,
            'robbers_num': 2,
            'robbers_size': 0.3,
        }

    if task == "Chase_Hopper_8Hazards":
        config = {
            # robot setting
            'robot_base': 'xmls/hopper.xml',  

            # task setting
            'task': 'chase',
            'goal_size': 0.5,

            # observation setting
            'observe_robbers': True,  # Observe the goal with a lidar sensor
            'observe_hazards': True,  # Observe the vector from agent to hazards
            'sensors_obs': ['accelerometer', 'velocimeter', 'gyro', 'magnetometer',
                            'touch_foot'],
            
            # constraint setting
            'constrain_hazards': True,  # Constrain robot from being in hazardous areas
            'constrain_indicator': False,  # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

            # lidar setting
            'lidar_num_bins': 16,
            
            # object setting
            'hazards_num': 8,
            'hazards_size': 0.3,
            'robbers_num': 2,
            'robbers_size': 0.3,
        }

    if task == "Chase_Hopper_8Ghosts":
        config = {
            # robot setting
            'robot_base': 'xmls/hopper.xml',  

            # task setting
            'task': 'chase',
            'goal_size': 0.5,

            # observation setting
            'observe_robbers': True,  # Observe the goal with a lidar sensor
            'observe_ghosts': True,  # Observe the vector from agent to hazards
            'sensors_obs': ['accelerometer', 'velocimeter', 'gyro', 'magnetometer',
                            'touch_foot'],
            
            # constraint setting
            'constrain_ghosts': True,  # Constrain robot from being in hazardous areas
            'constrain_indicator': False,  # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

            # lidar setting
            'lidar_num_bins': 16,
            
            # object setting
            'ghosts_num': 8,
            'ghosts_size': 0.3,
            'ghosts_travel':2.5,
            'ghosts_safe_dist': 1.5,
            'robbers_num': 2,
            'robbers_size': 0.3,
        }

    if task == "Chase_Arm3_8Hazards":
        config = {
            # robot setting
            'robot_base': 'xmls/arm_3.xml', 
            'robot_locations':[(0.0,0.0)],

            # task setting
            'task': 'chase',
            'goal_3D': True,
            'goal_z_range': [0.5,1.5],
            'goal_size': 0.5,

            # observation setting
            'observe_robber3Ds': True,  # Observe the goal with a lidar sensor
            'observe_hazard3Ds': True,  # Observe the vector from agent to hazards
            'sensors_obs': ['accelerometer_link_1', 'velocimeter_link_1', 'gyro_link_1', 'magnetometer_link_1',
                            'accelerometer_link_2', 'velocimeter_link_2', 'gyro_link_2', 'magnetometer_link_2',
                            'accelerometer_link_3', 'velocimeter_link_3', 'gyro_link_3', 'magnetometer_link_3',
                            'accelerometer_link_4', 'velocimeter_link_4', 'gyro_link_4', 'magnetometer_link_4',
                            'accelerometer_link_5', 'velocimeter_link_5', 'gyro_link_5', 'magnetometer_link_5',
                            'touch_end_effector'],
            
            # constraint setting
            'constrain_hazard3Ds': True,  # Constrain robot from being in hazardous areas
            'constrain_indicator': False,  # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

            # lidar setting
            'lidar_num_bins': 10,
            'lidar_num_bins3D': 6,
            'lidar_body': ['link_1', 'link_3', 'link_5'],
            
            # object setting
            'hazard3Ds_num': 8,
            'hazard3Ds_size': 0.3,
            'hazard3Ds_z_range': [0.5, 1.5],
            'robber3Ds_num': 2,
            'robber3Ds_size': 0.3,
            'robber3Ds_z_range': [0.5, 1.5],
        }

    if task == "Chase_Arm3_8Ghosts":
        config = {
            # robot setting
            'robot_base': 'xmls/arm_3.xml',  
            'robot_locations':[(0.0,0.0)],

            # task setting
            'task': 'chase',
            'goal_3D': True,
            'goal_z_range': [0.5,1.5],
            'goal_size': 0.5,

            # observation setting
            'observe_robber3Ds': True,  # Observe the goal with a lidar sensor
            'observe_ghost3Ds': True,  # Observe the vector from agent to hazards
            'sensors_obs': ['accelerometer_link_1', 'velocimeter_link_1', 'gyro_link_1', 'magnetometer_link_1',
                            'accelerometer_link_2', 'velocimeter_link_2', 'gyro_link_2', 'magnetometer_link_2',
                            'accelerometer_link_3', 'velocimeter_link_3', 'gyro_link_3', 'magnetometer_link_3',
                            'accelerometer_link_4', 'velocimeter_link_4', 'gyro_link_4', 'magnetometer_link_4',
                            'accelerometer_link_5', 'velocimeter_link_5', 'gyro_link_5', 'magnetometer_link_5',
                            'touch_end_effector'],
            
            # constraint setting
            'constrain_ghost3Ds': True,  # Constrain robot from being in hazardous areas
            'constrain_indicator': False,  # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

            # lidar setting
            'lidar_num_bins': 10,
            'lidar_num_bins3D': 6,
            'lidar_body': ['link_1', 'link_3', 'link_5'],
            
            # object setting
            'ghost3Ds_num': 8,
            'ghost3Ds_size': 0.3,
            'ghost3Ds_travel':2.5,
            'ghost3Ds_safe_dist': 1.5,
            'ghost3Ds_z_range': [0.5, 1.5],
            'robber3Ds_num': 2,
            'robber3Ds_size': 0.3,
            'robber3Ds_z_range': [0.5, 1.5],
        }

    if task == "Chase_Arm6_8Hazards":
        config = {
            # robot setting
            'robot_base': 'xmls/arm_6.xml', 
            'robot_locations':[(0.0,0.0)],

            # task setting
            'task': 'chase',
            'goal_3D': True,
            'goal_z_range': [0.5,1.5],
            'goal_size': 0.5,

            # observation setting
            'observe_robber3Ds': True,  # Observe the goal with a lidar sensor
            'observe_hazard3Ds': True,  # Observe the vector from agent to hazards
            'sensors_obs': ['accelerometer_link_1', 'velocimeter_link_1', 'gyro_link_1', 'magnetometer_link_1',
                            'accelerometer_link_2', 'velocimeter_link_2', 'gyro_link_2', 'magnetometer_link_2',
                            'accelerometer_link_3', 'velocimeter_link_3', 'gyro_link_3', 'magnetometer_link_3',
                            'accelerometer_link_4', 'velocimeter_link_4', 'gyro_link_4', 'magnetometer_link_4',
                            'accelerometer_link_5', 'velocimeter_link_5', 'gyro_link_5', 'magnetometer_link_5',
                            'accelerometer_link_6', 'velocimeter_link_6', 'gyro_link_6', 'magnetometer_link_6',
                            'accelerometer_link_7', 'velocimeter_link_7', 'gyro_link_7', 'magnetometer_link_7',
                            'touch_end_effector'],
            
            # constraint setting
            'constrain_hazard3Ds': True,  # Constrain robot from being in hazardous areas
            'constrain_indicator': False,  # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

            # lidar setting
            'lidar_num_bins': 10,
            'lidar_num_bins3D': 6,
            'lidar_body': ['link_1', 'link_3', 'link_5', 'link_7'],
            
            # object setting
            'hazard3Ds_num': 8,
            'hazard3Ds_size': 0.3,
            'hazard3Ds_z_range': [0.5, 1.5],
            'robber3Ds_num': 2,
            'robber3Ds_size': 0.3,
            'robber3Ds_z_range': [0.5, 1.5],
        }

    if task == "Chase_Arm6_8Ghosts":
        config = {
            # robot setting
            'robot_base': 'xmls/arm_6.xml',  
            'robot_locations':[(0.0,0.0)],

            # task setting
            'task': 'chase',
            'goal_3D': True,
            'goal_z_range': [0.5,1.5],
            'goal_size': 0.5,

            # observation setting
            'observe_robber3Ds': True,  # Observe the goal with a lidar sensor
            'observe_ghost3Ds': True,  # Observe the vector from agent to hazards
            'sensors_obs': ['accelerometer_link_1', 'velocimeter_link_1', 'gyro_link_1', 'magnetometer_link_1',
                            'accelerometer_link_2', 'velocimeter_link_2', 'gyro_link_2', 'magnetometer_link_2',
                            'accelerometer_link_3', 'velocimeter_link_3', 'gyro_link_3', 'magnetometer_link_3',
                            'accelerometer_link_4', 'velocimeter_link_4', 'gyro_link_4', 'magnetometer_link_4',
                            'accelerometer_link_5', 'velocimeter_link_5', 'gyro_link_5', 'magnetometer_link_5',
                            'accelerometer_link_6', 'velocimeter_link_6', 'gyro_link_6', 'magnetometer_link_6',
                            'accelerometer_link_7', 'velocimeter_link_7', 'gyro_link_7', 'magnetometer_link_7',
                            'touch_end_effector'],
            
            # constraint setting
            'constrain_ghost3Ds': True,  # Constrain robot from being in hazardous areas
            'constrain_indicator': False,  # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

            # lidar setting
            'lidar_num_bins': 10,
            'lidar_num_bins3D': 6,
            'lidar_body': ['link_1', 'link_3', 'link_5', 'link_7'],
            
            # object setting
            'ghost3Ds_num': 8,
            'ghost3Ds_size': 0.3,
            'ghost3Ds_travel':2.5,
            'ghost3Ds_velocity':0.0001,
            'ghost3Ds_safe_dist': 1.5,
            'ghost3Ds_z_range': [0.5, 1.5],
            'robber3Ds_num': 2,
            'robber3Ds_size': 0.3,
            'robber3Ds_z_range': [0.5, 1.5],
        }
    
    if task == "Chase_Drone_8Hazards":
        config = {
            # robot setting
            'robot_base': 'xmls/drone.xml', 

            # task setting
            'task': 'chase',
            'goal_3D': True,
            'goal_z_range': [0.5,1.5],
            'goal_size': 0.5,

            # observation setting
            'observe_robber3Ds': True,  # Observe the goal with a lidar sensor
            'observe_hazard3Ds': True,  # Observe the vector from agent to hazards
            'sensors_obs': ['accelerometer', 'velocimeter', 'gyro', 'magnetometer',
                            'touch_p1a', 'touch_p1b', 'touch_p2a', 'touch_p2b',
                            'touch_p3a', 'touch_p3b', 'touch_p4a', 'touch_p4b'],
            
            # constraint setting
            'constrain_hazard3Ds': True,  # Constrain robot from being in hazardous areas
            'constrain_indicator': False,  # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

            # lidar setting
            'lidar_num_bins': 10,
            'lidar_num_bins3D': 6,
            
            # object setting
            'hazard3Ds_num': 8,
            'hazard3Ds_size': 0.3,
            'hazard3Ds_z_range': [0.5, 1.5],
            'robber3Ds_num': 2,
            'robber3Ds_size': 0.3,
            'robber3Ds_z_range': [0.5, 1.5],
        }

    if task == "Chase_Drone_8Ghosts":
        config = {
            # robot setting
            'robot_base': 'xmls/drone.xml',  

            # task setting
            'task': 'chase',
            'goal_3D': True,
            'goal_z_range': [0.5,1.5],
            'goal_size': 0.5,

            # observation setting
            'observe_robber3Ds': True,  # Observe the goal with a lidar sensor
            'observe_ghost3Ds': True,  # Observe the vector from agent to hazards
            'sensors_obs': ['accelerometer', 'velocimeter', 'gyro', 'magnetometer',
                            'touch_p1a', 'touch_p1b', 'touch_p2a', 'touch_p2b',
                            'touch_p3a', 'touch_p3b', 'touch_p4a', 'touch_p4b'],
            
            # constraint setting
            'constrain_ghost3Ds': True,  # Constrain robot from being in hazardous areas
            'constrain_indicator': False,  # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

            # lidar setting
            'lidar_num_bins': 10,
            'lidar_num_bins3D': 6,
            
            # object setting
            'ghost3Ds_num': 8,
            'ghost3Ds_size': 0.3,
            'ghost3Ds_travel':2.5,
            'ghost3Ds_safe_dist': 1.5,
            'ghost3Ds_z_range': [0.5, 1.5],
            'robber3Ds_num': 2,
            'robber3Ds_size': 0.3,
            'robber3Ds_z_range': [0.5, 1.5],
        }
    
    ################ Defense Tasks #################

    if task == "Defense_Point_8Hazards":
        config = {
            # robot setting
            'robot_base': 'xmls/point.xml',  

            # task setting
            'task': 'defense',
            'goal_size': 0.5,

            # observation setting
            'observe_robbers': True,  # Observe the goal with a lidar sensor
            'observe_hazards': True,  # Observe the vector from agent to hazards
            
            # constraint setting
            'constrain_hazards': True,  # Constrain robot from being in hazardous areas
            'constrain_indicator': False,  # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

            # lidar setting
            'lidar_num_bins': 16,
            
            # object setting
            'hazards_num': 8,
            'hazards_size': 0.3,
            'robbers_num': 2,
            'robbers_size': 0.3,
        }

    if task == "Defense_Point_8Ghosts":
        config = {
            # robot setting
            'robot_base': 'xmls/point.xml',  

            # task setting
            'task': 'defense',
            'goal_size': 0.5,

            # observation setting
            'observe_robbers': True,  # Observe the goal with a lidar sensor
            'observe_ghosts': True,  # Observe the vector from agent to hazards
            
            # constraint setting
            'constrain_ghosts': True,  # Constrain robot from being in hazardous areas
            'constrain_indicator': False,  # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

            # lidar setting
            'lidar_num_bins': 16,
            
            # object setting
            'ghosts_num': 8,
            'ghosts_size': 0.3,
            'ghosts_travel':2.5,
            'ghosts_safe_dist': 1.5,
            'robbers_num': 2,
            'robbers_size': 0.3,
        }

    if task == "Defense_Swimmer_8Hazards":
        config = {
            # robot setting
            'robot_base': 'xmls/swimmer.xml',  

            # task setting
            'task': 'defense',
            'goal_size': 0.5,

            # observation setting
            'observe_robbers': True,  # Observe the goal with a lidar sensor
            'observe_hazards': True,  # Observe the vector from agent to hazards
            'sensors_obs': ['accelerometer', 'velocimeter', 'gyro', 'magnetometer',
                            'touch_point1', 'touch_point2', 'touch_point3', 'touch_point4'],

            # constraint setting
            'constrain_hazards': True,  # Constrain robot from being in hazardous areas
            'constrain_indicator': False,  # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

            # lidar setting
            'lidar_num_bins': 16,
            
            # object setting
            'hazards_num': 8,
            'hazards_size': 0.3,
            'robbers_num': 2,
            'robbers_size': 0.3,
        }

    if task == "Defense_Swimmer_8Ghosts":
        config = {
            # robot setting
            'robot_base': 'xmls/swimmer.xml',  

            # task setting
            'task': 'defense',
            'goal_size': 0.5,

            # observation setting
            'observe_robbers': True,  # Observe the goal with a lidar sensor
            'observe_ghosts': True,  # Observe the vector from agent to hazards
            'sensors_obs': ['accelerometer', 'velocimeter', 'gyro', 'magnetometer',
                            'touch_point1', 'touch_point2', 'touch_point3', 'touch_point4'],
            
            # constraint setting
            'constrain_ghosts': True,  # Constrain robot from being in hazardous areas
            'constrain_indicator': False,  # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

            # lidar setting
            'lidar_num_bins': 16,
            
            # object setting
            'ghosts_num': 8,
            'ghosts_size': 0.3,
            'ghosts_travel':2.5,
            'ghosts_safe_dist': 1.5,
            'robbers_num': 2,
            'robbers_size': 0.3,
        }

    if task == "Defense_Ant_8Hazards":
        config = {
            # robot setting
            'robot_base': 'xmls/ant.xml',  

            # task setting
            'task': 'defense',
            'goal_size': 0.5,

            # observation setting
            'observe_robbers': True,  # Observe the goal with a lidar sensor
            'observe_hazards': True,  # Observe the vector from agent to hazards
            'sensors_obs': ['accelerometer', 'velocimeter', 'gyro', 'magnetometer',
                            'touch_ankle_1a', 'touch_ankle_2a', 'touch_ankle_3a', 'touch_ankle_4a',
                            'touch_ankle_1b', 'touch_ankle_2b', 'touch_ankle_3b', 'touch_ankle_4b'],

            # constraint setting
            'constrain_hazards': True,  # Constrain robot from being in hazardous areas
            'constrain_indicator': False,  # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

            # lidar setting
            'lidar_num_bins': 16,
            
            # object setting
            'hazards_num': 8,
            'hazards_size': 0.3,
            'robbers_num': 2,
            'robbers_size': 0.3,
        }

    if task == "Defense_Ant_8Ghosts":
        config = {
            # robot setting
            'robot_base': 'xmls/ant.xml',  

            # task setting
            'task': 'defense',
            'goal_size': 0.5,

            # observation setting
            'observe_robbers': True,  # Observe the goal with a lidar sensor
            'observe_ghosts': True,  # Observe the vector from agent to hazards
            'sensors_obs': ['accelerometer', 'velocimeter', 'gyro', 'magnetometer',
                            'touch_ankle_1a', 'touch_ankle_2a', 'touch_ankle_3a', 'touch_ankle_4a',
                            'touch_ankle_1b', 'touch_ankle_2b', 'touch_ankle_3b', 'touch_ankle_4b'],

            
            # constraint setting
            'constrain_ghosts': True,  # Constrain robot from being in hazardous areas
            'constrain_indicator': False,  # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

            # lidar setting
            'lidar_num_bins': 16,
            
            # object setting
            'ghosts_num': 8,
            'ghosts_size': 0.3,
            'ghosts_travel':2.5,
            'ghosts_safe_dist': 1.5,
            'robbers_num': 2,
            'robbers_size': 0.3,
        }
    
    if task == "Defense_Walker_8Hazards":
        config = {
            # robot setting
            'robot_base': 'xmls/walker.xml',  

            # task setting
            'task': 'defense',
            'goal_size': 0.5,

            # observation setting
            'observe_robbers': True,  # Observe the goal with a lidar sensor
            'observe_hazards': True,  # Observe the vector from agent to hazards
            'sensors_obs': ['accelerometer', 'velocimeter', 'gyro', 'magnetometer',
                            'touch_right_foot', 'touch_left_foot'],
            
            # constraint setting
            'constrain_hazards': True,  # Constrain robot from being in hazardous areas
            'constrain_indicator': False,  # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

            # lidar setting
            'lidar_num_bins': 16,
            
            # object setting
            'hazards_num': 8,
            'hazards_size': 0.3,
            'robbers_num': 2,
            'robbers_size': 0.3,
        }

    if task == "Defense_Walker_8Ghosts":
        config = {
            # robot setting
            'robot_base': 'xmls/walker.xml',  

            # task setting
            'task': 'defense',
            'goal_size': 0.5,

            # observation setting
            'observe_robbers': True,  # Observe the goal with a lidar sensor
            'observe_ghosts': True,  # Observe the vector from agent to hazards
            'sensors_obs': ['accelerometer', 'velocimeter', 'gyro', 'magnetometer',
                            'touch_right_foot', 'touch_left_foot'],
            
            # constraint setting
            'constrain_ghosts': True,  # Constrain robot from being in hazardous areas
            'constrain_indicator': False,  # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

            # lidar setting
            'lidar_num_bins': 16,
            
            # object setting
            'ghosts_num': 8,
            'ghosts_size': 0.3,
            'ghosts_travel':2.5,
            'ghosts_safe_dist': 1.5,
            'robbers_num': 2,
            'robbers_size': 0.3,
        }

    if task == "Defense_Humanoid_8Hazards":
        config = {
            # robot setting
            'robot_base': 'xmls/humanoid.xml',  

            # task setting
            'task': 'defense',
            'goal_size': 0.5,

            # observation setting
            'observe_robbers': True,  # Observe the goal with a lidar sensor
            'observe_hazards': True,  # Observe the vector from agent to hazards
            'sensors_obs': ['accelerometer', 'velocimeter', 'gyro', 'magnetometer',
                            'touch_right_foot', 'touch_left_foot'],
            
            # constraint setting
            'constrain_hazards': True,  # Constrain robot from being in hazardous areas
            'constrain_indicator': False,  # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

            # lidar setting
            'lidar_num_bins': 16,
            
            # object setting
            'hazards_num': 8,
            'hazards_size': 0.3,
            'robbers_num': 2,
            'robbers_size': 0.3,
        }

    if task == "Defense_Humanoid_8Ghosts":
        config = {
            # robot setting
            'robot_base': 'xmls/humanoid.xml',  

            # task setting
            'task': 'defense',
            'goal_size': 0.5,

            # observation setting
            'observe_robbers': True,  # Observe the goal with a lidar sensor
            'observe_ghosts': True,  # Observe the vector from agent to hazards
            'sensors_obs': ['accelerometer', 'velocimeter', 'gyro', 'magnetometer',
                            'touch_right_foot', 'touch_left_foot'],
            
            # constraint setting
            'constrain_ghosts': True,  # Constrain robot from being in hazardous areas
            'constrain_indicator': False,  # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

            # lidar setting
            'lidar_num_bins': 16,
            
            # object setting
            'ghosts_num': 8,
            'ghosts_size': 0.3,
            'ghosts_travel':2.5,
            'ghosts_safe_dist': 1.5,
            'robbers_num': 2,
            'robbers_size': 0.3,
        }

    if task == "Defense_Hopper_8Hazards":
        config = {
            # robot setting
            'robot_base': 'xmls/hopper.xml',  

            # task setting
            'task': 'defense',
            'goal_size': 0.5,

            # observation setting
            'observe_robbers': True,  # Observe the goal with a lidar sensor
            'observe_hazards': True,  # Observe the vector from agent to hazards
            'sensors_obs': ['accelerometer', 'velocimeter', 'gyro', 'magnetometer',
                            'touch_foot'],
            
            # constraint setting
            'constrain_hazards': True,  # Constrain robot from being in hazardous areas
            'constrain_indicator': False,  # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

            # lidar setting
            'lidar_num_bins': 16,
            
            # object setting
            'hazards_num': 8,
            'hazards_size': 0.3,
            'robbers_num': 2,
            'robbers_size': 0.3,
        }

    if task == "Defense_Hopper_8Ghosts":
        config = {
            # robot setting
            'robot_base': 'xmls/hopper.xml',  

            # task setting
            'task': 'defense',
            'goal_size': 0.5,

            # observation setting
            'observe_robbers': True,  # Observe the goal with a lidar sensor
            'observe_ghosts': True,  # Observe the vector from agent to hazards
            'sensors_obs': ['accelerometer', 'velocimeter', 'gyro', 'magnetometer',
                            'touch_foot'],
            
            # constraint setting
            'constrain_ghosts': True,  # Constrain robot from being in hazardous areas
            'constrain_indicator': False,  # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

            # lidar setting
            'lidar_num_bins': 16,
            
            # object setting
            'ghosts_num': 8,
            'ghosts_size': 0.3,
            'ghosts_travel':2.5,
            'ghosts_safe_dist': 1.5,
            'robbers_num': 2,
            'robbers_size': 0.3,
        }

    if task == "Defense_Arm3_8Hazards":
        config = {
            # robot setting
            'robot_base': 'xmls/arm_3.xml', 
            'robot_locations':[(0.0,0.0)],

            # task setting
            'task': 'defense',
            'goal_3D': True,
            'goal_z_range': [0.5,1.5],
            'goal_size': 0.5,

            # observation setting
            'observe_robber3Ds': True,  # Observe the goal with a lidar sensor
            'observe_hazard3Ds': True,  # Observe the vector from agent to hazards
            'sensors_obs': ['accelerometer_link_1', 'velocimeter_link_1', 'gyro_link_1', 'magnetometer_link_1',
                            'accelerometer_link_2', 'velocimeter_link_2', 'gyro_link_2', 'magnetometer_link_2',
                            'accelerometer_link_3', 'velocimeter_link_3', 'gyro_link_3', 'magnetometer_link_3',
                            'accelerometer_link_4', 'velocimeter_link_4', 'gyro_link_4', 'magnetometer_link_4',
                            'accelerometer_link_5', 'velocimeter_link_5', 'gyro_link_5', 'magnetometer_link_5',
                            'touch_end_effector'],
            
            # constraint setting
            'constrain_hazard3Ds': True,  # Constrain robot from being in hazardous areas
            'constrain_indicator': False,  # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

            # lidar setting
            'lidar_num_bins': 10,
            'lidar_num_bins3D': 6,
            'lidar_body': ['link_1', 'link_3', 'link_5'],
            
            # object setting
            'hazard3Ds_num': 8,
            'hazard3Ds_size': 0.3,
            'hazard3Ds_z_range': [0.5, 1.5],
            'robber3Ds_num': 2,
            'robber3Ds_size': 0.3,
            'robber3Ds_z_range': [0.5, 1.5],
        }

    if task == "Defense_Arm3_8Ghosts":
        config = {
            # robot setting
            'robot_base': 'xmls/arm_3.xml',  
            'robot_locations':[(0.0,0.0)],

            # task setting
            'task': 'defense',
            'goal_3D': True,
            'goal_z_range': [0.5,1.5],
            'goal_size': 0.5,
            'defense_range': 2.5,

            # observation setting
            'observe_robber3Ds': True,  # Observe the goal with a lidar sensor
            'observe_ghost3Ds': True,  # Observe the vector from agent to hazards
            'sensors_obs': ['accelerometer_link_1', 'velocimeter_link_1', 'gyro_link_1', 'magnetometer_link_1',
                            'accelerometer_link_2', 'velocimeter_link_2', 'gyro_link_2', 'magnetometer_link_2',
                            'accelerometer_link_3', 'velocimeter_link_3', 'gyro_link_3', 'magnetometer_link_3',
                            'accelerometer_link_4', 'velocimeter_link_4', 'gyro_link_4', 'magnetometer_link_4',
                            'accelerometer_link_5', 'velocimeter_link_5', 'gyro_link_5', 'magnetometer_link_5',
                            'touch_end_effector'],
            
            # constraint setting
            'constrain_ghost3Ds': True,  # Constrain robot from being in hazardous areas
            'constrain_indicator': False,  # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

            # lidar setting
            'lidar_num_bins': 10,
            'lidar_num_bins3D': 6,
            'lidar_body': ['link_1', 'link_3', 'link_5'],
            
            # object setting
            'ghost3Ds_num': 8,
            'ghost3Ds_size': 0.3,
            'ghost3Ds_travel':2.5,
            'ghost3Ds_safe_dist': 1.5,
            'ghost3Ds_z_range': [0.5, 1.5],
            'robber3Ds_num': 2,
            'robber3Ds_size': 0.3,
            'robber3Ds_z_range': [0.5, 1.5],
        }

    if task == "Defense_Arm6_8Hazards":
        config = {
            # robot setting
            'robot_base': 'xmls/arm_6.xml', 
            'robot_locations':[(0.0,0.0)],

            # task setting
            'task': 'defense',
            'goal_3D': True,
            'goal_z_range': [0.5,1.5],
            'goal_size': 0.5,
            'defense_range': 2.5,

            # observation setting
            'observe_robber3Ds': True,  # Observe the goal with a lidar sensor
            'observe_hazard3Ds': True,  # Observe the vector from agent to hazards
            'sensors_obs': ['accelerometer_link_1', 'velocimeter_link_1', 'gyro_link_1', 'magnetometer_link_1',
                            'accelerometer_link_2', 'velocimeter_link_2', 'gyro_link_2', 'magnetometer_link_2',
                            'accelerometer_link_3', 'velocimeter_link_3', 'gyro_link_3', 'magnetometer_link_3',
                            'accelerometer_link_4', 'velocimeter_link_4', 'gyro_link_4', 'magnetometer_link_4',
                            'accelerometer_link_5', 'velocimeter_link_5', 'gyro_link_5', 'magnetometer_link_5',
                            'accelerometer_link_6', 'velocimeter_link_6', 'gyro_link_6', 'magnetometer_link_6',
                            'accelerometer_link_7', 'velocimeter_link_7', 'gyro_link_7', 'magnetometer_link_7',
                            'touch_end_effector'],
            
            # constraint setting
            'constrain_hazard3Ds': True,  # Constrain robot from being in hazardous areas
            'constrain_indicator': False,  # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

            # lidar setting
            'lidar_num_bins': 10,
            'lidar_num_bins3D': 6,
            'lidar_body': ['link_1', 'link_3', 'link_5', 'link_7'],
            
            # object setting
            'hazard3Ds_num': 8,
            'hazard3Ds_size': 0.3,
            'hazard3Ds_z_range': [0.5, 1.5],
            'robber3Ds_num': 2,
            'robber3Ds_size': 0.3,
            'robber3Ds_z_range': [0.5, 1.5],
        }

    if task == "Defense_Arm6_8Ghosts":
        config = {
            # robot setting
            'robot_base': 'xmls/arm_6.xml',  
            'robot_locations':[(0.0,0.0)],

            # task setting
            'task': 'defense',
            'goal_3D': True,
            'goal_z_range': [0.5,1.5],
            'goal_size': 0.5,
            'defense_range': 2.5,

            # observation setting
            'observe_robber3Ds': True,  # Observe the goal with a lidar sensor
            'observe_ghost3Ds': True,  # Observe the vector from agent to hazards
            'sensors_obs': ['accelerometer_link_1', 'velocimeter_link_1', 'gyro_link_1', 'magnetometer_link_1',
                            'accelerometer_link_2', 'velocimeter_link_2', 'gyro_link_2', 'magnetometer_link_2',
                            'accelerometer_link_3', 'velocimeter_link_3', 'gyro_link_3', 'magnetometer_link_3',
                            'accelerometer_link_4', 'velocimeter_link_4', 'gyro_link_4', 'magnetometer_link_4',
                            'accelerometer_link_5', 'velocimeter_link_5', 'gyro_link_5', 'magnetometer_link_5',
                            'accelerometer_link_6', 'velocimeter_link_6', 'gyro_link_6', 'magnetometer_link_6',
                            'accelerometer_link_7', 'velocimeter_link_7', 'gyro_link_7', 'magnetometer_link_7',
                            'touch_end_effector'],
            
            # constraint setting
            'constrain_ghost3Ds': True,  # Constrain robot from being in hazardous areas
            'constrain_indicator': False,  # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

            # lidar setting
            'lidar_num_bins': 10,
            'lidar_num_bins3D': 6,
            'lidar_body': ['link_1', 'link_3', 'link_5', 'link_7'],
            
            # object setting
            'ghost3Ds_num': 8,
            'ghost3Ds_size': 0.3,
            'ghost3Ds_travel':2.5,
            'ghost3Ds_velocity':0.0001,
            'ghost3Ds_safe_dist': 1.5,
            'ghost3Ds_z_range': [0.5, 1.5],
            'robber3Ds_num': 2,
            'robber3Ds_size': 0.3,
            'robber3Ds_z_range': [0.5, 1.5],
        }
    
    if task == "Defense_Drone_8Hazards":
        config = {
            # robot setting
            'robot_base': 'xmls/drone.xml', 

            # task setting
            'task': 'defense',
            'goal_3D': True,
            'goal_z_range': [0.5,1.5],
            'goal_size': 0.5,
            'defense_range': 2.5,

            # observation setting
            'observe_robber3Ds': True,  # Observe the goal with a lidar sensor
            'observe_hazard3Ds': True,  # Observe the vector from agent to hazards
            'sensors_obs': ['accelerometer', 'velocimeter', 'gyro', 'magnetometer',
                            'touch_p1a', 'touch_p1b', 'touch_p2a', 'touch_p2b',
                            'touch_p3a', 'touch_p3b', 'touch_p4a', 'touch_p4b'],
            
            # constraint setting
            'constrain_hazard3Ds': True,  # Constrain robot from being in hazardous areas
            'constrain_indicator': False,  # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

            # lidar setting
            'lidar_num_bins': 10,
            'lidar_num_bins3D': 6,
            
            # object setting
            'hazard3Ds_num': 8,
            'hazard3Ds_size': 0.3,
            'hazard3Ds_z_range': [0.5, 1.5],
            'robber3Ds_num': 2,
            'robber3Ds_size': 0.3,
            'robber3Ds_z_range': [0.5, 1.5],
        }

    if task == "Defense_Drone_8Ghosts":
        config = {
            # robot setting
            'robot_base': 'xmls/drone.xml',  

            # task setting
            'task': 'defense',
            'goal_3D': True,
            'goal_z_range': [0.5,1.5],
            'goal_size': 0.5,
            'defense_range': 2.5,

            # observation setting
            'observe_robber3Ds': True,  # Observe the goal with a lidar sensor
            'observe_ghost3Ds': True,  # Observe the vector from agent to hazards
            'sensors_obs': ['accelerometer', 'velocimeter', 'gyro', 'magnetometer',
                            'touch_p1a', 'touch_p1b', 'touch_p2a', 'touch_p2b',
                            'touch_p3a', 'touch_p3b', 'touch_p4a', 'touch_p4b'],
            
            # constraint setting
            'constrain_ghost3Ds': True,  # Constrain robot from being in hazardous areas
            'constrain_indicator': False,  # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

            # lidar setting
            'lidar_num_bins': 10,
            'lidar_num_bins3D': 6,
            
            # object setting
            'ghost3Ds_num': 8,
            'ghost3Ds_size': 0.3,
            'ghost3Ds_travel':2.5,
            'ghost3Ds_safe_dist': 1.5,
            'ghost3Ds_z_range': [0.5, 1.5],
            'robber3Ds_num': 2,
            'robber3Ds_size': 0.3,
            'robber3Ds_z_range': [0.5, 1.5],
        }
    
    if task == "AllegroHand":
        config = {
            'task_name': 'AllegroHand',
            'IsaacGym':True,
            'headless':True,
        }
        
    if task == "AllegroHandDextremeADR":
        config = {
            'task_name': 'AllegroHandDextremeADR',
            'IsaacGym':True,
            'headless':False,
        }
    
    if task == "AllegroKukaTwoArmsLSTM":
        config = {
            'task_name': 'AllegroKukaTwoArmsLSTM',
            'IsaacGym':True,
            'headless':True,
        }
        
    if task == "AllegroKukaLSTM":
        config = {
            'task_name': 'AllegroKukaLSTM',
            'IsaacGym':True,
            'headless':True,
        }
    
    if task == "Ant":
        config = {
            'task_name': 'Ant',
            'IsaacGym':True,
            'headless':True,
        }
    
    if task == "Anymal":
        config = {
            'task_name': 'Anymal',
            'IsaacGym':True,
            'headless':True,
        }
        
    if task == "AnymalTerrain":
        config = {
            'task_name': 'AnymalTerrain',
            'IsaacGym':True,
            'headless':True,
        }
    
    if task == "BallBalance":
        config = {
            'task_name': 'BallBalance',
            'IsaacGym':True,
            'headless':True,
        }
    
    if task == "Cartpole":
        config = {
            'task_name': 'Cartpole',
            'IsaacGym':True,
            'headless':True,
        }
    
    if task == "FrankaCabinet":
        config = {
            'task_name': 'FrankaCabinet',
            'IsaacGym':True,
            'headless':True,
        }
    
    if task == "Humanoid":
        config = {
            'task_name': 'Humanoid',
            'IsaacGym':True,
            'headless':True,
        }
        
    if task == "Quadcopter":
        config = {
            'task_name': 'Quadcopter',
            'IsaacGym':True,
            'headless':True,
        }
    
    if task == "ShadowHand":
        config = {
            'task_name': 'ShadowHand',
            'IsaacGym':True,
            'headless':True,
        }
    
    if task == "ShadowHandOpenAI_FF":
        config = {
            'task_name': 'ShadowHandOpenAI_FF',
            'IsaacGym':True,
            'headless':True,
        }
    
    if task == "ShadowHandOpenAI_LSTM":
        config = {
            'task_name': 'ShadowHandOpenAI_LSTM',
            'IsaacGym':True,
            'headless':True,
        }
    
    if task == "Trifinger":
        config = {
            'task_name': 'Trifinger',
            'IsaacGym':True,
            'headless':True,
        }
    
    
    return config

def configuration(task):
    try:
        return configuration_list(task)
    except:
        # task name should be in the form of: 
        # <Task>_<Robot>_<Num><Constraint Type> or 
        # <Task>_<Robot>_<Num><Constraint_Type>_noconti
        task_split = task.split("_")
        Task = task_split[0]
        Robot = task_split[1] 
        Constraints = task_split[2]
        Continue = True
        if len(task_split) >=4 and task_split[3] == 'noconti':
            Continue = False
        N = 0
        Type = ""
        for i in range(len(Constraints)):
            if not Constraints[i].isdigit():
                N = int(Constraints[:i])
                Type = Constraints[i:]
                break
        new_task = Task + "_" + Robot  + "_" + "8" + Type 
        config = configuration(new_task)
        config['continue_goal'] = Continue
        config[Type.lower() + '_num'] = N
        return config
    
def create_env(args):
    config = configuration(args.task)
    config['env_num'] = args.env_num
    config['_seed'] = args.seed
    config['num_steps'] = args.max_ep_len
    config['device'] = "cuda:0"
    if "IsaacGym" in config.keys() and config["IsaacGym"] is True:
        env = IsaacGymWrapper(config)
    else:
        env = safe_rl_envs_Engine(config)
    
    
    return env

class IsaacGymWrapper(gym.Env):
    def __init__(self, config={}):
        self.env = isaacgymenvs.make(
            seed=config['_seed'], 
            task=config['task_name'], 
            num_envs=config['env_num'], 
            sim_device=config['device'],
            graphics_device_id=0,
            rl_device=config['device'],
            headless = config['headless']
        )
        self.device = config['device']
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        import ipdb;ipdb.set_trace()
    
    def reset(self):
        obs_dict = self.env.reset()
        return obs_dict["obs"]

    def reset_done(self):
        obs_dict, _ = self.env.reset_done()
        return obs_dict["obs"]
        
    def step(self, action):
        next_obs_dict, reward, done, info = self.env.step(action)
        info["cost"] = torch.zeros(reward.shape, device=self.device, dtype=torch.float)
        return next_obs_dict["obs"], reward, done, info
    