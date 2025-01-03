from ppo import ppo_video
import argparse

if __name__ == '__main__':
    
    parser = ppo_video.video_parser()
    
    from omni.isaac.lab.app import AppLauncher

    # parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
    parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
    parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
    parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
    parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
    parser.add_argument(
        "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
    )
    parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
    # parser.add_argument("--task", type=str, default=None, help="Name of the task.")
    # parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
    parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
    parser.add_argument('--env_num', type=int, default=100)
    parser.add_argument('--max_ep_len', type=int, default=24)
    parser.add_argument('--seed', '-s', type=int, default=0)
    # # append AppLauncher cli args
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    # # launch omniverse app
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
        
    # """Rest everything follows."""

    import gymnasium as isaaclabgym
    from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
    from omni.isaac.lab.utils.dict import print_dict
    from omni.isaac.lab.utils.io import dump_pickle, dump_yaml

    import omni.isaac.lab_tasks  # noqa: F401
    from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
    
    
    env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(
        args.task, use_gpu=1, num_envs=args.env_num, use_fabric=not args.disable_fabric
    )
    
    env_unwrapped = isaaclabgym.make(args.task, cfg=env_cfg, render_mode="rgb_array" if args.video else None)
    
    args.env_unwrapped = env_unwrapped
    ppo_video.video_player(args)
    
    simulation_app.close() 
    # python train.py --task Isaac-Velocity-Flat-G1-v0 --headless --env_num 4096 --hid 128 --l 3 --max_ep_len 24 --epochs 1500 --target_kl 0.01