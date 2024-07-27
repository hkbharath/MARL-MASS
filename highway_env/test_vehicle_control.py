import argparse
from argparse import ArgumentParser
import gym
from highway_env.envs.control_test_env import ControlTestEnv
from common.utils import init_wandb

def parse_args():
    parser = ArgumentParser(description=('Train or evaluate policy on RL environment '
                                                  'using mappo'))
    parser.add_argument('--control', type=str, required=False,
                        default="steer", help="Lateral control type")
    parser.add_argument('--lc-dir', type=str, required=False,
                        default="left", help="Lateral control direction")
    
    return parser.parse_args()

def log_profile(config:dict, args:argparse.Namespace, state_hist:dict, action_hist:dict) -> None:
    # TODO: call wandb log in loop
    # print("\n----------------")
    # print(action_hist)
    # print(state_hist)
    # print("----------------\n")

    project_name = "Control profile"
    exp_name = "av-"+ args.control + "-" + args.lc_dir
    wandb = init_wandb(config=config, project_name=project_name, exp_name=exp_name)

    if wandb:
        wandb.define_metric("t_step")
        # define inputs
        wandb.define_metric("action.steering", step_metric="t_step")
        wandb.define_metric("action.acceleration", step_metric="t_step")
        wandb.define_metric("action.lc_action", step_metric="t_step")

        # define state parameters
        wandb.define_metric("state.steering_angle", step_metric="t_step")
        wandb.define_metric("state.heading", step_metric="t_step")
        wandb.define_metric("state.speed", step_metric="t_step")
        wandb.define_metric("state.y", step_metric="t_step")
        wandb.define_metric("state.x", step_metric="t_step")

        for action_rec in action_hist:
            log_entry = {
                "action": action_rec,
                "t_step": action_rec["t_step"]
            }
            wandb.log(log_entry)

        for state_rec in state_hist:
            log_entry = {
                "state": state_rec,
                "t_step": state_rec["t_step"]
            }
            wandb.log(log_entry)

def main():
    args = parse_args()

    env_id = "control-test-v0"
    if args.control == "steer_vel":
        env_id = "control-test-steer_vel-v0"
    env:ControlTestEnv = gym.make(env_id)

    state_hist = {}
    action_hist = {}
    if args.lc_dir == "left":
        state_hist, action_hist = env.make_left_lc()
    elif args.lc_dir == "right":
        state_hist, action_hist = env.make_right_lc()
    elif args.lc_dir == "z_left":
        state_hist, action_hist = env.make_zigzag_lc(init_lane=0)
    elif args.lc_dir == "z_right":
        state_hist, action_hist = env.make_zigzag_lc(init_lane=1)

    log_profile(config = env.config, args = args, state_hist=state_hist, action_hist=action_hist)

if __name__ == "__main__":
    main()