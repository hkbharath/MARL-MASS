from argparse import ArgumentParser
import gym
from highway_env.envs.control_test_env import ControlTestEnv

def parse_args():
    parser = ArgumentParser(description=('Train or evaluate policy on RL environment '
                                                  'using mappo'))
    parser.add_argument('--control', type=str, required=False,
                        default="steer", help="Lateral control type")
    parser.add_argument('--lc-dir', type=str, required=False,
                        default="left", help="Lateral control direction")
    
    return parser.parse_args()

def log_profile(state_hist:dict, action_hist:dict):
    # TODO: call wandb log in loop
    print("\n----------------")
    print(action_hist)
    print(state_hist)
    print("----------------\n")
    
    None

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

    log_profile(state_hist=state_hist, action_hist=action_hist)

if __name__ == "__main__":
    main()