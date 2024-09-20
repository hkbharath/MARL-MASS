import argparse
from argparse import ArgumentParser
import gym
from test.av_control.control_test_env import ControlTestEnv
from highway_env.vehicle.safe_controller import MDPLCVehicle
from common.utils import init_wandb


def parse_args():
    parser = ArgumentParser(
        description=("Train or evaluate policy on RL environment " "using mappo")
    )
    parser.add_argument(
        "--control",
        type=str,
        required=False,
        default="steer",
        help="Lateral control type",
    )
    parser.add_argument(
        "--lc-dir",
        type=str,
        required=False,
        default="left",
        help="Lateral control direction",
    )
    parser.add_argument(
        "--kp",
        type=float,
        required=False,
        default=15,
        help="KP_steer value for steer_vel control",
    )
    parser.add_argument(
        "--rf",
        type=float,
        required=False,
        default=0.15,
        help="Reduction factor for steer_vel control",
    )
    parser.add_argument(
        "--exp", action="store_true", help="Store expected steering velocity ref values"
    )
    parser.add_argument(
        "--midlc",
        action="store_true",
        help="Delay lane change by a specified INIT_STEPS",
    )
    parser.add_argument(
        "--fast", action="store_true", help="Set higher inital speed to 35 m/s"
    )
    parser.add_argument(
        "--slow", action="store_true", help="Set lower inital speed to 15 m/s"
    )

    return parser.parse_args()


def log_profile(
    config: dict, args: argparse.Namespace, state_hist: dict, action_hist: dict
) -> None:
    # TODO: call wandb log in loop
    # print("\n----------------")
    # print(action_hist)
    # print(state_hist)
    # print("----------------\n")

    project_name = "Control profile"
    exp_name = "av-" + args.control + "-" + args.lc_dir
    exp_name = exp_name + "-exp" if args.exp else exp_name
    exp_name = exp_name + "-midlc" if args.midlc else exp_name
    exp_name = exp_name + "-fast" if args.fast else exp_name
    exp_name = exp_name + "-slow" if args.slow else exp_name

    if args.control == "steer_vel":
        exp_name = "{0}-kp:{1}-rf:{2:1.3f}".format(
            exp_name, MDPLCVehicle.KP_STEER, MDPLCVehicle.STEER_TARGET_RF
        )

    wandb_config = {"env_config": config, "test_config": vars(args)}
    wandb = init_wandb(
        config=wandb_config, project_name=project_name, exp_name=exp_name
    )

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
            log_entry = {"action": action_rec, "t_step": action_rec["t_step"]}
            wandb.log(log_entry)

        for state_rec in state_hist:
            log_entry = {"state": state_rec, "t_step": state_rec["t_step"]}
            wandb.log(log_entry)


def main():
    args = parse_args()

    env_id = "control-test-v0"
    if args.control == "steer_vel":
        env_id = "control-test-steer_vel-v0"
        MDPLCVehicle.KP_STEER = args.kp
        MDPLCVehicle.STEER_TARGET_RF = args.rf
        MDPLCVehicle.RECORD_EXPECTED = args.exp

    if args.midlc:
        ControlTestEnv.INIT_STEPS = 3  # 1s simulation time
    if args.fast:
        ControlTestEnv.INIT_SPEED = 35  # 1s simulation time
        MDPLCVehicle.SPEED_MAX = 35
    if args.slow:
        ControlTestEnv.INIT_SPEED = 15  # 1s simulation time

    env: ControlTestEnv = gym.make(env_id)

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

    env.close()
    log_profile(
        config=env.config, args=args, state_hist=state_hist, action_hist=action_hist
    )


if __name__ == "__main__":
    main()
