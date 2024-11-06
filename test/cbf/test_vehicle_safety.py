import argparse
from argparse import ArgumentParser
import gym
from typing import Any, Union
from uuid import uuid4
from highway_env.vehicle.safety.cbf import CBFType

from test.cbf import CBFTestEnv, CBFMATestEnv
from common.utils import init_wandb, log_profiles
from highway_env.vehicle.safe_controller import MDPLCVehicle


def parse_args():
    parser = ArgumentParser(description=("Test CBF constraints"))
    parser.add_argument(
        "--test-type",
        type=str,
        required=False,
        default="lon",
        help="Type of crash scenario being tested",
    )

    parser.add_argument(
        "--safety",
        type=str,
        required=False,
        default="none",
        choices=["none", "cbf-avlon", "cbf-av", "cbf-avs", "cbf-cav"],
        help="Type of CBF constraint being applied",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        required=False,
        default=1.174,
        help="Gamma used in CBF optimisation constraints",
    )
    parser.add_argument(
        "--extreme",
        action="store_true",
        help="Simulate vehicles with extreme speeds",
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        default="safety_test",
        help="Experiment name to be used for wandb logging",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Execute single simulation step to debug CBF values",
    )
    parser.add_argument(
        "--speeds",
        type=str,
        required=False,
        # default="25,25,20",
        help="Speed for the vehicles in the env",
    )
    parser.add_argument(
        "--ix",
        type=str,
        required=False,
        # default="25,50,75",
        help="Initial x position for the vehicles in the env",
    )
    parser.add_argument(
        "--safe-dist",
        type=str,
        required=False,
        default="theadway",
        choices=["theadway", "braking"],
        help="Type of safe distance used in CBF constraints",
    )
    parser.add_argument(
        "--obstacle",
        action="store_true",
        help="Add obstacle at 225m in the test simulation",
    )

    return parser.parse_args()


def setup_logging(
    config: dict,
    args: argparse.Namespace,
    uid: str,
    sfx: str,
    project_name: Union[str, None] = None,
    exp_name: Union[str, None] = None,
) -> Any:
    project_name = "Safety layer"
    if args.exp_name.endswith("-test"):
        project_name += " Tests"
    exp_name = args.exp_name + "-" + args.safety + "-" + args.test_type

    if args.speeds is not None:
        exp_name = exp_name + "-" + args.speeds
    if args.ix is not None:
        exp_name = exp_name + "-" + args.ix

    if args.extreme:
        exp_name = exp_name + "-extr"
    if args.safety != "none":
        exp_name = exp_name + "-g:" + str(args.gamma)

    config["run_id"] = uid
    exp_name = exp_name + "-" + sfx

    wandb_run = init_wandb(config=config, project_name=project_name, exp_name=exp_name)
    return wandb_run


def main():
    args = parse_args()
    run_id = str(uuid4()).split("-")[-1]

    env_id = "cbf-test-v0"
    if args.test_type == "lon":
        env_id = "cbf-test-v0"
        if args.extreme:
            CBFTestEnv.VEHICLE_SPEEDS = [30, 15]
            CBFTestEnv.USE_RANDOM = False
    elif args.test_type in (
        "lat_adj_left_lc",
        "lat_adj_right_lc",
        "lat_ego_left_lc",
        "lat_ego_right_lc",
    ):
        env_id = "cbf-test-v1"
        if args.speeds is not None:
            CBFMATestEnv.VEHICLE_SPEEDS = [int(v) for v in args.speeds.split(",")]
        if args.ix is not None:
            CBFMATestEnv.INIT_POS = [int(v) for v in args.ix.split(",")]

    CBFTestEnv.DEBUG_CBF = args.debug
    MDPLCVehicle.SAFE_DIST = args.safe_dist

    env: CBFTestEnv = gym.make(env_id)
    env.config["safety_guarantee"] = args.safety
    env.config["obstacle"] = args.obstacle

    wandb_run = None
    if not args.debug:
        wandb_run = setup_logging(env.config, args, run_id, "log")

    CBFType.GAMMA_B = args.gamma

    if args.test_type == "lon":
        cprofiles = env.simulate_lon_crash()
    elif args.test_type == "lat_adj_left_lc":
        cprofiles = env.simulate_adj_lc_crash(init_lane=0)
    elif args.test_type == "lat_adj_right_lc":
        cprofiles = env.simulate_adj_lc_crash(init_lane=1)
    elif args.test_type == "lat_ego_left_lc":
        cprofiles = env.simulate_ego_lc_crash(init_lane=1)
    elif args.test_type == "lat_ego_right_lc":
        cprofiles = env.simulate_ego_lc_crash(init_lane=0)
    else:
        raise ValueError("CBF type '{0}' not supported".format(args.test_type))

    env.close()

    if wandb_run:
        wandb_run.finish()

    if not args.debug:
        log_profiles(
            config=env.config,
            args=args,
            cp=cprofiles,
            run_id=run_id,
            init_logging_f=setup_logging,
        )


if __name__ == "__main__":
    main()
