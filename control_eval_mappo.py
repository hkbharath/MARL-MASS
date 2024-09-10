from marl.mappo import MAPPOControlEval
from common.utils import (
    agg_double_list,
    init_wandb,
    get_config_file,
    set_torch_seed,
    log_profiles,
)

from uuid import uuid4
from typing import Any, Union, Callable
import gym
import argparse
import configparser
import os
from highway_env.envs.merge_env_v1 import MergeEnvLCMARL


def parse_args():
    default_base_dir = "./results/"
    default_config = "configs/configs_marl-cav.ini"
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate control profiles of the trained MAPPO policy using single evalluation runs"
        )
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=False,
        default="",
        help="pretrained model path",
    )
    parser.add_argument(
        "--seed",
        type=str,
        required=False,
        default="319",
        help="random seeds for evaluation, split by ,",
    )
    parser.add_argument(
        "--checkpoint", type=int, default=None, required=False, help="Checkpoint number"
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        required=False,
        default=None,
        help="WandB experiment run name",
    )
    parser.add_argument(
        "--src-url",
        type=str,
        required=False,
        default=None,
        help="WandB URL to link evaluations to source training runs",
    )
    args = parser.parse_args()
    return args


def init_logging(
    config: dict,
    args: argparse.Namespace,
    uid: str,
    sfx: str,
    project_name: Union[str, None] = None,
    exp_name: Union[str, None] = None,
) -> Any:
    if args.exp_name is not None:
        exp_name = args.exp_name
    if args.checkpoint is not None:
        exp_name = exp_name + ":cp-{:d}".format(args.checkpoint)

    exp_name = exp_name + "-" + config["env"]["safety_guarantee"]
    exp_name = exp_name + "-" + sfx

    if project_name is None:
        project_name = "Multi-Vehicle Control"

    config["run_id"] = uid
    wandb_run = init_wandb(config=config, project_name=project_name, exp_name=exp_name)

    return wandb_run


def evaluate(args):
    run_id = str(uuid4()).split("-")[-1]

    if os.path.exists(args.model_dir):
        model_dir = args.model_dir + "/models/"
    else:
        raise Exception("Sorry, no pretrained models")

    config_file = get_config_file(args.model_dir + "/configs/")
    config = configparser.ConfigParser()
    if os.path.exists(config_file):
        config.read(config_file)
    else:
        print("Config file:'{0}' not found!".format(config_file))

    # make the torch seed for reproducibility
    torch_seed = config.getint("MODEL_CONFIG", "torch_seed")
    set_torch_seed(torch_seed=torch_seed)

    video_dir = args.model_dir + "/eval_videos"

    # model configs
    BATCH_SIZE = config.getint("MODEL_CONFIG", "BATCH_SIZE")
    MEMORY_CAPACITY = config.getint("MODEL_CONFIG", "MEMORY_CAPACITY")
    ROLL_OUT_N_STEPS = config.getint("MODEL_CONFIG", "ROLL_OUT_N_STEPS")
    reward_gamma = config.getfloat("MODEL_CONFIG", "reward_gamma")
    actor_hidden_size = config.getint("MODEL_CONFIG", "actor_hidden_size")
    critic_hidden_size = config.getint("MODEL_CONFIG", "critic_hidden_size")
    MAX_GRAD_NORM = config.getfloat("MODEL_CONFIG", "MAX_GRAD_NORM")
    ENTROPY_REG = config.getfloat("MODEL_CONFIG", "ENTROPY_REG")
    reward_type = config.get("MODEL_CONFIG", "reward_type")
    TARGET_UPDATE_STEPS = config.getint("MODEL_CONFIG", "TARGET_UPDATE_STEPS")
    TARGET_TAU = config.getfloat("MODEL_CONFIG", "TARGET_TAU")

    # train configs
    actor_lr = config.getfloat("TRAIN_CONFIG", "actor_lr")
    critic_lr = config.getfloat("TRAIN_CONFIG", "critic_lr")
    EPISODES_BEFORE_TRAIN = config.getint("TRAIN_CONFIG", "EPISODES_BEFORE_TRAIN")
    reward_scale = config.getfloat("TRAIN_CONFIG", "reward_scale")

    # init env
    env_id = config.get("ENV_CONFIG", "env_name", fallback="merge-multi-agent-v0")
    env: "MergeEnvLCMARL" = gym.make(env_id)

    env.config["seed"] = config.getint("ENV_CONFIG", "seed")
    env.config["simulation_frequency"] = config.getint(
        "ENV_CONFIG", "simulation_frequency"
    )
    env.config["duration"] = config.getint("ENV_CONFIG", "duration")
    env.config["policy_frequency"] = config.getint("ENV_CONFIG", "policy_frequency")
    env.config["COLLISION_REWARD"] = config.getint("ENV_CONFIG", "COLLISION_REWARD")
    env.config["HIGH_SPEED_REWARD"] = config.getint("ENV_CONFIG", "HIGH_SPEED_REWARD")
    env.config["HEADWAY_COST"] = config.getint("ENV_CONFIG", "HEADWAY_COST")
    env.config["HEADWAY_TIME"] = config.getfloat("ENV_CONFIG", "HEADWAY_TIME")
    env.config["MERGING_LANE_COST"] = config.getint("ENV_CONFIG", "MERGING_LANE_COST")
    env.config["traffic_density"] = config.getint("ENV_CONFIG", "traffic_density")
    traffic_density = config.getint("ENV_CONFIG", "traffic_density")
    env.config["action_masking"] = config.getboolean("MODEL_CONFIG", "action_masking")
    env.config["safety_guarantee"] = config.get("ENV_CONFIG", "safety_guarantee")
    # TEST Safety layer
    env.config["safety_guarantee"] = "cbf-avlon"
    env.config["lateral_control"] = config.get(
        "ENV_CONFIG", "lateral_control", fallback="steer"
    )
    env.config["mixed_traffic"] = config.getboolean("ENV_CONFIG", "mixed_traffic")

    # init wnadb logging
    project_name = config.get("PROJECT_CONFIG", "name", fallback=None) + "-evaluations"
    exp_name = config.get("PROJECT_CONFIG", "exp_name", fallback="default")

    wb_config = {"env": env.config, "marl": config._sections}
    if args.src_url is not None:
        wb_config["src-url"] = args.src_url
    wandb_run = init_logging(
        config=wb_config,
        args=args,
        uid=run_id,
        sfx="marl_eval",
        project_name=project_name,
        exp_name=exp_name,
    )

    assert env.T % ROLL_OUT_N_STEPS == 0
    state_dim = env.n_s
    action_dim = env.n_a
    test_seeds = args.seed

    mappo = MAPPOControlEval(
        env=env,
        memory_capacity=MEMORY_CAPACITY,
        state_dim=state_dim,
        action_dim=action_dim,
        batch_size=BATCH_SIZE,
        entropy_reg=ENTROPY_REG,
        roll_out_n_steps=ROLL_OUT_N_STEPS,
        actor_hidden_size=actor_hidden_size,
        critic_hidden_size=critic_hidden_size,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        reward_scale=reward_scale,
        target_update_steps=TARGET_UPDATE_STEPS,
        target_tau=TARGET_TAU,
        reward_gamma=reward_gamma,
        reward_type=reward_type,
        max_grad_norm=MAX_GRAD_NORM,
        test_seeds=test_seeds,
        episodes_before_train=EPISODES_BEFORE_TRAIN,
        traffic_density=traffic_density,
    )

    # load the model if exist
    mappo.load(model_dir, train_mode=False, global_step=args.checkpoint)
    rewards, _, ext_info = mappo.evaluation(env, video_dir)
    avg_speeds = ext_info["avg_speeds"]
    crash_count = ext_info["crash_count"]
    step_time = ext_info["step_time"]
    avg_speed_mu, avg_speed_std = agg_double_list(avg_speeds)
    rewards_mu, rewards_std = agg_double_list(rewards)
    crash_count = sum(crash_count)
    step_time_mu, _ = agg_double_list(step_time)
    if wandb_run:
        wandb_run.log(
            {
                "reward": rewards_mu,
                "average_speed": avg_speed_mu,
                "crash_count": crash_count,
                "time_per_step": step_time_mu,
            }
        )
        wandb_run.finish()

    env.close()

    # log control profile of each vehicles
    cprofiles = ext_info["control_profile"]

    log_profiles(
        config=wb_config,
        args=args,
        cp=cprofiles,
        run_id=run_id,
        exp_name=exp_name,
        init_logging_f=init_logging,
    )


if __name__ == "__main__":
    args = parse_args()
    # train or eval
    evaluate(args)
