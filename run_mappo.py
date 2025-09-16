from marl.mappo import MAPPO
from marl.mappo_gi import MAPPO_GI
from common.utils import (
    agg_double_list,
    copy_file_ppo,
    init_dir,
    init_wandb,
    get_config_file,
    set_torch_seed,
)
import sys

import gym
import matplotlib.pyplot as plt
import argparse
import configparser
import os
from datetime import datetime
from highway_env.envs.merge_env_v1 import MergeEnvMARL
from highway_env.vehicle.safety.cbf import CBFType

DEFAULT_EVAL_SEEDS_30 = "132,730,103,874,343,348,235,199,185,442,849,55,784,737,992,854,546,639,902,192,222,622,102,540,771,92,604,556,81,965"
DEFAULT_EVAL_SEEDS_100 = "132,730,103,874,343,348,235,199,185,442,849,55,784,737,992,854,546,639,902,192,222,622,102,540,771,92,604,556,81,965,450,867,762,495,915,149,469,361,429,298,222,354,26,480,611,903,375,447,993,589,977,108,683,401,276,577,205,149,316,143,105,725,515,476,827,317,211,331,845,404,319,116,171,744,272,938,312,961,606,405,329,453,199,373,726,51,459,979,718,854,675,312,39,921,204,919,504,940,663,408"

DEFAULT_EVAL_SEEDS = DEFAULT_EVAL_SEEDS_100


def parse_args():
    default_base_dir = "./results/"
    default_config = "configs/configs_marl-cav.ini"
    parser = argparse.ArgumentParser(
        description=("Train or evaluate policy on RL environment " "using mappo")
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        required=False,
        default=default_base_dir,
        help="experiment base dir",
    )
    parser.add_argument(
        "--option", type=str, required=False, default="train", help="train or evaluate"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        default=default_config,
        help="experiment config path",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=False,
        default="",
        help="pretrained model path",
    )
    parser.add_argument(
        "--evaluation-seeds",
        type=str,
        required=False,
        default=DEFAULT_EVAL_SEEDS,
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
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render simulation and record video",
    )
    args = parser.parse_args()
    return args


def train(args):
    base_dir = args.base_dir
    config_file = args.config
    config = configparser.ConfigParser()
    config.read(config_file)

    # make the torch seed for reproducibility
    torch_seed = config.getint("MODEL_CONFIG", "torch_seed")
    set_torch_seed(torch_seed=torch_seed)

    # create an experiment folder
    now = datetime.utcnow().strftime("%b_%d_%H_%M_%S")
    output_dir = base_dir + now
    dirs = init_dir(output_dir)
    copy_file_ppo(dirs["configs"], configs=config_file)

    if os.path.exists(args.model_dir):
        model_dir = args.model_dir
    else:
        model_dir = dirs["models"]

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
    shared_network = config.getboolean("MODEL_CONFIG", "shared_network", fallback=False)

    # train configs
    actor_lr = config.getfloat("TRAIN_CONFIG", "actor_lr")
    critic_lr = config.getfloat("TRAIN_CONFIG", "critic_lr")
    MAX_EPISODES = config.getint("TRAIN_CONFIG", "MAX_EPISODES")
    EPISODES_BEFORE_TRAIN = config.getint("TRAIN_CONFIG", "EPISODES_BEFORE_TRAIN")
    EVAL_INTERVAL = config.getint("TRAIN_CONFIG", "EVAL_INTERVAL")
    EVAL_EPISODES = config.getint("TRAIN_CONFIG", "EVAL_EPISODES")
    reward_scale = config.getfloat("TRAIN_CONFIG", "reward_scale")

    # CBF conf
    CBFType.GAMMA_B = config.getfloat("ENV_CONFIG", "cbf_eta", fallback=0.0)
    CBFType.TAU = config.getfloat("ENV_CONFIG", "HEADWAY_TIME", fallback=1.2)

    # init env
    env_id = config.get("ENV_CONFIG", "env_name", fallback="merge-multi-agent-v0")
    env: MergeEnvMARL = gym.make(env_id)

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
    env.config["lateral_control"] = config.get(
        "ENV_CONFIG", "lateral_control", fallback="steer"
    )
    env.config["mixed_traffic"] = config.getboolean(
        "ENV_CONFIG", "mixed_traffic", fallback=None
    )
    env.config["traffic_type"] = config.get(
        "ENV_CONFIG", "traffic_type", fallback="cav"
    )
    env.config["agent_reward"] = config.get(
        "ENV_CONFIG", "agent_reward", fallback="default"
    )
    assert env.T % ROLL_OUT_N_STEPS == 0

    env_eval: MergeEnvMARL = gym.make(env_id)
    env_eval.config["seed"] = config.getint("ENV_CONFIG", "seed") + 1
    env_eval.config["simulation_frequency"] = config.getint(
        "ENV_CONFIG", "simulation_frequency"
    )
    env_eval.config["duration"] = config.getint("ENV_CONFIG", "duration")
    env_eval.config["policy_frequency"] = config.getint(
        "ENV_CONFIG", "policy_frequency"
    )
    env_eval.config["COLLISION_REWARD"] = config.getint(
        "ENV_CONFIG", "COLLISION_REWARD"
    )
    env_eval.config["HIGH_SPEED_REWARD"] = config.getint(
        "ENV_CONFIG", "HIGH_SPEED_REWARD"
    )
    env_eval.config["HEADWAY_COST"] = config.getint("ENV_CONFIG", "HEADWAY_COST")
    env_eval.config["HEADWAY_TIME"] = config.getfloat("ENV_CONFIG", "HEADWAY_TIME")
    env_eval.config["MERGING_LANE_COST"] = config.getint(
        "ENV_CONFIG", "MERGING_LANE_COST"
    )
    env_eval.config["traffic_density"] = config.getint("ENV_CONFIG", "traffic_density")
    env_eval.config["action_masking"] = config.getboolean(
        "MODEL_CONFIG", "action_masking"
    )
    env_eval.config["safety_guarantee"] = config.get("ENV_CONFIG", "safety_guarantee")
    env_eval.config["lateral_control"] = config.get(
        "ENV_CONFIG", "lateral_control", fallback="steer"
    )
    env_eval.config["mixed_traffic"] = config.getboolean(
        "ENV_CONFIG", "mixed_traffic", fallback=None
    )
    env_eval.config["traffic_type"] = config.get(
        "ENV_CONFIG", "traffic_type", fallback="cav"
    )
    env_eval.config["agent_reward"] = config.get(
        "ENV_CONFIG", "agent_reward", fallback="default"
    )

    state_dim = env.n_s
    action_dim = env.n_a
    test_seeds = ",".join([str(i) for i in range(0, 600, 20)])

    # init wnadb logging
    project_name = config.get("PROJECT_CONFIG", "name", fallback=None)
    exp_name = config.get("PROJECT_CONFIG", "exp_name", fallback=None)
    if args.exp_name is not None:
        exp_name = args.exp_name
    wb_config = {
        "env": env.config,
        "marl": config._sections,
        "base_dir": os.path.abspath(output_dir),
    }
    wandb = init_wandb(config=wb_config, project_name=project_name, exp_name=exp_name)

    if not shared_network:
        mappo = MAPPO(
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
    else:
        mappo = MAPPO_GI(
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
            shared_network=shared_network,
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
    mappo.load(model_dir, train_mode=True)
    env.seed = env.config["seed"]
    env.unwrapped.seed = env.config["seed"]
    eval_rewards = []

    video_dir = None
    if args.render:
        video_dir = dirs["train_videos"]

    while mappo.n_episodes < MAX_EPISODES:
        mappo.interact()
        if mappo.n_episodes >= EPISODES_BEFORE_TRAIN:
            mappo.train()
        if mappo.episode_done and ((mappo.n_episodes + 1) % EVAL_INTERVAL == 0):
            rewards, _, ext_info = mappo.evaluation(env_eval, video_dir, EVAL_EPISODES)
            avg_speeds = ext_info["avg_speeds"]
            crash_count = ext_info["crash_count"]
            step_time = ext_info["step_time"]

            rewards_mu, rewards_std = agg_double_list(rewards)
            print(
                "Episode %d, Average Reward %.2f" % (mappo.n_episodes + 1, rewards_mu)
            )
            eval_rewards.append(rewards_mu)

            avg_speed_mu, avg_speed_std = agg_double_list(avg_speeds)
            traffic_speed_mu, traffic_speed_std = agg_double_list(
                ext_info["traffic_speeds"]
            )
            crash_count = sum(crash_count)
            step_time_mu, _ = agg_double_list(step_time)
            episode_len_mu, episode_len_std = agg_double_list(ext_info["steps"])
            merge_percent_mu, merge_percent_stde = agg_double_list(
                ext_info["merge_percents"]
            )
            if wandb:
                wandb.log(
                    {
                        "reward": rewards_mu,
                        "average_speed": avg_speed_mu,
                        "crash_count": crash_count,
                        "time_per_step": step_time_mu,
                        "traffic_speed": traffic_speed_mu,
                        "min_headway": ext_info["min_headway"],
                        "min_headway_training": mappo.train_min_headway,
                        "episode": mappo.n_episodes + 1,
                        "episode_len": episode_len_mu,
                        "episode_len_std": episode_len_std,
                        "merge_percent": merge_percent_mu,
                        "merge_percent_stde": merge_percent_stde,
                    }
                )
            # Reset min headway
            mappo.train_min_headway = float("inf")
            # save the model
            mappo.save(dirs["models"], mappo.n_episodes + 1)

    # save the model
    mappo.save(dirs["models"], MAX_EPISODES + 2)

    if wandb:
        wandb.finish()

    # plt.figure()
    # plt.plot(eval_rewards)
    # plt.xlabel("Episode")
    # plt.ylabel("Average Reward")
    # plt.legend(["MAPPO"])
    # plt.show()


def evaluate(args):
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

    video_dir = None
    if args.render:
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
    shared_network = config.getboolean("MODEL_CONFIG", "shared_network", fallback=False)

    # train configs
    actor_lr = config.getfloat("TRAIN_CONFIG", "actor_lr")
    critic_lr = config.getfloat("TRAIN_CONFIG", "critic_lr")
    EPISODES_BEFORE_TRAIN = config.getint("TRAIN_CONFIG", "EPISODES_BEFORE_TRAIN")
    reward_scale = config.getfloat("TRAIN_CONFIG", "reward_scale")

    # CBF conf
    CBFType.GAMMA_B = config.getfloat("ENV_CONFIG", "cbf_eta", fallback=0.0)
    CBFType.TAU = config.getfloat("ENV_CONFIG", "HEADWAY_TIME", fallback=1.2)

    # init env
    env_id = config.get("ENV_CONFIG", "env_name", fallback="merge-multi-agent-v0")
    env: MergeEnvMARL = gym.make(env_id)

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
    env.config["lateral_control"] = config.get(
        "ENV_CONFIG", "lateral_control", fallback="steer"
    )
    env.config["mixed_traffic"] = config.getboolean(
        "ENV_CONFIG", "mixed_traffic", fallback=None
    )
    env.config["traffic_type"] = config.get(
        "ENV_CONFIG", "traffic_type", fallback="cav"
    )
    env.config["agent_reward"] = config.get(
        "ENV_CONFIG", "agent_reward", fallback="default"
    )

    # init wnadb logging
    project_name = config.get("PROJECT_CONFIG", "name", fallback=None) + "-evaluations"
    exp_name = config.get("PROJECT_CONFIG", "exp_name", fallback="default")
    if args.exp_name is not None:
        exp_name = args.exp_name
    if args.checkpoint is not None:
        exp_name = exp_name + ":cp-{:d}".format(args.checkpoint)

    wb_config = {"env": env.config, "marl": config._sections}
    if args.src_url is not None:
        wb_config["src-url"] = args.src_url
    wandb = init_wandb(config=wb_config, project_name=project_name, exp_name=exp_name)

    assert env.T % ROLL_OUT_N_STEPS == 0
    state_dim = env.n_s
    action_dim = env.n_a
    test_seeds = args.evaluation_seeds
    seeds = [int(s) for s in test_seeds.split(",")]

    if not shared_network:
        mappo = MAPPO(
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
    else:
        mappo = MAPPO_GI(
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
            shared_network=shared_network,
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
    rewards, _, ext_info = mappo.evaluation(env, video_dir, len(seeds), is_train=False)
    avg_speeds = ext_info["avg_speeds"]
    crash_count = ext_info["crash_count"]
    step_time = ext_info["step_time"]
    avg_speed_mu, avg_speed_stde = agg_double_list(avg_speeds)
    rewards_mu, rewards_stde = agg_double_list(rewards)
    traffic_speed_mu, traffic_speed_stde = agg_double_list(ext_info["traffic_speeds"])
    crash_count = sum(crash_count)
    step_time_mu, _ = agg_double_list(step_time)
    episode_len_mu, episode_len_stde = agg_double_list(ext_info["steps"])
    merge_percent_mu, merge_percent_stde = agg_double_list(ext_info["merge_percents"])
    if wandb:
        wandb.log(
            {
                "reward": rewards_mu,
                "reward_stde": rewards_stde,
                "average_speed": avg_speed_mu,
                "average_speed_stde": avg_speed_stde,
                "crash_count": crash_count,
                "time_per_step": step_time_mu,
                "traffic_speed": traffic_speed_mu,
                "traffic_speed_stde": traffic_speed_stde,
                "min_headway": ext_info["min_headway"],
                "episode_len": episode_len_mu,
                "episode_len_stde": episode_len_stde,
                "merge_percent": merge_percent_mu,
                "merge_percent_stde": merge_percent_stde,
            }
        )
        wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    # train or eval
    if args.option == "train":
        train(args)
    else:
        evaluate(args)
