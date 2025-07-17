from marl.mappo import MAPPO
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
import time
from datetime import datetime
from highway_env.envs.merge_env_v1 import MergeEnvMARL
from common.utils import VideoRecorder

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
        "--config",
        type=str,
        required=False,
        default=default_config,
        help="experiment config path",
    )
    parser.add_argument(
        "--evaluation-seeds",
        type=str,
        required=False,
        default=DEFAULT_EVAL_SEEDS,
        help="random seeds for evaluation, split by ,",
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        required=False,
        default=None,
        help="WandB experiment run name",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render simulation and record video",
    )
    args = parser.parse_args()
    return args

def idm_evaluation(env:MergeEnvMARL, output_dir, eval_seeds=[0]):
    rewards = []
    infos = []
    avg_speeds = []
    traffic_speeds = []
    steps = []
    vehicle_speed = []
    vehicle_position = []
    video_recorder = None
    crash_count = []
    step_time = []
    min_headways = []
    merge_percents = []
    video_filename = None

    env.controlled_vehicles.append(None)

    for i in range(len(eval_seeds)):
        avg_speed = 0
        traffic_speed = None
        step = 0
        rewards_i = []
        infos_i = []
        done = False
        step_time_i = 0
        min_headway_i = float('inf')
        state, action_mask = env.reset(is_training=False, testing_seeds=eval_seeds[i])
        
        if output_dir is not None:
            rendered_frame = env.render(mode="rgb_array")
            video_filename = os.path.join(output_dir,
                                        "testing_episode{}".format(self.n_episodes + 1) + '_{}'.format(i) +
                                        '.mp4')
        # Init video recording
        if video_filename is not None:
            print("Recording video to {} ({}x{}x{}@{}fps)".format(video_filename, *rendered_frame.shape,
                                                                    5))
            video_recorder = VideoRecorder(video_filename,
                                            frame_size=rendered_frame.shape, fps=5)
            video_recorder.add_frame(rendered_frame)
        else:
            video_recorder = None

        while not done:
            step += 1
            s_start = time.process_time()
            state, reward, done, info = env.step(None)
            s_time = time.process_time() - s_start

            avg_speed += info["average_speed"]
            min_headway_i = max(0, min(min_headway_i, info["min_headway"]))
            if "traffic_speed" in info:
                traffic_speed = traffic_speed + info["traffic_speed"] \
                                if traffic_speed is not None \
                                else info["traffic_speed"]

            if video_recorder is not None:
                rendered_frame = env.render(mode="rgb_array")
                video_recorder.add_frame(rendered_frame)

            rewards_i.append(reward)
            infos_i.append(info)
            step_time_i += s_time

        # Capture the seeds with crash
        if any(vehicle.crashed for vehicle in env.road.vehicles):
            print("Vehicle crashed in simulation with seed: ", eval_seeds[i])
        rewards.append(rewards_i)
        infos.append(infos_i)
        steps.append(step)
        avg_speeds.append(avg_speed / step)
        crash_count.append(env.is_crashed())
        step_time.append(step_time_i/ step)
        if traffic_speed is not None:
            traffic_speeds.append(traffic_speed / step)
        if "merge_percent" in infos_i[-1]:
            merge_percents.append(infos_i[-1]["merge_percent"])

        min_headways.append(min_headway_i)

    if video_recorder is not None:
        video_recorder.release()
    env.close()

    ext_info = {"steps":steps,
                "avg_speeds": avg_speeds,
                "crash_count": crash_count,
                "step_time": step_time,
                "min_headways": min_headways,
                "traffic_speeds": traffic_speeds,
                "merge_percents": merge_percents}
    # Debug safety violation
    return rewards, (vehicle_speed, vehicle_position), ext_info

def evaluate(args):
    base_dir = args.base_dir
    
    # create an experiment folder
    now = datetime.utcnow().strftime("%b_%d_%H_%M_%S")
    output_dir = base_dir + now
    dirs = init_dir(output_dir)

    config_file = args.config
    config = configparser.ConfigParser()
    if os.path.exists(config_file):
        config.read(config_file)
    else:
        print("Config file:'{0}' not found!".format(config_file))

    video_dir = None
    if args.render:
        video_dir = dirs["eval_videos"]

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

    wb_config = {"env": env.config, "marl": config._sections}

    wandb = init_wandb(config=wb_config, project_name=project_name, exp_name=exp_name)

    state_dim = env.n_s
    action_dim = env.n_a
    test_seeds = args.evaluation_seeds
    seeds = [int(s) for s in test_seeds.split(",")]

    rewards, _, ext_info = idm_evaluation(env, video_dir, eval_seeds=seeds)
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
    min_headway = min(ext_info["min_headways"])
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
                "min_headway": min_headway,
                "episode_len": episode_len_mu,
                "episode_len_stde": episode_len_stde,
                "merge_percent": merge_percent_mu,
                "merge_percent_stde": merge_percent_stde,
            }
        )
        wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
