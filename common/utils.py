import cv2, os
import torch as th
from torch.autograd import Variable
import numpy as np
from shutil import copy
import torch.nn as nn
import fnmatch

from typing import Callable, Union, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from argparse import Namespace


def entropy(p):
    return -th.sum(p * th.log(p), 1)


def kl_log_probs(log_p1, log_p2):
    return -th.sum(th.exp(log_p1) * (log_p2 - log_p1), 1)


# Necessary for my KFAC implementation.
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


def index_to_one_hot(index, dim):
    if isinstance(index, np.int) or isinstance(index, np.int64):
        one_hot = np.zeros(dim)
        one_hot[index] = 1.0
    else:
        one_hot = np.zeros((len(index), dim))
        one_hot[np.arange(len(index)), index] = 1.0
    return one_hot


def to_tensor_var(x, use_cuda=True, dtype="float"):
    FloatTensor = th.cuda.FloatTensor if use_cuda else th.FloatTensor
    LongTensor = th.cuda.LongTensor if use_cuda else th.LongTensor
    ByteTensor = th.cuda.ByteTensor if use_cuda else th.ByteTensor
    if dtype == "float":
        x = np.array(x, dtype=np.float64).tolist()
        return Variable(FloatTensor(x))
    elif dtype == "long":
        x = np.array(x, dtype=np.long).tolist()
        return Variable(LongTensor(x))
    elif dtype == "byte":
        x = np.array(x, dtype=np.byte).tolist()
        return Variable(ByteTensor(x))
    else:
        x = np.array(x, dtype=np.float64).tolist()
        return Variable(FloatTensor(x))


def agg_double_list(l):
    # l: [ [...], [...], [...] ]
    # l_i: result of each step in the i-th episode
    s = [np.sum(np.array(l_i), 0) for l_i in l]
    s_mu = np.mean(np.array(s), 0)
    s_std = np.std(np.array(s), 0)
    return s_mu, s_std


class VideoRecorder:
    """This is used to record videos of evaluations"""

    def __init__(self, filename, frame_size, fps):
        self.video_writer = cv2.VideoWriter(
            filename,
            cv2.VideoWriter_fourcc(*"mp4v"),
            int(fps),
            (frame_size[1], frame_size[0]),
        )

        if not self.video_writer.isOpened():
            print(f"Failed to open video writer for {filename}")

    def add_frame(self, frame):
        self.video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def release(self):
        self.video_writer.release()

    def __del__(self):
        self.release()


def copy_file(tar_dir):
    # env = '.highway-env/envs/common/abstract.py'
    # copy(env, tar_dir)
    # env1 = '.highway_env/envs/merge_env_v1.py'
    # copy(env1, tar_dir)

    env2 = "configs/configs.ini"
    copy(env2, tar_dir)

    models = "MAA2C.py"
    copy(models, tar_dir)
    main = "run_ma2c.py"
    copy(main, tar_dir)
    c1 = "common/Agent.py"
    copy(c1, tar_dir)
    c2 = "common/Memory.py"
    copy(c2, tar_dir)
    c3 = "common/Model.py"
    copy(c3, tar_dir)


def copy_file_ppo(tar_dir, configs=None):
    # env = '.highway-env/envs/common/abstract.py'
    # copy(env, tar_dir)
    # env1 = '.highway_env/envs/merge_env_v1.py'
    # copy(env1, tar_dir)

    env2 = configs if configs else "configs/configs_ppo.ini"
    copy(env2, tar_dir)

    models = "marl/mappo.py"
    copy(models, tar_dir)
    main = "run_mappo.py"
    copy(main, tar_dir)
    c1 = "marl/single_agent/Agent_common.py"
    copy(c1, tar_dir)
    c2 = "marl/single_agent/Memory_common.py"
    copy(c2, tar_dir)
    c3 = "marl/single_agent/Model_common.py"
    copy(c3, tar_dir)


def copy_file_akctr(tar_dir):
    # env = '.highway-env/envs/common/abstract.py'
    # copy(env, tar_dir)
    # env1 = '.highway_env/envs/merge_env_v1.py'
    # copy(env1, tar_dir)

    env2 = "configs/configs_acktr.ini"
    copy(env2, tar_dir)

    models = "MAACKTR.py"
    copy(models, tar_dir)
    main = "run_maacktr.py"
    copy(main, tar_dir)
    c1 = "single_agent/Agent_common.py"
    copy(c1, tar_dir)
    c2 = "single_agent/Memory_common.py"
    copy(c2, tar_dir)
    c3 = "single_agent/Model_common.py"
    copy(c3, tar_dir)


def init_dir(
    base_dir, pathes=["train_videos", "configs", "models", "eval_videos", "eval_logs"]
):
    if not os.path.exists("./results/"):
        os.mkdir("./results/")
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    print("Base dir: ", base_dir)
    dirs = {}
    for path in pathes:
        cur_dir = base_dir + "/%s/" % path
        if not os.path.exists(cur_dir):
            os.mkdir(cur_dir)
        dirs[path] = cur_dir
    return dirs


def init_wandb(config: dict, project_name: str, exp_name: str) -> Any:
    try:
        import wandb
        wandb_var = wandb
        if project_name is None or exp_name is None:
            return None

        # start a new wandb run to track this script
        run = wandb_var.init(
            # set the wandb project where this run will be logged
            project=project_name,
            name=exp_name,
            # track hyperparameters and run metadata
            config=config,
        )

        return run
    except ImportError:
        print("wandb not available logging parameters in the terminal only.")
        return None


def init_logging(
    config: dict,
    args: "Namespace",
    uid: str,
    sfx: Union[str, None] = None,
    project_name: Union[str, None] = None,
    exp_name: Union[str, None] = None,
) -> Any:

    if sfx is not None and exp_name is not None:
        exp_name = exp_name + "-" + sfx

    if project_name is None:
        project_name = "Untracked-default"

    config["run_id"] = uid
    config["cmd_args"] = vars(args)
    wandb_run = init_wandb(config=config, project_name=project_name, exp_name=exp_name)

    return wandb_run


def log_profiles(
    config: dict,
    args: "Namespace",
    cp: dict,
    run_id: str,
    init_logging_f: Callable = init_logging,
    exp_name: Union[str, None] = None,
) -> None:

    for v_id, v_info in cp.items():
        action_hist = v_info["action_hist"]
        state_hist = v_info["state_hist"]
        wandb_run = init_logging_f(
            config=config, args=args, uid=run_id, sfx=v_id, exp_name=exp_name
        )
        if wandb_run:
            wandb_run.define_metric("t_step")
            # define inputs
            wandb_run.define_metric("action.steering", step_metric="t_step")
            wandb_run.define_metric("action.acceleration", step_metric="t_step")
            wandb_run.define_metric("action.ull_acceleration", step_metric="t_step")
            wandb_run.define_metric("action.lc_action", step_metric="t_step")
            wandb_run.define_metric("action.safe_diff.steering", step_metric="t_step")
            wandb_run.define_metric(
                "action.safe_diff.acceleration", step_metric="t_step"
            )

            # define state parameters
            wandb_run.define_metric("state.steering_angle", step_metric="t_step")
            wandb_run.define_metric("state.heading", step_metric="t_step")
            wandb_run.define_metric("state.speed", step_metric="t_step")
            wandb_run.define_metric("state.y", step_metric="t_step")
            wandb_run.define_metric("state.x", step_metric="t_step")
            wandb_run.define_metric("state.vx", step_metric="t_step")
            wandb_run.define_metric("state.vy", step_metric="t_step")

            for action_rec, state_rec in zip(action_hist, state_hist):
                log_entry = {
                    "state": state_rec,
                    "action": action_rec,
                    "t_step": action_rec["t_step"],
                }
                wandb_run.log(log_entry)
            wandb_run.finish()


def get_config_file(base_directory):
    pattern = "*.ini"

    for root, _, filenames in os.walk(base_directory):
        for filename in fnmatch.filter(filenames, pattern):
            return os.path.join(root, filename)
    return ""


def set_torch_seed(torch_seed: int = 0):
    th.manual_seed(torch_seed)
    th.backends.cudnn.benchmark = False
    th.backends.cudnn.deterministic = True
