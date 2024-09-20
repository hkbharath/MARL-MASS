"""
This environment is built on HighwayEnv to test the control inputs.
Bharathkumar Hegde: hegdeb@tcd.ie
Date: 23/07/2024
"""

import numpy as np
from gym.envs.registration import register
from typing import Tuple
import time


from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.envs.common.action import Action


class ControlTestEnv(AbstractEnv):
    """
    An test environment to execute left lane change or right lane change.
    This environment can be used to generate control profiles while excuting the lane change manoeuvres.
    """

    INIT_STEPS = -1
    INIT_SPEED = 25
    n_a = 5

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {
                    "type": "KinematicLC",
                    "vehicles_count": 1,
                },
                "action": {
                    "type": "DiscreteMetaActionLC",
                    "longitudinal": True,
                    "lateral": True,
                },
                "controlled_vehicles": 1,
                "screen_width": 1000,
                "screen_height": 100,
                "centering_position": [1.5, 0.5],
                "scaling": 5,
                "simulation_frequency": 15,  # [Hz]
                "duration": 5,  # time step
                "policy_frequency": 5,  # [Hz]
                "action_masking": False,
                "show_trajectories": True,
                "lateral_control": "steer",
                "safety_guarantee": "none",
            }
        )
        return config

    def _is_terminal(self) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return (
            any(vehicle.crashed for vehicle in self.controlled_vehicles)
            or self.steps >= self.config["duration"] * self.config["policy_frequency"]
        )

    def _reward(self, action: Action) -> np.float:
        return 0

    def reset(self, testing_seeds=0, init_lane: int = 1) -> np.ndarray:
        self._reset_env(init_lane=init_lane)
        return super().reset(is_training=False, testing_seeds=testing_seeds, num_CAV=0)

    def _reset(self, num_CAV=0):
        # Created for comptability
        return

    def _reset_env(self, init_lane: int = 1) -> None:
        self._make_road()
        self._make_vehicle(init_lane=init_lane)

        self.T = int(self.config["duration"] * self.config["policy_frequency"])

    def _make_road(
        self,
    ) -> None:
        """
        Make a straight road with 2 lanes 100m length.
        """
        net = RoadNetwork()

        # Highway lanes
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
        net.add_lane("a", "b", StraightLane([0, 0], [300, 0], line_types=[c, s]))
        net.add_lane("a", "b", StraightLane([0, 4], [300, 4], line_types=[n, c]))

        road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )
        self.road = road

    def _make_vehicle(self, init_lane: int = 1) -> None:

        road = self.road
        self.controlled_vehicles = []

        init_pos = road.network.get_lane(("a", "b", init_lane)).position(25, 0)
        init_speed = self.INIT_SPEED

        safety_layer = self.config["safety_guarantee"]
        lateral_ctrl = self.config["lateral_control"]

        ego_vehicle = self.action_type.vehicle_class(
            safety_layer=safety_layer,
            lateral_ctrl=lateral_ctrl,
            store_profile=True,
            road=road,
            position=init_pos,
            speed=init_speed,
        )
        ego_vehicle.id = 1
        self.controlled_vehicles.append(ego_vehicle)
        road.vehicles.append(ego_vehicle)

    def make_left_lc(self, test_seed=0) -> Tuple[dict, dict]:
        done = False
        obs, _ = self.reset(testing_seeds=test_seed, init_lane=1)
        sim_step = 0

        while not done:
            action = self.ACTIONS_ALL["IDLE"]
            if sim_step > self.INIT_STEPS:
                action = self.ACTIONS_ALL["LANE_LEFT"]
            obs, reward, done, info = self.step(action=action)
            self.render()
            time.sleep(0.1)
            sim_step += 1

        return self.vehicle.state_hist, self.vehicle.action_hist

    def make_right_lc(self, test_seed=0) -> Tuple[dict, dict]:
        done = False
        obs, _ = self.reset(testing_seeds=test_seed, init_lane=0)
        sim_step = 0

        while not done:
            action = self.ACTIONS_ALL["IDLE"]
            if sim_step > self.INIT_STEPS:
                action = self.ACTIONS_ALL["LANE_RIGHT"]
            obs, reward, done, info = self.step(action=action)
            self.render()
            time.sleep(0.1)
            sim_step += 1

        return self.vehicle.state_hist, self.vehicle.action_hist

    def make_zigzag_lc(self, test_seed=0, init_lane=0) -> Tuple[dict, dict]:
        done = False
        obs, _ = self.reset(testing_seeds=test_seed, init_lane=init_lane)

        # Alternate lane every second
        step_count = 0
        curr_action = self.ACTIONS_ALL["LANE_RIGHT"]
        next_action = self.ACTIONS_ALL["LANE_LEFT"]
        sim_step = 0

        if init_lane == 1:
            curr_action, next_action = next_action, curr_action

        while not done:
            action = self.ACTIONS_ALL["IDLE"]
            if sim_step > self.INIT_STEPS:
                action = curr_action
            obs, reward, done, info = self.step(action=action)
            step_count += 1
            self.render()
            time.sleep(0.1)
            sim_step += 1
            sim_step += 1

            # swap lane change action to make alternate lane change
            if step_count == self.config["policy_frequency"]:
                step_count = 0
                curr_action, next_action = next_action, curr_action

        return self.vehicle.state_hist, self.vehicle.action_hist


class ControlTestStreerVEnv(ControlTestEnv):
    """
    An test environment to execute left lane change or right lane change.
    This environment can be used to generate control profiles while excuting the lane change manoeuvres.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "lateral_control": "steer_vel",
            }
        )
        return config


register(
    id="control-test-v0",
    entry_point="test.av_control:ControlTestEnv",
)

register(
    id="control-test-steer_vel-v0",
    entry_point="test.av_control:ControlTestStreerVEnv",
)
