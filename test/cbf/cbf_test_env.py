"""
This environment is built on HighwayEnv to test the control inputs.
Bharathkumar Hegde: hegdeb@tcd.ie
Date: 23/07/2024
"""

import numpy as np
from gym.envs.registration import register
from typing import Tuple
import time
from highway_env.utils import class_from_path
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.envs.common.action import Action
from highway_env.vehicle.safe_controller import MDPLCVehicle
from highway_env.road.objects import Obstacle


class CBFTestEnv(AbstractEnv):
    """
    An test environment to simulate a crash on the same lane. Using CBF safety layer should avoid the crash.
    """

    n_a = 5
    VEHICLE_SPEEDS = [25, 20]
    INIT_POS = [25, 65]
    USE_RANDOM = True
    DEBUG_CBF = False

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "action": {
                    "type": "MultiAgentAction",
                    "action_config": {
                        "type": "DiscreteMetaActionLC",
                        "lateral": True,
                        "longitudinal": True,
                    },
                },
                "observation": {
                    "type": "MultiAgentObservation",
                    "observation_config": {"type": "KinematicLC"},
                },
                "controlled_vehicles": 2,
                "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicleHist",
                "screen_width": 1000,
                "screen_height": 100,
                "centering_position": [1, 0.5],
                "scaling": 3,
                "simulation_frequency": 15,  # [Hz]
                "duration": 10,  # time step
                "policy_frequency": 5,  # [Hz]
                "action_masking": False,
                "show_trajectories": False,
                "lateral_control": "steer_vel",
                "safety_guarantee": "cbf-avlon",  # Options: "none", "priority", "cbf-avlon", "cbf-av", "cbf-avs", "cbf-cav"
                "obstacle": True,
            }
        )
        return config

    def _is_terminal(self) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        if (
            any(vehicle.crashed for vehicle in self.controlled_vehicles)
            or self.steps >= self.config["duration"] * self.config["policy_frequency"]
        ):
            time.sleep(5)
            return True
        return False

    def _reward(self, action: Action) -> np.float:
        return 0

    def reset(self, testing_seeds=0, init_lane: int = 0) -> np.ndarray:
        self._reset_env(init_lane=init_lane)
        return super().reset(is_training=False, testing_seeds=testing_seeds, num_CAV=0)

    def _reset(self, num_CAV=0):
        # Created for comptability
        return

    def _reset_env(self, init_lane: int = 0) -> None:
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
        net.add_lane("a", "b", StraightLane([0, 0], [500, 0], line_types=[c, s]))

        rab = StraightLane([0, 4], [500, 4], line_types=[n, c])
        net.add_lane("a", "b", rab)

        road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )
        if self.config["obstacle"]:
            road.objects.append(Obstacle(road, rab.position(225, 0)))
        self.road = road

    def _make_vehicle(self, init_lane: int = 0) -> None:

        road = self.road
        self.controlled_vehicles = []

        init_pos = road.network.get_lane(("a", "b", init_lane)).position(
            self.INIT_POS[0], 0
        )
        init_speed = self.VEHICLE_SPEEDS[0]
        if self.USE_RANDOM:
            init_speed = init_speed + np.random.rand() * 2

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

        init_pos_o = road.network.get_lane(("a", "b", init_lane)).position(
            self.INIT_POS[1], 0
        )
        init_speed_o = self.VEHICLE_SPEEDS[1]
        if self.USE_RANDOM:
            init_speed_o = init_speed_o + np.random.rand() * 2

        lead_vehicle = self.action_type.vehicle_class(
            safety_layer=safety_layer,
            lateral_ctrl=lateral_ctrl,
            store_profile=True,
            road=road,
            position=init_pos_o,
            speed=init_speed_o,
        )
        lead_vehicle.id = 0
        self.controlled_vehicles.append(lead_vehicle)
        road.vehicles.append(lead_vehicle)

    def simulate_lon_crash(self, test_seed=0) -> Tuple[dict, dict]:
        done = False
        obs, _ = self.reset(testing_seeds=test_seed, init_lane=0)

        step = 0

        while not done:
            obs, reward, done, info = self.step((self.ACTIONS_ALL["FASTER"], self.ACTIONS_ALL["IDLE"]))
            self.render()
            time.sleep(0.1)
            step += 1
            if self.DEBUG_CBF and step > 1:
                done = True
        time.sleep(1)

        return self._vehicle_profiles()

    def _vehicle_profiles(self):
        cprofiles = {}
        for v in self.road.vehicles:
            if isinstance(v, MDPLCVehicle):
                cprofiles["av" + str(v.id)] = {
                    "state_hist": v.state_hist,
                    "action_hist": v.action_hist,
                }
            else:
                cprofiles["hdv" + str(v.id)] = {
                    "state_hist": v.state_hist,
                    "action_hist": v.action_hist,
                }
        return cprofiles


class CBFMATestEnv(CBFTestEnv):
    """
    An test environment to simulate a crash on the same lane. Using CBF safety layer should avoid the crash.
    """

    n_a = 5

    EXTR_VEHICLE_SPEEDS = [30, 30, 15]
    VEHICLE_SPEEDS = [25, 25, 20]
    """ Speed of ego, adjacent, and leading vehicles"""
    INIT_POS = [25, 65, 105]
    """ Initial positions of ego, adjacent, and leading vehicles"""

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "action": {
                    "type": "MultiAgentAction",
                    "action_config": {
                        "type": "DiscreteMetaActionLC",
                        "lateral": True,
                        "longitudinal": True,
                    },
                },
                "observation": {
                    "type": "MultiAgentObservation",
                    "observation_config": {"type": "KinematicLC"},
                },
                "controlled_vehicles": 3,
                "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicleL",
                "screen_width": 1000,
                "screen_height": 100,
                "centering_position": [1, 0.5],
                "scaling": 3,
                "simulation_frequency": 15,  # [Hz]
                "duration": 10,  # time step
                "policy_frequency": 5,  # [Hz]
                "action_masking": False,
                "show_trajectories": False,
                "lateral_control": "steer_vel",
                "safety_guarantee": "cbf-av",  # Options: "none", "priority", "cbf-avlon", "cbf-av", "cbf-cav"
            }
        )
        return config

    def _make_vehicle(self, init_lane: int = 0) -> None:

        road = self.road
        self.controlled_vehicles = []
        self.init_lane = init_lane

        # vid info: 2: leading, 1: adjacent, 0:ego
        for vid in range(len(self.INIT_POS) - 1, -1, -1):
            # This works only in two lane cases
            if vid % 2 == 0:
                spawn_init_lane = self.init_lane
            else:
                spawn_init_lane = 1 - self.init_lane

            init_pos = road.network.get_lane(("a", "b", spawn_init_lane)).position(
                self.INIT_POS[vid], 0
            )
            init_speed = self.VEHICLE_SPEEDS[vid]

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
            ego_vehicle.id = vid
            self.controlled_vehicles.append(ego_vehicle)
            road.vehicles.append(ego_vehicle)

    def simulate_adj_lc_crash(self, test_seed=0, init_lane=0) -> Tuple[dict, dict]:
        done = False
        obs, _ = self.reset(testing_seeds=test_seed, init_lane=init_lane)

        step = 0

        adj_action = (
            self.ACTIONS_ALL["LANE_LEFT"]
            if self.init_lane == 0
            else self.ACTIONS_ALL["LANE_RIGHT"]
        )

        while not done:
            action = (
                self.ACTIONS_ALL["IDLE"],
                self.ACTIONS_ALL["IDLE"],
                self.ACTIONS_ALL["FASTER"],
            )


            action = (
                self.ACTIONS_ALL["IDLE"],
                adj_action,
                self.ACTIONS_ALL["FASTER"],
            )
            obs, reward, done, info = self.step(action)
            self.render()
            time.sleep(0.1)
            step += 1
            if self.DEBUG_CBF and step > 1:
                done = True
        time.sleep(1)

        return self._vehicle_profiles()

    def simulate_ego_lc_crash(self, test_seed=0, init_lane=0) -> Tuple[dict, dict]:
        done = False
        obs, _ = self.reset(testing_seeds=test_seed, init_lane=init_lane)

        step = 0

        ego_action = (
            self.ACTIONS_ALL["LANE_RIGHT"]
            if self.init_lane == 0
            else self.ACTIONS_ALL["LANE_LEFT"]
        )

        while not done:
            obs, reward, done, info = self.step(
                (self.ACTIONS_ALL["IDLE"], self.ACTIONS_ALL["IDLE"], ego_action)
            )
            self.render()
            time.sleep(0.1)
            step += 1
            if self.DEBUG_CBF and step > 1:
                done = True
        time.sleep(1)

        return self._vehicle_profiles()

    def simulate_random_lcs(self, test_seed=0, init_lane=0):
        done = False
        obs, _ = self.reset(testing_seeds=test_seed, init_lane=init_lane)

        step = 0

        while not done:
            if step < 30:
                actions = np.random.choice(
                    len(self.ACTIONS_ALL),
                    size=len(self.controlled_vehicles),
                    replace=True,
                )
            else:
                actions = [self.ACTIONS_ALL["FASTER"]] * len(self.controlled_vehicles)
            obs, reward, done, info = self.step(tuple(actions))
            self.render()
            time.sleep(0.1)
            step += 1
            if self.DEBUG_CBF and step > 1:
                done = True
        time.sleep(1)

        return self._vehicle_profiles()


register(
    id="cbf-test-v0",
    entry_point="test.cbf:CBFTestEnv",
)

register(
    id="cbf-test-v1",
    entry_point="test.cbf:CBFMATestEnv",
)
