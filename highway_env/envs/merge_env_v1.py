"""
This environment is built on HighwayEnv with one main road and one merging lane.
Dong Chen: chendon9@msu.edu
Date: 01/05/2021
"""
import numpy as np
from gym.envs.registration import register
from typing import Tuple

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv, MultiAgentWrapper
from highway_env.road.lane import LineType, StraightLane, SineLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import ControlledVehicle, MDPVehicle
from highway_env.road.objects import Obstacle
from highway_env.vehicle.kinematics import Vehicle


class MergeEnv(AbstractEnv):
    """
    A highway merge negotiation environment.

    The ego-vehicle is driving on a highway and approached a merge, with some vehicles incoming on the access ramp.
    It is rewarded for maintaining a high speed and avoiding collisions, but also making room for merging
    vehicles.
    """
    n_a = 5
    n_s = 25

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics"},
            "action": {
                "type": "DiscreteMetaAction",
                "longitudinal": True,
                "lateral": True},
            "controlled_vehicles": 1,
            "screen_width": 600,
            "screen_height": 120,
            "centering_position": [0.3, 0.5],
            "scaling": 3,
            "simulation_frequency": 15,  # [Hz]
            "duration": 20,  # time step
            "policy_frequency": 5,  # [Hz]
            "reward_speed_range": [10, 30],
            "COLLISION_REWARD": 200,  # default=200
            "HIGH_SPEED_REWARD": 1,  # default=0.5
            "HEADWAY_COST": 4,  # default=1
            "HEADWAY_TIME": 1.2,  # default=1.2[s]
            "MERGING_LANE_COST": 4,  # default=4
            "traffic_density": 1,  # easy or hard modes
        })
        return config

    def _reward(self, action: int) -> float:
        # Cooperative multi-agent reward
        return sum(self._agent_reward(action, vehicle) for vehicle in self.controlled_vehicles) \
               / len(self.controlled_vehicles)

    def _agent_reward(self, action: int, vehicle: Vehicle) -> float:
        """
            The vehicle is rewarded for driving with high speed on lanes to the right and avoiding collisions
            But an additional altruistic penalty is also suffered if any vehicle on the merging lane has a low speed.
            :param action: the action performed
            :return: the reward of the state-action transition
       """
        # the optimal reward is 0
        scaled_speed = utils.lmap(vehicle.speed, self.config["reward_speed_range"], [0, 1])
        # compute cost for staying on the merging lane
        if vehicle.lane_index == ("b", "c", 1):
            Merging_lane_cost = - np.exp(-(vehicle.position[0] - sum(self.ends[:3])) ** 2 / (
                    10 * self.ends[2]))
        else:
            Merging_lane_cost = 0

        # compute headway cost
        headway_distance = self._compute_headway_distance(vehicle)
        Headway_cost = np.log(
            headway_distance / (self.config["HEADWAY_TIME"] * vehicle.speed)) if vehicle.speed > 0 else 0
        # compute overall reward
        reward = self.config["COLLISION_REWARD"] * (-1 * vehicle.crashed) \
                 + (self.config["HIGH_SPEED_REWARD"] * np.clip(scaled_speed, 0, 1)) \
                 + self.config["MERGING_LANE_COST"] * Merging_lane_cost \
                 + self.config["HEADWAY_COST"] * (Headway_cost if Headway_cost < 0 else 0)
        return reward

    def _regional_reward(self):
        for vehicle in self.controlled_vehicles:
            neighbor_vehicle = []

            # vehicle is on the main road
            if vehicle.lane_index == ("a", "b", 0) or vehicle.lane_index == ("b", "c", 0) or vehicle.lane_index == (
                    "c", "d", 0):
                v_fl, v_rl = self.road.surrounding_vehicles(vehicle)
                if len(self.road.network.side_lanes(vehicle.lane_index)) != 0:
                    v_fr, v_rr = self.road.surrounding_vehicles(vehicle,
                                                                self.road.network.side_lanes(
                                                                    vehicle.lane_index)[0])
                # assume we can observe the ramp on this road
                elif vehicle.lane_index == ("a", "b", 0) and vehicle.position[0] > self.ends[0]:
                    v_fr, v_rr = self.road.surrounding_vehicles(vehicle, ("k", "b", 0))
                else:
                    v_fr, v_rr = None, None
            else:
                # vehicle is on the ramp
                v_fr, v_rr = self.road.surrounding_vehicles(vehicle)
                if len(self.road.network.side_lanes(vehicle.lane_index)) != 0:
                    v_fl, v_rl = self.road.surrounding_vehicles(vehicle,
                                                                self.road.network.side_lanes(
                                                                    vehicle.lane_index)[0])
                # assume we can observe the straight road on the ramp
                elif vehicle.lane_index == ("k", "b", 0):
                    v_fl, v_rl = self.road.surrounding_vehicles(vehicle, ("a", "b", 0))
                else:
                    v_fl, v_rl = None, None
            for v in [v_fl, v_fr, vehicle, v_rl, v_rr]:
                if v is not None and isinstance(v, MDPVehicle):
                    neighbor_vehicle.append(v)
            regional_reward = sum(v.local_reward for v in neighbor_vehicle)
            vehicle.regional_reward = regional_reward / sum(1 for _ in filter(None.__ne__, neighbor_vehicle))

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        agent_info = []
        # min_headway = 1e5

        obs, reward, done, info = super().step(action)
        info["agents_dones"] = tuple(self._agent_is_terminal(vehicle) for vehicle in self.controlled_vehicles)
        for v in self.controlled_vehicles:
            agent_info.append([v.position[0], v.position[1], v.speed])
            # if hasattr(v, "min_headway"):
            #     min_headway = min(min_headway, v.min_headway)
        info["agents_info"] = agent_info
        # info["min_headway"] = min_headway

        for vehicle in self.controlled_vehicles:
            vehicle.local_reward = self._agent_reward(action, vehicle)
        # local reward
        info["agents_rewards"] = tuple(vehicle.local_reward for vehicle in self.controlled_vehicles)
        # regional reward
        self._regional_reward()
        info["regional_rewards"] = tuple(vehicle.regional_reward for vehicle in self.controlled_vehicles)

        obs = np.asarray(obs).reshape((len(obs), -1))
        return obs, reward, done, info

    def _is_terminal(self) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return any(vehicle.crashed for vehicle in self.controlled_vehicles) \
               or self.steps >= self.config["duration"] * self.config["policy_frequency"]\
               or any (vehicle.position[0] < 0 for vehicle in self.controlled_vehicles)

    def _agent_is_terminal(self, vehicle: Vehicle) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return vehicle.crashed \
               or self.steps >= self.config["duration"] * self.config["policy_frequency"] \
               or vehicle.position[0] < 0

    def _num_vehicles(self, num_CAV=0)->Tuple[int, int]:
        if self.config["traffic_density"] == 1:
            # easy mode: 1-3 CAVs + 1-3 HDVs
            if num_CAV == 0:
                num_CAV = np.random.choice(np.arange(1, 4), 1)[0]
            else:
                num_CAV = num_CAV
            num_HDV = np.random.choice(np.arange(1, 4), 1)[0]

        elif self.config["traffic_density"] == 2:
            # hard mode: 2-4 CAVs + 2-4 HDVs
            if num_CAV == 0:
                num_CAV = np.random.choice(np.arange(2, 5), 1)[0]
            else:
                num_CAV = num_CAV
            num_HDV = np.random.choice(np.arange(2, 5), 1)[0]

        elif self.config["traffic_density"] == 3:
            # hard mode: 4-6 CAVs + 3-5 HDVs
            if num_CAV == 0:
                num_CAV = np.random.choice(np.arange(4, 7), 1)[0]
            else:
                num_CAV = num_CAV
            num_HDV = np.random.choice(np.arange(3, 6), 1)[0]
        
        if (self.config["mixed_traffic"] is not None 
            and not self.config["mixed_traffic"]):
            num_CAV = num_CAV + num_HDV
            num_HDV = 0
        
        return num_CAV, num_HDV
    

    def _reset(self, num_CAV=0) -> None:
        self._make_road()

        num_CAV, num_HDV = self._num_vehicles(num_CAV=num_CAV)    
        self._make_vehicles(num_CAV, num_HDV)
        self.action_is_safe = True
        self.T = int(self.config["duration"] * self.config["policy_frequency"])

    def _make_road(self, ) -> None:
        """
        Make a road composed of a straight highway and a merging lane.
        :return: the road
        """
        net = RoadNetwork()

        # Highway lanes
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
        net.add_lane("a", "b", StraightLane([0, 0], [sum(self.ends[:2]), 0], line_types=[c, c]))
        net.add_lane("b", "c",
                     StraightLane([sum(self.ends[:2]), 0], [sum(self.ends[:3]), 0], line_types=[c, s]))
        net.add_lane("c", "d", StraightLane([sum(self.ends[:3]), 0], [sum(self.ends), 0], line_types=[c, c]))

        # Merging lane
        amplitude = 3.25
        ljk = StraightLane([0, 6.5 + 4], [self.ends[0], 6.5 + 4], line_types=[c, c], forbidden=True)
        lkb = SineLane(ljk.position(self.ends[0], -amplitude), ljk.position(sum(self.ends[:2]), -amplitude),
                       amplitude, 2 * np.pi / (2 * self.ends[1]), np.pi / 2, line_types=[c, c], forbidden=True)
        lbc = StraightLane(lkb.position(self.ends[1], 0), lkb.position(self.ends[1], 0) + [self.ends[2], 0],
                           line_types=[n, c], forbidden=True)
        net.add_lane("j", "k", ljk)
        net.add_lane("k", "b", lkb)
        net.add_lane("b", "c", lbc)
        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        road.objects.append(Obstacle(road, lbc.position(self.ends[2], 0)))
        self.road = road

    def _make_ego_vehicle(self, road:Road, position:np.ndarray, speed:float, veh_id:int=0):
        return self.action_type.vehicle_class(road=road, 
                                            position = position, 
                                            speed=speed)
    
    def _make_hdv_vehicle(self, road:Road, position:np.ndarray, speed:float, veh_id:int=0):
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])

        return other_vehicles_type(road, position=position, speed=speed)
        

    def _record_vehicle_count(self, n_merge:int=0, n_highway:int=0) -> None:
        return

    def _make_vehicles(self, num_CAV=4, num_HDV=3) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.
        :return: the ego-vehicle
        """
        road = self.road
        self.controlled_vehicles = []

        # print("CAVs:{0}, HDVs:{1}".format(num_CAV, num_HDV))
        
        # default spawn points
        # spawn_points_s = [10, 50, 90, 130, 170, 210]
        # spawn_points_m = [5, 45, 85, 125, 165, 205]

        # mix 60 diff = td 1 [30-32 m/s?]
        # spawn_points_s = [10, 70, 130, 190, 250]
        # spawn_points_m = [5, 65, 130, 195, 255]

        # mix 50 diff = td 2 [25-27 m/s]
        spawn_points_s = [10, 60, 110, 160, 210, 260]
        spawn_points_m = [5, 55, 105, 155, 205, 255]

        # mix 30 diff = td 3 [15-17 m/s]
        # spawn_points_s = [10, 40, 70, 100, 130, 160]
        # spawn_points_m = [5, 35, 65, 105, 135, 165]
        
        # spawn_points_s = [np.random.randint(pt, pt+5) for pt in range(50, 110, 10)]
        # spawn_points_m = [np.random.randint(pt, pt+6) for pt in range(50, 110, 10)]
        
        """Spawn points for CAV"""
        num_s_c = num_CAV // 2 if num_CAV != 1 else np.random.choice(2)
        num_m_c = num_CAV - num_s_c
        # spawn point indexes on the straight road
        spawn_point_s_c = np.random.choice(spawn_points_s, num_s_c, replace=False)
        # spawn point indexes on the merging road
        spawn_point_m_c = np.random.choice(spawn_points_m, num_m_c,
                                           replace=False)
        spawn_point_s_c = list(spawn_point_s_c)
        spawn_point_m_c = list(spawn_point_m_c)
        # remove the points to avoid duplicate
        for a in spawn_point_s_c:
            spawn_points_s.remove(a)
        for b in spawn_point_m_c:
            spawn_points_m.remove(b)

        """Spawn points for HDV"""
        num_s_h = num_HDV // 2 if num_HDV != 1 else np.random.choice(2)
        num_m_h = num_HDV - num_s_h
        # spawn point indexes on the straight road
        spawn_point_s_h = np.random.choice(spawn_points_s, num_s_h, replace=False)
        # spawn point indexes on the merging road
        spawn_point_m_h = np.random.choice(spawn_points_m, num_m_h,
                                           replace=False)
        spawn_point_s_h = list(spawn_point_s_h)
        spawn_point_m_h = list(spawn_point_m_h)

        # initial speed with noise and location noise
        initial_speed = np.random.rand(num_CAV + num_HDV) * 2 + 25  # range from [25, 27]
        loc_noise = np.random.rand(num_CAV + num_HDV) * 8 - 4  # range from [-4, 4]
        initial_speed = list(initial_speed)
        loc_noise = list(loc_noise)

        """spawn the CAV on the straight road first"""
        for _ in range(num_s_c):
            ego_vehicle = self._make_ego_vehicle(road=road, 
                                                         position = road.network.get_lane(("a", "b", 0))
                                                            .position(spawn_point_s_c.pop(0) + loc_noise.pop(0), 0), 
                                                         speed=initial_speed.pop(0),
                                                         veh_id=len(road.vehicles))
            self.controlled_vehicles.append(ego_vehicle)
            road.vehicles.append(ego_vehicle)
        """spawn the rest CAV on the merging road"""
        for _ in range(num_m_c):
            ego_vehicle = self._make_ego_vehicle(road = road, position = road.network.get_lane(("j", "k", 0))
                                                            .position(spawn_point_m_c.pop(0) + loc_noise.pop(0), 0), 
                                                         speed=initial_speed.pop(0),
                                                         veh_id=len(road.vehicles))
            self.controlled_vehicles.append(ego_vehicle)
            road.vehicles.append(ego_vehicle)

        """spawn the HDV on the main road first"""
        for _ in range(num_s_h):
            hd_vehicle = self._make_hdv_vehicle(road=road,
                                                position=road.network.get_lane(("a", "b", 0)).position(
                                                    spawn_point_s_h.pop(0) + loc_noise.pop(0), 0),
                                                speed=initial_speed.pop(0),
                                                veh_id=len(road.vehicles))
            road.vehicles.append(hd_vehicle)
            

        """spawn the rest HDV on the merging road"""
        for _ in range(num_m_h):
            hd_vehicle = self._make_hdv_vehicle(road=road,
                                                position=road.network.get_lane(("j", "k", 0)).position(
                                                    spawn_point_m_h.pop(0) + loc_noise.pop(0), 0),
                                                speed=initial_speed.pop(0),
                                                veh_id=len(road.vehicles))
            road.vehicles.append(hd_vehicle)
        
        self._record_vehicle_count(n_merge=num_m_c+num_m_h, n_highway=num_s_c+num_s_h)

    def terminate(self):
        return

    def init_test_seeds(self, test_seeds):
        self.test_num = len(test_seeds)
        self.test_seeds = test_seeds


class MergeEnvMARL(MergeEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "action": {
                "type": "MultiAgentAction",
                "action_config": {
                    "type": "DiscreteMetaAction",
                    "lateral": True,
                    "longitudinal": True
                }},
            "observation": {
                "type": "MultiAgentObservation",
                "observation_config": {
                    "type": "Kinematics"
                }},
            "controlled_vehicles": 4
        })
        return config
    
class MergeEnvLCMARL(MergeEnv):

    n_a = 5
    n_s = 30
    n_merge = 0
    
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "action": {
                "type": "MultiAgentAction",
                "action_config": {
                    "type": "DiscreteMetaActionLC",
                    "lateral": True,
                    "longitudinal": True
                }},
            "observation": {
                "type": "MultiAgentObservation",
                "observation_config": {
                    "type": "KinematicLC"
                }},
            "action_masking": False,
            "lateral_control": "steer",
            "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicleHist",
            "traffic_type": "cav", # supported option "cav", "mixed", "av", "hdv"
            "agent_reward": "default" # supported "srew"
        })
        return config
    
    def _agent_reward(self, action: int, vehicle: Vehicle) -> float:
        """
            The vehicle is rewarded for driving with high speed
            But an additional altruistic penalty is also suffered if any vehicle on the merging lane has a low speed.
            :param action: the action performed
            :return: the reward of the state-action transition
       """
        if self.config["agent_reward"] != "srew":
            return super()._agent_reward(action=action, vehicle=vehicle)
        
        # the optimal reward is 0
        scaled_speed = utils.lmap(vehicle.speed, self.config["reward_speed_range"], [0, 1])
        # compute cost for staying on the merging lane
        if vehicle.lane_index == ("b", "c", 1):
            Merging_lane_cost = - np.exp(-(vehicle.position[0] - sum(self.ends[:3])) ** 2 / (
                    10 * self.ends[2]))
        else:
            Merging_lane_cost = 0

        # compute headway cost
        headway_distance = self._compute_headway_distance(vehicle)
        Headway_cost = -1* np.log(
            headway_distance / (self.config["HEADWAY_TIME"] * vehicle.speed)) if vehicle.speed > 0 else 0
        # compute overall reward
        reward = self.config["COLLISION_REWARD"] * (-1 * vehicle.crashed) \
                 + (self.config["HIGH_SPEED_REWARD"] * np.clip(scaled_speed, 0, 1)) \
                 + self.config["MERGING_LANE_COST"] * Merging_lane_cost \
                 + self.config["HEADWAY_COST"] * (Headway_cost if Headway_cost < 0 else 0)
        return reward
    
    def _num_vehicles(self, num_CAV=0)->Tuple[int, int]:
        if self.config["traffic_type"] == "mixed":
            # Backward compatability
            self.config["mixed_traffic"] = True
            return super()._num_vehicles(num_CAV)
        elif self.config["traffic_type"] == "cav":
            self.config["mixed_traffic"] = False
            return super()._num_vehicles(num_CAV)
        elif self.config["traffic_type"] == "av":
            num_CAV, num_HDV = super()._num_vehicles(num_CAV)
            num_HDV = num_CAV + num_HDV -1
            num_CAV = 1
            return num_CAV, num_HDV
        elif self.config["traffic_type"] == "hdv":
            num_CAV, num_HDV = super()._num_vehicles(num_CAV)
            num_HDV = num_CAV + num_HDV
            num_CAV = 0
            return num_CAV, num_HDV

        return super()._num_vehicles(num_CAV)

        
    def _record_vehicle_count(self, n_merge:int=0, n_highway:int=0) -> None:
        self.n_merge = n_merge
        return
   
    def _make_ego_vehicle(self, road:Road, position:np.ndarray, speed:float, veh_id:int=0):
        safety_layer = self.config["safety_guarantee"]
        lateral_ctrl = self.config["lateral_control"]
        ego_v = self.action_type.vehicle_class(safety_layer = safety_layer,
                                              lateral_ctrl = lateral_ctrl,
                                            road=road, 
                                            position = position, 
                                            speed=speed)
        ego_v.id = veh_id
        return ego_v

    def _make_hdv_vehicle(self, road:Road, position:np.ndarray, speed:float, veh_id:int=0):
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])

        hdv_v = other_vehicles_type(road, position=position,
                                    speed=speed)
        hdv_v.id = veh_id
        return hdv_v
    
    def step(self, action):
        traffic_speed = 0

        obs, reward, done, info =  super().step(action)
        for v in self.road.vehicles:
            traffic_speed += v.speed
        traffic_speed = traffic_speed / len(self.road.vehicles)
        info["traffic_speed"] = traffic_speed
        info["min_headway"] = self._compute_min_time_headway()

        if done:
            # Evaluate percentage of vehicle merged into the highway steam.
            n_rem_merge = 0
            for ve in self.road.vehicles:
                if ve.lane_index in [("b", "c", 1), ("k", "b", 0), ("j", "k", 0)]:
                    n_rem_merge = n_rem_merge + 1
            info["merge_percent"] = (self.n_merge - n_rem_merge)/self.n_merge * 100
        
        return obs, reward, done, info
    
    def _reward(self, action: int) -> float:
        # Cooperative multi-agent reward
        if ("traffic_type" in self.config 
            and self.config["traffic_type"] == "hdv"):
            return sum(self._agent_reward(action, vehicle) for vehicle in self.road.vehicles) \
               / len(self.road.vehicles)
        return sum(self._agent_reward(action, vehicle) for vehicle in self.controlled_vehicles) \
               / len(self.controlled_vehicles)
    
    def _compute_min_time_headway(self):
        min_headway = float('inf')
        for veh in self.controlled_vehicles:
            headway_distance = super()._compute_headway_distance(veh)
            for ob in self.road.objects:
                if (abs(ob.position[1] - veh.position[1]) <= 2) and (
                    ob.position[0] > veh.position[0]
                ):
                    hd = ob.position[0] - veh.position[0]
                    if hd < headway_distance:
                        headway_distance = hd
            headway_distance = headway_distance - veh.LENGTH
            min_headway = min(min_headway, headway_distance/(veh.velocity[0] if veh.velocity[0] > 1 else 1))
        return min_headway

class MergeEnvMARLSteerVel(MergeEnvLCMARL):
    n_a = 5
    n_s = 25
    
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "action": {
                "type": "MultiAgentAction",
                "action_config": {
                    "type": "DiscreteMetaActionLC",
                    "lateral": True,
                    "longitudinal": True
                }},
            "observation": {
                "type": "MultiAgentObservation",
                "observation_config": {
                    "type": "Kinematics"
                }},
            "controlled_vehicles": 4,
            "lateral_control": "steer"
        })
        return config

class MergeEnvLCHDV(MergeEnvLCMARL):

    n_a = 5
    n_s = 30
    n_merge = 0
    
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "action": {
                "type": "MultiAgentAction",
                "action_config": {
                    "type": "DiscreteMetaActionLC",
                    "lateral": True,
                    "longitudinal": True
                }},
            "observation": {
                "type": "MultiAgentObservationHDV",
                "observation_config": {
                    "type": "KinematicLC"
                }},
            "action_masking": False,
            "lateral_control": "steer",
            "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicleHist",
            "traffic_type": "hdv", # supported option "cav", "mixed", "av", "hdv"
            "agent_reward": "default" # supported "srew"
        })
        return config
    
    @property
    def vehicle(self) -> Vehicle:
        """First (default) controlled vehicle."""
        return self.road.vehicles[0] if self.road is not None and self.road.vehicles else None
    
    def _compute_min_time_headway_hdvs(self):
        min_headway = float('inf')
        for veh in self.road.vehicles:
            if veh in self.controlled_vehicles:
                continue
            headway_distance = super()._compute_headway_distance(veh)
            for ob in self.road.objects:
                if (abs(ob.position[1] - veh.position[1]) <= 2) and (
                    ob.position[0] > veh.position[0]
                ):
                    hd = ob.position[0] - veh.position[0]
                    if hd < headway_distance:
                        headway_distance = hd
            headway_distance = headway_distance - veh.LENGTH
            min_headway = min(min_headway, headway_distance/(veh.velocity[0] if veh.velocity[0] > 1 else 1))
        return min_headway
    
    def step(self, action):
        """
        Perform an action and step the environment dynamics.

        The action is executed by the ego-vehicle, and all other vehicles on the road performs their default behaviour
        for several simulation timesteps until the next decision making step.

        :param action: the action performed by the ego-vehicle
        :return: a tuple (observation, reward, terminal, info)
        """
        average_speed = 0
        if self.road is None or self.vehicle is None:
            raise NotImplementedError("The road and vehicle must be initialized in the environment implementation")

        self.steps += 1

        self.new_action = action

        # action is a tuple, e.g., (2, 3, 0, 1)
        self._simulate(self.new_action)

        obs = self.observation_type.observe()
        reward = self._reward(action)
        terminal = self._is_terminal()


        for v in self.road.vehicles:
            average_speed += v.speed
        average_speed = average_speed / len(self.road.vehicles)

        self.vehicle_speed.append([v.speed for v in self.road.vehicles])
        self.vehicle_pos.append(([v.position[0] for v in self.road.vehicles]))
        info = {
            "speed": self.vehicle.speed,
            "crashed": self.vehicle.crashed,
            "average_speed": average_speed,
        }

        # if terminal:
        #     # print("steps, action, new_action: ", self.steps, action, self.new_action)
        #     print(self.steps)

        try:
            info["cost"] = self._cost(action)
        except NotImplementedError:
            pass

        info["traffic_speed"] = average_speed
        info["min_headway"] = self._compute_min_time_headway_hdvs()

        if terminal:
            # Evaluate percentage of vehicle merged into the highway steam.
            n_rem_merge = 0
            for ve in self.road.vehicles:
                if ve.lane_index in [("b", "c", 1), ("k", "b", 0), ("j", "k", 0)]:
                    n_rem_merge = n_rem_merge + 1
            info["merge_percent"] = (self.n_merge - n_rem_merge)/self.n_merge * 100

        # print(self.steps)
        return obs, reward, terminal, info
    
    def is_crashed(self):
        return any(vehicle.crashed for vehicle in self.road.vehicles)
    
    def _is_terminal(self) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return any(vehicle.crashed for vehicle in self.road.vehicles) \
               or self.steps >= self.config["duration"] * self.config["policy_frequency"]
    
register(
    id='merge-v1',
    entry_point='highway_env.envs:MergeEnv',
)

register(
    id='merge-multi-agent-v0',
    entry_point='highway_env.envs:MergeEnvMARL',
)

register(
    id='merge-multi-agent-v1',
    entry_point='highway_env.envs:MergeEnvLCMARL',
)

register(
    id='merge-multi-agent-hdv-v1',
    entry_point='highway_env.envs:MergeEnvLCHDV',
)

register(
    id='merge-multi-agent-v05',
    entry_point='highway_env.envs:MergeEnvMARLSteerVel',
)