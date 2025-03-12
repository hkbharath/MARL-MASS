import copy
import numpy as np
from queue import PriorityQueue
from typing import TYPE_CHECKING

from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.controller import MDPVehicle
from highway_env.vehicle.kinematics import Vehicle
from highway_env.envs.common.idm_controller import idm_controller, generate_actions
from highway_env.envs.common.mdp_controller import mdp_controller
from highway_env.road.objects import Obstacle

if TYPE_CHECKING:
    from highway_env.envs.common.abstract import AbstractEnv

def safety_supervisor(env:"AbstractEnv", actions):
    """"
    implementation of safety supervisor
    """
    # make a deep copy of the environment
    actions = list(actions)
    env_copy = copy.deepcopy(env)
    n_points = int(env.config["simulation_frequency"] // env.config["policy_frequency"]) * env.config[
        "n_step"]
    """compute the priority of controlled vehicles"""
    q = PriorityQueue()
    vehicles_and_actions = []  # original vehicle and action

    # reset the trajectories
    for v in env_copy.road.vehicles:
        v.trajectories = []

    index = 0
    for vehicle, action in zip(env_copy.controlled_vehicles, actions):
        """ 1: ramp > straight road
            2: distance to the merging end
            2: small safety room > large safety room
        """
        priority_number = 0

        # v_fl, v_rl = env_copy.road.neighbour_vehicles(vehicle)
        # print(env_copy.road.network.next_lane(vehicle.lane_index, position=vehicle.position))

        # vehicle is on the ramp or not
        if vehicle.lane_index == ("b", "c", 1):
            priority_number = -0.5
            distance_to_merging_end = env.distance_to_merging_end(vehicle)
            priority_number -= (env.ends[2] - distance_to_merging_end) / env.ends[2]
            headway_distance = env._compute_headway_distance(vehicle)
            priority_number += 0.5 * np.log(headway_distance
                                        / (env.config["HEADWAY_TIME"] * vehicle.speed)) if vehicle.speed > 0 else 0
        else:
            headway_distance = env._compute_headway_distance(vehicle)
            priority_number += 0.5 * np.log(headway_distance
                                        / (env.config["HEADWAY_TIME"] * vehicle.speed)) if vehicle.speed > 0 else 0

        priority_number += np.random.rand() * 0.001  # to avoid the same priority number for two vehicles
        q.put((priority_number, [vehicle, action, index]))
        index += 1

    # q is ordered from large to small numbers
    while not q.empty():
        next_item = q.get()
        vehicles_and_actions.append(next_item[1])

    for i, vehicle_and_action in enumerate(vehicles_and_actions):
        first_change = True  # only do the first change

        # if the vehicle is stepped before, reset it
        if len(vehicle_and_action[0].trajectories) == n_points:
            action = vehicle_and_action[1]
            index = vehicle_and_action[2]
            env_copy.controlled_vehicles[index] = copy.deepcopy(env.controlled_vehicles[index])
            vehicle = env_copy.controlled_vehicles[index]
            env_copy.road.vehicles[index] = vehicle
        else:
            vehicle = vehicle_and_action[0]
            action = vehicle_and_action[1]
            index = vehicle_and_action[2]

        available_actions = env._get_available_actions(vehicle, env_copy)
        # vehicle is on the main lane
        if vehicle.lane_index == ("a", "b", 0) or vehicle.lane_index == ("b", "c", 0) or vehicle.lane_index == (
                "c", "d", 0):
            v_fl, v_rl = env_copy.road.surrounding_vehicles(vehicle)
            if len(env_copy.road.network.side_lanes(vehicle.lane_index)) != 0:
                v_fr, v_rr = env_copy.road.surrounding_vehicles(vehicle,
                                                                env_copy.road.network.side_lanes(
                                                                    vehicle.lane_index)[0])
            # assume we can observe the ramp on this road
            elif vehicle.lane_index == ("a", "b", 0) and vehicle.position[0] > env.ends[0]:
                v_fr, v_rr = env_copy.road.surrounding_vehicles(vehicle, ("k", "b", 0))
            else:
                v_fr, v_rr = None, None

        # vehicle is on the ramp
        else:
            v_fr, v_rr = env_copy.road.surrounding_vehicles(vehicle)
            if len(env_copy.road.network.side_lanes(vehicle.lane_index)) != 0:
                v_fl, v_rl = env_copy.road.surrounding_vehicles(vehicle,
                                                                env_copy.road.network.side_lanes(
                                                                    vehicle.lane_index)[0])
            # assume we can observe the straight road on the ramp
            elif vehicle.lane_index == ("k", "b", 0):
                v_fl, v_rl = env_copy.road.surrounding_vehicles(vehicle, ("a", "b", 0))
            else:
                v_fl, v_rl = None, None

        # propograte the vehicle for n steps
        for t in range(n_points):
            # consider the front vehicles first
            for v in [v_fl, v_fr, vehicle, v_rl, v_rr]:
                if isinstance(v, Obstacle) or v is None:
                    continue

                # skip if the vehicle has been stepped before
                if len(v.trajectories) == n_points and i != 0 and v is not vehicle:
                    pass

                # other surrounding vehicles
                else:
                    if type(v) is IDMVehicle:
                        # determine the action in the first time step
                        if t == 0:
                            a = generate_actions(v, env_copy)
                            idm_controller(v, env_copy, a)
                        else:
                            idm_controller(v, env_copy, v.action)

                    elif isinstance(v, MDPVehicle) and v is not vehicle:
                        # use the previous action: idle
                        mdp_controller(v, env_copy,  actions[v.id])
                    elif isinstance(v, MDPVehicle) and v is vehicle:
                        if actions[index] == action:
                            mdp_controller(v, env_copy, action)
                        else:
                            # take the safe action after replace
                            mdp_controller(v, env_copy, actions[index])

            # check collision for every time step TODO: Check
            for other in [v_fl, v_rl, v_fr, v_rr]:
                if isinstance(other, Vehicle):
                    env.check_collision(vehicle, other, other.trajectories[t])

            for other in env_copy.road.objects:
                env.check_collision(vehicle, other, [other.position, other.heading, other.speed])

            if vehicle.crashed:
                # TODO: check multiple collisions during n_points
                # replace with a safety action
                safety_rooms = []
                updated_vehicles = []
                candidate_actions = []
                for a in available_actions:
                    vehicle_copy = copy.deepcopy(env.controlled_vehicles[index])
                    safety_room = env.check_safety_room(vehicle_copy, a, [v_fl, v_rl, v_fr, v_rr],
                                                            env_copy, t)
                    updated_vehicles.append(vehicle_copy)
                    candidate_actions.append(a)
                    safety_rooms.append(safety_room)

                # reset the vehicle trajectory associated with the new action
                env_copy.controlled_vehicles[index] = updated_vehicles[safety_rooms.index(max(safety_rooms))]
                vehicle = env_copy.controlled_vehicles[index]
                env_copy.road.vehicles[index] = vehicle
                if first_change:
                    first_change = False
                    actions[index] = candidate_actions[safety_rooms.index(max(safety_rooms))]
                # TODO: check the collision after replacing the action
                # reset its neighbor's crashed as False if True
                for other in [v_fl, v_rl, v_fr, v_rr]:
                    if isinstance(other, Vehicle) and other.crashed:
                        other.crashed = False

    return tuple(actions)
