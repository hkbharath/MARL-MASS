import copy
import numpy as np
from queue import PriorityQueue
from typing import TYPE_CHECKING
from concurrent.futures import ThreadPoolExecutor, as_completed

from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.controller import MDPVehicle
from highway_env.vehicle.kinematics import Vehicle
from highway_env.envs.common.idm_controller import idm_controller, generate_actions
from highway_env.envs.common.mdp_controller import mdp_controller
from highway_env.road.objects import Obstacle

if TYPE_CHECKING:
    from highway_env.envs.common.abstract import AbstractEnv


def _evaluate_vehicle_action(
    index,
    vehicle,
    original_vehicle,
    available_actions,
    env_copy: "AbstractEnv",
    n_points,
    neighbours,
):
    """
    Helper function to evaluate and update the action for a single vehicle.
    Used for parallel execution.
    Assumes all vehicles have already been propagated for n_points steps.
    """
    sel_action = None

    vehicle_copy = copy.deepcopy(vehicle)
    vehicle_copy.crashed = False
    for t in range(n_points):
        # check the collision for each point trajectory for each neighbour
        vehicle_copy.position = vehicle.trajectories[t][0]
        vehicle_copy.heading = vehicle.trajectories[t][1]
        for neighbour in neighbours:
            if isinstance(neighbour, Vehicle):
                env_copy.check_collision(vehicle_copy, neighbour, neighbour.trajectories[t])

        for other in env_copy.road.objects:
            env_copy.check_collision(
                vehicle_copy, other, [other.position, other.heading, other.speed]
            )

        # Check for crash in the propagated trajectory
        crashed = vehicle_copy.crashed
        if crashed:
            safety_rooms = []
            candidate_actions = []
            # Try all available actions to find the safest
            for a in available_actions:
                vehicle_orig_copy = copy.deepcopy(original_vehicle)
                # Evaluate safety room at each time step
                safety_room = 0
                for t in range(n_points):
                    safety_room += env_copy.check_safety_room(
                        vehicle_orig_copy, a, neighbours, env_copy, t
                    )
                candidate_actions.append(a)
                safety_rooms.append(safety_room)
            # Pick the action with the largest safety room
            best_idx = safety_rooms.index(max(safety_rooms))
            sel_action = candidate_actions[best_idx]
            break

    return index, sel_action


def safety_layer_dmc(env: "AbstractEnv", actions):
    """
    Implementation of safety supervisor with parallel evaluation of actions.
    First, propagate all vehicles for n_points steps.
    Then, in parallel, evaluate and update actions if needed.
    """
    actions = list(actions)
    env_copy = copy.deepcopy(env)
    n_points = (
        int(env.config["simulation_frequency"] // env.config["policy_frequency"])
        * env.config["n_step"]
    )
    q = PriorityQueue()
    vehicles_and_actions = []

    for v in env_copy.road.vehicles:
        v.trajectories = []

    index = 0
    for vehicle, action in zip(env_copy.controlled_vehicles, actions):
        priority_number = 0
        if vehicle.lane_index == ("b", "c", 1):
            priority_number = -0.5
            distance_to_merging_end = env.distance_to_merging_end(vehicle)
            priority_number -= (env.ends[2] - distance_to_merging_end) / env.ends[2]
            headway_distance = env._compute_headway_distance(vehicle)
            priority_number += (
                0.5
                * np.log(
                    headway_distance / (env.config["HEADWAY_TIME"] * vehicle.speed)
                )
                if vehicle.speed > 0
                else 0
            )
        else:
            headway_distance = env._compute_headway_distance(vehicle)
            priority_number += (
                0.5
                * np.log(
                    headway_distance / (env.config["HEADWAY_TIME"] * vehicle.speed)
                )
                if vehicle.speed > 0
                else 0
            )
        priority_number += np.random.rand() * 0.001
        q.put((priority_number, [vehicle, action, index]))
        index += 1

    priority_map = {}

    # --- Step 1: Propagate all vehicles for n_points steps (serially) ---
    while not q.empty():
        next_item = q.get()
        vehicles_and_actions.append(next_item[1])
        vehicle, action, index = next_item[1]
        priority_map[vehicle] = len(priority_map)

        for t in range(n_points):
            if isinstance(vehicle, IDMVehicle):
                if t == 0:
                    a = generate_actions(vehicle, env_copy)
                    idm_controller(vehicle, env_copy, a)
                else:
                    idm_controller(vehicle, env_copy, vehicle.action)
            elif isinstance(vehicle, MDPVehicle):
                mdp_controller(vehicle, env_copy, actions[index])

    # --- Step 2: Update each vehicle's action with respect their neighbours with less priority in parallel ---
    results = actions
    with ThreadPoolExecutor() as executor:
        futures = []
        for vehicle_and_action in vehicles_and_actions:
            vehicle, action, index = vehicle_and_action
            # vehicle is on the main lane
            if (
                vehicle.lane_index == ("a", "b", 0)
                or vehicle.lane_index == ("b", "c", 0)
                or vehicle.lane_index == ("c", "d", 0)
            ):
                v_fl, v_rl = env_copy.road.surrounding_vehicles(vehicle)
                if len(env_copy.road.network.side_lanes(vehicle.lane_index)) != 0:
                    v_fr, v_rr = env_copy.road.surrounding_vehicles(
                        vehicle, env_copy.road.network.side_lanes(vehicle.lane_index)[0]
                    )
                # assume we can observe the ramp on this road
                elif (
                    vehicle.lane_index == ("a", "b", 0)
                    and vehicle.position[0] > env.ends[0]
                ):
                    v_fr, v_rr = env_copy.road.surrounding_vehicles(
                        vehicle, ("k", "b", 0)
                    )
                else:
                    v_fr, v_rr = None, None

            # vehicle is on the ramp
            else:
                v_fr, v_rr = env_copy.road.surrounding_vehicles(vehicle)
                if len(env_copy.road.network.side_lanes(vehicle.lane_index)) != 0:
                    v_fl, v_rl = env_copy.road.surrounding_vehicles(
                        vehicle, env_copy.road.network.side_lanes(vehicle.lane_index)[0]
                    )
                # assume we can observe the straight road on the ramp
                elif vehicle.lane_index == ("k", "b", 0):
                    v_fl, v_rl = env_copy.road.surrounding_vehicles(
                        vehicle, ("a", "b", 0)
                    )
                else:
                    v_fl, v_rl = None, None

            neighbours = []
            for vn in [v_fl, v_rl, v_fr, v_rr]:
                if vn is None or priority_map.get(vn, -10e4) < priority_map[vehicle]:
                    neighbours.append(vn)
                else:
                    neighbours.append(None)

            original_vehicle = env.controlled_vehicles[index]
            available_actions = env._get_available_actions(original_vehicle, env_copy)
            futures.append(
                executor.submit(
                    _evaluate_vehicle_action,
                    index,
                    vehicle,
                    original_vehicle,
                    available_actions,
                    env_copy,
                    n_points,
                    neighbours,
                )
            )

        # Override the action if a new safe action is found
        for future in as_completed(futures):
            idx, new_action = future.result()
            if new_action is not None:
                results[idx] = new_action

    return tuple(results)
