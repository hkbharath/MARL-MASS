import copy
import numpy as np
from typing import TYPE_CHECKING, List, Union

from highway_env.vehicle.safety.cbf import CBFType, cbf_factory

if TYPE_CHECKING:
    from highway_env.road.road import Road
    from highway_env.vehicle.safe_controller import MDPLCVehicle
    from highway_env.vehicle.controller import ControlledVehicle
    from highway_env.vehicle.behavior import IDMVehicleL


def is_same_lane(road: "Road", vehicle:"ControlledVehicle", lane_index_2):
    lane_index_1 = vehicle.lane_index
    return lane_index_1 == lane_index_2 or lane_index_2 == road.network.next_lane(
        lane_index_1, position=vehicle.position
    )


def is_adj_lane(road: "Road", vehicle:"ControlledVehicle", lane_index_2):
    lane_index_1 = vehicle.lane_index
    next_lane = road.network.next_lane(lane_index_1, position=vehicle.position)
    return (
        lane_index_1[:-1] == lane_index_2[:-1]
        and abs(lane_index_1[-1] - lane_index_2[-1]) == 1
    ) or (
        lane_index_2[:-1] == next_lane[:-1]
        and abs(lane_index_2[-1] - next_lane[-1]) == 1
    )


def safe_action_longitudinal(
    cbf: "CBFType",
    action,
    vehicle: "MDPLCVehicle",
    road: "Road",
    dt: float,
    perception_dist,
):

    # Identify leading vehicle
    perception_dist = 6 * vehicle.SPEED_MAX

    s_e = vehicle.to_dict()
    sf_e = {k: s_e[k] for k in CBFType.STATE_SPACE}

    sf_o = {k: s_e[k] + 100 for k in CBFType.STATE_SPACE}

    leading_vehicle: List[ControlledVehicle] = road.close_vehicles_to(
        vehicle, perception_dist, count=5, see_behind=False
    )

    for veh in leading_vehicle:
        if veh.lane_index == vehicle.lane_index:
            s_o = veh.to_dict()
            sf_o = {k: s_o[k] if k in s_o else 0 for k in CBFType.STATE_SPACE}
            break

    fgp_e = vehicle.fg_params
    if fgp_e is None:
        raise AttributeError(
            "fg_params not found in the the vehicle class: {0}".format(type(vehicle))
        )
    fp_e = fgp_e["f"]
    gp_e = fgp_e["g"]

    # Worst case acceleration is assumed for observed vehicle
    # Unactuated dynamics of heading and steering angle of observed vehicle are ignored here
    f = np.ravel(
        np.array(
            [
                [fp_e["x"] * dt, fp_e["y"] * dt, 0, 0, fp_e["heading"], 0],
                [
                    sf_o["vx"] * dt,
                    sf_o["vy"] * dt,
                    CBFType.ACCELERATION_RANGE[0] * dt,
                    0,
                    0,
                    0,
                ],
            ]
        )
    )

    g = np.reshape(
        np.array(
            [
                [0, 0, gp_e["vx"] * dt, gp_e["vy"] * dt, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1 * dt],
                [0, 0, 0, 0, 0, 0],
            ]
        ),
        (cbf.action_size, -1),
    )
    g = np.transpose(g)

    x = np.ravel(np.array([list(sf_e.values()), list(sf_o.values())]))

    # TODO: Check shapes of f,g,x
    # assert(f.shape, (len(CBFType.STATE_SPACE)*2),)
    # assert(g.shape, (len(CBFType.STATE_SPACE)*2, cbf.action_size))
    # assert(x.shape, (len(CBFType.STATE_SPACE)*2),)

    u_ll = np.array([action["acceleration"], action["steering"]])
    u_safe = cbf.control_barrier(u_ll, f, g, x)

    print("u_safe: ", u_safe)
    safe_action = {"acceleration": u_safe[0], "steering": u_safe[1]}
    safe_diff = {"acceleration": u_safe[0] - u_ll[0], "steering": u_safe[1] - u_ll[1]}
    return safe_action, safe_diff, cbf.get_status()


def safe_action_av(
    cbf: "CBFType",
    action,
    vehicle: "MDPLCVehicle",
    road: "Road",
    dt: float,
    safe_dist: str,
    perception_dist,
):

    # print(
    #     "========================Vehicle:{}=======================".format(vehicle.id)
    # )
    perception_dist = 6 * vehicle.SPEED_MAX

    s_e = vehicle.to_dict()
    sf_e = {k: s_e[k] for k in cbf.STATE_SPACE}

    # Assume a virtual vehicle stopped beyond the perception
    sf_ol = {}
    for k in cbf.STATE_SPACE:
        if k == "x":
            sf_ol[k] = s_e[k] + perception_dist + 1
        elif k=="y":
            sf_ol[k] = s_e[k]
        else:
            sf_ol[k] = 0.0

    sf_oa = {}
    for k in cbf.STATE_SPACE:
        if k == "x":
            sf_oa[k] = s_e[k] + perception_dist + 1
        elif k == "y":
            if vehicle.lane_index[2] == 1:
                sf_oa[k] = s_e[k] - vehicle.lane.DEFAULT_WIDTH
            elif vehicle.lane_index[2] == 0:
                sf_oa[k] = s_e[k] + vehicle.lane.DEFAULT_WIDTH
        else:
            sf_oa[k] = 0.0
    # Leading vehicles are ordered by increasing distance from the ego vehicle
    leading_vehicles: List[Union["MDPLCVehicle", "IDMVehicleL"]] = (
        road.close_vehicles_to(vehicle, perception_dist, count=5, see_behind=True)
    )

    sf_oar = copy.deepcopy(sf_oa)
    sf_oar["x"] = s_e["x"] - perception_dist - 1

    s_ol, s_oa, s_oar = None, None, None

    for veh in leading_vehicles:
        # rear vehicle in the adjacent lane
        if vehicle.lane_distance_to(veh) < 0:
            if s_oar is None and is_adj_lane(road, veh, vehicle.lane_index):
                # print(
                #     "=====================Rear adjacent Vehicle: {}=====================".format(
                #         veh.id
                #     )
                # )
                s_oar = veh.to_dict()
                sf_oar = {k: s_oar[k] if k in s_oar else 0 for k in cbf.STATE_SPACE}
            continue
        # Vehicle in the adjacent lane
        # AND is at dist shorter then lane change dist
        # AND found before the leading vehicle
        if s_oa is None and is_adj_lane(road, vehicle, veh.lane_index):
            # This vehicle would have already changed state therefore, use old state
            # s_oa = veh.to_dict()
            s_oa = veh.state_hist[-1]
            sf_oa = {k: s_oa[k] if k in s_oa else 0 for k in cbf.STATE_SPACE}
            # print(
            #     "========================Adjacent Vehicle: {}=======================".format(
            #         veh.id
            #     )
            # )
        # Leading vehicle in the same target lane
        if s_ol is None and is_same_lane(road, vehicle, veh.lane_index):
            # print(
            #     "========================Leading Vehicle: {}=======================".format(
            #         veh.id
            #     )
            # )
            # This vehicle would have already changed state therefore, use old state
            # s_ol = veh.to_dict()
            s_ol = veh.state_hist[-1]
            sf_ol = {k: s_ol[k] if k in s_ol else 0 for k in cbf.STATE_SPACE}

    # Check for obstacles ahead in the lane
    for other in road.objects:
        if (s_ol is None or other.position[0] <= s_ol["x"]) and (abs(other.position[1] - vehicle.position[1]) <= 2):
            # print(
            #     "========================Leading Obstacle: at {}=======================".format(
            #         other.position
            #     )
            # )
            s_ol = other.to_dict()
            sf_ol = {k: s_ol[k] if k in s_ol else 0 for k in cbf.STATE_SPACE}
        if (s_oa is None or other.position[0] <= s_oa["x"]) and (2 < abs(other.position[1] - vehicle.position[1]) <= 4):
            # print(
            #     "========================Adjacent Obstacle: at {}=======================".format(
            #         other.position
            #     )
            # )
            s_oa = other.to_dict()
            sf_oa = {k: s_oa[k] if k in s_oa else 0 for k in cbf.STATE_SPACE}

    fgp_e = vehicle.fg_params
    if fgp_e is None:
        raise AttributeError(
            "fg_params not found in the the vehicle class: {0}".format(type(vehicle))
        )
    fp_e = fgp_e["f"]
    gp_e = fgp_e["g"]

    # Worst case acceleration is assumed for observed vehicle
    # Unactuated dynamics of heading and steering angle of observed vehicle are ignored here
    f = np.ravel(
        np.array(
            [
                [0, 0],
                [
                    (
                        (s_ol["vx"] * dt + 0.5 * cbf.ACCELERATION_RANGE[0] * dt**2)
                        if s_ol is not None
                        else 0
                    ),
                    0,
                ],
                [
                    (
                        (s_oa["vx"] * dt + 0.5 * cbf.ACCELERATION_RANGE[0] * dt**2)
                        if s_oa is not None
                        else 0
                    ),
                    0,
                ],
                [
                    (
                        (s_oar["vx"] * dt + 0.5 * cbf.ACCELERATION_RANGE[1] * dt**2)
                        if s_oar is not None
                        else 0
                    ),
                    0,
                ],
            ]
        )
    )

    g = np.reshape(
        np.array(
            [
                [gp_e["vx"] * dt, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 1 * dt],
                [0, 0],
                [0, 0],
                [0, 0],
            ]
        ),
        (cbf.action_size, -1),
    )
    g = np.transpose(g)

    x = np.ravel(
        np.array(
            [
                list(sf_e.values()),
                list(sf_ol.values()),
                list(sf_oa.values()),
                list(sf_oar.values()),
            ],
            dtype=np.float,
        ),
    )

    v_ol = s_ol["vx"] if s_ol is not None else 0
    v_oa = s_oa["vx"] if s_oa is not None else 0
    v_oar = s_oar["vx"] if s_oar is not None else 0

    if safe_dist == "braking":
        # Evaluate braking distance for longitudinal safety
        sd_l = abs(
            v_ol**2 / (2 * cbf.ACCELERATION_RANGE[0])
            - s_e["vx"] ** 2 / (2 * cbf.ACCELERATION_RANGE[0])
        )
        sd_a = abs(
            v_oa**2 / (2 * cbf.ACCELERATION_RANGE[0])
            - s_e["vx"] ** 2 / (2 * cbf.ACCELERATION_RANGE[0])
        )
        sd_ar = abs(
            s_e["vx"] ** 2 / (2 * cbf.ACCELERATION_RANGE[0])
            - v_oar**2 / (2 * cbf.ACCELERATION_RANGE[0])
        )
        cbf.safe_dists = [sd_l, sd_a, sd_ar]
        # print("safe distance: braking:[lead,adj,rear_adj]: ", cbf.safe_dists)
        # Headway distance in [m]
        vehicle.set_min_headway(sf_ol["x"] - sf_e["x"])

    elif safe_dist == "theadway":
        # Safe dist using Time hadway [s]
        v_oar = v_oar + cbf.ACCELERATION_RANGE[1] * dt
        cbf.safe_dists = [s_e["vx"] * cbf.TAU, s_e["vx"] * cbf.TAU, v_oar * cbf.TAU]
        # print("safe distance: theadway:[lead,adj,rear_adj]: ", cbf.safe_dists)
        # Time headway in [s]
        vehicle.set_min_headway((sf_ol["x"] - s_e["x"]) / s_e["vx"])
    else:
        raise ValueError("safe_dist type {} not supported".format(safe_dist))

    # v_ll = s_e["vx"] + action["acceleration"] * dt

    # beta = np.arctan(0.5*np.tan(action["steering"]))
    # dpsi_ll = (s_e["vx"]/vehicle.LENGTH * np.sin(beta)) + s_e["heading"]

    pred_v = copy.deepcopy(vehicle)
    pred_v.safety_layer = "none"
    # pvs = pred_v.predict_trajectory([action], dt, dt, dt)
    pred_v.act(action=action)
    pred_v.step(dt=dt)
    ps_e = pred_v.to_dict()
    dpsi_ll = (ps_e["heading"] - s_e["heading"]) / dt
    u_ll = np.array([ps_e["vx"], dpsi_ll])
    u_safe = cbf.control_barrier(u_ll, f, g, x, dt)

    # Lateral control is not constrained yet.
    u_safe[1] = action["steering"]

    # Avoid lane change if adjacent vehicle is close
    if not cbf.is_lc_allowed(f=f, g=g, x=x, u=u_safe):
        # print("Avoiding lane change")
        vehicle.target_lane_index = vehicle.lane_index
        u_safe[1] = vehicle.steering_control(vehicle.target_lane_index)

    # print("u_safe: ", u_safe)

    u_safe[0] = (u_safe[0] - vehicle.speed) / dt

    safe_action = {"acceleration": u_safe[0], "steering": u_safe[1]}
    safe_diff = {
        "acceleration": u_safe[0] - action["acceleration"],
        "steering": u_safe[1] - action["steering"],
    }
    return safe_action, safe_diff, cbf.get_status()

def safe_action_av_state(
    cbf: "CBFType",
    action,
    vehicle: "MDPLCVehicle",
    road: "Road",
    dt: float,
    safe_dist: str,
    perception_dist,
):

    print(
        "========================Vehicle:{}=======================".format(vehicle.id)
    )
    perception_dist = 6 * vehicle.SPEED_MAX

    s_e = vehicle.to_dict()
    sf_e = {k: s_e[k] for k in cbf.STATE_SPACE}

    # Assume a virtual vehicle stopped beyond the perception
    sf_ol = {}
    for k in cbf.STATE_SPACE:
        if k == "x":
            sf_ol[k] = s_e[k] + perception_dist + 1
        elif k=="y":
            sf_ol[k] = s_e[k]
        else:
            sf_ol[k] = 0.0

    sf_oa = {}
    for k in cbf.STATE_SPACE:
        if k == "x":
            sf_oa[k] = s_e[k] + perception_dist + 1
        elif k == "y":
            if vehicle.lane_index[2] == 1:
                sf_oa[k] = s_e[k] - vehicle.lane.DEFAULT_WIDTH
            elif vehicle.lane_index[2] == 0:
                sf_oa[k] = s_e[k] + vehicle.lane.DEFAULT_WIDTH
        else:
            sf_oa[k] = 0.0
    # Leading vehicles are ordered by increasing distance from the ego vehicle
    leading_vehicles: List[Union["MDPLCVehicle", "IDMVehicleL"]] = (
        road.close_vehicles_to(vehicle, perception_dist, count=5, see_behind=True)
    )

    sf_oar = copy.deepcopy(sf_oa)
    sf_oar["x"] = s_e["x"] - perception_dist - 1

    s_ol, s_oa, s_oar = None, None, None

    for veh in leading_vehicles:
        # rear vehicle in the adjacent lane
        if vehicle.lane_distance_to(veh) < 0:
            if s_oar is None and is_adj_lane(road, veh, vehicle.lane_index):
                # print(
                #     "=====================Rear adjacent Vehicle: {}=====================".format(
                #         veh.id
                #     )
                # )
                s_oar = veh.to_dict()
                sf_oar = {k: s_oar[k] if k in s_oar else 0 for k in cbf.STATE_SPACE}
            continue
        # Vehicle in the adjacent lane
        # AND is at dist shorter then lane change dist
        # AND found before the leading vehicle
        if s_oa is None and is_adj_lane(road, vehicle, veh.lane_index):
            # This vehicle would have already changed state therefore, use old state
            # s_oa = veh.to_dict()
            s_oa = veh.state_hist[-1]
            sf_oa = {k: s_oa[k] if k in s_oa else 0 for k in cbf.STATE_SPACE}
            # print(
            #     "========================Adjacent Vehicle: {}=======================".format(
            #         veh.id
            #     )
            # )
        # Leading vehicle in the same target lane
        if s_ol is None and is_same_lane(road, vehicle, veh.lane_index):
            print(
                "========================Leading Vehicle: {}=======================".format(
                    veh.id
                )
            )
            # This vehicle would have already changed state therefore, use old state
            # s_ol = veh.to_dict()
            s_ol = veh.state_hist[-1]
            sf_ol = {k: s_ol[k] if k in s_ol else 0 for k in cbf.STATE_SPACE}

    # Check for obstacles ahead in the lane
    for other in road.objects:
        if (s_ol is None or other.position[0] <= s_ol["x"]) and (abs(other.position[1] - vehicle.position[1]) <= 2):
            # print(
            #     "========================Leading Obstacle: at {}=======================".format(
            #         other.position
            #     )
            # )
            s_ol = other.to_dict()
            sf_ol = {k: s_ol[k] if k in s_ol else 0 for k in cbf.STATE_SPACE}
        if (s_oa is None or other.position[0] <= s_oa["x"]) and (2 < abs(other.position[1] - vehicle.position[1]) <= 4):
            # print(
            #     "========================Adjacent Obstacle: at {}=======================".format(
            #         other.position
            #     )
            # )
            s_oa = other.to_dict()
            sf_oa = {k: s_oa[k] if k in s_oa else 0 for k in cbf.STATE_SPACE}

    fgp_e = vehicle.fg_params
    if fgp_e is None:
        raise AttributeError(
            "fg_params not found in the the vehicle class: {0}".format(type(vehicle))
        )
    fp_e = fgp_e["f"]
    gp_e = fgp_e["g"]

    # Worst case acceleration is assumed for observed vehicle
    # Unactuated dynamics of heading and steering angle of observed vehicle are ignored here
    f = np.ravel(
        np.array(
            [
                [0, 0],
                [
                    (
                        (s_ol["vx"] * dt + 0.5 * cbf.ACCELERATION_RANGE[0] * dt**2)
                        if s_ol is not None
                        else 0
                    ),
                    0,
                ],
                [
                    (
                        (s_oa["vx"] * dt + 0.5 * cbf.ACCELERATION_RANGE[0] * dt**2)
                        if s_oa is not None
                        else 0
                    ),
                    0,
                ],
                [
                    (
                        (s_oar["vx"] * dt + 0.5 * cbf.ACCELERATION_RANGE[1] * dt**2)
                        if s_oar is not None
                        else 0
                    ),
                    0,
                ],
            ]
        )
    )

    g = np.reshape(
        np.array(
            [
                [gp_e["vx"] * dt, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 1 * dt],
                [0, 0],
                [0, 0],
                [0, 0],
            ]
        ),
        (cbf.action_size, -1),
    )
    g = np.transpose(g)

    x = np.ravel(
        np.array(
            [
                list(sf_e.values()),
                list(sf_ol.values()),
                list(sf_oa.values()),
                list(sf_oar.values()),
            ],
            dtype=np.float,
        ),
    )

    f = f + x
    v_ol = s_ol["vx"] if s_ol is not None else 0
    v_oa = s_oa["vx"] if s_oa is not None else 0
    v_oar = s_oar["vx"] if s_oar is not None else 0

    if safe_dist == "braking":
        # Evaluate braking distance for longitudinal safety
        sd_l = abs(
            v_ol**2 / (2 * cbf.ACCELERATION_RANGE[0])
            - s_e["vx"] ** 2 / (2 * cbf.ACCELERATION_RANGE[0])
        )
        sd_a = abs(
            v_oa**2 / (2 * cbf.ACCELERATION_RANGE[0])
            - s_e["vx"] ** 2 / (2 * cbf.ACCELERATION_RANGE[0])
        )
        sd_ar = abs(
            s_e["vx"] ** 2 / (2 * cbf.ACCELERATION_RANGE[0])
            - v_oar**2 / (2 * cbf.ACCELERATION_RANGE[0])
        )
        cbf.safe_dists = [sd_l, sd_a, sd_ar]
        # print("safe distance: braking:[lead,adj,rear_adj]: ", cbf.safe_dists)
        # Headway distance in [m]
        vehicle.set_min_headway(sf_ol["x"] - sf_e["x"])

    elif safe_dist == "theadway":
        # Safe dist using Time hadway [s]
        v_oar = v_oar + cbf.ACCELERATION_RANGE[1] * dt
        cbf.safe_dists = [s_e["vx"] * cbf.TAU, s_e["vx"] * cbf.TAU, v_oar * cbf.TAU]
        # print("safe distance: theadway:[lead,adj,rear_adj]: ", cbf.safe_dists)
        # Time headway in [s]
        vehicle.set_min_headway((sf_ol["x"] - s_e["x"] - vehicle.LENGTH) / s_e["vx"])
    else:
        raise ValueError("safe_dist type {} not supported".format(safe_dist))

    # v_ll = s_e["vx"] + action["acceleration"] * dt

    # beta = np.arctan(0.5*np.tan(action["steering"]))
    # dpsi_ll = (s_e["vx"]/vehicle.LENGTH * np.sin(beta)) + s_e["heading"]

    pred_v = copy.deepcopy(vehicle)
    pred_v.safety_layer = "none"
    # pvs = pred_v.predict_trajectory([action], dt, dt, dt)
    pred_v.act(action=action)
    pred_v.step(dt=dt)
    ps_e = pred_v.to_dict()
    dpsi_ll = (ps_e["heading"] - s_e["heading"]) / dt
    u_ll = np.array([ps_e["vx"], dpsi_ll])
    u_safe = cbf.control_barrier(u_ll, f, g, x, dt)

    # Lateral control is not constrained yet.
    u_safe[1] = action["steering"]

    # Avoid lane change if adjacent vehicle is close
    if not cbf.is_lc_allowed(f=f, g=g, x=x, u=u_safe):
        # print("Avoiding lane change")
        vehicle.target_lane_index = vehicle.lane_index
        u_safe[1] = vehicle.steering_control(vehicle.target_lane_index)

    # print("u_safe: ", u_safe)

    u_safe[0] = (u_safe[0] - vehicle.speed) / dt

    safe_action = {"acceleration": u_safe[0], "steering": u_safe[1]}
    safe_diff = {
        "acceleration": u_safe[0] - action["acceleration"],
        "steering": u_safe[1] - action["steering"],
    }
    return safe_action, safe_diff, cbf.get_status()


def safe_action_av_lateral(
    cbf: "CBFType",
    action,
    vehicle: "MDPLCVehicle",
    road: "Road",
    dt: float,
    perception_dist,
):

    print(
        "========================Vehicle:{}=======================".format(vehicle.id)
    )
    perception_dist = 6 * vehicle.SPEED_MAX

    s_e = vehicle.to_dict()
    sf_e = {k: s_e[k] for k in CBFType.STATE_SPACE}

    # Assume a virtual vehicle stopped beyond the perception
    sf_oa = {}
    for k in CBFType.STATE_SPACE:
        if k == "x":
            sf_oa[k] = s_e[k] + perception_dist + 1
        elif k == "y":
            sf_oa[k] = s_e[k] + 2 * vehicle.lane.DEFAULT_WIDTH
        else:
            sf_oa[k] = 0

    sf_ol = {
        k: s_e[k] + perception_dist + 1 if (k == "x") else 0
        for k in CBFType.STATE_SPACE
    }

    # Leading vehicles are ordered by increasing distance from the ego vehicle
    leading_vehicles: List[ControlledVehicle] = road.close_vehicles_to(
        vehicle, perception_dist, count=5, see_behind=False
    )

    s_ol, s_oa = (
        None,
        None,
    )

    for veh in leading_vehicles:
        if vehicle.lane_distance_to(veh) < 0:
            continue
        # Adjacent vehicle changing lanes to ego vehicles lane
        # OR Adjacent vehicle in the target lane while the ego vehicle is changing lane
        if (
            (
                veh.lane_index != veh.target_lane_index
                and veh.target_lane_index == vehicle.lane_index
            )
            or (
                vehicle.lane_index != vehicle.target_lane_index
                and veh.lane_index == vehicle.target_lane_index
            )
            and vehicle.lane_distance_to(veh) <= 50
            and s_oa is None
        ):
            s_oa = veh.to_dict()
            sf_oa = {k: s_oa[k] if k in s_oa else 0 for k in CBFType.STATE_SPACE}
            print(
                "========================Adjacent Vehicle: {}=======================".format(
                    veh.id
                )
            )
        # Leading vehicle in the same lane
        elif veh.lane_index == vehicle.lane_index and s_ol is None:
            print(
                "========================Leading Vehicle: {}=======================".format(
                    veh.id
                )
            )
            s_ol = veh.to_dict()
            sf_ol = {k: s_ol[k] if k in s_ol else 0 for k in CBFType.STATE_SPACE}

    fgp_e = vehicle.fg_params
    if fgp_e is None:
        raise AttributeError(
            "fg_params not found in the the vehicle class: {0}".format(type(vehicle))
        )
    fp_e = fgp_e["f"]
    gp_e = fgp_e["g"]

    # Worst case acceleration is assumed for observed vehicle
    # Unactuated dynamics of heading and steering angle of observed vehicle are ignored here
    f = np.ravel(
        np.array(
            [
                [fp_e["x"] * dt, fp_e["y"] * dt, 0, 0, fp_e["heading"], 0],
                [
                    sf_ol["vx"] * dt,
                    sf_ol["vy"] * dt,
                    CBFType.ACCELERATION_RANGE[0] * dt,
                    0,
                    0,  # TODO: Consider max steering angle towards ego vehicle
                    0,
                ],
                [
                    sf_oa["vx"] * dt,
                    sf_oa["vy"] * dt,
                    CBFType.ACCELERATION_RANGE[0] * dt,
                    0,
                    0,  # TODO: Consider max steering angle towards ego vehicle
                    0,
                ],
            ]
        )
    )

    g = np.reshape(
        np.array(
            [
                # [0.5*dt**2*gp_e["vx"], 0.5*dt**2*gp_e["vy"], gp_e["vx"] * dt, gp_e["vy"] * dt, 0, 0],
                [0, 0, gp_e["vx"] * dt, gp_e["vy"] * dt, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1 * dt],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        ),
        (cbf.action_size, -1),
    )
    g = np.transpose(g)

    x = np.ravel(
        np.array([list(sf_e.values()), list(sf_ol.values()), list(sf_oa.values())])
    )

    # TODO: Check shapes of f,g,x
    # assert(f.shape, (len(CBFType.STATE_SPACE)*2),)
    # assert(g.shape, (len(CBFType.STATE_SPACE)*2, cbf.action_size))
    # assert(x.shape, (len(CBFType.STATE_SPACE)*2),)

    u_ll = np.array([action["acceleration"], action["steering"]])
    u_safe = cbf.control_barrier(u_ll, f, g, x)

    print("u_safe: ", u_safe)
    safe_action = {"acceleration": u_safe[0], "steering": u_safe[1]}
    safe_diff = {"acceleration": u_safe[0] - u_ll[0], "steering": u_safe[1] - u_ll[1]}
    return safe_action, safe_diff, cbf.get_status()


# def safe_action_cav(cbf: CBFType, env: "AbstractEnv", vehicle: "MDPLCVehicle", action):
#     return


def safety_layer(
    safety_type: str,
    action: dict,
    vehicle: "MDPLCVehicle",
    dt: float,
    safe_dist: str = "theadway",
    **kwargs
):
    """
    Implements decentralised safety layer to evaluate safe actions using CBF.
    """

    if safety_type == "avlon":
        cbf: CBFType = cbf_factory(
            safety_type,
            action_size=len(action),
            action_bound=[(vehicle.MIN_ACC, vehicle.MAX_ACC), (-4 * np.pi, 4 * np.pi)],
            vehicle_size=[vehicle.LENGTH, vehicle.WIDTH],
            vehicle_lane=vehicle.lane_index[2],
        )
        return safe_action_longitudinal(
            cbf=cbf, action=action, vehicle=vehicle, dt=dt**kwargs
        )
    elif safety_type == "av":
        v_min = vehicle.speed + vehicle.MIN_ACC * dt
        v_max = vehicle.speed + vehicle.MAX_ACC * dt
        cbf: CBFType = cbf_factory(
            safety_type,
            action_size=len(action),
            action_bound=[(v_min, v_max), (-4 * np.pi, 4 * np.pi)],
            vehicle_size=[vehicle.LENGTH, vehicle.WIDTH],
            vehicle_lane=vehicle.lane_index[2],
        )
        return safe_action_av(
            cbf=cbf,
            action=action,
            vehicle=vehicle,
            dt=dt,
            safe_dist=safe_dist,
            **kwargs
        )
    elif safety_type == "avs":
        v_min = vehicle.speed + vehicle.MIN_ACC * dt
        v_max = vehicle.speed + vehicle.MAX_ACC * dt
        cbf: CBFType = cbf_factory(
            safety_type,
            action_size=len(action),
            action_bound=[(v_min, v_max), (-4 * np.pi, 4 * np.pi)],
            vehicle_size=[vehicle.LENGTH, vehicle.WIDTH],
            vehicle_lane=vehicle.lane_index[2],
        )
        return safe_action_av_state(
            cbf=cbf,
            action=action,
            vehicle=vehicle,
            dt=dt,
            safe_dist=safe_dist,
            **kwargs
        )
    # elif safety_type == "cav":
    #     return safe_action_cav(cbf=cbf, **kwargs)
    else:
        raise ValueError("Undefined safety_type:{0}".format(safety_type))
