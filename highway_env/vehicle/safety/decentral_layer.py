import copy
import numpy as np
from typing import TYPE_CHECKING, List

from highway_env.vehicle.safety.cbf import CBFType, cbf_factory

if TYPE_CHECKING:
    from highway_env.road.road import Road
    from highway_env.vehicle.safe_controller import MDPLCVehicle
    from highway_env.vehicle.controller import ControlledVehicle


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
    perception_dist,
):

    print(
        "========================Vehicle:{}=======================".format(vehicle.id)
    )
    perception_dist = 6 * vehicle.SPEED_MAX

    s_e = vehicle.to_dict()
    sf_e = {k: s_e[k] for k in CBFType.STATE_SPACE}

    # Assume a virtual vehicle stopped beyond the perception
    sf_ol = {}
    for k in CBFType.STATE_SPACE:
        if k == "x":
            sf_ol[k] = s_e[k] + perception_dist + 1
        else:
            sf_ol[k] = 0.0

    sf_oa = {}
    for k in CBFType.STATE_SPACE:
        if k == "x":
            sf_oa[k] = s_e[k] + perception_dist + 1
        elif k == "y":
            if vehicle.lane_index[2] == 1:
                sf_oa[k] = s_e[k] - 2 * vehicle.lane.DEFAULT_WIDTH
            elif vehicle.lane_index[2] == 0:
                sf_oa[k] = s_e[k] + 2 * vehicle.lane.DEFAULT_WIDTH
        else:
            sf_oa[k] = 0.0
    # Leading vehicles are ordered by increasing distance from the ego vehicle
    leading_vehicles: List[ControlledVehicle] = road.close_vehicles_to(
        vehicle, perception_dist, count=5, see_behind=True
    )

    sf_oar = copy.deepcopy(sf_oa)
    sf_oar["x"] = s_e[k] - perception_dist - 1

    s_ol, s_oa, s_oar = None, None, None

    for veh in leading_vehicles:
        # rear vehicle in the adjacent lane
        if vehicle.lane_distance_to(veh) < 0:
            if (
                veh.lane_index[:-1] == vehicle.lane_index[:-1]
                and abs(veh.lane_index[-1] - vehicle.lane_index[-1]) == 1
                # and -1 * vehicle.lane_distance_to(veh) <= (veh.speed * CBFType.TAU + vehicle.LENGTH + 9)
                and s_oar is None
            ):
                print(
                    "=====================Rear adjacent Vehicle: {}=====================".format(
                        veh.id
                    )
                )
                s_oar = veh.to_dict()
                sf_oar = {k: s_oar[k] if k in s_oar else 0 for k in CBFType.STATE_SPACE}
            continue
        # Vehicle in the adjacent lane
        # AND is at dist shorter then lane change dist
        # AND found before the leading vehicle
        if (
            (
                veh.lane_index[:-1] == vehicle.lane_index[:-1]
                and abs(veh.lane_index[-1] - vehicle.lane_index[-1]) == 1
            )
            and s_oa is None
        ):
            s_oa = veh.to_dict()
            sf_oa = {k: s_oa[k] if k in s_oa else 0 for k in CBFType.STATE_SPACE}
            print(
                "========================Adjacent Vehicle: {}=======================".format(
                    veh.id
                )
            )
        # Leading vehicle in the same target lane
        if veh.lane_index == vehicle.lane_index and s_ol is None:
            print(
                "========================Leading Vehicle: {}=======================".format(
                    veh.id
                )
            )
            s_ol = veh.to_dict()
            sf_ol = {k: s_ol[k] if k in s_ol else 0 for k in CBFType.STATE_SPACE}
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
                    sf_ol["vx"] * dt,
                    sf_ol["vy"] * dt,
                    CBFType.ACCELERATION_RANGE[0] * dt,
                    0,
                    0,
                    0,
                ],
                [
                    sf_oa["vx"] * dt,
                    sf_oa["vy"] * dt,
                    CBFType.ACCELERATION_RANGE[0] * dt,
                    0,
                    0,
                    0,
                ],
                [
                    sf_oar["vx"] * dt,
                    sf_oar["vy"] * dt,
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
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1 * dt],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
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

    u_ll = np.array([action["acceleration"], action["steering"]])
    u_safe = cbf.control_barrier(u_ll, f, g, x, dt)

    # Avoid lane change if adjacent vehicle is close
    if not cbf.is_lc_allowed(f=f, g=g, x=x, u=u_safe):
        print("Avoiding lane change")
        vehicle.target_lane_index = vehicle.lane_index
        u_safe[1] = vehicle.steering_control(vehicle.target_lane_index)

    print("u_safe: ", u_safe)
    safe_action = {"acceleration": u_safe[0], "steering": u_safe[1]}
    safe_diff = {"acceleration": u_safe[0] - u_ll[0], "steering": u_safe[1] - u_ll[1]}
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


def safety_layer(safety_type: str, action: dict, vehicle: "MDPLCVehicle", **kwargs):
    """
    Implements decentralised safety layer to evaluate safe actions using CBF.
    """

    cbf: CBFType = cbf_factory(
        safety_type,
        action_size=len(action),
        action_bound=[(vehicle.MIN_ACC, vehicle.MAX_ACC), (-4 * np.pi, 4 * np.pi)],
        vehicle_size=[vehicle.LENGTH, vehicle.WIDTH],
        vehicle_lane=vehicle.lane_index[2],
    )

    if safety_type == "avlon":
        return safe_action_longitudinal(
            cbf=cbf, action=action, vehicle=vehicle, **kwargs
        )
    elif safety_type == "av":
        return safe_action_av(cbf=cbf, action=action, vehicle=vehicle, **kwargs)
    # elif safety_type == "cav":
    #     return safe_action_cav(cbf=cbf, **kwargs)
    else:
        raise ValueError("Undefined safety_type:{0}".format(safety_type))
