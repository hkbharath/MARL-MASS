import copy
import numpy as np
from typing import TYPE_CHECKING, List

from highway_env.vehicle.safety.cbf import CBFType, cbf_factory

if TYPE_CHECKING:
    from highway_env.road.road import Road
    from highway_env.vehicle.safe_controller import MDPLCVehicle
    from highway_env.vehicle.kinematics import Vehicle


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

    leading_vehicle: List[Vehicle] = road.close_vehicles_to(
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
    return safe_action, safe_diff


# def safe_action_av(cbf: "CBFType", env: "AbstractEnv", vehicle: "MDPLCVehicle", action):
#     return


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
        vehicle_size=[vehicle.LENGTH, vehicle.WIDTH]
    )

    if safety_type == "avlon":
        return safe_action_longitudinal(
            cbf=cbf, action=action, vehicle=vehicle, **kwargs
        )
    # elif safety_type == "av":
    #     return safe_action_av(cbf=cbf, **kwargs)
    # elif safety_type == "cav":
    #     return safe_action_cav(cbf=cbf, **kwargs)
    else:
        raise ValueError("Undefined safety_type:{0}".format(safety_type))
