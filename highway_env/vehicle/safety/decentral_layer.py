import copy
import numpy as np

from highway_env.vehicle.controller import MDPLCVehicle
from highway_env.vehicle.kinematics import Vehicle
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.road import Road
from highway_env.vehicle.safety.cbf import CBFType, cbf_factory


def safe_action_longitudinal(cbf: CBFType, env: AbstractEnv, vehicle: MDPLCVehicle, action):

    road: Road = env.road
    dt = 1 / env.config["simulation_frequency"]

    # Identify leading vehicle
    leading_vehicle: Vehicle = road.close_vehicles_to(
        vehicle, env.PERCEPTION_DISTANCE, count=1, see_behind=False
    )

    s_e = vehicle.to_dict()
    sf_e = {k: s_e[k] for k in CBFType.STATE_SPACE}

    s_o = leading_vehicle.to_dict()
    sf_o = {k: s_o[k] for k in CBFType.STATE_SPACE}

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
                    CBFType.ACCELERATION_RANGE(0) * dt,
                    CBFType.ACCELERATION_RANGE(0) * dt,
                    0,
                    0,
                ],
            ]
        )
    )
    # f = np.transpose(f)

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
    # x = np.transpose(x)

    # TODO: Check shapes of f,g,x
    # assert(f.shape, (len(CBFType.STATE_SPACE)*2),)
    # assert(g.shape, (len(CBFType.STATE_SPACE)*2, cbf.action_size))
    # assert(x.shape, (len(CBFType.STATE_SPACE)*2),)

    u_ll = np.array([action["acceleration"], action["steering"]])
    u_safe = cbf.control_barrier(u_ll, f, g, x)
    return u_safe


def safe_action_av(cbf: CBFType, env: AbstractEnv, vehicle: MDPLCVehicle, action):
    return


def safe_action_cav(cbf: CBFType, env: AbstractEnv, vehicle: MDPLCVehicle, action):
    return


def safety_layer(safety_type: str, **kwargs):
    """
    Implements decentralised safety layer to evaluate safe actions using CBF.
    """

    cbf: CBFType = cbf_factory(safety_type)

    if safety_type == "avlon":
        return safe_action_longitudinal(cbf, **kwargs)
    elif safety_type == "av":
        return safe_action_av(cbf, **kwargs)
    elif safety_type == "cav":
        return safe_action_cav(cbf, **kwargs)
    else:
        raise ValueError("Undefined safety_type:{0}".format(safety_type))
