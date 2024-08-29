import copy
import numpy as np

from highway_env.vehicle.controller import MDPLCVehicle
from highway_env.vehicle.kinematics import Vehicle
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.road import Road
from highway_env.vehicle.safety.cbf import CBFType, cbf_factory

def cbf_av_longitudinal(cbf:CBFType, env:AbstractEnv, vehicle:MDPLCVehicle, action):
    
    road:Road = env.road
    # Identify leading vehicle
    leading_vehicle:Vehicle = road.close_vehicles_to(vehicle,
                                             env.PERCEPTION_DISTANCE,
                                             count=1,
                                             see_behind=False)
    
    f = np.zeros()
    g = np.zeros()
    
    x = np.ravel(np.array([list(vehicle.to_dict().values()), 
                           list(leading_vehicle.to_dict().values())]))

    return action 


def cbf_av(cbf:CBFType, env:AbstractEnv, vehicle:MDPLCVehicle, action):
    return

def cbf_cav(cbf:CBFType, env:AbstractEnv, vehicle:MDPLCVehicle, action):
    return

def safety_layer(safety_type:str, **kwargs):
    """
    Implements decentralised safety layer to evaluate safe actions using CBF.
    """

    cbf:CBFType = cbf_factory(safety_type)

    if safety_type == "avlon":
        return cbf_av_longitudinal(cbf, **kwargs)
    elif safety_type == "av":
        return cbf_av(cbf, **kwargs)
    elif safety_type == "cav":
        return cbf_cav(cbf, **kwargs)
    else:
        raise ValueError("Undefined safety_type:{0}".format(safety_type))

