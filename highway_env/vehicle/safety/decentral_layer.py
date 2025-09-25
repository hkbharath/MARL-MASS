import copy
import numpy as np
from typing import TYPE_CHECKING, List, Union, Tuple

from highway_env.vehicle.safety.cbf import CBFType, CBF_CAV, cbf_factory
from highway_env.utils import CBF_DEBUG

if TYPE_CHECKING:
    from highway_env.road.road import Road
    from highway_env.vehicle.safe_controller import MDPLCVehicle
    from highway_env.vehicle.controller import ControlledVehicle
    from highway_env.vehicle.behavior import IDMVehicleHist


def is_same_lane(vehicle: "ControlledVehicle", lane_index_2):
    road: "Road" = vehicle.road
    lane_index_1 = vehicle.lane_index
    next_lane = road.network.next_lane(lane_index_1, position=vehicle.position)
    # should not require a lane change in the next lane.
    return (lane_index_1 == lane_index_2) or (lane_index_2 == next_lane)


def is_adj_lane(vehicle: "ControlledVehicle", lane_index_2):
    road: "Road" = vehicle.road
    lane_index_1 = vehicle.lane_index
    next_lane = road.network.next_lane(lane_index_1, position=vehicle.position)

    if (
        lane_index_1[:-1] == lane_index_2[:-1]
        and abs(lane_index_1[-1] - lane_index_2[-1]) == 1
    ):
        return lane_index_1[-1] - lane_index_2[-1]
    elif (
        next_lane[:-1] == lane_index_2[:-1]
        and abs(next_lane[-1] - lane_index_2[-1]) == 1
    ):
        return next_lane[-1] - lane_index_2[-1]

    return 0


def is_on_ramp_adj(lane_index_1, lane_index_2):
    return lane_index_1 == ("a", "b", 0) and lane_index_2 == ("k", "b", 0)


def is_approaching_same_lane(ve: "ControlledVehicle", vl: "ControlledVehicle"):
    if ve.lane_distance_to(vl) < 0:
        return False
    y_dist = vl.position[1] - ve.position[1]
    dist_cond = abs(y_dist) <= 3.5
    heading_cond = False
    if y_dist < 0:
        heading_cond = vl.heading > 0.037
    else:
        heading_cond = vl.heading < -0.037
    ret = dist_cond and heading_cond
    return ret


# Define a function to check if the vehicle is close to lane edge
def is_close_to_lane_edge(vehicle: "ControlledVehicle") -> bool:
    lane_pos = vehicle.lane.local_coordinates(vehicle.position)
    edge_threshold = 1.5  # meters

    # Check if the vehicle is close to the left or right edge of the lane
    if abs(lane_pos[1]) > edge_threshold:
        return True
    return False


def simplified_control(
    s: dict, action: dict, vl: float, dt: float
) -> Tuple[float, float]:
    if s is None:
        return 0, 0

    speed = 0
    if "speed" in s:
        speed = s["speed"]
    elif "cos_h" in s:
        speed = s["vx"] / s["cos_h"]  # Assume that "vx" exists

    v = s["vx"] + action["acceleration"] * dt
    v = max(0, v)
    beta = np.arctan(0.5 * np.tan(action["steering"]))
    dpsi = (speed / vl * np.sin(beta)) + s["heading"]

    return v, dpsi


def derived_acceleration(safe_v: float, speed: float, dt: float):
    safe_a = (safe_v - speed) / dt
    return safe_a


def multi_agent_state(
    cbf: "CBFType",
    vehicle: "MDPLCVehicle",
    road: "Road",
    perception_dist: float,
    is_ma_dynamics=False,
) -> Union[Tuple[dict, dict, dict], Tuple[dict, dict, dict, dict, dict, dict]]:
    s_ol, s_oa, s_oar = None, None, None
    a_ol = {"steering": 0, "acceleration": 0}
    a_oa = {"steering": 0, "acceleration": 0}
    gp = {"ol": {"vx": 0}, "oa": {"vx": 0}}

    # Leading vehicles are ordered by increasing distance from the ego vehicle
    surrounding_vehicles: List[Union["MDPLCVehicle", "IDMVehicleHist"]] = (
        road.close_vehicles_to(vehicle, perception_dist, count=5, see_behind=True)
    )

    for veh in surrounding_vehicles:
        v_a = is_adj_lane(vehicle, veh.lane_index)
        a_v = is_adj_lane(veh, vehicle.lane_index)

        # True adjacent
        if (not is_approaching_same_lane(ve=vehicle, vl=veh)) and (v_a or a_v):
            # rear vehicle in the adjacent lane
            if s_oar is None and vehicle.lane_distance_to(veh) < 0:
                if CBF_DEBUG:
                    print(
                        "=====================Rear adjacent Vehicle: {}=====================".format(
                            veh.id
                        )
                    )
                s_oar = veh.to_dict()
            elif s_oa is None and vehicle.lane_distance_to(veh) >= 0:
                # This vehicle would have already changed state therefore, use old state
                # s_oa = veh.to_dict()
                if CBF_DEBUG:
                    print(
                        "========================Adjacent Vehicle: {}=======================".format(
                            veh.id
                        )
                    )
                s_oa = veh.state_hist[-2]
                if is_ma_dynamics:
                    a_oa = (
                        veh.safe_action
                        if hasattr(veh, "safe_action")
                        else {"steering": 0, "acceleration": cbf.ACCELERATION_RANGE[0]}
                    )
                    gp["oa"] = (
                        veh.fg_params["g"] if hasattr(veh, "fg_params") else {"vx": 1}
                    )

                    # left adj changing to right lane OR right adjacent changing to left lane
                    if hasattr(veh, "hl_action"):
                        # vehicle (or veh) must have initiated a lane change to consider for constraining adjacent vehicle
                        cbf.constrain_adj = vehicle.collaborate_adj and (
                            is_same_lane(
                                vehicle=vehicle,
                                lane_index_2=get_target_lane(veh, veh.hl_action),
                            )
                            or is_same_lane(
                                vehicle=veh,
                                lane_index_2=get_target_lane(
                                    vehicle, vehicle.hl_action
                                ),
                            )
                        )

                    veh_corner = None
                    # Decide to constrain if the vehicle is entering the current vehicles lane
                    if v_a == -1 or a_v == 1:
                        # Adj vehicle is on right side lane, therefore capture left corner.
                        veh_corner = veh.get_corner(dir="L")
                    elif v_a == 1 or a_v == -1:
                        # Adj vehicle is on left side lane, therefore capture right corner.
                        veh_corner = veh.get_corner(dir="R")
                    if veh_corner is not None:
                        cbf.constrain_adj = not veh.lane.on_lane(position=veh_corner)
        # adjacent hdv in merging lane
        elif (
            not hasattr(veh, "safe_action")
            and is_on_ramp_adj(vehicle.lane_index, veh.lane_index)
            and vehicle.lane_distance_to(veh) >= 0
        ):
            if CBF_DEBUG:
                print(
                    "========================Adjacent On-Ramp HDV Vehicle: {}=======================".format(
                        veh.id
                    )
                )
            s_oa = veh.state_hist[-2]

            # As ego vehicle is constrained with respect to this HDV,
            # consider a digital twin at 0.5 s headway away from
            # the current position of the ego vehicle.
            s_oa["x"] = s_oa["x"] + 0.5 * vehicle.velocity[0]
            cbf.constrain_adj = True

            a_oa = {"steering": 0, "acceleration": cbf.ACCELERATION_RANGE[0]}
            gp["oa"] = veh.fg_params["g"] if hasattr(veh, "fg_params") else {"vx": 1}
        # leading vehicle
        elif (
            s_ol is None
            and (
                is_same_lane(vehicle, veh.lane_index)
                or (is_approaching_same_lane(ve=vehicle, vl=veh))
            )
            and vehicle.lane_distance_to(veh) > 0
        ):
            if CBF_DEBUG:
                print(
                    "========================Leading Vehicle: {}=======================".format(
                        veh.id
                    )
                )
            # This vehicle would have already changed state therefore, use old state
            # s_ol = veh.to_dict()
            s_ol = veh.state_hist[-2]
            if is_ma_dynamics:
                a_ol = (
                    veh.safe_action
                    if hasattr(veh, "safe_action")
                    else {"steering": 0, "acceleration": cbf.ACCELERATION_RANGE[0]}
                )
                gp["ol"] = (
                    veh.fg_params["g"] if hasattr(veh, "fg_params") else {"vx": 1}
                )

    # Check for obstacles ahead in the lane
    for other in road.objects:
        if vehicle.position[0] > other.position[0]:
            continue

        if (s_ol is None or other.position[0] <= s_ol["x"]) and (
            abs(other.position[1] - vehicle.position[1]) <= 2
        ):
            if CBF_DEBUG:
                print(
                    "========================Leading Obstacle: at {}=======================".format(
                        other.position
                    )
                )
            s_ol = other.to_dict()
            s_ol["heading"] = 0
            if is_ma_dynamics:
                a_ol = {"steering": 0, "acceleration": 0}
                gp["ol"] = {"vx": 0}
        if (s_oa is None or other.position[0] <= s_oa["x"]) and (
            2 < abs(other.position[1] - vehicle.position[1]) <= 4
        ):
            if CBF_DEBUG:
                print(
                    "========================Adjacent Obstacle: at {}=======================".format(
                        other.position
                    )
                )
            s_oa = other.to_dict()
            s_oa["heading"] = 0
            if is_ma_dynamics:
                a_oa = {"steering": 0, "acceleration": 0}
                gp["oa"] = {"vx": 0}
                cbf.constrain_adj = False

    if is_ma_dynamics:
        if CBF_DEBUG and s_oa is not None:
            print(
                "Constrain adjacent vehicle: {0} at {1}".format(
                    cbf.constrain_adj, s_oa["x"]
                )
            )
        return s_ol, s_oa, s_oar, a_ol, a_oa, gp

    return s_ol, s_oa, s_oar


def get_target_lane(vehicle: "MDPLCVehicle", hl_action: str) -> Tuple:
    veh_target_lane_index = vehicle.lane_index
    # LANE_RIGHT = 2
    if hl_action == "LANE_RIGHT":
        _from, _to, _id = vehicle.lane_index
        target_lane_index = (
            _from,
            _to,
            np.clip(_id + 1, 0, len(vehicle.road.network.graph[_from][_to]) - 1),
        )
        if vehicle.road.network.get_lane(target_lane_index).is_reachable_from(
            vehicle.position
        ):
            veh_target_lane_index = target_lane_index
    # LANE_LEFT = 0
    elif hl_action == "LANE_LEFT":
        _from, _to, _id = vehicle.lane_index
        target_lane_index = (
            _from,
            _to,
            np.clip(_id - 1, 0, len(vehicle.road.network.graph[_from][_to]) - 1),
        )
        if vehicle.road.network.get_lane(target_lane_index).is_reachable_from(
            vehicle.position
        ):
            veh_target_lane_index = target_lane_index

    return veh_target_lane_index


def safe_action_hss(
    cbf: "CBFType",
    action,
    vehicle: "MDPLCVehicle",
    road: "Road",
    dt: float,
    safe_dist: str,
    perception_dist=None,
):

    if CBF_DEBUG:
        print(
            "========================Vehicle:{}=======================".format(
                vehicle.id
            )
        )
    perception_dist = (
        6 * vehicle.SPEED_MAX if perception_dist is None else perception_dist
    )

    s_e = vehicle.to_dict()
    # Set min velocity for calculations in extreme case
    s_e["vx"] = s_e["vx"] if s_e["vx"] > 1 else 1
    sf_e = {k: s_e[k] for k in cbf.STATE_SPACE}

    # Assume a virtual vehicle stopped beyond the perception
    sf_ol = {}
    for k in cbf.STATE_SPACE:
        if k == "x":
            sf_ol[k] = s_e[k] + perception_dist + 1
        elif k == "y":
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

    sf_oar = copy.deepcopy(sf_oa)
    sf_oar["x"] = s_e["x"] - perception_dist - 1

    s_ol, s_oa, s_oar = multi_agent_state(
        cbf=cbf,
        vehicle=vehicle,
        road=road,
        perception_dist=perception_dist,
    )

    if s_oa is not None:
        sf_oa = {k: s_oa[k] if k in s_oa else 0 for k in cbf.STATE_SPACE}

    if s_ol is not None:
        sf_ol = {k: s_ol[k] if k in s_ol else 0 for k in cbf.STATE_SPACE}

    if s_oar is not None:
        sf_oar = {k: s_oar[k] if k in s_oar else 0 for k in cbf.STATE_SPACE}

    fgp_e = vehicle.fg_params
    if fgp_e is None:
        raise AttributeError(
            "fg_params not found in the the vehicle class: {0}".format(type(vehicle))
        )

    gp_e = fgp_e["g"]

    # Unactuated dynamics of heading and steering angle of observed vehicle are ignored here
    f = np.ravel(
        np.array(
            [
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
            ]
        )
    )

    # Representation: [[s_e1, s_e2], [s_ol1, s_ol2], [s_oa1, s_oa2], [s_oar1, s_aor2] x no. of inpus in u]
    g = np.reshape(
        np.array(
            [
                [
                    [gp_e["vx"] * dt, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                ],
                [
                    [0, 1 * dt],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                ],
                [
                    [0, 0],
                    [1 * dt, 0],
                    [0, 0],
                    [0, 0],
                ],
                [
                    [0, 0],
                    [0, 1 * dt],
                    [0, 0],
                    [0, 0],
                ],
                [
                    [0, 0],
                    [0, 0],
                    [1 * dt, 0],
                    [0, 0],
                ],
                [
                    [0, 0],
                    [0, 0],
                    [0, 1 * dt],
                    [0, 0],
                ],
                [
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [1 * dt, 0],
                ],
                [
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 1 * dt],
                ],
            ]
        ),
        (cbf.action_size * 4, -1),
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

    if safe_dist == "theadway":
        # Safe dist using Time hadway [s]
        sv_oar = s_oar["vx"] if s_oar is not None else 0
        sv_oar = sv_oar + cbf.ACCELERATION_RANGE[1] * dt
        sv_oar = (
            sv_oar if sv_oar > 1 else 1
        )  # Set min velocity for calculations in extreme case

        buffer = (cbf.ACCELERATION_RANGE[1] + 0.1) * dt * cbf.TAU
        cbf.safe_dists = [
            s_e["vx"] * cbf.TAU + vehicle.LENGTH + buffer,
            s_e["vx"] * cbf.TAU + vehicle.LENGTH + buffer,
            sv_oar * cbf.TAU + vehicle.LENGTH + buffer,
        ]

        if CBF_DEBUG:
            print("safe distance: theadway:[lead, adj, rear_adj]: ", cbf.safe_dists)
        # Time headway in [s]
        vehicle.set_min_headway(
            (sf_ol["x"] - s_e["x"] - vehicle.LENGTH) / s_e["vx"], cbf.TAU
        )
    else:
        raise ValueError("safe_dist type {} not supported".format(safe_dist))

    # Worst case deceleration for observed leading vehicle
    wcl_action = {"steering": 0, "acceleration": cbf.ACCELERATION_RANGE[0]}
    # Worst case acceleration for rear vehicle
    wcr_action = {"steering": 0, "acceleration": cbf.ACCELERATION_RANGE[1]}

    v_ll, dpsi_ll = simplified_control(s=s_e, action=action, vl=vehicle.LENGTH, dt=dt)
    v_ol, dpsi_ol = simplified_control(
        s=s_ol, action=wcl_action, vl=vehicle.LENGTH, dt=dt
    )
    v_oa, dpsi_oa = simplified_control(
        s=s_oa, action=wcl_action, vl=vehicle.LENGTH, dt=dt
    )
    v_oar, dpsi_oar = simplified_control(
        s=s_oar, action=wcr_action, vl=vehicle.LENGTH, dt=dt
    )

    u_ma = np.array([v_ll, dpsi_ll, v_ol, dpsi_ol, v_oa, dpsi_oa, v_oar, dpsi_oar])

    u_safe = cbf.control_barrier(u_ma, f, g, x, dt)

    # Lateral control is not constrained yet.
    u_safe[1] = action["steering"]

    u_safe_ma = np.append(u_safe, u_ma[2:])

    vehicle.is_collaborating = cbf.constrain_adj
    vehicle.is_lc_safe = True

    # Avoid lane change if adjacent vehicle is close
    if not cbf.is_lc_allowed(f=f, g=g, x=x, u=u_safe_ma):
        if CBF_DEBUG:
            print("Avoiding lane change")
        vehicle.target_lane_index = vehicle.lane_index
        u_safe[1] = vehicle.steering_control(vehicle.target_lane_index)
        vehicle.is_lc_safe = False

    if CBF_DEBUG:
        print("u_safe: ", u_safe)

    u_safe[0] = derived_acceleration(u_safe[0], s_e["vx"], dt)

    safe_action = {"acceleration": u_safe[0], "steering": u_safe[1]}
    safe_diff = {
        "acceleration": u_safe[0] - action["acceleration"],
        "steering": u_safe[1] - action["steering"],
    }
    return safe_action, safe_diff, cbf.get_status()


def safe_action_mass(
    cbf: "CBF_CAV",
    action,
    vehicle: "MDPLCVehicle",
    road: "Road",
    dt: float,
    safe_dist: str,
    perception_dist=None,
):

    if CBF_DEBUG:
        print(
            "========================Vehicle:{}=======================".format(
                vehicle.id
            )
        )
    perception_dist = (
        6 * vehicle.SPEED_MAX if perception_dist is None else perception_dist
    )

    s_e = vehicle.to_dict()
    # Set min velocity for calculations in extreme case
    s_e["vx"] = s_e["vx"] if s_e["vx"] > 1 else 1
    sf_e = {k: s_e[k] for k in cbf.STATE_SPACE}

    # Assume a virtual vehicle stopped beyond the perception
    sf_ol = {}
    for k in cbf.STATE_SPACE:
        if k == "x":
            sf_ol[k] = s_e[k] + perception_dist + 1
        elif k == "y":
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

    sf_oar = copy.deepcopy(sf_oa)
    sf_oar["x"] = s_e["x"] - perception_dist - 1

    s_ol, s_oa, s_oar, a_ol, a_oa, gp = multi_agent_state(
        cbf=cbf,
        vehicle=vehicle,
        road=road,
        perception_dist=perception_dist,
        is_ma_dynamics=cbf.is_ma_dynamics,
    )

    if s_oa is not None:
        sf_oa = {k: s_oa[k] if k in s_oa else 0 for k in cbf.STATE_SPACE}

    if s_ol is not None:
        sf_ol = {k: s_ol[k] if k in s_ol else 0 for k in cbf.STATE_SPACE}

    if s_oar is not None:
        sf_oar = {k: s_oar[k] if k in s_oar else 0 for k in cbf.STATE_SPACE}

    fgp_e = vehicle.fg_params
    if fgp_e is None:
        raise AttributeError(
            "fg_params not found in the the vehicle class: {0}".format(type(vehicle))
        )

    gp_e = fgp_e["g"]
    gp_ol = gp["ol"]
    gp_oa = gp["oa"]

    # Unactuated dynamics of heading and steering angle of observed vehicle are ignored here
    f = np.ravel(
        np.array(
            [
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
            ]
        )
    )

    # Representation: [[s_e1, s_e2], [s_ol1, s_ol2], [s_oa1, s_oa2], [s_oar1, s_aor2] x no. of inpus in u]
    g = np.reshape(
        np.array(
            [
                [
                    [gp_e["vx"] * dt, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                ],
                [
                    [0, 1 * dt],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                ],
                [
                    [0, 0],
                    [gp_ol["vx"] * dt, 0],
                    [0, 0],
                    [0, 0],
                ],
                [
                    [0, 0],
                    [0, 1 * dt],
                    [0, 0],
                    [0, 0],
                ],
                [
                    [0, 0],
                    [0, 0],
                    [gp_oa["vx"] * dt, 0],
                    [0, 0],
                ],
                [
                    [0, 0],
                    [0, 0],
                    [0, 1 * dt],
                    [0, 0],
                ],
                [
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [1 * dt, 0],
                ],
                [
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 1 * dt],
                ],
            ]
        ),
        (cbf.action_size * 4, -1),
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

    if safe_dist == "theadway":
        # Safe dist using Time hadway [s]
        sv_oar = s_oar["vx"] if s_oar is not None else 0
        sv_oar = sv_oar + cbf.ACCELERATION_RANGE[1] * dt
        sv_oar = (
            sv_oar if sv_oar > 1 else 1
        )  # Set min velocity for calculations in extreme case

        buffer = (cbf.ACCELERATION_RANGE[1] + 0.1) * dt * cbf.TAU
        cbf.safe_dists = [
            s_e["vx"] * cbf.TAU + vehicle.LENGTH + buffer,
            s_e["vx"] * cbf.TAU + vehicle.LENGTH + buffer,
            sv_oar * cbf.TAU + vehicle.LENGTH + buffer,
        ]

        if CBF_DEBUG:
            print("safe distance: theadway:[lead, adj, rear_adj]: ", cbf.safe_dists)
        # Time headway in [s]
        vehicle.set_min_headway(
            (sf_ol["x"] - s_e["x"] - vehicle.LENGTH) / s_e["vx"], cbf.TAU
        )
    else:
        raise ValueError("safe_dist type {} not supported".format(safe_dist))

    v_ll, dpsi_ll = simplified_control(s=s_e, action=action, vl=vehicle.LENGTH, dt=dt)
    v_ol, dpsi_ol = simplified_control(s=s_ol, action=a_ol, vl=vehicle.LENGTH, dt=dt)
    v_oa, dpsi_oa = simplified_control(s=s_oa, action=a_oa, vl=vehicle.LENGTH, dt=dt)

    # Worst case acceleration is assumed for observed rear vehicle as it has not made its decision yet
    wcr_action = {"steering": 0, "acceleration": cbf.ACCELERATION_RANGE[1]}
    v_oar, dpsi_oar = simplified_control(
        s=s_oar, action=wcr_action, vl=vehicle.LENGTH, dt=dt
    )

    u_ma = np.array([v_ll, dpsi_ll, v_ol, dpsi_ol, v_oa, dpsi_oa, v_oar, dpsi_oar])

    u_safe = cbf.control_barrier(u_ma, f, g, x, dt)

    # Lateral control is not constrained yet.
    u_safe[1] = action["steering"]

    u_safe_ma = np.append(u_safe, u_ma[2:])

    vehicle.is_collaborating = cbf.constrain_adj
    vehicle.is_lc_safe = True

    # Avoid lane change if adjacent vehicle is close
    if is_close_to_lane_edge(vehicle=vehicle) and (
        not cbf.is_lc_allowed(f=f, g=g, x=x, u=u_safe_ma)
    ):
        if CBF_DEBUG:
            print("Avoiding lane change")
        vehicle.target_lane_index = vehicle.lane_index
        u_safe[1] = vehicle.steering_control(vehicle.target_lane_index)
        vehicle.is_lc_safe = False
    # Avoid longitudinal constraint if vehicle is slow and changing lanes
    elif (
        vehicle.hl_action in ["LANE_RIGHT", "LANE_LEFT"]
        and vehicle.speed < vehicle.STOPPING_SPEED
    ):
        u_safe[0] = v_ll

    vehicle.collaborate_adj = cbf.can_collaborate_adj(f=f, g=g, x=x, u=u_safe_ma)

    if CBF_DEBUG:
        print("u_safe: ", u_safe)

    u_safe[0] = derived_acceleration(u_safe[0], s_e["vx"], dt)

    safe_action = {"acceleration": u_safe[0], "steering": u_safe[1]}
    safe_diff = {
        "acceleration": u_safe[0] - action["acceleration"],
        "steering": u_safe[1] - action["steering"],
    }
    return safe_action, safe_diff, cbf.get_status()


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

    if safety_type in ("hss", "av", "avs", "avs_cint"):
        v_min = vehicle.speed + vehicle.MIN_ACC * dt
        v_max = vehicle.speed + vehicle.MAX_ACC * dt
        cbf: CBFType = cbf_factory(
            safety_type,
            action_size=len(action),
            action_bound=[(v_min, v_max), (-4 * np.pi, 4 * np.pi)],
            vehicle_size=[vehicle.LENGTH, vehicle.WIDTH],
            vehicle_lane=vehicle.lane_index[2],
        )
        return safe_action_hss(
            cbf=cbf,
            action=action,
            vehicle=vehicle,
            dt=dt,
            safe_dist=safe_dist,
            **kwargs
        )
    elif safety_type in ("mass", "cav"):
        v_min = vehicle.speed + vehicle.MIN_ACC * dt
        v_min = max(0, v_min)
        v_max = vehicle.speed + vehicle.MAX_ACC * dt
        cbf: CBFType = cbf_factory(
            safety_type,
            action_size=len(action),
            action_bound=[(v_min, v_max), (-4 * np.pi, 4 * np.pi)],
            vehicle_size=[vehicle.LENGTH, vehicle.WIDTH],
            vehicle_lane=vehicle.lane_index[2],
        )
        return safe_action_mass(
            cbf=cbf,
            action=action,
            vehicle=vehicle,
            dt=dt,
            safe_dist=safe_dist,
            **kwargs
        )
    else:
        raise ValueError("Undefined safety_type:{0}".format(safety_type))
