import numpy as np
from highway_env.road.road import LaneIndex
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.controller import MDPVehicle
from highway_env.vehicle.safety.decentral_layer import safety_layer
from typing import Any, Tuple, Dict, Union


class MDPLCVehicle(MDPVehicle):

    KP_STEER = 20
    """ Proportional constant for steering velocity control"""
    STEER_TARGET_RF = 0.125
    """ Reduction factor to define a smaller target to reach."""

    # [m/s2] 0-100 km/h in 4.63s
    MAX_ACC: float = 6

    # [m/s2] to safety maintain 1.2s time headway to reach from max speed to lowest speed.
    MIN_ACC: float = -12.5
    PERCEPTION_DIST = 6 * MDPVehicle.SPEED_MAX

    SAFE_DIST: str = "theadway"

    def __init__(
        self,
        safety_layer: str = None,
        lateral_ctrl: str = "steer",
        store_profile: bool = True,
        **kwargs
    ) -> None:
        """_summary_

        Args:
            safety_layer (_type_, optional): Specify the class used to ensure safety of the vehicle. Defaults to None.
            lateral_ctrl (str, optional): Set `lateral_ctrl = 'steer'` for 1st-order response on the steering wheel dynamics and
                                            `lateral_ctrl = 'steer_vel'` for 2nd-order response to steering wheel based on steering velocity.
                                            Defaults to 'steer'.
        """
        super().__init__(**kwargs)
        self.safety_layer = safety_layer
        self.lateral_ctrl = lateral_ctrl
        self.store_profile = store_profile
        self.t_step = 0.0
        self.action_hist = []
        self.state_hist = []
        self.hl_action = None
        self.collaborate_adj = False
        self.safe_action = self.action

        # Addition state parameter store current steering state
        self.steering_angle = 0

        self.min_headway = self.PERCEPTION_DIST / self.MAX_SPEED  # 6

        self.fg_params = None

    def act(self, action: Union[dict, str] = None) -> None:
        if isinstance(action, str):
            self.hl_action = action
        super().act(action)

    def to_dict(
        self, origin_vehicle: "Vehicle" = None, observe_intentions: bool = True
    ) -> dict:
        d = super().to_dict(origin_vehicle, observe_intentions)

        if self.lateral_ctrl == "steer_vel":
            d["steering_angle"] = self.steering_angle

        # set heading relative to the lane heading
        vlocal_pos = self.lane.local_coordinates(self.position)
        d["heading"] = self.lane.heading_at(vlocal_pos[0]) - d["heading"]

        if origin_vehicle:
            origin_dict = origin_vehicle.to_dict()
            for key in ["heading"]:
                d[key] -= origin_dict[key]
        return d

    def steering_control(self, target_lane_index: LaneIndex) -> float:
        """Generate a steering velocity using proportional controller. KP must be tune appropriately for this process.

        Args:
            target_lane_index (LaneIndex): target lane index

        Returns:
            float: steering control or steering velocity control
        """
        steering_ref = super().steering_control(target_lane_index)
        if self.lateral_ctrl == "steer_vel":
            steering_ref = steering_ref * self.STEER_TARGET_RF
            return self.KP_STEER * (steering_ref - self.steering_angle)

        return steering_ref

    def clip_actions(self) -> None:
        super().clip_actions()
        self.action["acceleration"] = np.clip(
            self.action["acceleration"], self.MIN_ACC, self.MAX_ACC
        )

    def step(self, dt: float) -> None:
        """
        Propagate the vehicle state given its actions.

        Extends a modified bicycle model handle 1st-order response on the steering wheel dynamics and
        2nd-order response to steering wheel based on steering velocity.

        If the vehicle is crashed, the actions are overridden with erratic steering and braking until complete stop.
        The vehicle's current lane is updated.


        :param dt: timestep of integration of the model [s]
        """

        self.clip_actions()
        safe_action, safe_diff, safe_status = self.get_safe_action(dt)
        self.safe_action = safe_action

        if self.lateral_ctrl == "steer_vel":

            beta = np.arctan(1 / 2 * np.tan(self.steering_angle))
            v = self.speed * np.array(
                [np.cos(self.heading + beta), np.sin(self.heading + beta)]
            )
            self.position += v * dt

            d_heading = self.speed * np.sin(beta) / (self.LENGTH / 2)
            self.heading += d_heading

            self.speed += safe_action["acceleration"] * dt
            self.steering_angle += safe_action["steering"] * dt
            self.speed = max(0, self.speed)

            self.fg_params = {
                "f": {
                    "x": self.velocity[0],
                    "y": self.velocity[1],
                    "heading": d_heading,
                    "beta": beta,
                },
                "g": {
                    "vx": np.cos(self.heading + beta),
                    "vy": np.sin(self.heading + beta),
                },
            }
        elif self.lateral_ctrl == "steer":
            delta_f = safe_action["steering"]
            beta = np.arctan(1 / 2 * np.tan(delta_f))
            v = self.speed * np.array(
                [np.cos(self.heading + beta), np.sin(self.heading + beta)]
            )
            self.position += v * dt
            self.heading += self.speed * np.sin(beta) / (self.LENGTH / 2) * dt
            self.speed += safe_action["acceleration"] * dt
            self.speed = max(0, self.speed)

            self.fg_params = {
                "f": {
                    "x": self.velocity[0],
                    "y": self.velocity[1],
                    "beta": beta,
                },
                "g": {
                    "vx": np.cos(self.heading + beta),
                    "vy": np.sin(self.heading + beta),
                },
            }
        else:
            raise AttributeError(
                "Lateral control: {0} is not supported".format(self.lateral_ctrl)
            )
        self.on_state_update()

        self.t_step += dt
        a_info = {
            "safe_action": safe_action,
            "safe_diff": safe_diff,
            "safe_status": safe_status,
        }
        self.log_step(additional_info=a_info)

    def log_step(self, additional_info: dict = None):
        if not self.store_profile:
            return

        state_rec: dict = self.to_dict()
        state_rec.update({"speed": self.speed})
        if self.lateral_ctrl != "steer_vel":
            state_rec.update({"steering_angle": self.action["steering"]})

        if (
            additional_info is not None
            and "safe_status" in additional_info
            and additional_info["safe_status"] is not None
        ):
            state_rec["safe_status"] = additional_info["safe_status"]

        state_rec["t_step"] = self.t_step
        state_rec["headway"] = self.min_headway

        self.state_hist.append(state_rec)

        safe_action = None
        safe_diff = None
        if additional_info is not None and all(
            k in additional_info for k in ["safe_action", "safe_diff"]
        ):
            safe_action = additional_info["safe_action"]
            safe_diff = additional_info["safe_diff"]

        action_rec = self.action
        if safe_action is not None:
            action_rec = safe_action
        action_rec["ull_acceleration"] = self.action["acceleration"]
        action_rec["ull_steering"] = self.action["steering"]

        # Fix this step to log high level behaviour
        action_rec["lc_action"] = self.get_hl_action_map(self.hl_action)

        if safe_diff is not None:
            action_rec["safe_diff"] = safe_diff
        action_rec["t_step"] = self.t_step
        self.action_hist.append(action_rec)

    def get_safe_action(
        self, dt: float
    ) -> Tuple[Dict, Union[Dict, None], Union[Dict, None]]:
        if (
            self.safety_layer in ["none", "priority"]
            or self.lateral_ctrl not in ["steer_vel", "steer"]
            or "cbf-" not in self.safety_layer
            or self.fg_params is None
            or len(self.state_hist) < 2
        ):
            return self.action, None, None

        safety_type = self.safety_layer.split("-")[1]
        return safety_layer(
            safety_type=safety_type,
            road=self.road,
            vehicle=self,
            dt=dt,
            perception_dist=self.PERCEPTION_DIST,
            action=self.action,
            safe_dist=self.SAFE_DIST,
        )

    def get_hl_action_map(self, a_str: str) -> int:
        # picked from AbstractEnv
        ACTIONS_ALL = {
            "LANE_LEFT": 0,
            "IDLE": 1,
            "LANE_RIGHT": 2,
            "FASTER": 3,
            "SLOWER": 4,
        }

        return ACTIONS_ALL[a_str]

    def set_min_headway(self, headway: float, tau: float):
        self.min_headway = headway
        if self.min_headway < tau:
            print(
                "Minimum headway violation by vehicle {2}: observed {0}, but expected >{1}".format(
                    self.min_headway, tau, self.id
                )
            )
            print("Vehicle state:\n", self.to_dict())
