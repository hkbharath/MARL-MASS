import numpy as np
from highway_env.road.road import LaneIndex
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.controller import MDPVehicle

class MDPLCVehicle(MDPVehicle):
    # This is set to 1/simulation_freq as the process of steering can be assumed to be a simple proportional process.
    KP_STEER = 15
    STEER_TARGET_RF = 0.150  # Reduction factor to define a smaller target to reach.

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
        self.action_hist = None
        self.state_hist = None
        self.t_step = 0.0

        if store_profile:
            self.action_hist = []
            self.state_hist = []

        # Addition state parameter store current steering state
        self.steering_angle = 0

        self.fg_params = {}

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

    def step(self, dt: float) -> None:
        """
        Propagate the vehicle state given its actions.

        Extends a modified bicycle model handle 1st-order response on the steering wheel dynamics and
        2nd-order response to steering wheel based on steering velocity.

        If the vehicle is crashed, the actions are overridden with erratic steering and braking until complete stop.
        The vehicle's current lane is updated.


        :param dt: timestep of integration of the model [s]
        """

        state_var: dict = {"t_step": self.t_step}
        if self.lateral_ctrl == "steer_vel":
            self.clip_actions()
            self.steering_angle += self.action["steering"] * dt
            beta = np.arctan(1 / 2 * np.tan(self.steering_angle))
            v = self.speed * np.array(
                [np.cos(self.heading + beta), np.sin(self.heading + beta)]
            )
            self.position += v * dt

            d_heading = self.speed * np.sin(beta) / (self.LENGTH / 2)
            self.heading += d_heading

            self.speed += self.action["acceleration"] * dt

            self.fg_params = {
                "f": {
                    "x": self.velocity[0],
                    "y": self.velocity[1],
                    "heading": d_heading,
                },
                "g": {
                    "vx": np.cos(self.heading + beta),
                    "vy": np.sin(self.heading + beta),
                },
            }

            self.on_state_update()
        else:
            super().step(dt)
            state_var.update({"steering_angle": self.action["steering"]})

        if self.action_hist is not None:
            action_rec = self.action
            action_rec["t_step"] = self.t_step
            action_rec["lc_action"] = 1  # 'IDLE': 1
            if self.lane_index != self.target_lane_index:
                # lc_actions: 'LANE_LEFT': 0, 'LANE_RIGHT': 2,
                # lane_index: left:0, right:1
                _f, _t, _id = self.target_lane_index
                action_rec["lc_action"] = 0 if _id == 0 else 2
            self.action_hist.append(self.action)

        if self.state_hist is not None:
            state_var.update(self.to_dict())
            state_var.update({"speed": self.speed})
            self.state_hist.append(state_var)

        self.t_step += dt
