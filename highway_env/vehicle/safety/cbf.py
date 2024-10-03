import numpy as np
from cvxopt import matrix
from cvxopt import solvers
from typing import List, Any, Tuple, Union, Dict


class CBFType:

    STATE_SPACE = ["x", "y", "vx", "vy", "heading", "steering_angle"]

    ACCELERATION_RANGE = (-6, 6)
    """Acceleration range: [-x, x], in m/s²."""

    STEERING_ANGLE_RANGE = (-np.pi / 3, np.pi / 3)
    """ Steering range : (-del, del) in rad """

    GAMMA_B = 0
    """Gamma used to enforce stricness of constraint in CBF optimisation"""

    KD = 1
    """Used for robust constraints"""

    TAU = 1.2
    """Safe time headway"""

    def __init__(
        self, action_size: int, action_bound: List[Tuple], vehicle_size: List[int]
    ):
        # P representes the coefficients of the input variables x^2.
        # Since we aim to minimise both velocity at first only one value is set to 1.
        # For both values set both values to 1.
        self.action_size = action_size
        self.action_bound = action_bound
        self.vehicle_size = vehicle_size

        # Additional size is used to assist optimisation process
        self.u_size = self.action_size + 1
        np_P = np.diag(np.ones(self.u_size))
        np_P[self.u_size - 1][self.u_size - 1] = 1e18
        self.P = matrix(np_P, tc="d")

        # q is used to define coefficients of x1x2 terms. In our optimisation problem these coefficients are 0.
        self.q = matrix(np.zeros(self.u_size), tc="d")

        self.is_safe: Union[bool, None] = None
        self.is_invariant: Union[bool, None] = None
        self.is_optimal: Union[bool, None] = None

    def get_G(self, g):
        """_summary_

        Args:
            g (matrix): actuated dynamics

        Raises:
            NotImplementedError: Must be implemented in the extended class
        """
        raise NotImplementedError("subclass must implement get_G")

    def get_h(self, f, g, x, u_ll, eta=None):
        """
        Args:
            f: matrix representing unactuated dynamics
            g: matrix representing actuated dynamics
            x: state variables
            u_ll: low-level input provided by the PID controller
            std: standard deviation ()?

        Raises:
            NotImplementedError: Must be implemented in the extended class

        Returns:
            CBF constraints
        """
        raise NotImplementedError("subclass must implement get_G")

    def get_Ab(self, f, g, x, u_ll):
        return None, None

    def check_bounds(self, u_safe):
        if u_safe[0] - 0.001 > self.action_bound[0][1]:
            raise ValueError(
                "Error in QP. Invalid accceleration: {0}".format(u_safe[0])
            )
        elif u_safe[0] + 0.001 < self.action_bound[0][0]:
            raise ValueError(
                "Error in QP. Invalid accceleration: {0}".format(u_safe[0])
            )

    def check_dims(self, G: np.ndarray, h: np.ndarray):
        """Check dimensions of G and h matricies

        Args:
            G (np.array): coefficients multiplied with inpput `u`
            h (np.array): constraint bounding values
        """
        raise NotImplementedError("subclass must implement check_dims")

    def define_pq(self, x: np.array) -> None:
        return

    def control_barrier(self, u_ll, f, g, x, dt=0):
        u_ll = np.squeeze(u_ll)
        # Set up Quadratic Program to satisfy CBF

        print("f: ", f)
        print("g: ", g)
        print("x: ", x)

        self.define_pq(x=x)

        G = self.get_G(g=g)
        h = self.get_h(f=f, g=g, x=x, u_ll=u_ll)

        # Convert numpy arrays to cvx matrices to set up QP
        G = matrix(G, tc="d")
        h = matrix(h, tc="d")

        A, b = self.get_Ab(f=f, g=g, x=x, u_ll=u_ll)
        if (A is not None) and (b is not None):
            A = matrix(A, tc="d")
            b = matrix(b, tc="d")

        solvers.options["show_progress"] = True
        sol = solvers.qp(self.P, self.q, G, h, A, b)
        u_bar = sol["x"]

        u_safe = np.add(np.squeeze(u_ll), np.squeeze(u_bar)[:2])
        self.check_bounds(u_safe)

        is_opt = sol["status"] != "unknown"

        # If dt is specified, solve qp again with 1/dt eta value
        # if not is_opt and dt > 0:
        #     h = self.get_h(f=f, g=g, x=x, u_ll=u_ll, eta=1/dt)
        #     h = matrix(h, tc="d")

        #     sol = solvers.qp(self.P, self.q, G, h, A, b)
        #     u_bar = sol["x"]

        #     u_safe = np.add(np.squeeze(u_ll), np.squeeze(u_bar)[:2])
        #     self.check_bounds(u_safe)

        #     is_opt = sol["status"] != "unknown"

        print("u_ll , u_bar: ", np.squeeze(u_ll), np.squeeze(u_bar)[:2])
        self.update_status(is_opt=is_opt, f=f, g=g, x=x, u_safe=u_safe)

        return np.array(u_safe)

    def is_lc_allowed(self, f=None, g=None, x=None, u=None):
        return True

    def hs(self, p, q, x):
        return np.dot(p, x) + q

    def hds(self, p, q, f, g, u):
        return np.dot(p, f) + np.dot(np.squeeze(np.dot(p, g)), u) + q

    def update_status(self, is_opt, f, g, x, u_safe):
        self.is_optimal = is_opt

    def get_status(self) -> Dict[str, bool]:
        sol_state = {"is_optimal": float(self.is_optimal)}
        if self.is_safe is not None:
            sol_state["is_safe"] = float(self.is_safe)
        if self.is_invariant is not None:
            sol_state["is_invariant"] = float(self.is_invariant)

        return sol_state


class CBF_AV_Longitudinal(CBFType):
    """Single agent CBF for AVs defined in Wang 2020, but only for longitudinal control. This CBF consideres states from two vehicles
    and defines three constraints. The first constraint is the cbf constraint for longitudinal motion.
    Remaining two constraints limit the control inputs to allowed action_bounds.
    The vehicle state space must contain 6 variables, specified in CBFType.STATE_SPACE

    Args:
        int: action size for setting up CBF matrices
        np.array: action bounds
    """

    # GAMMA_B = 1.6251

    def __init__(
        self,
        action_size: int,
        action_bound: List[Tuple],
        vehicle_size: List[int],
        **kwargs
    ) -> Any:
        super().__init__(action_size, action_bound, vehicle_size)

        # Supporting matrix
        # \delta x between two vehicle is evaluated from this matrix
        self.dx = np.ravel(np.array([[-1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]]))

        # \delta vx between two vehicle is evaluated from this matrix
        self.dvx = np.ravel(np.array([[0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0]]))

        # Logitudinal CBF: h_lon
        self.p_lon = self.dx - self.TAU * self.dvx

        # reduce one vehicle length, as position correspond to centre of the car
        self.q_lon = -self.vehicle_size[0]

    def get_G(self, g):

        G = np.concatenate(
            (np.expand_dims(-np.dot(self.p_lon, g), axis=0), [[1, 0]], [[-1, 0]])
        )

        # This row added to accomodate for the extra input varibale used to stabilise the optimisation process
        G = np.concatenate((G, [[-1], [0], [0]]), axis=1)

        print("G: ", G)

        # assert (G.shape, (3, self.action_size))
        return G

    def get_h(self, f, g, x, u_ll, eta=None):

        h = np.array(
            [
                np.dot(self.p_lon, f)
                + (self.GAMMA_B - 1) * np.dot(self.p_lon, x)
                + self.GAMMA_B * self.q_lon
                + np.dot(np.squeeze(np.dot(self.p_lon, g)), u_ll),
                self.action_bound[0][1] - u_ll[0],
                -self.action_bound[0][0] + u_ll[0],
            ]
        )
        print("h: ", h)
        # assert (h.shape, (3, 1))
        return h

    def update_status(self, is_opt, f, g, x, u_safe):
        super().update_status(is_opt, f, g, x, u_safe)
        hls = np.dot(self.p_lon, x) + self.q_lon
        hlds = (
            np.dot(self.p_lon, f)
            + np.dot(np.squeeze(np.dot(self.p_lon, g)), u_safe)
            + self.q_lon
        )
        self.is_safe = hls >= 0
        self.is_invariant = (hlds + (self.GAMMA_B - 1) * hls) >= 0

        print("is safe: ", self.is_safe)
        print("is invariant: ", self.is_invariant)
        print("h_lon(s), h_lon(s'): ", hls, hlds)
        print("eta: ", self.GAMMA_B)


class CBF_AV(CBFType):
    """Single agent CBF for individual AVs defined in Wang 2020. The lateral and longitudinal safe distance constrains are implemented in this class"""

    # GAMMA_B = 1.6251
    # GAMMA_LAT = 1.625

    STATE_SPACE = ["x", "heading"]

    ACCELERATION_RANGE = (-12.5, 6)
    """Acceleration range: in m/s²."""

    def __init__(
        self,
        action_size: int,
        action_bound: List[Tuple],
        vehicle_size: List[int],
        vehicle_lane: int,
    ):
        """Lateral andlongitudinal safety constraints from Wang 2020

        Args:
            action_size (int): number of actions
            action_bound (List[Tuple]): acceptable bounds for the control actions
            vehicle_size (List[int]): dimensions of the vehicle
            vehicle_lane (int): vehicles current lane

        Raises:
            ValueError: If vehicle in an environment with more than 2 lanes
        """
        super().__init__(action_size, action_bound, vehicle_size)

        # Supporting matrices
        # \delta x with leading vehicle
        self.dx_l = np.ravel(
            np.array(
                [
                    [-1, 0],
                    [1, 0],
                    [0, 0],
                    [0, 0],
                ]
            )
        )

        # \delta x with adjacent vehicle
        self.dx_a = np.ravel(
            np.array(
                [
                    [-1, 0],
                    [0, 0],
                    [1, 0],
                    [0, 0],
                ]
            )
        )

        # \delta x with rear vehicle
        self.dx_r = np.ravel(
            np.array(
                [
                    [1, 0],
                    [0, 0],
                    [0, 0],
                    [-1, 0],
                ]
            )
        )

        self.safe_dists:List[int] = [0,0,0]

    def define_pq(self, x: np.array) -> None:
        # print("x_safe_dists: ", self.safe_dists)

        # Logitudinal CBF: h_lon
        self.p_lon = self.dx_l
        self.p_lona = self.dx_a
        self.p_lonr = self.dx_r

        # print("p_lon: ", self.p_lon)
        # print("p_lona: ", self.p_lona)
        # print("p_lonr: ", self.p_lonr)

        # reduce one vehicle length, as position correspond to centre of the car
        self.q_lon = -self.vehicle_size[0] - self.safe_dists[0]
        self.q_lona = -self.vehicle_size[0] - self.safe_dists[1]
        self.q_lonr = -self.vehicle_size[0] - self.safe_dists[2]

        # print("q_lon: ", self.q_lon)
        # print("q_lona: ", self.q_lona)
        # print("q_lonr: ", self.q_lonr)
        
    def hds(self, p, q, f, g, u):
        return np.dot(p, f) + np.dot(np.squeeze(np.dot(p, g)), u)

    def get_G(self, g):

        G = np.concatenate(
            (
                np.expand_dims(-np.dot(self.p_lon, g), axis=0),
                # np.expand_dims(-np.dot(self.p_lat, g), axis=0),
                [[1, 0]],
                [[-1, 0]],
            )
        )

        # This row added to accomodate for the extra input varibale used to stabilise the optimisation process
        # G = np.concatenate((G, [[-1], [-1], [0], [0]]), axis=1)
        G = np.concatenate((G, [[-1], [0], [0]]), axis=1)

        print("G: ", G)

        # assert (G.shape, (3, self.action_size))
        return G

    def get_h(self, f, g, x, u_ll, eta=None):
        if eta is None:
            eta = self.GAMMA_B
        h = np.array(
            [
                np.dot(self.p_lon, f)
                + (eta - 1) * np.dot(self.p_lon, x)
                + (eta - 1) * self.q_lon
                + np.dot(np.squeeze(np.dot(self.p_lon, g)), u_ll),
                # np.dot(self.p_lat, f)
                # + (eta - 1) * np.dot(self.p_lat, x)
                # + eta * self.q_lat
                # + np.dot(np.squeeze(np.dot(self.p_lat, g)), u_ll),
                self.action_bound[0][1] - u_ll[0],
                -self.action_bound[0][0] + u_ll[0],
            ]
        )
        print("h: ", h)
        # assert (h.shape, (3, 1))
        return h

    def is_lc_allowed(self, f, g, x, u):
        eta = self.GAMMA_B
        hls_lona = self.hs(p=self.p_lona, q=self.q_lona, x=x)
        hlds_lona = self.hds(p=self.p_lona, q=self.q_lona, f=f, g=g, u=u)

        hls_lonr = self.hs(p=self.p_lonr, q=self.q_lonr, x=x)
        hlds_lonr = self.hds(p=self.p_lonr, q=self.q_lonr, f=f, g=g, u=u)

        print("h_lonr(s), h_lonr(s'): ", hls_lonr, hlds_lonr)

        # if either lateral or longitudinal condition is satisified for both vehicle in front and read, lc is allowed
        return ((hlds_lona + (eta - 1) * hls_lona) >= 0) and (
            (hlds_lonr + (eta - 1) * hls_lonr) >= 0
        )

    def update_status(self, is_opt, f, g, x, u_safe, eta=None):
        if eta is None:
            eta = self.GAMMA_B
        super().update_status(is_opt, f, g, x, u_safe)
        hls_lon = self.hs(
            p=self.p_lon, q=self.q_lon, x=x
        )  # np.dot(self.p_lon, x) + self.q_lon
        hlds_lon = self.hds(p=self.p_lon, q=self.q_lon, f=f, g=g, u=u_safe)

        self.is_safe = hls_lon >= 0
        self.is_invariant = (hlds_lon + (eta - 1) * hls_lon) >= 0

        print("is safe: ", self.is_safe)
        print("is invariant: ", self.is_invariant)
        print("h_lon(s), h_lon(s'): ", hls_lon, hlds_lon)
        print("eta: ", eta)


class CBF_AV_Lateral(CBFType):
    # TODO: Define this class to consider lateral safe space.
    """Single agent CBF for AVs defined in Wang 2020.

    Args:
        int: action size for setting up CBF matrices
        np.array: action bounds
    """

    def __init__(
        self,
        action_size: int,
        action_bound: List[Tuple],
        vehicle_size: List[int],
        vehicle_lane: int,
    ):
        super().__init__(action_size, action_bound, vehicle_size)

        # Supporting matrix
        # \delta x between two vehicle is evaluated from this matrix
        self.dx_l = np.ravel(
            np.array([[-1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
        )
        self.dx_a = np.ravel(
            np.array([[-1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]])
        )

        # \delta y between two vehicle is evaluated from this matrix
        if vehicle_lane == 0:
            self.dy_a = np.ravel(
                np.array([[0, -1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])
            )
        elif vehicle_lane == 1:
            self.dy_a = np.ravel(
                np.array([[0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, -1, 0, 0, 0, 0]])
            )
        else:
            raise ValueError("CBF constraint is implemented for 2 lanes only")

        # vx of ego vehicle
        self.vx = np.ravel(
            np.array([[0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
        )

        # steering angle of ego vehicle
        self.st = np.ravel(
            np.array([[0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
        )

        # safe longitudinal distance
        self.x_safe = self.TAU * self.vx

        # safe lateral distance
        self.y_safe = 4 - vehicle_size[1]  # AbstractLane.DEFAULT_WIDTH - Vehicle width

        # Logitudinal CBF: h_lon
        self.p_lon = self.dx_l - self.x_safe

        # reduce one vehicle length, as position correspond to centre of the car
        self.q_lon = -self.vehicle_size[0]

        self.p_lat = (self.dx_a / (self.TAU * 30)) + (self.dy_a / self.y_safe)
        print("p_lat:", self.p_lat)
        self.q_lat = -1

        # TODO: Add steering angle bound.
        # self.p_st_lb = self.st
        # self.q_st_lb = -self.STEERING_ANGLE_RANGE[0]

        # self.p_st_ub = -self.st
        # self.q_st_ub = self.STEERING_ANGLE_RANGE[1]

    def get_G(self, g):

        G = np.concatenate(
            (
                np.expand_dims(-np.dot(self.p_lon, g), axis=0),
                np.expand_dims(-np.dot(self.p_lat, g), axis=0),
                [[1, 0]],
                [[-1, 0]],
            )
        )

        # This row added to accomodate for the extra input varibale used to stabilise the optimisation process
        G = np.concatenate((G, [[-1], [-1], [0], [0]]), axis=1)

        print("G: ", G)

        # assert (G.shape, (3, self.action_size))
        return G

    def get_h(self, f, g, x, u_ll, eta=None):
        h = np.array(
            [
                np.dot(self.p_lon, f)
                + (self.GAMMA_B - 1) * np.dot(self.p_lon, x)
                + self.GAMMA_B * self.q_lon
                + np.dot(np.squeeze(np.dot(self.p_lon, g)), u_ll),
                np.dot(self.p_lat, f)
                + (self.GAMMA_B - 1) * np.dot(self.p_lat, x)
                + self.GAMMA_B * self.q_lat
                + np.dot(np.squeeze(np.dot(self.p_lat, g)), u_ll),
                self.action_bound[0][1] - u_ll[0],
                -self.action_bound[0][0] + u_ll[0],
            ]
        )
        print("h: ", h)
        # assert (h.shape, (3, 1))
        return h

    def get_Ab(self, f, g, x, u_ll):
        # hls_lat = np.dot(self.p_lat, x) + self.q_lat
        # hlds_lat = (
        #     np.dot(self.p_lat, f)
        #     + np.dot(np.squeeze(np.dot(self.p_lat, g)), u_ll)
        #     + self.q_lat
        # )

        # if ((hlds_lat + (self.GAMMA_B - 1) * hls_lat) < 0):
        #     A = np.concatenate(([np.dot(self.st, g)], [[-1]]), axis=1)
        #     b = np.array([0])
        #     # b = np.array([-np.dot(np.squeeze(np.dot(self.st, g)), u_ll)])
        #     print("A: ", A)
        #     print("b: ", b)
        #     return A,b

        return None, None

    def update_status(self, is_opt, f, g, x, u_safe):
        super().update_status(is_opt, f, g, x, u_safe)
        hls_lon = np.dot(self.p_lon, x) + self.q_lon
        hlds_lon = (
            np.dot(self.p_lon, f)
            + np.dot(np.squeeze(np.dot(self.p_lon, g)), u_safe)
            + self.q_lon
        )

        hls_lat = np.dot(self.p_lat, x) + self.q_lat
        hlds_lat = (
            np.dot(self.p_lat, f)
            + np.dot(np.squeeze(np.dot(self.p_lat, g)), u_safe)
            + self.q_lat
        )

        self.is_safe = (hls_lon >= 0) and (hls_lat >= 0)
        self.is_invariant = ((hlds_lon + (self.GAMMA_B - 1) * hls_lon) >= 0) and (
            (hlds_lat + (self.GAMMA_B - 1) * hls_lat) >= 0
        )

        print("is safe: ", self.is_safe)
        print("is invariant: ", self.is_invariant)
        print("h_lon(s), h_lon(s'): ", hls_lon, hlds_lon)
        print("h_lat(s), h_lat(s'): ", hls_lat, hlds_lat)
        print("eta: ", self.GAMMA_B)


class CBF_CAV(CBFType):
    """CBF for CAVs solved from Chen et al. 2019. This is used only for reference at this moment.

    Args:
        CBF_CAV(int): action size for setting up CBF matrices
    """

    def __init__(
        self, action_size: int, action_bound: List[Tuple], vehicle_size: List[int]
    ):
        super().__init__(action_size, action_bound, vehicle_size)
        self.P = matrix(np.diag([1.0, 1]), tc="d")

    def get_G(self, g):
        pass

    def get_h(self, f, g, x, u_ll, std):
        pass


def cbf_factory(cbf_type: str, **kwargs) -> CBFType:

    if cbf_type == "avlon":
        return CBF_AV_Longitudinal(**kwargs)
    elif cbf_type == "av":
        return CBF_AV(**kwargs)
    if cbf_type == "cav":
        return CBF_CAV(**kwargs)
    else:
        raise ValueError("Undefined cbf_type:{0}".format(cbf_type))
