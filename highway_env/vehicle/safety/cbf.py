import numpy as np
from cvxopt import matrix
from cvxopt import solvers
from typing import List, Any, Tuple, Union, Dict


class CBFType:

    STATE_SPACE = ["x", "y", "vx", "vy", "heading", "steering_angle"]

    ACCELERATION_RANGE = (-6, 6)
    """Acceleration range: [-x, x], in m/s²."""

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

    def get_h(self, f, g, x, u_ll, std):
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

    def control_barrier(self, u_ll, f, g, x, std=0):
        u_ll = np.squeeze(u_ll)
        # Set up Quadratic Program to satisfy CBF

        print("f: ", f)
        print("g: ", g)
        print("x: ", x)

        G = self.get_G(g=g)
        h = self.get_h(f=f, g=g, x=x, u_ll=u_ll, std=std)

        # Convert numpy arrays to cvx matrices to set up QP
        G = matrix(G, tc="d")
        h = matrix(h, tc="d")

        solvers.options["show_progress"] = True
        sol = solvers.qp(self.P, self.q, G, h)
        u_bar = sol["x"]

        print("u_ll , u_bar: ", np.squeeze(u_ll), np.squeeze(u_bar)[:2])
        u_safe = np.add(np.squeeze(u_ll), np.squeeze(u_bar)[:2])
        self.check_bounds(u_safe)

        is_opt = sol["status"] != "unknown"
        self.update_status(is_opt=is_opt, f=f, g=g, x=x, u_safe=u_safe)

        return np.array(u_safe)

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
        self, action_size: int, action_bound: List[Tuple], vehicle_size: List[int]
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

    def get_h(self, f, g, x, u_ll, std=0):

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
    """Single agent CBF for AVs defined in Wang 2020.

    Args:
        int: action size for setting up CBF matrices
        np.array: action bounds
    """

    def __init__(
        self, action_size: int, action_bound: List[Tuple], vehicle_size: List[int]
    ):
        super().__init__(action_size, action_bound, vehicle_size)
        self.P = matrix(np.diag([1.0, 1]), tc="d")
        # TODO: Redefine these calues

    def get_G(self, g):
        # TODO: redefine this
        pass

    def get_h(self, f, g, x, u_ll, std):
        # TODO: redefine this
        pass


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
