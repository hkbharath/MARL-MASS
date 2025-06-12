import numpy as np
from cvxopt import matrix
from cvxopt import solvers
from typing import List, Any, Tuple, Union, Dict
from highway_env.utils import CBF_DEBUG


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

    TAU = 0.5
    """Safe time headway"""

    ADJ_BUFFER = 2.0134
    """Buffer distance with adjcent vehicle to account for worst case assumption"""

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

        self.is_ma_dynamics = False
        self.constrain_adj = False

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
        return

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

        if CBF_DEBUG:
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

        solvers.options["show_progress"] = False
        sol = solvers.qp(self.P, self.q, G, h, A, b)
        u_bar = sol["x"]

        u_safe = np.add(np.squeeze(u_ll)[:2], np.squeeze(u_bar)[:2])
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

        if CBF_DEBUG:
            print("u_ll , u_bar: ", np.squeeze(u_ll), np.squeeze(u_bar)[:2])

        u_status = np.append(u_safe, u_ll[2:])
        self.update_status(is_opt=is_opt, f=f, g=g, x=x, u_safe=u_status)

        return np.array(u_safe)

    def is_lc_allowed(self, f=None, g=None, x=None, u=None):
        return True

    def hs(self, p, q, x):
        return np.dot(p, x) + q

    def hds(self, p, q, f, g, u):
        return np.dot(p, f) + np.dot(np.squeeze(np.dot(p, g)), u) + q

    def eval_safety_invariance(self, p, q, x, f, g, u, eta, p_str=None):

        hls = self.hs(p=p, q=q, x=x)
        hlds = self.hds(p=p, q=q, f=f, g=g, u=u)

        is_safe = hls >= -1e-6
        is_invariant = (hlds + (eta - 1) * hls) >= -1e-6

        if CBF_DEBUG and p_str is not None:
            print(p_str, hls, hlds)

        return is_safe, is_invariant

    def update_status(self, is_opt, f, g, x, u_safe):
        self.is_optimal = is_opt

    def get_status(self) -> Dict[str, bool]:
        sol_state = {"is_optimal": float(self.is_optimal)}
        if self.is_safe is not None:
            sol_state["is_safe"] = float(self.is_safe)
        if self.is_invariant is not None:
            sol_state["is_invariant"] = float(self.is_invariant)

        return sol_state

class CBF_AV(CBFType):
    """Single agent CBF for individual AVs defined in Wang 2020. The lateral and longitudinal safe distance constrains are implemented in this class"""

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
        """Lateral and Longitudinal safety constraints

        Args:
            action_size (int): number of actions
            action_bound (List[Tuple]): acceptable bounds for the control actions
            vehicle_size (List[int]): dimensions of the vehicle
            vehicle_lane (int): vehicles current lane

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

        self.safe_dists: List[int] = [0, 0, 0]

    def define_pq(self, x: np.array) -> None:
        # print("x_safe_dists: ", self.safe_dists)

        # Logitudinal CBF: h_lon
        self.p_lon = self.dx_l
        self.p_lona = self.dx_a
        self.p_lonr = self.dx_r

        if CBF_DEBUG:
            print("p_lon: ", self.p_lon)
            print("p_lona: ", self.p_lona)
            print("p_lonr: ", self.p_lonr)

        # reduce one vehicle length, as position correspond to centre of the car
        self.q_lon = -self.vehicle_size[0] - self.safe_dists[0]
        self.q_lona = -self.vehicle_size[0] - self.safe_dists[1]
        self.q_lonr = -self.vehicle_size[0] - self.safe_dists[2]

        if CBF_DEBUG:
            print("q_lon: ", self.q_lon)
            print("q_lona: ", self.q_lona)
            print("q_lonr: ", self.q_lonr)

    def hds(self, p, q, f, g, u):
        return np.dot(p, f) + np.dot(np.squeeze(np.dot(p, g)), u) + q

    def get_G(self, g):

        G = np.concatenate(
            (
                np.expand_dims(-np.dot(self.p_lon, g[:, :2]), axis=0),
                [[1, 0]],
                [[-1, 0]],
            )
        )

        # This row added to accomodate for the extra input varibale used to stabilise the optimisation process
        G = np.concatenate((G, [[-1], [0], [0]]), axis=1)

        if CBF_DEBUG:
            print("G: ", G)

        return G

    def get_h(self, f, g, x, u_ll, eta=None):
        if eta is None:
            eta = self.GAMMA_B
        h = np.array(
            [
                np.dot(self.p_lon, f)
                + (eta - 1) * np.dot(self.p_lon, x)
                + eta * self.q_lon
                + np.dot(self.p_lon, np.squeeze(np.dot(g, u_ll))),
                self.action_bound[0][1] - u_ll[0],
                -self.action_bound[0][0] + u_ll[0],
            ]
        )
        if CBF_DEBUG:
            print("h: ", h)

        return h

    def is_lc_allowed(self, f, g, x, u):
        eta = self.GAMMA_B
        hls_lona = self.hs(p=self.p_lona, q=self.q_lona, x=x)
        hlds_lona = self.hds(p=self.p_lona, q=self.q_lona, f=f, g=g, u=u)

        hls_lonr = self.hs(p=self.p_lonr, q=self.q_lonr, x=x)
        hlds_lonr = self.hds(p=self.p_lonr, q=self.q_lonr, f=f, g=g, u=u)

        if CBF_DEBUG:
            print("h_lona(s), h_lona(s'): ", hls_lona, hlds_lona)
            print("h_lonr(s), h_lonr(s'): ", hls_lonr, hlds_lonr)

        # if either lateral or longitudinal condition is satisified for both vehicle in front and read, lc is allowed
        return ((hls_lona >= 0) and (hlds_lona + (eta - 1) * hls_lona) >= 0) and (
            (hls_lonr >= 0) and (hlds_lonr + (eta - 1) * hls_lonr) >= 0
        )

    def update_status(self, is_opt, f, g, x, u_safe, eta=None):
        if eta is None:
            eta = self.GAMMA_B
        super().update_status(is_opt, f, g, x, u_safe)
        hls_lon = self.hs(
            p=self.p_lon, q=self.q_lon, x=x
        )  # np.dot(self.p_lon, x) + self.q_lon
        hlds_lon = self.hds(p=self.p_lon, q=self.q_lon, f=f, g=g, u=u_safe)

        self.is_safe = hls_lon >= -1e-6
        self.is_invariant = (hlds_lon + (eta - 1) * hls_lon) >= -1e-6

        if CBF_DEBUG:
            print("is safe: ", self.is_safe)
            print("is invariant: ", self.is_invariant)
            print("h_lon(s), h_lon(s'): ", hls_lon, hlds_lon)
            print("eta: ", eta)

class CBF_CAV(CBF_AV):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.constrain_adj = False
        self.is_ma_dynamics = True

    def define_pq(self, x: np.array) -> None:
        super().define_pq(x=x)

        if self.constrain_adj:
            self.q_lona = -self.vehicle_size[0] - self.safe_dists[1] - self.ADJ_BUFFER

            if CBF_DEBUG:
                print("Override ma -> q_lona: ", self.q_lona)

    def get_G(self, g):

        G = np.concatenate(
            (
                np.expand_dims(-np.dot(self.p_lon, g[:, :2]), axis=0),
                [[1, 0]],
                [[-1, 0]],
            )
        )

        # This row added to accomodate for the extra input varibale used to stabilise the optimisation process
        G = np.concatenate((G, [[-1], [0], [0]]), axis=1)

        if self.constrain_adj:
            g_adj = np.expand_dims(-np.dot(self.p_lona, g[:, :2]), axis=0)
            G = np.concatenate((G, np.concatenate((g_adj, [[-1]]), axis=1)))

        if CBF_DEBUG:
            print("G: ", G)

        return G

    def get_h(self, f, g, x, u_ll, eta=None):
        u_ma = u_ll
        if eta is None:
            eta = self.GAMMA_B
        h = np.array(
            [
                np.dot(self.p_lon, f)
                + (eta - 1) * np.dot(self.p_lon, x)
                + eta * self.q_lon
                + np.dot(self.p_lon, np.squeeze(np.dot(g, u_ma))),
                self.action_bound[0][1] - u_ma[0],
                -self.action_bound[0][0] + u_ma[0],
            ]
        )

        if self.constrain_adj:
            h = np.append(
                h,
                np.dot(self.p_lona, f)
                + (eta - 1) * np.dot(self.p_lona, x)
                + eta * self.q_lona
                + np.dot(self.p_lona, np.squeeze(np.dot(g, u_ma))),
            )

        if CBF_DEBUG:
            print("h: ", h)
        return h

    def can_collaborate_adj(self, f, g, x, u, eta=None):
        if eta is None:
            eta = self.GAMMA_B
        saf, inv = self.eval_safety_invariance(
            p=self.p_lona, q=self.q_lona, x=x, f=f, g=g, u=u, eta=eta
        )
        return inv


def cbf_factory(cbf_type: str, **kwargs) -> CBFType:

    if cbf_type in ("hss", "av", "avs", "avs_cint"):
        return CBF_AV(**kwargs)
    elif cbf_type in ("mass", "cav"):
        return CBF_CAV(**kwargs)
    else:
        raise ValueError("Undefined cbf_type:{0}".format(cbf_type))
