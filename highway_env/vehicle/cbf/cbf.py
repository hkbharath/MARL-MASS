import numpy as np
from cvxopt import matrix
from cvxopt import solvers

class CBFType():
    def __init__(self, action_size):
        self.P = matrix(np.diag([1., 1e18]), tc='d')
        self.q = matrix(np.zeros(action_size+1))
        self.gamma_b = 0.5
        self.kd = 1

    def get_G(self, g):
        raise NotImplementedError("subclass must implement get_G")
    
    def get_h(self, f, g, x, u_rl, std):
        raise NotImplementedError("subclass must implement get_G")

    def check_bounds(self, u_chk):
        # if (np.add(np.squeeze(u_rl), np.squeeze(u_bar[0])) - 0.001 >= self.torque_bound):
        #     u_bar[0] = self.torque_bound - u_rl
        #     print("Error in QP")
        # elif (np.add(np.squeeze(u_rl), np.squeeze(u_bar[0])) + 0.001 <= -self.torque_bound):
        #     u_bar[0] = -self.torque_bound - u_rl
        #     print("Error in QP")
        return

    def control_barrier(self, obs, u_rl, f, g, x, std):
        u_rl = np.squeeze(u_rl)
        #Set up Quadratic Program to satisfy CBF
        
        G = self.get_G(g=g)
        h = self.get_h(f=f, g=g, x=x, u_rl=u_rl, std=std)
        
        #Convert numpy arrays to cvx matrices to set up QP
        G = matrix(G,tc='d')
        h = matrix(h,tc='d')

        solvers.options['show_progress'] = False
        sol = solvers.qp(self.P, self.q, G, h)
        u_bar = sol['x']

        u_safe = np.add(np.squeeze(u_rl), np.squeeze(u_bar[0]))
        u_safe = self.check_bounds(u_safe)

        return np.expand_dims(np.array(u_safe), 0)

class CBFRobust(CBFType):
    """Robust CBF solved from Chen et al. 2019. This is used only for reference at this moment.

    Args:
        CBFRobust(int): action size for setting up CBF matrices
    """
    def __init__(self, action_size, action_bound):
        super().__init__(action_size)
        self.action_bound = action_bound
        self.H1 = np.array([0, 0, 0, 0, 0, 0, 1, 0.001, 0, -1, -0.001, 0, 0, 0, 0])
        self.H2 = np.array([0, 0, 0, 0, 0, 0, 1, 0.001, 0, -1, 0.001, 0, 0, 0, 0])
        self.H3 = np.array([0, 0, 0, 0, 0, 0, 1, -0.001, 0, -1, -0.001, 0, 0, 0, 0])
        self.H4 = np.array([0, 0, 0, 0, 0, 0, 1, -0.001, 0, -1, 0.001, 0, 0, 0, 0])
        self.H5 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0.001, 0, -1, 0.001, 0])
        self.H6 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0.001, 0, -1, -0.001, 0])
        self.H7 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -0.001, 0, -1, 0.001, 0])
        self.H8 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -0.001, 0, -1, -0.001, 0])
        self.F = -2.
    
    def get_G(self, g):
        G = np.array([[-np.dot(self.H1,g), -np.dot(self.H2,g), -np.dot(self.H3,g), -np.dot(self.H4,g), -np.dot(self.H5,g), 
                   -np.dot(self.H6,g), -np.dot(self.H7,g), -np.dot(self.H8,g), 1, -1] , [-1, -1, -1, -1, -1, -1, -1, -1, 0, 0]])
        G = np.transpose(G)
        return G
    
    def get_h(self, f, g, x, u_rl, std):
        h = np.array([self.gamma_b*self.F + np.dot(self.H1,f) + np.dot(self.H1,g)*u_rl - (1-self.gamma_b)*np.dot(self.H1,x) - self.kd*np.dot(np.abs(self.H1),std),
                  self.gamma_b*self.F + np.dot(self.H2,f) + np.dot(self.H2,g)*u_rl - (1-self.gamma_b)*np.dot(self.H2,x) - self.kd*np.dot(np.abs(self.H2),std),
                  self.gamma_b*self.F + np.dot(self.H3,f) + np.dot(self.H3,g)*u_rl - (1-self.gamma_b)*np.dot(self.H3,x) - self.kd*np.dot(np.abs(self.H3),std),
                  self.gamma_b*self.F + np.dot(self.H4,f) + np.dot(self.H4,g)*u_rl - (1-self.gamma_b)*np.dot(self.H4,x) - self.kd*np.dot(np.abs(self.H4),std),
                  self.gamma_b*self.F + np.dot(self.H5,f) + np.dot(self.H5,g)*u_rl - (1-self.gamma_b)*np.dot(self.H5,x) - self.kd*np.dot(np.abs(self.H5),std),
                  self.gamma_b*self.F + np.dot(self.H6,f) + np.dot(self.H6,g)*u_rl - (1-self.gamma_b)*np.dot(self.H6,x) - self.kd*np.dot(np.abs(self.H6),std),
                  self.gamma_b*self.F + np.dot(self.H7,f) + np.dot(self.H7,g)*u_rl - (1-self.gamma_b)*np.dot(self.H7,x) - self.kd*np.dot(np.abs(self.H7),std),
                  self.gamma_b*self.F + np.dot(self.H8,f) + np.dot(self.H8,g)*u_rl - (1-self.gamma_b)*np.dot(self.H8,x) - self.kd*np.dot(np.abs(self.H8),std),
                  -u_rl + self.action_bound,
                  u_rl + self.action_bound])
        return h
    
class CBFAV(CBFType):
    """Single agent CBF for AVs defined in Wang 2020.

    Args:
        int: action size for setting up CBF matrices
        np.array: action bounds
    """
    def __init__(self, action_size, action_bound):
        super().__init__(action_size)
        self.action_bound = action_bound
        # TODO: Redefine these calues
        self.H1 = np.array([0, 0, 0, 0, 0, 0, 1, 0.001, 0, -1, -0.001, 0, 0, 0, 0])
        self.H2 = np.array([0, 0, 0, 0, 0, 0, 1, 0.001, 0, -1, 0.001, 0, 0, 0, 0])
        self.H3 = np.array([0, 0, 0, 0, 0, 0, 1, -0.001, 0, -1, -0.001, 0, 0, 0, 0])
        self.H4 = np.array([0, 0, 0, 0, 0, 0, 1, -0.001, 0, -1, 0.001, 0, 0, 0, 0])
        self.H5 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0.001, 0, -1, 0.001, 0])
        self.H6 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0.001, 0, -1, -0.001, 0])
        self.H7 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -0.001, 0, -1, 0.001, 0])
        self.H8 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -0.001, 0, -1, -0.001, 0])
        self.F = -2.
    
    def get_G(self, g):
        # TODO: redefine this
        G = np.array([[-np.dot(self.H1,g), -np.dot(self.H2,g), -np.dot(self.H3,g), -np.dot(self.H4,g), -np.dot(self.H5,g), 
                   -np.dot(self.H6,g), -np.dot(self.H7,g), -np.dot(self.H8,g), 1, -1] , [-1, -1, -1, -1, -1, -1, -1, -1, 0, 0]])
        G = np.transpose(G)
        return G
    
    def get_h(self, f, g, x, u_rl, std):
        # TODO: redefine this
        h = np.array([self.gamma_b*self.F + np.dot(self.H1,f) + np.dot(self.H1,g)*u_rl - (1-self.gamma_b)*np.dot(self.H1,x) - self.kd*np.dot(np.abs(self.H1),std),
                  self.gamma_b*self.F + np.dot(self.H2,f) + np.dot(self.H2,g)*u_rl - (1-self.gamma_b)*np.dot(self.H2,x) - self.kd*np.dot(np.abs(self.H2),std),
                  self.gamma_b*self.F + np.dot(self.H3,f) + np.dot(self.H3,g)*u_rl - (1-self.gamma_b)*np.dot(self.H3,x) - self.kd*np.dot(np.abs(self.H3),std),
                  self.gamma_b*self.F + np.dot(self.H4,f) + np.dot(self.H4,g)*u_rl - (1-self.gamma_b)*np.dot(self.H4,x) - self.kd*np.dot(np.abs(self.H4),std),
                  self.gamma_b*self.F + np.dot(self.H5,f) + np.dot(self.H5,g)*u_rl - (1-self.gamma_b)*np.dot(self.H5,x) - self.kd*np.dot(np.abs(self.H5),std),
                  self.gamma_b*self.F + np.dot(self.H6,f) + np.dot(self.H6,g)*u_rl - (1-self.gamma_b)*np.dot(self.H6,x) - self.kd*np.dot(np.abs(self.H6),std),
                  self.gamma_b*self.F + np.dot(self.H7,f) + np.dot(self.H7,g)*u_rl - (1-self.gamma_b)*np.dot(self.H7,x) - self.kd*np.dot(np.abs(self.H7),std),
                  self.gamma_b*self.F + np.dot(self.H8,f) + np.dot(self.H8,g)*u_rl - (1-self.gamma_b)*np.dot(self.H8,x) - self.kd*np.dot(np.abs(self.H8),std),
                  -u_rl + self.action_bound,
                  u_rl + self.action_bound])
        return h

def cbf_factory(cbf_type:str, **kwargs) -> CBFType:
    if cbf_type == "robust":
        return CBFRobust(**kwargs)
    elif cbf_type == "av":
        return CBFAV(**kwargs)