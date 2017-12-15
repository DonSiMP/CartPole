"""
Dynamics setup of a cart-pole system.
See Dynamics class docstring for more details.

"""
from __future__ import division
import numpy as np; npl = np.linalg


class Dynamics(object):
    """
    Class for the dynamics of a cart-pole system.
    State: q == [pos, ang, vel, angvel]
    Input: u == force_on_cart
    (ang is 0 when hanging pole and pi when upright pole)

    l:          scalar pole length
    m:          tuple of cart and pole point masses
    b:          tuple of rail and joint damping
    g:          scalar downwards gravity
    rail_lims:  tuple of (min, max) limits on cart position
    force_lims: tuple of (min, max) limits for push-force on cart

    """
    def __init__(self, l=1, m=(0.3, 0.2), b=(0.5, 0.05), g=9.81, rail_lims=(-3, 3), force_lims=(-5, 5)):
        self.l = np.float64(l)
        self.m = np.array(m, dtype=np.float64)
        self.b = np.array(b, dtype=np.float64)
        self.g = np.float64(g)
        self.rail_lims = np.array(rail_lims, dtype=np.float64)
        self.force_lims = np.array(force_lims, dtype=np.float64)

        # System dimensions
        self.n_u = 1  # inputs
        self.n_d = 2  # degrees of freedom
        self.n_q = 2*self.n_d  # states

        # Verify that f and F correspond
        Q = np.random.sample((3, 4))
        U = np.random.sample((3, 1))
        for q, u, qdot in zip(Q, U, self.F(Q, U)):
            if not np.array_equal(self.f(q, u), qdot):
                raise AssertionError("Invalid Dynamics class source-code; f and F definitions don't correspond!")

    def step(self, q, u, dt, disturb=0.0):
        """
        Integrates cart-pole dynamics one timestep by partial-Verlet method, q_next = step(q, u, dt).
        Applies rail and force constraints.

        q:       array current state [pos, ang, vel, angvel]
        u:       scalar current input force on cart
        dt:      timestep for integration
        disturb: optional scalar external disturbance force on cart

        """
        q = np.array(q, dtype=np.float64)

        # Enforce input saturation and add disturbance
        u_act = np.clip(np.float64(u), self.force_lims[0], self.force_lims[1]) + disturb

        # Get numpy views of pose and twist, and compute accel
        pose = q[:2]
        twist = q[2:]
        accel = self.f(q, u_act)[2:]

        # Partial-Verlet integrate state and enforce rail constraints
        pose_next = pose + dt*twist + 0.5*(dt**2)*accel
        if pose_next[0] < self.rail_lims[0]:
            pose_next[0] = self.rail_lims[0]
            twist_next = np.array([0.0, twist[1] + dt*self.f(q, 0.0)[3]])
        elif pose_next[0] > self.rail_lims[1]:
            pose_next[0] = self.rail_lims[1]
            twist_next = np.array([0.0, twist[1] + dt*self.f(q, 0.0)[3]])
        else:
            twist_next = twist + dt*accel

        # Return next state = [pose, twist]
        return np.append(pose_next, twist_next)

    def f(self, q, u):
        """
        Cart-pole continuous-time design-dynamics, qdot = f(q, u).
        Ignores any rail or force limits.

        q: array state [pos, ang, vel, angvel]
        u: scalar input force on cart

        """
        # Memoize
        sq1, cq1 = np.sin(q[1]), np.cos(q[1])
        cq1cq1 = cq1*cq1
        q3q3 = q[3]*q[3]

        # Analytical dynamics in state-space form
        return np.array([q[2],
                         q[3],
                         (self.l*self.m[1]*sq1*q3q3 + u - self.b[0]*q[2] + self.m[1]*self.g*cq1*sq1) / (self.m[0] + self.m[1]*(1-cq1cq1)),
                         -(self.l*self.m[1]*cq1*sq1*q3q3 + u*cq1 + (self.m[0]+self.m[1])*self.g*sq1 + self.b[1]*q[3]) / (self.l*(self.m[0] + self.m[1]*(1-cq1cq1)))], dtype=np.float64)

    def F(self, Q, U):
        """
        Vectorized version of cart-pole continuous-time design-dynamics, Qdot = F(Q, U).
        Ignores any rail or force limits.

        Q: array N-by-4 state timeseries
        U: array N-by-1 input timeseries

        """
        # Memoize
        sQ1, cQ1 = np.sin(Q[:, 1]), np.cos(Q[:, 1])
        cQ1cQ1 = cQ1*cQ1
        Q3Q3 = Q[:, 3]*Q[:, 3]

        # Analytical dynamics in state-space form
        return np.column_stack((Q[:, 2],
                                Q[:, 3],
                                (self.l*self.m[1]*sQ1*Q3Q3 + U[:, 0] - self.b[0]*Q[:, 2] + self.m[1]*self.g*cQ1*sQ1) / (self.m[0] + self.m[1]*(1.0-cQ1cQ1)),
                                -(self.l*self.m[1]*cQ1*sQ1*Q3Q3 + U[:, 0]*cQ1 + (self.m[0]+self.m[1])*self.g*sQ1 + self.b[1]*Q[:, 3]) / (self.l*(self.m[0] + self.m[1]*(1.0-cQ1cQ1)))))

    def linearize(self, Q, U):
        """
        Returns a tuple (list_of_A_matrices, list_of_B_matrices) where A = df/dq(qi, ui) and B = df/du(qi, ui) for all
        the (qi, ui) pairs along the trajectory (Q, U), i.e. a piecewise LTV approximation of F about (Q, U).

        Q: array N-by-4 state timeseries
        U: array N-by-1 input timeseries

        """
        # Memoize
        Z = np.zeros(Q.shape[0], dtype=np.float64)
        I = np.ones(Q.shape[0], dtype=np.float64)
        sQ1, cQ1 = np.sin(Q[:, 1]), np.cos(Q[:, 1])
        s2Q1, c2Q1 = np.sin(2*Q[:, 1]), np.cos(2*Q[:, 1])
        cQ1cQ1 = cQ1*cQ1
        Q3Q3 = Q[:, 3]*Q[:, 3]

        # Stack-up analytical jacobians in a very vectorized manner
        return (np.rollaxis(np.array([[Z,Z,I,Z],
                                      [Z,Z,Z,I],
                                      [Z,(self.m[1]*(-2*self.g*self.m[1] + 2*self.g*c2Q1*(2*self.m[0] + self.m[1]) + self.l*cQ1*(4*self.m[0] - self.m[1])*Q3Q3 + self.l*np.cos(3*Q[:, 1])*self.m[1]*Q3Q3 - 4*U[:, 0]*s2Q1 + 4*self.b[0]*Q[:, 2]*s2Q1))/np.power(2*self.m[0] + self.m[1] - c2Q1*self.m[1],2),-(self.b[0]/(self.m[0] + self.m[1] - cQ1cQ1*self.m[1])),(2*self.l*self.m[1]*Q[:, 3]*sQ1)/(self.m[0] + self.m[1] - cQ1cQ1*self.m[1])],
                                      [Z,(-((2*self.m[0] + self.m[1] - c2Q1*self.m[1])*(self.g*cQ1*(self.m[0] + self.m[1]) + self.l*c2Q1*self.m[1]*Q3Q3 - U[:, 0]*sQ1))/2. + 2*cQ1*self.m[1]*sQ1*(self.b[1]*Q[:, 3] + self.g*(self.m[0] + self.m[1])*sQ1 + cQ1*(U[:, 0] + self.l*self.m[1]*Q3Q3*sQ1)))/(self.l*np.power(self.m[0] + self.m[1] - cQ1cQ1*self.m[1],2)),Z,(-2*(self.b[1] + self.l*self.m[1]*Q[:, 3]*s2Q1))/(self.l*(2*self.m[0] + self.m[1] - c2Q1*self.m[1]))]]), 2),
                np.rollaxis(np.array([[Z],
                                      [Z],
                                      [1/(self.m[0] + self.m[1] - cQ1cQ1*self.m[1])],
                                      [-(cQ1/(self.l*(self.m[0] + self.m[1] - cQ1cQ1*self.m[1])))]]), 2))
