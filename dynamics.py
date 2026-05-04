"""
Full 6-DOF Quadrotor Dynamics

Quaternion-based attitude (no gimbal lock).
First-order motor lag.
Newton-Euler translational + rotational dynamics.

Motor layout (top view, X-config):
    Front
  M1      M2       (front-left, front-right)
   
  M4      M3       (rear-left, rear-right)

M1, M3 spin CW.  M2, M4 spin CCW.
"""
import numpy as np
from config import QUAD, SIM


# ─────────────────────────────────────────────
# Quaternion Utilities
# Convention: [w, x, y, z]
# ─────────────────────────────────────────────

def quat_multiply(q1, q2):
    """Hamilton product q1 * q2."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quat_conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_normalize(q):
    n = np.linalg.norm(q)
    if n < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


def quat_to_rotation_matrix(q):
    """Quaternion [w,x,y,z] to 3x3 rotation matrix (body to world)."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),       1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),       2*(y*z + w*x),      1 - 2*(x*x + y*y)],
    ])


def quat_to_euler(q):
    """Quaternion to Euler [roll, pitch, yaw] in radians."""
    w, x, y, z = q
    sinr_cosp = 2.0 * (w*x + y*z)
    cosr_cosp = 1.0 - 2.0 * (x*x + y*y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w*y - z*x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    siny_cosp = 2.0 * (w*z + x*y)
    cosy_cosp = 1.0 - 2.0 * (y*y + z*z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw])


def euler_to_quat(rpy):
    """Euler [roll, pitch, yaw] to quaternion [w,x,y,z]."""
    r, p, y = rpy * 0.5
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    return np.array([
        cr*cp*cy + sr*sp*sy,
        sr*cp*cy - cr*sp*sy,
        cr*sp*cy + sr*cp*sy,
        cr*cp*sy - sr*sp*cy,
    ])


def rotate_vector_by_quat(q, v):
    """Rotate vector v by quaternion q: result = q * [0,v] * q_conj."""
    v_quat = np.array([0.0, v[0], v[1], v[2]])
    rotated = quat_multiply(quat_multiply(q, v_quat), quat_conjugate(q))
    return rotated[1:]


def gravity_in_body(q):
    """Get gravity direction in body frame (for RL observation)."""
    q_inv = quat_conjugate(q)
    return rotate_vector_by_quat(q_inv, np.array([0.0, 0.0, -1.0]))


# ─────────────────────────────────────────────
# Mixer Matrix
# ─────────────────────────────────────────────

def build_mixer_matrix(arm_length, k_f, k_t):
    """
    4x4 allocation matrix:
    [total_thrust, tau_x, tau_y, tau_z] = M @ [f1, f2, f3, f4]
    """
    L = arm_length / np.sqrt(2)   # effective arm for X-config
    c = k_t / k_f                 # torque-to-thrust ratio

    M = np.array([
        [1.0,   1.0,   1.0,   1.0],       # total thrust
        [ L,    -L,    -L,     L ],        # roll torque
        [ L,     L,    -L,    -L ],        # pitch torque
        [-c,     c,    -c,     c ],        # yaw torque (reaction)
    ])
    return M


# ─────────────────────────────────────────────
# State Container
# ─────────────────────────────────────────────

class QuadrotorState:
    """Holds the full state of the quadrotor."""
    __slots__ = ['pos', 'vel', 'quat', 'omega', 'motor_rpm']

    def __init__(self):
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.omega = np.zeros(3)
        self.motor_rpm = np.zeros(4)

    def copy(self):
        s = QuadrotorState()
        s.pos = self.pos.copy()
        s.vel = self.vel.copy()
        s.quat = self.quat.copy()
        s.omega = self.omega.copy()
        s.motor_rpm = self.motor_rpm.copy()
        return s

    @property
    def euler(self):
        return quat_to_euler(self.quat)

    @property
    def rotation_matrix(self):
        return quat_to_rotation_matrix(self.quat)


# ─────────────────────────────────────────────
# Main Dynamics Class
# ─────────────────────────────────────────────

class QuadrotorDynamics:
    """
    Full 6-DOF dynamics with:
    - Quaternion attitude propagation
    - First-order motor lag
    - Configurable parameters (for domain randomization)
    """

    def __init__(self, params=None):
        p = params if params is not None else QUAD.copy()
        self.mass = p['mass']
        self.g = p['g']
        self.I = p['I'].copy()
        self.I_inv = np.linalg.inv(self.I)
        self.k_f = p['k_f']
        self.k_t = p['k_t']
        self.arm_length = p['arm_length']
        self.max_rpm = p['max_rpm']
        self.min_rpm = p['min_rpm']
        self.motor_tau = p['motor_tau']
        self.drag_trans = p['drag_coef_trans'].copy()
        self.drag_rot = p['drag_coef_rot'].copy()

        self.mixer = build_mixer_matrix(self.arm_length, self.k_f, self.k_t)
        self.mixer_inv = np.linalg.inv(self.mixer)

        self.state = QuadrotorState()

    def reset(self, pos=None, vel=None, quat=None, omega=None):
        """Reset to given initial conditions."""
        self.state = QuadrotorState()
        if pos is not None:
            self.state.pos = np.array(pos, dtype=np.float64)
        if vel is not None:
            self.state.vel = np.array(vel, dtype=np.float64)
        if quat is not None:
            self.state.quat = quat_normalize(np.array(quat, dtype=np.float64))
        if omega is not None:
            self.state.omega = np.array(omega, dtype=np.float64)
        # Start motors at hover RPM
        self.state.motor_rpm = np.ones(4) * self.get_hover_rpm()
        return self.state

    def step(self, rpm_commands, dt):
        """
        Advance physics by dt seconds.
        rpm_commands: [4] desired RPM (after fault applied externally).
        """
        s = self.state

        # 1. Motor dynamics (first-order lag)
        alpha = dt / (self.motor_tau + dt)
        rpm_cmd_clipped = np.clip(rpm_commands, self.min_rpm, self.max_rpm)
        s.motor_rpm = s.motor_rpm + alpha * (rpm_cmd_clipped - s.motor_rpm)

        # 2. Forces from motor RPMs
        thrusts = self.k_f * s.motor_rpm**2          # per-motor thrust
        wrench = self.mixer @ thrusts                  # [F_total, tau_x, tau_y, tau_z]
        total_thrust = wrench[0]
        torques = wrench[1:4]

        # 3. Translational dynamics (world frame)
        R = quat_to_rotation_matrix(s.quat)
        thrust_world = R @ np.array([0.0, 0.0, total_thrust])
        gravity_world = np.array([0.0, 0.0, -self.mass * self.g])
        drag_world = -self.drag_trans * s.vel
        acc = (thrust_world + gravity_world + drag_world) / self.mass

        # 4. Rotational dynamics (body frame)
        # J * omega_dot = torques - omega x (J*omega) - drag
        gyro = np.cross(s.omega, self.I @ s.omega)
        rot_drag = -self.drag_rot * s.omega
        omega_dot = self.I_inv @ (torques - gyro + rot_drag)

        # 5. Integrate (semi-implicit Euler)
        s.vel = s.vel + acc * dt
        s.pos = s.pos + s.vel * dt

        s.omega = s.omega + omega_dot * dt

        # 6. Quaternion integration
        omega_quat = np.array([0.0, s.omega[0], s.omega[1], s.omega[2]])
        q_dot = 0.5 * quat_multiply(s.quat, omega_quat)
        s.quat = s.quat + q_dot * dt
        s.quat = quat_normalize(s.quat)
        s.vel = np.clip(s.vel, -20.0, 20.0)
        s.omega = np.clip(s.omega, -50.0, 50.0)
        s.pos = np.clip(s.pos, -10.0, 10.0)
        return s

    def get_hover_rpm(self):
        """RPM needed per motor for level hover."""
        return np.sqrt(self.mass * self.g / (4 * self.k_f))

    def rpm_from_wrench(self, wrench):
        """Convert [thrust, tau_x, tau_y, tau_z] to per-motor RPMs."""
        thrusts = self.mixer_inv @ wrench
        thrusts = np.clip(thrusts, 0.0, self.k_f * self.max_rpm**2)
        rpms = np.sqrt(thrusts / self.k_f)
        return np.clip(rpms, self.min_rpm, self.max_rpm)