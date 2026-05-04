"""
Central configuration for the entire project.
Every tunable number lives here. Nothing hardcoded elsewhere.
"""
import numpy as np

# ─────────────────────────────────────────────
# Quadrotor Physical Parameters (Crazyflie 2.x scale)
# ─────────────────────────────────────────────
QUAD = dict(
    mass=0.027,                          # kg
    g=9.81,                              # m/s^2
    arm_length=0.0397,                   # m (center to motor)
    k_f=3.16e-10,                        # thrust coef N/(rad/s)^2
    k_t=7.94e-12,                        # torque coef N*m/(rad/s)^2
    I=np.diag([1.4e-5, 1.4e-5, 2.17e-5]),  # inertia tensor kg*m^2
    max_rpm=21702.0,                     # max motor speed (rad/s)
    min_rpm=0.0,
    motor_tau=0.02,                      # motor lag time constant (s)
    drag_coef_trans=np.array([0.0, 0.0, 0.0]),
    drag_coef_rot=np.array([0.0, 0.0, 0.0]),
)

# Derived: hover RPM per motor (healthy, level flight)
QUAD['hover_rpm'] = np.sqrt(QUAD['mass'] * QUAD['g'] / (4 * QUAD['k_f']))

# ─────────────────────────────────────────────
# Simulation Timing
# ─────────────────────────────────────────────
SIM = dict(
    dt=1.0 / 500.0,                     # physics timestep (500 Hz)
    ctrl_dt=1.0 / 50.0,                 # RL control frequency (50 Hz)
    episode_duration=10.0,               # seconds per episode
    ctrl_steps_per_physics=10,           # 500/50 = 10 physics steps per RL step
)
SIM['max_steps'] = int(SIM['episode_duration'] / SIM['ctrl_dt'])

# ─────────────────────────────────────────────
# PID Gains (cascaded: outer=position, inner=attitude)
# ─────────────────────────────────────────────
PID = dict(
    pos_P=np.array([6.0, 6.0, 9.0]),
    pos_I=np.array([0.6, 0.6, 0.90]),
    pos_D=np.array([3.5, 3.5, 5.25]),
    pos_I_max=np.array([0.3, 0.3, 0.5]),

    att_P=np.array([0.02, 0.02, 0.01]),
    att_I=np.array([0.001, 0.001, 0.0005]),
    att_D=np.array([0.002, 0.002, 0.001]),
    att_I_max=np.array([0.005, 0.005, 0.002]),

    max_tilt_cmd=0.35,
)

# ─────────────────────────────────────────────
# Fault Model
# ─────────────────────────────────────────────
FAULT = dict(
    min_efficiency=0.65,                 # worst case: 35% RPM loss
    max_motors_faulted=1,                # single motor faults
)

# ─────────────────────────────────────────────
# RL Observation & Action
# ─────────────────────────────────────────────
RL = dict(
    obs_dim=20,
    act_dim=4,
    compensation_fraction=0.30,          # RL can add +/-30% of max RPM

    learning_rate=3e-4,
    n_steps=2048,
    batch_size=512,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_layers=[256, 256],
    total_timesteps_per_level=3_000_000,
)

# ─────────────────────────────────────────────
# Reward Weights
# ─────────────────────────────────────────────
REWARD = dict(
    k_pos=4.0,
    k_vel=0.15,
    k_att=0.8,
    k_omega=0.02,
    k_action_rate=0.02,
    k_action_mag=0.005,
    alive_bonus=0.1,
    goal_radius=0.15,
    near_radius=0.40,
    goal_bonus=2.0,
    near_bonus=0.2,
    sustained_time=1.0,
    crash_penalty=-15.0,
    max_tilt_deg=70.0,
    max_pos_error=2.0,
)

# ─────────────────────────────────────────────
# Curriculum Levels
# ─────────────────────────────────────────────
CURRICULUM = [
    dict(
        name="healthy",
        fault_range=(1.0, 1.0),
        init_pos_range=0.3,
        init_att_range_deg=5.0,
        domain_rand=False,
        inject_mid_episode=False,
        promotion_reward=200.0,
        promotion_success_rate=0.95,
    ),
    dict(
        name="mild_fault",
        fault_range=(0.85, 0.95),
        init_pos_range=0.3,
        init_att_range_deg=5.0,
        domain_rand=False,
        inject_mid_episode=False,
        promotion_reward=150.0,
        promotion_success_rate=0.92,
    ),
    dict(
        name="moderate_fault",
        fault_range=(0.70, 0.90),
        init_pos_range=0.4,
        init_att_range_deg=8.0,
        domain_rand=False,
        inject_mid_episode=True,
        promotion_reward=120.0,
        promotion_success_rate=0.90,
    ),
    dict(
        name="severe_fault",
        fault_range=(0.65, 0.85),
        init_pos_range=0.5,
        init_att_range_deg=10.0,
        domain_rand=False,
        inject_mid_episode=True,
        promotion_reward=999.0,       # deliberately unreachable — trains full 3M steps
        promotion_success_rate=0.99,
    ),
]