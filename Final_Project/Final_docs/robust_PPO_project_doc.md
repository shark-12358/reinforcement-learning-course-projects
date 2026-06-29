# Robust PPO Ledge-Climbing Robot Project Documentation

This document describes the current robust PPO implementation for the ledge-climbing robot project. It is written for the current robust code version:

- `envs_robust.py`
- `Ledge_Climb_Robot_PPO_robust.py`

The goal of this document is to clearly explain what the environment does, what observations and rewards are used, how PPO processes the data, how robustness perturbations are added, and what files are generated in the `results` folder.

## 1. Project Objective

The task is to train a ledge-climbing robot with PPO. The robot does not learn a full joint trajectory from zero. Instead, the environment provides a hand-designed reference gait, and the PPO policy learns a residual correction added to the reference joint-position command.

The robust version extends the baseline PPO task with optional:

- Multi-environment training using `env1` to `env4`.
- Unseen validation on `env5`.
- Friction domain randomization.
- Sensor observation noise.
- PPO-side low-pass observation filtering.
- More complete CSV and plot outputs for report and debugging.

The main robust training target is to obtain one model that can generalize across `env1` to `env4`, and then validate on the mixed/unseen `env5`.

## 2. Main Files

### 2.1 Environment file

```text
envs_robust.py
```

This file defines the Gymnasium/PyBullet environment:

```python
class LedgeClimbEnv(gym.Env):
```

It handles:

- Loading the robot URDF.
- Loading the selected ledge environment URDF.
- Building the reference gait.
- Applying residual joint-position actions.
- Computing observations.
- Adding sensor noise to observations when enabled.
- Applying friction randomization when enabled.
- Computing rewards, success, fall, timeout, cycle metrics, energy, and COT.
- Recording video frames during rollout.

### 2.2 PPO file

```text
Ledge_Climb_Robot_PPO_robust.py
```

This file handles:

- PPO training.
- Single-env and multi-env training modes.
- Loading and recording existing best models.
- PPO-side observation filtering.
- RunningMeanStd observation normalization.
- Tanh-squashed Gaussian action sampling.
- GAE return calculation.
- PPO clipped-policy update.
- Model saving.
- CSV and plot generation.
- Final best-model rollout video generation.

## 3. Environment Configuration

### 3.1 Available environments

The environment configuration dictionary supports five environment IDs:

```text
env1, env2, env3, env4, env5
```

Each environment has:

- URDF path.
- Robot initial position.
- Environment base position.
- Target definition.
- Success margin.

The current configuration uses:

```python
"target_x": None
```

When `target_x` is `None`, the environment computes the target from the environment AABB:

```python
target_x = env_aabb_max_x - success_margin
```

The current success margin is:

```python
success_margin = 0.05  # m
```

### 3.2 Environment geometry is not directly given to the policy

The agent observation does not include:

- `env_id`
- ledge URDF name
- ledge shape parameters
- `target_x`
- `ledge_height`
- AABB values
- friction coefficients
- contact force
- contact points

Some of those quantities appear in `info` for logging and debugging, but they are not concatenated into the policy observation.

The policy can still react to different terrain geometries indirectly through the robot state, such as base pose, gripper pose, and joint states. This is not information leakage; it is feedback from the simulated robot state.

## 4. Physics and Timing

### 4.1 PyBullet timing

The environment uses:

```python
PHYSICS_DT = 1.0 / 240.0
FRAME_SKIP = 8
CONTROL_DT = PHYSICS_DT * FRAME_SKIP
```

Therefore:

```text
physics frequency = 240 Hz
control frequency = 30 Hz
control time step = 1 / 30 s
```

The policy outputs one residual action every control step. PyBullet advances by `FRAME_SKIP = 8` internal physics steps for each PPO action.

### 4.2 Episode length

The maximum episode duration is:

```python
MAX_EPISODE_SECONDS = 120.0
```

At 30 Hz control frequency:

```text
max_episode_steps = 120.0 / (1 / 30) = 3600 control steps
```

This is different from the speed target used in the reward. The speed target encourages fast completion, while `MAX_EPISODE_SECONDS` is only the hard timeout limit.

### 4.3 Reset settling

At reset, the simulation uses a short settling period:

```python
RESET_SETTLE_SECONDS = 0.25
```

The robot is first initialized to default joint states and allowed to settle before the reference trajectory begins.

## 5. Robot Joints and Action Space

### 5.1 Controlled joints

The environment controls the movable joints found in the robot URDF. The expected joint names are:

```text
left_upper_claw
left_lower_claw
right_upper_claw
right_lower_claw
left_elbow
shoulder
right_elbow
left_wrist
right_wrist
```

The exact ordering is taken from the URDF joint order after fixed joints are skipped. The PPO action dimension is:

```text
action_dim = number of controlled joints = 9
```

### 5.2 PPO action meaning

The PPO policy does not output the full joint command directly. It outputs a residual joint-position correction:

```text
residual_action
```

The environment adds this residual to the reference gait command:

```text
unclipped_applied = current_reference_action + residual_action
applied_action = clip(unclipped_applied, joint_low, joint_high)
```

The applied command is then sent to PyBullet position control.

### 5.3 Residual action limits

The values `5`, `10`, and `15` in the code are angle limits in degrees. They are not torque limits.

The current residual action limits are:

```python
RESIDUAL_ACTION_LIMITS_DEG = {
    "left_upper_claw": 5.0,
    "left_lower_claw": 5.0,
    "right_upper_claw": 5.0,
    "right_lower_claw": 5.0,
    "left_wrist": 10.0,
    "right_wrist": 10.0,
    "left_elbow": 15.0,
    "shoulder": 15.0,
    "right_elbow": 15.0,
}
```

Therefore:

```text
claw joints           : residual limit ±5 deg
wrist joints          : residual limit ±10 deg
elbow/shoulder joints : residual limit ±15 deg
```

The global fallback residual limit is:

```python
RESIDUAL_ACTION_LIMIT_DEG = 5.0
```

### 5.4 Residual action smoothing and rate limiting

Before being applied, the PPO residual action is processed by:

1. Action-space clipping.
2. Exponential residual smoothing.
3. Per-step residual rate limiting.
4. Final joint-limit clipping after adding to the reference command.

The residual smoothing is:

```python
proposed_residual = alpha * commanded_residual + (1 - alpha) * previous_residual
```

with:

```python
RESIDUAL_FILTER_ALPHA = 0.35
```

The residual rate limits are:

```python
RESIDUAL_RATE_LIMITS_DEG_PER_STEP = {
    "left_upper_claw": 3.0,
    "left_lower_claw": 3.0,
    "right_upper_claw": 3.0,
    "right_lower_claw": 3.0,
    "left_wrist": 3.0,
    "right_wrist": 3.0,
    "left_elbow": 4.5,
    "shoulder": 4.5,
    "right_elbow": 4.5,
}
```

This prevents the residual correction from jumping too abruptly.

## 6. Motor Command

The environment uses PyBullet position control:

```python
p.setJointMotorControlArray(
    bodyUniqueId=self.robot_id,
    jointIndices=self.joint_indices,
    controlMode=p.POSITION_CONTROL,
    targetPositions=target_positions.tolist(),
    physicsClientId=self.client_id,
)
```

The current script does not explicitly pass:

```python
forces=
positionGains=
velocityGains=
```

Therefore, the script does not explicitly set per-joint maximum motor force or torque limits.

Important clarification:

```text
The values 5, 10, and 15 are residual angle limits in degrees, not motor torque limits in N·m.
```

The rollout CSV records `joint_torque_Nm` from PyBullet joint states:

```python
p.getJointState(...)[3]
```

This torque is used for logging, power, energy, and COT calculation. It is not a manually specified torque limit in the environment script.

## 7. Reference Gait

### 7.1 Reference trajectory concept

The environment builds a reference joint-position trajectory from keyframes. PPO learns a residual correction around this reference.

The reference trajectory is built in:

```python
_build_reference_trajectory()
```

The trajectory includes phases such as:

- Right claw opening/closing.
- Body joints moving between climbing postures.
- Left claw opening/closing.
- Short hold segments.

### 7.2 Duration scaling

The reference duration scale is:

```python
REFERENCE_DURATION_SCALE = 0.50
```

The keyframe durations are first scaled and then converted from the original 240 Hz demo time base into 30 Hz control steps.

### 7.3 Interpolation

The currently active interpolation is linear:

```python
alpha = (k + 1) / duration
target = (1.0 - alpha) * start + alpha * end
```

There is a quintic interpolation expression in the code comments, but it is not active in the current implementation.

### 7.4 Reference timing fix

The current robust environment includes a reference-observation timing fix.

The correct timing is:

```text
obs_t contains reference[k]
actor outputs residual[k]
step executes reference[k] + residual[k]
```

After a step is executed, the environment increments `reference_step`. The next observation is then built using the updated `reference_step`, so the next actor call sees the reference command that will actually be used by the next step.

Inside `_get_obs()`, the observation reference and phase are obtained together:

```python
obs_reference_action, _, reference_phase = self._get_reference_action()
```

This ensures that:

```text
reference_phase and reference_action in the observation are synchronized.
```

The `info["reference_action"]` field is used for logging and represents the reference action used in the current executed step. It is not the policy input for the next step.

## 8. Observation Design

### 8.1 Observation dimension

The observation dimension is:

```python
obs_dim = (2 * n_joints) + 12 + 12 + 6 + n_joints + 1 + n_joints
```

For `n_joints = 9`:

```text
obs_dim = 67
```

### 8.2 Observation block order

The actual observation order is:

```text
1. joint_angles
2. joint_velocities
3. base_state
4. gripper_pose
5. gripper_velocities
6. previous_residual_action
7. reference_phase
8. obs_reference_action
```

Where:

```text
base_state =
    base position (x, y, z)
    base Euler angle (roll, pitch, yaw)
    base linear velocity (vx, vy, vz)
    base angular velocity (wx, wy, wz)
```

```text
gripper_pose =
    left gripper position (x, y, z)
    left gripper Euler angle (roll, pitch, yaw)
    right gripper position (x, y, z)
    right gripper Euler angle (roll, pitch, yaw)
```

```text
gripper_velocities =
    left gripper linear velocity (vx, vy, vz)
    right gripper linear velocity (vx, vy, vz)
```

### 8.3 Sensor observation block

The first part of the observation is treated as a sensor measurement block:

```text
sensor_obs_dim = (2 * n_joints) + 12 + 12 + 6
```

For `n_joints = 9`:

```text
sensor_obs_dim = 48
```

The sensor block contains:

```text
joint angles
joint velocities
base position
base Euler angle
base linear velocity
base angular velocity
left/right gripper position
left/right gripper Euler angle
left/right gripper linear velocity
```

The following tail blocks are not treated as sensor measurements:

```text
previous residual action
reference phase
current/next reference action
```

These tail blocks are not sensor-noised and are not PPO-filtered.

## 9. Sensor Noise

### 9.1 Where noise is added

Sensor noise is added inside the environment after the clean raw observation is assembled and before the observation is returned to PPO.

The flow is:

```text
PyBullet true state
→ clean raw observation
→ add sensor noise to sensor block
→ noisy raw observation returned by env
→ PPO-side observation filter
→ RunningMeanStd normalization and clipping
→ policy network
```

Noise is added in raw physical units, before PPO normalization.

### 9.2 Clean and noisy observation storage

The current environment stores:

```python
self.last_clean_obs
self.last_noisy_obs
```

`last_clean_obs` is the observation before sensor noise is added.

`last_noisy_obs` is the observation after sensor noise is added.

The PPO rollout logger uses these arrays to produce the clean/noisy/filtered observation comparison CSV and plots.

The clean observation is not passed through `info["clean_observation"]`.

### 9.3 Noise distribution

Sensor noise is zero-mean Gaussian:

```text
noisy_obs_i = clean_obs_i + epsilon_i
```

with:

```text
epsilon_i ~ Normal(0, sigma_i^2)
```

The current implementation does not hard-clip sensor noise.

### 9.4 Default noise standard deviations

Current robust v6 defaults are:

```python
SENSOR_NOISE_STD = {
    "joint_angle": np.deg2rad(2.0),
    "joint_velocity": np.deg2rad(10.0),
    "base_position": 0.01,
    "base_euler": np.deg2rad(2.0),
    "base_linear_velocity": 0.05,
    "base_angular_velocity": np.deg2rad(10.0),
    "gripper_position": 0.01,
    "gripper_euler": np.deg2rad(2.0),
    "gripper_linear_velocity": 0.05,
}
```

In intuitive units:

```text
joint angle noise              : 2 deg standard deviation
joint velocity noise           : 10 deg/s standard deviation
base position noise            : 10 mm standard deviation
base attitude noise            : 2 deg standard deviation
base linear velocity noise     : 0.05 m/s standard deviation
base angular velocity noise    : 10 deg/s standard deviation
gripper position noise         : 10 mm standard deviation
gripper attitude noise         : 2 deg standard deviation
gripper linear velocity noise  : 0.05 m/s standard deviation
```

### 9.5 Noise diagnostics

The environment records:

```text
sensor_noise_mean_abs
sensor_noise_max_abs
```

These are useful to verify whether sensor noise is actually enabled. They mix different physical units, so they are diagnostic quantities rather than direct physical performance metrics.

## 10. Friction Domain Randomization

### 10.1 Purpose

Friction randomization is used to improve robustness to contact uncertainty between the robot and the ledge.

### 10.2 Randomized parameters

At reset, if enabled, the environment samples:

```python
FRICTION_RANDOMIZATION_RANGES = {
    "env_lateral_friction": (0.45, 0.90),
    "robot_lateral_friction": (0.45, 0.90),
    "spinning_friction": (0.0, 0.02),
    "rolling_friction": (0.0, 0.005),
}
```

The sampled values are applied with PyBullet `changeDynamics()` to the environment body and robot links.

### 10.3 Friction logging note

The environment stores sampled friction values in:

```python
info["friction_params"]
```

with keys:

```text
env_lateral_friction
robot_lateral_friction
spinning_friction
rolling_friction
```

The rollout debug logger writes these values dynamically as `friction_<key>` columns.

The episode-level logger still contains some legacy friction column names such as `lateral_friction`, `contact_damping`, and `contact_stiffness`. In the current v6 code, those episode columns may be blank or NaN because the current friction dictionary uses the newer key names listed above.

## 11. PPO-Side Observation Filtering

### 11.1 Filter location

In the current robust pipeline, the environment only adds sensor noise. The active low-pass filter is in the PPO script, not in the environment.

The environment-side `use_observation_filter` argument is kept only for backward compatibility. `make_ledge_env()` passes:

```python
use_observation_filter=False
```

The actual filter is:

```python
ObservationLowPassFilter
```

inside the PPO script.

### 11.2 Filter formula

The filter is an exponential moving average applied only to the sensor block:

```text
filtered_t = alpha * noisy_t + (1 - alpha) * filtered_{t-1}
```

The default is:

```python
OBSERVATION_FILTER_ALPHA = 0.8
```

With this definition:

```text
alpha close to 1.0  → light filtering, follows current measurement closely
alpha close to 0.0  → heavy filtering, smoother but more delayed
```

At episode reset, the filter state is initialized from the first noisy observation.

### 11.3 What is filtered

Only the sensor observation block is filtered:

```text
joint angles
joint velocities
base position
base Euler angle
base linear velocity
base angular velocity
gripper position
gripper Euler angle
gripper linear velocity
```

The following are not filtered:

```text
previous residual action
reference phase
reference action
```

### 11.4 Filter diagnostics

PPO annotates `info` with:

```text
ppo_observation_filter_enabled
ppo_observation_filter_alpha
ppo_observation_filter_delta_mean_abs
ppo_observation_filter_delta_max_abs
```

The delta is computed between filtered and noisy sensor observations.

## 12. Observation Normalization and Clipping

### 12.1 RunningMeanStd

The PPO script uses online observation normalization:

```python
RunningMeanStd
```

During training, the filtered observation is used to update the running mean and variance.

During evaluation and recording, the stored running statistics from the checkpoint are used without updating.

### 12.2 Normalization equation

The normalized observation is:

```text
obs_norm = (obs - mean) / sqrt(var + 1e-8)
```

Then it is clipped to:

```python
OBS_CLIP = 5.0
```

So each normalized observation component is clipped to:

```text
[-5, 5]
```

### 12.3 Invalid value protection

Before normalization, non-finite values are replaced:

```python
nan      → 0
+inf     → 1e6
-inf     → -1e6
```

The number of invalid observation values is logged as:

```text
invalid_obs_count
```

### 12.4 Policy input

The policy network receives:

```text
normalized(filtered(noisy(clean_observation)))
```

when noise and filtering are enabled.

When all robustness switches are disabled, the policy receives:

```text
normalized(clean_observation)
```

## 13. Reward Design

The reward is computed from the true PyBullet state, not from the noisy observation. Sensor noise affects what the policy sees, but it does not directly change the reward calculation, success condition, or fall condition.

The total reward is the sum of:

```text
progress reward
time penalty
action cost
smoothness cost
joint velocity cost
base posture/lateral costs
gripper posture/lateral costs
pace cost
success bonus
success speed bonus
fall penalty
timeout penalty
```

The reward weights are:

```python
REWARD_WEIGHTS = {
    "progress": 10.0,
    "success": 150.0,
    "success_speed": 450.0,
    "fall": -150.0,
    "timeout": -25.0,
    "time": -0.02,
    "pace": -0.08,
    "action": -0.01,
    "smoothness": -0.03,
    "joint_velocity": -0.012,
    "base_lateral": -0.025,
    "base_tilt": -0.020,
    "base_yaw": -0.012,
    "gripper_lateral": -0.028,
    "gripper_center_y": -0.024,
    "gripper_y_span": -0.010,
    "gripper_tilt": -0.020,
    "gripper_yaw": -0.010,
}
```

### 13.1 Progress reward

```text
progress_reward = 10.0 * delta_front_x
```

where `delta_front_x` is the change in the front-most x position among the base and grippers.

### 13.2 Time penalty

Every control step receives:

```text
time_penalty = -0.02
```

This encourages shorter episodes.

### 13.3 Action cost

The action cost penalizes the commanded residual action magnitude after action-space clipping:

```text
action_cost = -0.01 * mean((commanded_residual_action / residual_limit)^2)
```

This discourages large residual corrections.

### 13.4 Smoothness cost

The smoothness cost penalizes the change of the filtered/rate-limited residual action:

```text
smoothness_cost = -0.03 * mean(((residual_action - previous_residual_action) / residual_limit)^2)
```

This discourages sudden residual changes.

### 13.5 Joint velocity cost

The implementation first computes:

```text
mean_abs_joint_velocity = mean(abs(joint_velocity))
```

Then:

```text
joint_velocity_norm = min(mean_abs_joint_velocity / 20.0, 5.0)
joint_velocity_cost = -0.012 * joint_velocity_norm^2
```

This is not the mean of squared normalized joint velocities. It is the squared normalized mean absolute joint velocity.

### 13.6 Base penalties

Base-related terms use normalized squared costs:

```text
base_lateral_norm = min(abs(base_y) / 0.06, 5.0)
base_tilt_norm    = min(base_tilt_abs / deg2rad(10), 5.0)
base_yaw_norm     = min(abs(base_yaw) / deg2rad(18), 5.0)
```

```text
base_lateral_cost = -0.025 * base_lateral_norm^2
base_tilt_cost    = -0.020 * base_tilt_norm^2
base_yaw_cost     = -0.012 * base_yaw_norm^2
```

### 13.7 Gripper penalties

Gripper-related terms also use normalized squared costs:

```text
gripper_lateral_norm  = min(max_gripper_abs_y / 0.05, 5.0)
gripper_center_y_norm = min(abs(gripper_center_y) / 0.04, 5.0)
gripper_y_span_norm   = min(gripper_y_span / 0.10, 5.0)
gripper_tilt_norm     = min(max_gripper_tilt_abs / deg2rad(10), 5.0)
gripper_yaw_norm      = min(gripper_yaw_abs / deg2rad(22), 5.0)
```

```text
gripper_lateral_cost  = -0.028 * gripper_lateral_norm^2
gripper_center_y_cost = -0.024 * gripper_center_y_norm^2
gripper_y_span_cost   = -0.010 * gripper_y_span_norm^2
gripper_tilt_cost     = -0.020 * gripper_tilt_norm^2
gripper_yaw_cost      = -0.010 * gripper_yaw_norm^2
```

### 13.8 Pace cost

Pace cost compares time progress and spatial progress as fractions, not as meter-level position error.

The code computes:

```text
target_distance = target_x - start_front_x
progress_fraction = clip(forward_progress / target_distance, 0, 1)
target_elapsed_fraction = clip(step_counter / target_step_count, 0, 1)
pace_error = max(0, target_elapsed_fraction - progress_fraction)
pace_cost = -0.08 * pace_error^2
```

The target time for pace is:

```python
PACE_TARGET_SECONDS = 15.0
```

### 13.9 Success and success-speed bonus

A successful episode receives:

```text
success_bonus = 150.0
```

The success-speed bonus is:

```text
success_speed_bonus = 450.0 * max(0, 1 - elapsed_seconds / SUCCESS_SPEED_TARGET_SECONDS)
```

with:

```python
SUCCESS_SPEED_TARGET_SECONDS = 15.0
```

This bonus is only given if the task succeeds.

### 13.10 Fall and timeout penalties

If the robot falls:

```text
fall_penalty = -150.0
```

If the episode times out without success or fall:

```text
timeout_penalty = -25.0
```

## 14. Success, Fall, and Timeout Conditions

### 14.1 Success

Success requires:

```text
front_x >= target_x
```

and stable posture:

```text
abs(base_y) <= 0.10
base_tilt_abs <= 32 deg
abs(gripper_center_y) <= 0.09
max_gripper_tilt_abs <= 38 deg
```

### 14.2 Fall or unstable condition

Fall/unstable occurs if:

```text
base_z < ledge_height - FALL_MARGIN
```

with:

```python
FALL_MARGIN = 0.25
```

or if:

```text
max_gripper_abs_y > 0.20
max_gripper_tilt_abs > 50 deg
```

### 14.3 Timeout

Timeout occurs when:

```text
step_counter >= max_episode_steps
```

For the default 120 s episode length and 30 Hz control rate:

```text
max_episode_steps = 3600
```

## 15. Energy, Work, and COT

### 15.1 Joint power

At each PyBullet physics step, the environment reads joint velocity and torque:

```text
joint_velocity
joint_torque
```

It computes signed and absolute mechanical power:

```text
signed_power = sum(torque_i * velocity_i)
absolute_power = sum(abs(torque_i * velocity_i))
```

Then it integrates over the physics time step:

```text
episode_signed_work_j += signed_power * PHYSICS_DT
episode_abs_work_j    += absolute_power * PHYSICS_DT
```

### 15.2 COT definition

The main COT metric uses absolute mechanical work:

```text
COT_abs = episode_abs_work_j / (robot_mass * 9.81 * forward_progress)
```

Absolute work is used as the main COT because the simulation does not model actuator electrical losses or energy regeneration. Negative mechanical work is therefore treated as part of the total actuation effort rather than ignored.

Signed work is still logged:

```text
episode_signed_work_j
```

but it is not the main COT metric.

### 15.3 Cycle metrics

The environment also records cycle-level metrics:

```text
cycle duration
cycle forward distance
cycle average speed
cycle absolute work
cycle signed work
cycle COT_abs
```

Each cycle is based on completion of one reference trajectory cycle.

## 16. PPO Algorithm

### 16.1 Network architecture

The PPO agent uses separate actor and critic MLPs.

Each MLP has:

```text
input layer
hidden layer 1: 256 units, Tanh
hidden layer 2: 256 units, Tanh
output layer
```

The actor outputs the Gaussian mean. The log standard deviation is a learned state-independent parameter vector:

```python
log_std = nn.Parameter(torch.full((act_dim,), LOG_STD_INIT))
```

with:

```python
LOG_STD_INIT = -0.5
LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0
```

### 16.2 Action sampling and squashing

The actor samples a raw Gaussian action:

```text
raw_action ~ Normal(mean, std)
```

Then applies tanh squashing and action-limit scaling:

```text
action = tanh(raw_action) * action_limit
```

The resulting `action` is the residual joint-position command in radians.

### 16.3 Log-probability correction

The PPO log probability includes both:

- tanh squash correction
- action scale correction

The implementation stores raw Gaussian actions in the rollout buffer and recomputes corrected log probabilities during PPO update.

### 16.4 Deterministic action

During evaluation or recording, the policy uses:

```text
raw_action = mean
action = tanh(mean) * action_limit
```

No stochastic sampling is used in deterministic evaluation.

## 17. PPO Training Procedure

### 17.1 Main hyperparameters

Current defaults:

```python
TOTAL_TIMESTEPS = 400_000
ROLLOUT_LENGTH = 2048
MINIBATCH_SIZE = 256
PPO_EPOCHS = 10
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RATIO = 0.20
LEARNING_RATE = 3e-4
ENTROPY_COEF = 0.01
VALUE_COEF = 0.50
MAX_GRAD_NORM = 0.50
TARGET_KL = None
OBS_CLIP = 5.0
EVAL_INTERVAL = 10_000
EVAL_EPISODES = 5
```

### 17.2 Rollout data flow

During training:

```text
env returns noisy raw observation
→ PPO observation filter
→ RunningMeanStd update and normalization
→ actor-critic policy
→ residual action
→ env step
```

The rollout buffer stores normalized observations, raw Gaussian actions, log probabilities, value estimates, rewards, termination flags, done flags, and bootstrap next values.

### 17.3 GAE and bootstrap logic

The buffer computes GAE with:

```text
gamma = 0.99
lambda = 0.95
```

Terminated episodes do not bootstrap. Time-limit truncations may bootstrap from the final value estimate, but GAE does not propagate across episode reset.

### 17.4 PPO update

The PPO objective uses clipped policy ratio:

```text
ratio = exp(new_log_prob - old_log_prob)
```

```text
actor_loss = -mean(min(ratio * advantage, clipped_ratio * advantage))
```

The critic loss is:

```text
critic_loss = 0.5 * mean((return - value)^2)
```

The total loss is:

```text
total_loss = actor_loss + VALUE_COEF * critic_loss - ENTROPY_COEF * entropy
```

Gradient norm is clipped to:

```python
MAX_GRAD_NORM = 0.50
```

### 17.5 Advantage normalization

Before PPO update, advantages are normalized:

```text
advantage = (advantage - mean) / (std + 1e-8)
```

### 17.6 Target KL

The code supports optional KL early stopping:

```python
TARGET_KL
```

Current default:

```python
TARGET_KL = None
```

Therefore KL early stopping is disabled by default.

If enabled, early stopping occurs when:

```text
approx_kl > 1.5 * TARGET_KL
```

## 18. Training and Testing Modes

### 18.1 Run modes

The main mode is selected by:

```python
RUN_MODE
```

Valid values:

```text
train_single
train_multi
record
```

### 18.2 Single environment training

For single environment training:

```python
RUN_MODE = "train_single"
TRAIN_SINGLE_ENV_ID = "1"
```

### 18.3 Multi-environment training

For multi-environment training:

```python
RUN_MODE = "train_multi"
TRAIN_MULTI_ENVS = ["1", "2", "3", "4"]
```

The implementation uses episode-level environment randomization. At the start of each episode, one environment ID is randomly selected from `TRAIN_MULTI_ENVS`.

This is not parallel vectorized PPO. It is a single PyBullet environment whose URDF is switched at episode reset.

### 18.4 Training-time evaluation

Training-time evaluation environments are:

```python
EVAL_ENVS_SINGLE = ["1"]
EVAL_ENVS_MULTI = ["1", "2", "3", "4"]
```

The best model is selected based on evaluation metrics, mainly success rate and then secondary metrics such as return, progress, episode length, and posture stability.

### 18.5 Final rollout after training

After training, if:

```python
SAVE_FINAL_VIDEO = True
```

then the script records the best model on the selected rollout environments.

The environments are controlled by:

```python
TRAIN_RECORD_ROLLOUT_ENVS
```

If it is `None`, the code uses the evaluation environments. To record all environments after multi-env training, use:

```python
TRAIN_RECORD_ROLLOUT_ENVS = ["1", "2", "3", "4", "5"]
```

### 18.6 Record mode

To load an existing best model and test it:

```python
RUN_MODE = "record"
RECORD_TEST_ENVS = ["1", "2", "3", "4", "5"]
RECORD_MODEL_RESULT_DIR = "multi_env_with_noise"
RECORD_RUN_ID = "multi_env_with_noise"
```

For a no-noise model, use:

```python
RECORD_MODEL_RESULT_DIR = "multi_env_without_noise"
RECORD_RUN_ID = "multi_env_without_noise"
```

## 19. Robustness Switches

### 19.1 Training switches

```python
TRAIN_ENABLE_FRICTION_RANDOMIZATION = True
TRAIN_ENABLE_SENSOR_NOISE = True
TRAIN_ENABLE_OBSERVATION_FILTER = True
```

### 19.2 Evaluation switches

```python
EVAL_ENABLE_FRICTION_RANDOMIZATION = False
EVAL_ENABLE_SENSOR_NOISE = False
EVAL_ENABLE_OBSERVATION_FILTER = True
```

By default, training-time evaluation is deterministic and clean, except that PPO observation filtering remains enabled.

### 19.3 Record switches

```python
RECORD_ENABLE_FRICTION_RANDOMIZATION = False
RECORD_ENABLE_SENSOR_NOISE = False
RECORD_ENABLE_OBSERVATION_FILTER = True
```

### 19.4 How to run a clean baseline using the robust code

To make the robust code behave close to the baseline without disturbances, set all disturbance and filter switches to `False`:

```python
TRAIN_ENABLE_FRICTION_RANDOMIZATION = False
TRAIN_ENABLE_SENSOR_NOISE = False
TRAIN_ENABLE_OBSERVATION_FILTER = False

EVAL_ENABLE_FRICTION_RANDOMIZATION = False
EVAL_ENABLE_SENSOR_NOISE = False
EVAL_ENABLE_OBSERVATION_FILTER = False

RECORD_ENABLE_FRICTION_RANDOMIZATION = False
RECORD_ENABLE_SENSOR_NOISE = False
RECORD_ENABLE_OBSERVATION_FILTER = False
```

This disables:

- friction randomization
- sensor noise
- PPO observation filtering

It will be very close to the baseline environment, except that the current robust environment includes the reference timing fix and updated logging/output structure.

## 20. Results Folder Structure

### 20.1 Training output folders

For single-env training with noise enabled:

```text
results/PPO/with_robust/single_env_with_noise/env1/
```

For single-env training with noise disabled:

```text
results/PPO/with_robust/single_env_without_noise/env1/
```

For multi-env training with noise enabled:

```text
results/PPO/with_robust/multi_env_with_noise/
```

For multi-env training with noise disabled:

```text
results/PPO/with_robust/multi_env_without_noise/
```

The folder choice is currently based on whether `TRAIN_ENABLE_SENSOR_NOISE` is true or false. Friction randomization and PPO filter settings do not create additional folder names.

### 20.2 Main subfolders

Each run folder contains:

```text
best_model/
csv/report/
csv/debug/
plot/report/
plot/debug/
video/
```

The current v6 code does not create a separate `csv/raw/` folder.

### 20.3 Current CSV outputs

Current report CSV files include:

```text
csv/report/episode_log.csv
csv/report/update_log.csv
csv/report/eval_log.csv
csv/report/best_model_rollout_summary*.csv
csv/report/best_model_rollout_joint_tracking_log*.csv
csv/report/best_model_rollout_cycle_log*.csv
csv/report/best_model_rollout_observation_filter_log*.csv
```

Current debug CSV files include:

```text
csv/debug/best_model_rollout_debug_log*.csv
```

The `*` suffix appears when recording multiple test environments, for example:

```text
best_model_rollout_summary_env1.csv
best_model_rollout_summary_env5.csv
```

### 20.4 Current plot outputs

Training curves include:

```text
plot/report/episode_return.png
plot/report/episode_length.png
plot/report/success_rate.png
plot/report/forward_progress.png
plot/report/residual_action_magnitude.png
plot/report/losses.png
```

Debug training plots include optimizer and diagnostic plots under:

```text
plot/debug/
```

Best-rollout plots include joint tracking plots and observation comparison plots.

The observation comparison plots show:

```text
Clean
Noisy
Filtered
```

for selected observation groups:

```text
observation_joint_angles.png
observation_joint_velocities.png
observation_base_attitude.png
observation_gripper_attitude.png
```

### 20.5 Video outputs

For a single final rollout environment, the video is:

```text
video/best_model_run.mp4
```

For multiple rollout environments, videos are suffixed:

```text
video/best_model_run_env1.mp4
video/best_model_run_env2.mp4
...
video/best_model_run_env5.mp4
```

### 20.6 Best model test outputs

Record mode writes to:

```text
results/PPO/with_robust/best_model_test/<model_id>/envX/
```

For example:

```text
results/PPO/with_robust/best_model_test/multi_env_with_noise/env5/
```

A summary over all tested environments is written to:

```text
results/PPO/with_robust/best_model_test/<model_id>/csv/report/summary.csv
```

## 21. Important CSV Contents

### 21.1 `episode_log.csv`

This file logs one row per training episode. It includes:

- episode return
- episode length
- success/fall/timeout
- final progress
- base and gripper pose metrics
- cycle and energy metrics
- reward-term totals
- residual-action statistics
- joint-limit clip fraction
- noise and filter diagnostics
- friction randomization flag

### 21.2 `update_log.csv`

This file logs one row per PPO update. It includes:

- actor loss
- critic loss
- total loss
- entropy estimate
- approximate KL
- clip fraction
- explained variance
- number of minibatch updates
- KL early-stop status

### 21.3 `eval_log.csv`

This file logs deterministic evaluation metrics. It includes:

- evaluation environments
- mean return
- success rate
- mean episode length
- mean progress
- mean posture metrics
- mean cycle and COT metrics
- whether the best model was updated

### 21.4 `best_model_rollout_joint_tracking_log*.csv`

This file logs detailed best-rollout time series:

- demo reference per joint
- residual action per joint
- applied reference per joint
- feedback joint angle
- joint velocity
- PyBullet torque estimate
- base pose
- gripper pose
- cycle metrics

### 21.5 `best_model_rollout_cycle_log*.csv`

This file logs cycle-level metrics:

- cycle duration
- forward distance
- forward speed
- absolute work
- signed work
- COT
- gripper pose at cycle start and end

### 21.6 `best_model_rollout_observation_filter_log*.csv`

This file logs clean/noisy/filtered observation values for report plots.

For each selected signal, it stores:

```text
signal_clean
signal_noisy
signal_filtered
```

This file is used to verify:

```text
Clean vs Noisy     → sensor noise is being added
Noisy vs Filtered  → PPO-side filtering is active
```

## 22. Notes and Current Limitations

1. The environment does not directly give ledge geometry or environment ID to the policy observation.
2. The policy does receive absolute base and gripper positions. Therefore, in fixed environments, the model may implicitly associate world position with common terrain features. This is not direct geometry leakage, but it is a form of state information that can help feedback adaptation.
3. The current motor command does not explicitly set PyBullet force or torque limits in `setJointMotorControlArray()`.
4. The values `5`, `10`, and `15` are residual angle limits in degrees, not torque limits.
5. The current COT metric uses absolute mechanical work as the main energy estimate.
6. The current result structure has `csv/report` and `csv/debug`, but no separate `csv/raw` folder.
7. Episode-level friction columns contain some legacy names and may be NaN. Rollout debug logs preserve the current friction parameters with dynamic `friction_<key>` fields.
8. Sensor noise is not clipped. It is pure Gaussian noise with the configured standard deviation.
9. The active observation filter is in PPO, not in the environment.
10. If all robustness switches are disabled, the robust code is close to the baseline, but it still includes the reference timing fix and newer logging structure.

## 23. Recommended Report Interpretation

For reporting robust PPO, the most important claims supported by the current implementation are:

1. PPO is trained as a residual controller around a reference gait.
2. Robust training uses domain randomization through friction variation and noisy sensor observations.
3. The policy does not receive direct ledge geometry information.
4. PPO applies low-pass filtering before observation normalization.
5. Observation comparison plots show clean, noisy, and filtered signals.
6. The main COT metric is computed using absolute mechanical work.
7. Multi-env training uses episode-level randomization across `env1` to `env4`.
8. `env5` is used as an unseen validation environment combining features from the training environments.