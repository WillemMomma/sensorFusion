import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "/home/willemmomma/thesis/catkin_ws/src/iiwa-ros-imitation-learning/cor_tud_controllers/python")

from sensor_fusion.unscented_kalman_filter import UnscentedKalmanFilter

# True dynamics
def true_dynamics(state, dt):
    x, v = state
    x_new = x + v * dt
    return np.array([x_new, v])

# Measurement function (position only, with noise)
def measure(state, noise_std):
    x, _ = state
    measurement = x + np.random.normal(0, noise_std)
    return measurement

# Parameters
dt = 1.0  # time step
process_noise_std = 0.1
measurement_noise_std = 1.0
initial_state = np.array([0, 1])  # Starting at x=0 with velocity=1

# UKF initialization (simplified for this example)
ukf = UnscentedKalmanFilter(
    state_dim=2, 
    meas_dim=1, 
    process_noise=np.eye(2) * process_noise_std**2,
    measurement_noise=np.array([[measurement_noise_std**2]])
)

true_states = [initial_state]
measurements = []
estimates = []

for _ in range(25):
    # Propagate true state
    true_state = true_dynamics(true_states[-1], dt)
    true_states.append(true_state)

    # Generate measurement
    measurement = measure(true_state, measurement_noise_std)
    measurements.append(measurement)

    # UKF prediction and update
    ukf.predict(lambda s: true_dynamics(s, dt))
    ukf.update(np.array([measurement]), lambda s: np.array([s[0]]))
    estimates.append(ukf.state.copy())

# Plotting results
true_states = np.array(true_states)
estimates = np.array(estimates)
measurements = np.array(measurements)

plt.figure(figsize=(12, 6))
plt.plot(true_states[:, 0], label='True Position')
plt.scatter(range(25), measurements, color='red', label='Measurements')
plt.plot(estimates[:, 0], label='UKF Estimate')
plt.xlabel('Time Step')
plt.ylabel('Position')
plt.legend()
plt.show()
