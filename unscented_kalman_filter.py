"""
Author: Willem Momma <w.j.momma@student.tudelft.nl, willemmomma@gmail.com>
"""
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

class UnscentedKalmanFilter:
    """
    Class for the Unscented Kalman Filter 
    Derrived from The Unscented Kalman Filter for Nonlinear Estimation 
    by Eric A. Wan and Rudolph van der Merwe
    """
    def __init__(self, F, G, Q, P, H, R, x0, kappa, alpha, beta):
        """
        Initialize all matrices
        """
        self.F = F # State Transition Model
        self.G = G # Control Input Matrix
        self.Q = Q # Process Noise Covariance Matrix
        self.P = P # State Covariance Matrix 
        self.H = H # Measurement Matrix 
        self.R = R # Measurement Noise Matrix 
        self.x0 = x0 # Initial State 
        self.kappa = kappa # Kappa tuning parameter 
        self.alpha = alpha # Alpha tuning parameter
        self.beta = beta # Beta tuning parameter
        self.n = self.F.shape[0] # State dimension

        # Compute weights 
        self.compute_weights()

    def generate_sigma_points(self):
        """
        Part of the unscented transform, generates sigma points
        Sigma points are the mean and the points around the mean
        """
        lambda_ = self.alpha**2 * (self.n + self.kappa) - self.n
        sigma_points = np.zeros((2 * self.n + 1, self.n))
        sigma_points[0] = self.x0
        U = np.linalg.cholesky((self.n + lambda_) * self.P)

        for i in range(self.n):
            sigma_points[i + 1] = self.x0 + U[:, i]
            sigma_points[self.n + i + 1] = self.x0 - U[:, i]
        return sigma_points
    
    def compute_weights(self):
        """
        Compute weights for the unscented transform        
        """
        lambda_ = self.alpha**2 * (self.n + self.kappa) - self.n
        self.Wm = np.zeros(2 * self.n + 1)
        self.Wc = np.zeros(2 * self.n + 1)
        self.Wm[0] = lambda_ / (self.n + lambda_)
        self.Wc[0] = self.Wm[0] + (1 - self.alpha**2 + self.beta)
        for i in range(1, 2 * self.n + 1):
            self.Wm[i] = 1 / (2 * (self.n + lambda_))
            self.Wc[i] = 1 / (2 * (self.n + lambda_))
        
    def predict(self, state_transition_function, control_input=None):
        # Generate sigma points
        sigma_points = self.generate_sigma_points()
        print('Sigma Point: ', sigma_points)
        # Propagate sigma points through the state transition function
        propagated_sigma_points = np.zeros_like(sigma_points)
        for i, point in enumerate(sigma_points):
            if control_input is not None:
                propagated_sigma_points[i] = state_transition_function(point, control_input)
            else:
                propagated_sigma_points[i] = state_transition_function(point)
        print('Propagated Sigma Point: ', propagated_sigma_points)
        # Calculate predicted state mean (self.x_prior for clarity)
        self.x_prior = np.dot(self.Wm, propagated_sigma_points)

        # Calculate predicted state covariance (self.P_prior for clarity)
        self.P_prior = np.zeros_like(self.P)
        for i in range(2 * self.n + 1):
            y = propagated_sigma_points[i] - self.x_prior
            self.P_prior += self.Wc[i] * np.outer(y, y)
        self.P_prior += self.Q  # Add process noise covariance

        # After prediction, update the current state estimate and covariance
        self.x = self.x_prior
        self.P = self.P_prior

    def update(self, z):
        """
        Update state with a new measurement z.
        """
        # Predict measurement
        Z = np.array([self.H @ sigma_point for sigma_point in self.generate_sigma_points()])
        z_pred = np.dot(self.Wm, Z)
        
        # Measurement covariance
        S = np.zeros((self.H.shape[0], self.H.shape[0]))
        for i in range(2 * self.n + 1):
            z_diff = Z[i] - z_pred
            S += self.Wc[i] * np.outer(z_diff, z_diff)
        S += self.R  # Add measurement noise covariance

        # Cross covariance
        P_xz = np.zeros((self.n, self.H.shape[0]))
        for i in range(2 * self.n + 1):
            x_diff = self.generate_sigma_points()[i] - self.x
            z_diff = Z[i] - z_pred
            P_xz += self.Wc[i] * np.outer(x_diff, z_diff)

        # Kalman gain
        K = np.dot(P_xz, np.linalg.inv(S))

        # Update state estimate and covariance matrix
        z_diff = z - z_pred
        self.x += np.dot(K, z_diff)
        self.P -= np.dot(K, np.dot(S, K.T))

        return self.x, self.P

# Constants for the double pendulum
G = 9.8  # acceleration due to gravity, in m/s^2
L1 = 1.0  # length of pendulum 1 in m
L2 = 2.0  # length of pendulum 2 in m
M1 = 1.0  # mass of pendulum 1 in kg
M2 = 1.0  # mass of pendulum 2 in kg
t_stop = 2.5  # how many seconds to simulate
history_len = 500  # how many trajectory points to display

def derivs(t, state):
    dydx = np.zeros_like(state)
    delta = state[2] - state[0]
    den1 = (M1+M2) * L1 - M2 * L1 * np.cos(delta) * np.cos(delta)
    dydx[1] = ((M2 * L1 * state[1] ** 2 * np.sin(delta) * np.cos(delta)
                + M2 * G * np.sin(state[2]) * np.cos(delta)
                + M2 * L2 * state[3] ** 2 * np.sin(delta)
                - (M1+M2) * G * np.sin(state[0]))
               / den1)
    den2 = (L2/L1) * den1
    dydx[3] = ((- M2 * L2 * state[3] ** 2 * np.sin(delta) * np.cos(delta)
                + (M1+M2) * G * np.sin(state[0]) * np.cos(delta)
                - (M1+M2) * L1 * state[1] ** 2 * np.sin(delta)
                - (M1+M2) * G * np.sin(state[2]))
               / den2)
    dydx[0] = state[1]
    dydx[2] = state[3]
    return dydx

def generate_noisy_angle_measurements(angles, noise_level=0.01):
    noisy_measurements = angles + noise_level * np.random.randn(*angles.shape)
    return noisy_measurements

# Function to calculate end-effector position
def calculate_end_effector(theta1, theta2):
    x = L1 * np.sin(theta1) + L2 * np.sin(theta2)
    y = -L1 * np.cos(theta1) - L2 * np.cos(theta2)
    return x, y

# State Transition Function for the Unscented Kalman Filter
def state_transition_function(state, dt, _=None):
    t_span = [0, dt]
    sol = solve_ivp(derivs, t_span, state, method='RK45', t_eval=[dt])
    return sol.y.flatten()

# Main function to run the example
def main():
    # Initial state of the double pendulum
    initial_state = np.array([np.radians(120.0), 0.0, np.radians(-10.0), 0.0])
    dt = 0.01  # Time step
    
    # Time points to solve the ODE
    t = np.arange(0, t_stop, dt)
    
    # Solve the ODE for the true trajectory
    sol = solve_ivp(derivs, [t[0], t[-1]], initial_state, t_eval=t, method='RK45')
    true_trajectory = sol.y.T
    
    # Calculate end-effector positions for the true trajectory
    true_end_effector_positions = np.array([calculate_end_effector(theta1, theta2) 
                                            for theta1, theta2 in true_trajectory[:, [0, 2]]])
    
    # Generate noisy measurements of the joint angles
    true_angles = true_trajectory[:, [0, 2]]  # Extracting theta1 and theta2 from the true trajectory
    noisy_angle_measurements = generate_noisy_angle_measurements(true_angles, noise_level=0.01)

    
    # Initial state estimate (can be the same as the true initial state for this example)
    x0 = initial_state
    
    # Initial state covariance (reflects the initial uncertainty)
    P0 = np.diag([0.01, 0.01, 0.01, 0.01])
    
    # Process noise covariance (smaller values since we want to trust our model more)
    Q = np.diag([0.01, 0.01, 0.01, 0.01])
    
    # Measurement noise covariance (reduced to match the lower noise in measurements)
    R = np.diag([0.01, 0.01])
    
    # State transition matrix (identity if the state transition is modeled in the state_transition_function)
    F = np.eye(4)
    
    # Measurement matrix (maps the state to the measured variables)
    H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])  # Assuming we can only measure the angles directly
    
    # Initialize the Unscented Kalman Filter    
    ukf = UnscentedKalmanFilter(F=F, G=np.zeros((4, 4)), Q=Q, P=P0, H=H, R=R, x0=x0, kappa=0, alpha=1e-3, beta=2)

    # Placeholder for UKF estimates
    ukf_estimates = np.zeros((len(t), 4))

    for i, timestamp in enumerate(t):
        if i == 0:
            ukf_estimates[i] = x0
        else:
            ukf.predict(lambda state: state_transition_function(state, dt))
            ukf.update(noisy_angle_measurements[i])
            ukf_estimates[i] = ukf.x

    # Calculate end-effector positions for UKF estimated trajectory
    ukf_end_effector_positions = np.array([calculate_end_effector(theta1, theta2) for theta1, theta2 in ukf_estimates[:, [0, 2]]])

    # Plotting the results
    plt.figure(figsize=(10, 8))

    # Plot true end-effector path
    plt.plot(true_end_effector_positions[:, 0], true_end_effector_positions[:, 1], label='True End-Effector Path', color='blue')

    # Convert noisy angle measurements to end-effector positions for plotting
    noisy_end_effector_positions = np.array([calculate_end_effector(theta1, theta2) for theta1, theta2 in noisy_angle_measurements])
    plt.plot(noisy_end_effector_positions[:, 0], noisy_end_effector_positions[:, 1], label='Noisy Measured End-Effector Path', alpha=0.9, linestyle='dashed', color='orange')

    # Convert UKF estimates to end-effector positions for plotting
    ukf_end_effector_positions = np.array([calculate_end_effector(theta1, theta2) for theta1, theta2 in ukf_estimates[:, [0, 2]]])
    plt.plot(ukf_end_effector_positions[:, 0], ukf_end_effector_positions[:, 1], label='UKF Estimated End-Effector Path', alpha=0.9, linestyle='dotted', color='green')

    # Finalize the plot
    plt.legend()
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('Double Pendulum End-Effector Path Comparison')
    plt.axis('equal')
    plt.show()

if __name__ == '__main__':
    main()