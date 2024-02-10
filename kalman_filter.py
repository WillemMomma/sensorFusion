#!/usr/bin/python3

"""
Author: Willem Momma <w.j.momma@student.tudelft.nl, willemmomma@gmail.com>
"""

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import scipy.linalg 
class KalmanFilter:
    """
    Class to setup a Kalman filter for sensor fusion.
    """
    def __init__(self, F, G, Q, H, R, P):
        """
        Initialize all matrices 
        """
        self.F = F # State Transition Matrix
        self.G = G # Control Matrix 
        self.Q = Q # Process Noise Covariance 
        self.H = H # Measurement Matrix 
        self.R = R # Measurment Noise Covariance 
        self.P = P # State Covariance Matrix 
        self.x = np.zeros((F.shape[0], 1))

    def predict(self, u):
        """ 
        Extrapolates the state and uncertainty from F, G, P and Q
        """
        self.x = np.dot(self.F, self.x) + np.dot(self.G, u) 
        self.P = np.dot(np.dot(self.F,self.P), self.F.T) + self.Q
        return self.x
    
    def set_measurement_noise_covariance(self, R):
        """
        Set the measurement noise covariance
        """
        self.R = R

    def update(self, z):
        """
        Correcting the measurement 
        """
        # Computing residual between actual measurement and predicted measurement
        y = z - np.dot(self.H, self.x)

        # Computing residual covariance 
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R

        # Computing Kalman Gain
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        # Update the estimate with measurement
        self.x = self.x + np.dot(K, y)

        # Update uncertainties, state covariance matrix 
        I = np.eye(self.F.shape[0])
        self.P = np.dot(I - np.dot(K, self.H), self.P)  

        # TODO For numerical stability, use Joseph form instead

def main(example = 1): 

    """
    Running the kalman filter on multiple examples
    Input: select example to run: 
    """
    if example == 1:    
        # Example single joint robot in 2d
        # Define system parameters
        delta_t = 0.1  # Time step
        I = 1.0  # Moment of inertia
        torque = 0.1  # Constant torque

        # Define matrices for the Kalman Filter
        F = np.array([[1, delta_t], [0, 1]])
        G = np.array([[0], [delta_t / I]])
        Q = np.diag([0.001, 0.001])  # Process noise covariance
        H = np.eye(2)  # Measurement matrix
        R = np.diag([0.01, 0.01])  # Measurement noise covariance
        P = np.eye(2)  # Initial state covariance

        # Initialize the Kalman Filter
        kf = KalmanFilter(F, G, Q, H, R, P)

        # Simulate the system
        time_steps = 100
        actual_positions = []
        measured_positions = []
        kalman_positions = []

        actual_position = 0
        actual_velocity = 0

        for t in range(time_steps):
            # Simulate actual system dynamics
            actual_velocity += (torque / I) * delta_t
            actual_position += actual_velocity * delta_t

            # Simulate measurement (actual position + noise)
            measured_position = actual_position + np.random.normal(0, 0.1)  # Add noise
            measured_velocity = actual_velocity + np.random.normal(0, 0.2)  # Add noise

            # Kalman Filter prediction and update
            kf.predict(np.array([[torque]]))
            kf.update(np.array([[measured_position], [measured_velocity]]))
            # Store values for plotting
            actual_positions.append(actual_position)
            measured_positions.append(measured_position)
            kalman_positions.append(kf.x[0, 0])

        # Plotting the results
        plt.figure(figsize=(10, 6))
        plt.plot(actual_positions, label='Actual Position')
        plt.plot(measured_positions, label='Measured Position')
        plt.plot(kalman_positions, label='Kalman Position')
        plt.xlabel('Time Step')
        plt.ylabel('Position')
        plt.title('Kalman Filter Performance')
        plt.legend()
        plt.show()
    
    if example == 3:
        # Load data
        kuka_tool_data = pd.read_csv('/home/willemmomma/thesis/catkin_ws/src/iiwa-ros-imitation-learning/cor_tud_controllers/python/data/rosbags/data/tool_positions.csv')
        optitrack_data = pd.read_csv('/home/willemmomma/thesis/catkin_ws/src/iiwa-ros-imitation-learning/cor_tud_controllers/python/data/rosbags/data/optitrack_positions.csv')

        # variances calculated from the kuka_tool_data and optitrack_data
        variance_kuka_x = kuka_tool_data['X'].var()
        variance_kuka_y = kuka_tool_data['Y'].var()
        variance_kuka_z = kuka_tool_data['Z'].var()

        variance_optitrack_x = optitrack_data['X'].var()
        variance_optitrack_y = optitrack_data['Y'].var()
        variance_optitrack_z = optitrack_data['Z'].var()       

        # Kalman Filter setup
        F = np.eye(3)
        H = np.eye(3)
        Q = np.diag([0.000001, 0.000001, 0.000001]) # Example values
        P = np.eye(3) * 1000
        x = np.zeros((3, 1))
        G = np.zeros_like(F) 
        R1 = np.diag([variance_kuka_x, variance_kuka_y, variance_kuka_z])*10000 
        R2 = np.diag([variance_optitrack_x, variance_optitrack_y, variance_optitrack_z]) 
        kf = KalmanFilter(F, np.zeros_like(F), Q, H, R2, P) 

        fused_estimates = []

        # Debugging: Print variances
        print("KUKA variances:", variance_kuka_x, variance_kuka_y, variance_kuka_z)
        print("Optitrack variances:", variance_optitrack_x, variance_optitrack_y, variance_optitrack_z)

        # Debugging: Check some sample data
        print("Sample KUKA data:", kuka_tool_data.iloc[0])
        print("Sample Optitrack data:", optitrack_data.iloc[0])


        # Process each data point
        for i in range(len(kuka_tool_data)):
            z1 = kuka_tool_data.iloc[i].values.reshape(-1, 1)
            z2 = optitrack_data.iloc[i].values.reshape(-1, 1)

            # Calculate the combined noise covariance matrix
            R_combined_inv = np.linalg.inv(R1) + np.linalg.inv(R2)
            R_combined = np.linalg.inv(R_combined_inv)

            # Calculate the combined measurement
            z_combined = R_combined @ (np.linalg.inv(R1) @ z1 + np.linalg.inv(R2) @ z2)

            # Update the Kalman Filter's measurement noise covariance
            kf.set_measurement_noise_covariance(R_combined)

            # Kalman Filter update
            kf.predict(u=np.zeros((3, 1))) # No control input
            kf.update(z_combined)
            # Store the updated state
            fused_estimates.append(kf.x.flatten())

        # Convert to DataFrame for easier processing
        fused_estimates_df = pd.DataFrame(fused_estimates, columns=['X', 'Y', 'Z'])

        # Plotting
        plt.figure(figsize=(12, 8))
        plt.plot(kuka_tool_data['X'], label='KUKA X', alpha=0.7)
        plt.plot(optitrack_data['X'], label='Optitrack X', alpha=0.7)
        plt.plot(fused_estimates_df['X'], label='Fused X', alpha=0.7)
        plt.xlabel('Time')
        plt.ylabel('Position')
        plt.title('Sensor Fusion - Position X Comparison')
        plt.legend()
        plt.show()

        # Print mean and variance of the fused data
        mean_fused = fused_estimates_df.mean()
        variance_fused = fused_estimates_df.var()

        print("Mean of Fused Data:", mean_fused)
        print("Variance of Fused Data:", variance_fused)

if __name__ == "__main__":
    main(3)