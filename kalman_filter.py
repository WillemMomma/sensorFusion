#!/usr/bin/python3

"""
Author: Willem Momma <w.j.momma@student.tudelft.nl, willemmomma@gmail.com>
"""

import numpy as np 
import matplotlib.pyplot as plt
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

    """Running the kalman filter on an example of 2 dimensional torque controlled robot
      robot with 1 joint and 2 links.  
    """
    if example == 1:    
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
    

if __name__ == '__main__':
    # This so we can test it in file and import the file without running the main function
    main()
