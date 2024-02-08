"""
Author: Willem Momma <w.j.momma@student.tudelft.nl, willemmomma@gmail.com>
"""

import numpy as np 

class KalmanFilter:
    """
    Class to setup a Kalman filter for sensor fusion.
    """
    def __init__(self):
        pass

    def predict(self):
        pass

    def update(self):
        pass


def main(): 

    """Running the kalman filter on an example of 2 dimensional torque controlled robot
      robot with 1 joint and 2 links.  
    """
    
    l = 0.5 # each link is 1 meter 

    x = np.zeros(2) # theta and theta_dot 
    z = np.zeros(2) # measurement vector 
    u = np.zeros(1) # joint torque  
    F = np.zeros(2,2) # state transition matrix
    w = np.zeros(2) # process noise vector
    v = np.zeros(2) # measurmement noise vector 
    P = np.zeros(2,2) # state covariance matrix 
    Q = np.zeros(2,2) # process noise covariance 
    R = np.zeros(2,2) # measurement noise covariance 
    H = np.zeros(2,2) # observation matrix 
    K = np.zeros(2,2) # Kalman gain 
    G = np.zeros(2,1) # Control matrix


    