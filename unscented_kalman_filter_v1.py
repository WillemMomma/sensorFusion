import numpy as np

class UnscentedKalmanFilter:
    def __init__(self, state_dim, meas_dim, process_noise, measurement_noise):
        self.state_dim = state_dim
        self.meas_dim = meas_dim

        # State vector and covariance matrix
        self.state = np.zeros(state_dim)
        self.covariance = np.eye(state_dim)

        # Process and measurement noise covariance matrices
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

        # UKF parameters
        self.alpha = 0.001
        self.beta = 2
        self.kappa = 0

        # Compute lambda
        self.lambda_ = self.alpha**2 * (self.state_dim + self.kappa) - self.state_dim

    def generate_sigma_points(self):
        sigma_points = np.zeros((2 * self.state_dim + 1, self.state_dim))
        U = np.linalg.cholesky((self.state_dim + self.lambda_) * self.covariance + 1e-3 * np.eye(self.state_dim))

        sigma_points[0] = self.state
        for i in range(self.state_dim):
            sigma_points[i + 1] = self.state + U[:, i]
            sigma_points[self.state_dim + i + 1] = self.state - U[:, i]

        return sigma_points


    def predict(self, process_model):
        sigma_points = self.generate_sigma_points()

        # Predicted state and covariance
        predicted_state = np.zeros(self.state_dim)
        predicted_covariance = np.zeros((self.state_dim, self.state_dim))

        # Weights for mean and covariance
        w_m = np.full(2 * self.state_dim + 1, 1 / (2 * (self.state_dim + self.lambda_)))
        w_c = w_m.copy()
        w_m[0] = self.lambda_ / (self.state_dim + self.lambda_)
        w_c[0] = w_m[0] + (1 - self.alpha**2 + self.beta)

        # Propagate sigma points through the process model
        for i, point in enumerate(sigma_points):
            sigma_points[i] = process_model(point)

        # Calculate predicted state and covariance
        for i, point in enumerate(sigma_points):
            predicted_state += w_m[i] * point
            diff = point - predicted_state
            predicted_covariance += w_c[i] * np.outer(diff, diff)

        predicted_covariance += self.process_noise
        if not np.all(np.isfinite(predicted_covariance)):
            raise ValueError("Non-finite values encountered in the predicted covariance.")

        self.state = predicted_state
        self.covariance = predicted_covariance

    def update(self, measurement, measurement_model):
        sigma_points = self.generate_sigma_points()

        # Transform sigma points into measurement space
        transformed_sigma_points = np.array([measurement_model(point) for point in sigma_points])

        # Predicted measurement and covariance
        predicted_measurement = np.zeros(self.meas_dim)
        predicted_meas_covariance = np.zeros((self.meas_dim, self.meas_dim))
        cross_covariance = np.zeros((self.state_dim, self.meas_dim))

        # Weights for mean and covariance
        w_m = np.full(2 * self.state_dim + 1, 1 / (2 * (self.state_dim + self.lambda_)))
        w_c = w_m.copy()
        w_m[0] = self.lambda_ / (self.state_dim + self.lambda_)
        w_c[0] = w_m[0] + (1 - self.alpha**2 + self.beta)

        # Calculate predicted measurement and covariance
        for i, point in enumerate(transformed_sigma_points):
            predicted_measurement += w_m[i] * point
            diff_meas = point - predicted_measurement
            predicted_meas_covariance += w_c[i] * np.outer(diff_meas, diff_meas)

        predicted_meas_covariance += self.measurement_noise
        if not np.all(np.isfinite(predicted_meas_covariance)):
            raise ValueError("Non-finite values encountered in the predicted measurement covariance.")

        # Calculate cross covariance
        for i in range(2 * self.state_dim + 1):
            diff_state = sigma_points[i] - self.state
            diff_meas = transformed_sigma_points[i] - predicted_measurement
            cross_covariance += w_c[i] * np.outer(diff_state, diff_meas)

        # Kalman gain
        K = np.dot(cross_covariance, np.linalg.inv(predicted_meas_covariance))

        # Update state and covariance
        self.state += np.dot(K, (measurement - predicted_measurement))
        self.covariance -= np.dot(K, np.dot(predicted_meas_covariance, K.T))
