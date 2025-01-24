import numpy as np

class KalmanFilter2D:
    """
    Simple linear Kalman Filter for 2D position tracking (x, y).
    State: x_k = [x, y]
    Measurement: z_k = [x, y] + measurement noise
    """
    def __init__(self, dt=1.0, q_scale=1e-4, r_scale=0.1):
        # State transition matrix (assuming x and y are constant)
        self.F = np.eye(2)
        self.dt = dt
        
        # Measurement matrix (z = H x + v) -> H = I_2
        self.H = np.eye(2)
        
        # Initial state and covariance
        self.x_est = np.array([0.0, 0.0])
        self.P_est = np.eye(2) * 100.0
        
        # Q: system noise covariance (can be low if source is stationary)
        self.Q = np.eye(2) * q_scale
        
        # R: measurement noise covariance (measurement error)
        self.R = np.eye(2) * r_scale

    def predict(self):
        # Prior state estimate (x_{k|k-1})
        self.x_est = self.F @ self.x_est
        # Prior covariance estimate (P_{k|k-1})
        self.P_est = self.F @ self.P_est @ self.F.T + self.Q

    def update(self, z_meas):
        # Innovation
        y_k = z_meas - self.H @ self.x_est
        # Innovation covariance
        S_k = self.H @ self.P_est @ self.H.T + self.R
        # Kalman gain
        K_k = self.P_est @ self.H.T @ np.linalg.inv(S_k)
        # State update
        self.x_est = self.x_est + K_k @ y_k
        
        # Covariance update
        I = np.eye(2)
        self.P_est = (I - K_k @ self.H) @ self.P_est