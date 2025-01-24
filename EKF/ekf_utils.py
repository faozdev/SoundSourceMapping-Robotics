import numpy as np

def h_measurement(x_state, p1, p2):
    """
    x_state = [x, y]
    p1, p2 = array centers
    Output = [theta1, theta2] where
      theta1 = atan2(y - p1y, x - p1x)
      theta2 = atan2(y - p2y, x - p2x)
    """
    x, y = x_state
    theta1 = np.arctan2(y - p1[1], x - p1[0])
    theta2 = np.arctan2(y - p2[1], x - p2[0])
    return np.array([theta1, theta2])

def jacobian_h(x_state, p1, p2):
    """
    Jacobian of h_measurement(x) with respect to x = [x, y].
    """
    x, y = x_state
    dx1 = x - p1[0]
    dy1 = y - p1[1]
    dx2 = x - p2[0]
    dy2 = y - p2[1]
    r1sq = dx1**2 + dy1**2
    r2sq = dx2**2 + dy2**2

    # Prevent division by zero
    eps = 1e-12
    if r1sq < eps:
        r1sq = eps
    if r2sq < eps:
        r2sq = eps

    H = np.array([
        [-dy1 / r1sq,  dx1 / r1sq],
        [-dy2 / r2sq,  dx2 / r2sq]
    ])
    return H

def wrap_angle(a):
    """
    Wrap angle a to the interval (-pi, pi).
    """
    return (a + np.pi) % (2*np.pi) - np.pi


def ekf_update(x_est, P_est, z_k, p1, p2, R):
    """
    Function that performs EKF update.
    - x_est, P_est: Current state and covariance
    - z_k: measurement (measured angles)
    - p1, p2: centers of sensor arrays
    - R: measurement noise covariance
    Returns: (x_est_new, P_est_new)
    """
    # h(x_est)
    h_x = h_measurement(x_est, p1, p2)
    
    y_k = z_k - h_x
    # Wrap angle
    y_k[0] = wrap_angle(y_k[0])
    y_k[1] = wrap_angle(y_k[1])

    # Jacobian
    H_k = jacobian_h(x_est, p1, p2)
    S_k = H_k @ P_est @ H_k.T + R
    K_k = P_est @ H_k.T @ np.linalg.inv(S_k)
    x_est_new = x_est + K_k @ y_k
    P_est_new = (np.eye(2) - K_k @ H_k) @ P_est
    return x_est_new, P_est_new
