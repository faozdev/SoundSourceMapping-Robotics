import numpy as np
import matplotlib.pyplot as plt
import sys
import os

this_file_path = os.path.abspath(__file__)           # .../MyProject/EKF/main.py
ekf_dir = os.path.dirname(this_file_path)            # .../MyProject/EKF
myproject_dir = os.path.dirname(ekf_dir)             # .../MyProject
sys.path.append(myproject_dir)

from geometry_utils import generate_random_points_in_polygon
from geometry_utils import is_point_in_polygon
from array_utils import generate_circular_array, get_music_peak
from ekf_utils import ekf_update, wrap_angle
from music_utils import measure_covmat
import array_utils 

# -------------------------------------------------------------
# 1) Simulation Parameters
# -------------------------------------------------------------

def main():
    # Fix randomness
    np.random.seed(42)

    # Room polygon
    room_polygon = np.array([
        [-15, -15],
        [ 15, -15],
        [ 15,   0],
        [ 10,   0],
        [ 10,  15],
        [-15,  15],
        [-15, -15]
    ])

    # Single source position
    src_position = generate_random_points_in_polygon(room_polygon, 1)[0]
    snr = 5.0

    # Centers of two circular arrays
    P1 = np.array([ 0.0,  0.0])
    P2 = np.array([-10.0, -6.0])

    # Array size and radius
    N = 16
    array_radius = 1.0

    # Circular array coordinates
    array_2D_1 = generate_circular_array(P1, N, array_radius)
    array_2D_2 = generate_circular_array(P2, N, array_radius)

    # True angles
    theta1_true = np.arctan2(src_position[1] - P1[1], src_position[0] - P1[0])
    theta2_true = np.arctan2(src_position[1] - P2[1], src_position[0] - P2[0])
    alpha_true = np.sqrt(0.5) * (np.random.randn() + 1j * np.random.randn())

    print("True Source Position:", src_position)
    print(f"True Angles: theta1 = {theta1_true:.3f}, theta2 = {theta2_true:.3f}")

    # -------------------------------------------------------------
    # 2) EKF Initialization
    # -------------------------------------------------------------
    # Initialize x_est randomly within room or choose fixed position
    x_est = np.array([-5.0, 10.0])
    P_est = np.eye(2) * 100.0

    # Assuming fixed source: F = I, Q is small
    Q = np.eye(2) * 1e-4
    # Angle measurement noise:
    R = np.eye(2) * 1e-2

    # -------------------------------------------------------------
    # 3) Loop: Measurement + EKF
    # -------------------------------------------------------------
    Angles = np.linspace(-np.pi, np.pi, 360)
    num_time_steps = 30
    est_history = []

    for k in range(num_time_steps):
        # Measure CovMat (simulated)
        CovMat1 = measure_covmat(array_2D_1, theta1_true, alpha_true, snr, num_snapshot=200)
        CovMat2 = measure_covmat(array_2D_2, theta2_true, alpha_true, snr, num_snapshot=200)

        # DoA estimation with MUSIC
        doa1_est, _, _ = get_music_peak(CovMat1, array_2D_1, Angles)
        doa2_est, _, _ = get_music_peak(CovMat2, array_2D_2, Angles)

        z_k = np.array([doa1_est, doa2_est])  # Measurement vector

        # 3.1) Predict (x_est, P_est)
        # Source is fixed, so x_est doesn't change
        P_est = P_est + Q

        # 3.2) Update (EKF)
        x_est, P_est = ekf_update(x_est, P_est, z_k, P1, P2, R)

        est_history.append(x_est.copy())

        print(f"iter={k:2d} | Measured DoA=({doa1_est:.3f}, {doa2_est:.3f}) "
              f"-> EKF=({x_est[0]:.2f}, {x_est[1]:.2f})")

    est_history = np.array(est_history)

    # -------------------------------------------------------------
    # 4) Plot
    # -------------------------------------------------------------
    plt.figure(figsize=(8,6))
    plt.plot(room_polygon[:,0], room_polygon[:,1], 'k-')
    plt.fill(room_polygon[:,0], room_polygon[:,1], facecolor='none', edgecolor='k', alpha=0.3)

    plt.plot(src_position[0], src_position[1], 'r*', markersize=12, label='True Source')
    plt.plot(P1[0], P1[1], 'kx', markersize=10, label='Array1 Center')
    plt.plot(P2[0], P2[1], 'kx', markersize=10, label='Array2 Center')
    plt.plot(est_history[:,0], est_history[:,1], 'bo--', label='EKF Estimated Path')

    plt.title("Source Position Tracking with Kalman Filter (Single Source, 2 Measurements)")
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()