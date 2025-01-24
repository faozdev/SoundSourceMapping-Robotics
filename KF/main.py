import numpy as np
import matplotlib.pyplot as plt
import sys
import os

this_file_path = os.path.abspath(__file__)           # .../MyProject/KF/main.py
ekf_dir = os.path.dirname(this_file_path)            # .../MyProject/KF
myproject_dir = os.path.dirname(ekf_dir)             # .../MyProject
sys.path.append(myproject_dir)


from geometry_utils import generate_random_points_in_polygon
from geometry_utils import is_point_in_polygon
from array_utils import generate_circular_array, true_angle, get_music_peak_KL
from music_utils import measure_covmat
from position_estimation import estimate_position_from_angles
from kalman_filter import KalmanFilter2D

def main():
    np.random.seed(42)
    
    # -------------------------------------------------------------
    # 1) Room polygon
    # -------------------------------------------------------------
    room_polygon = np.array([
        [-15, -15],
        [ 15, -15],
        [ 15,   0],
        [ 10,   0],
        [ 10,  15],
        [-15,  15],
        [-15, -15]
    ])
    
    # Source position (single point)
    src_position = generate_random_points_in_polygon(room_polygon, 1)[0]
    print("True Source Position:", src_position)
    
    # -------------------------------------------------------------
    # 2) Array centers
    # -------------------------------------------------------------
    P1 = np.array([  0.0,   0.0])
    P2 = np.array([-10.0,  -6.0])
    P3 = np.array([  8.0,  10.0])
    array_centers = np.vstack([P1, P2, P3])
    
    # Generate circular arrays
    N = 16
    array_radius = 1.0
    array_2D_1 = generate_circular_array(P1, N, array_radius)
    array_2D_2 = generate_circular_array(P2, N, array_radius)
    array_2D_3 = generate_circular_array(P3, N, array_radius)
    
    # True angles
    theta1_true = true_angle(src_position, P1)
    theta2_true = true_angle(src_position, P2)
    theta3_true = true_angle(src_position, P3)
    print(f"True Angles: {theta1_true:.3f}, {theta2_true:.3f}, {theta3_true:.3f}")
    
    # Random complex amplitude
    alpha_true = np.sqrt(0.5) * (np.random.randn() + 1j*np.random.randn())
    
    # -------------------------------------------------------------
    # 3) Kalman Filter
    # -------------------------------------------------------------
    kf = KalmanFilter2D(dt=1.0, q_scale=1e-4, r_scale=0.1)
    # Initial estimate (optional)
    kf.x_est = np.array([5.0, -10.0])
    
    # -------------------------------------------------------------
    # 4) Time loop
    # -------------------------------------------------------------
    Angles = np.linspace(-np.pi, np.pi, 360)
    num_steps = 20
    est_history = []
    snr = 5.0
    
    for t in range(num_steps):
        # CovMat measurements
        CovMat1 = measure_covmat(array_2D_1, theta1_true, alpha_true, snr=snr, num_snapshot=200)
        CovMat2 = measure_covmat(array_2D_2, theta2_true, alpha_true, snr=snr, num_snapshot=200)
        CovMat3 = measure_covmat(array_2D_3, theta3_true, alpha_true, snr=snr, num_snapshot=200)
        
        # MUSIC angle estimation
        doa1_est = get_music_peak_KL(CovMat1, array_2D_1, Angles, L=1)
        doa2_est = get_music_peak_KL(CovMat2, array_2D_2, Angles, L=1)
        doa3_est = get_music_peak_KL(CovMat3, array_2D_3, Angles, L=1)
        
        # Get (x,y) measurement from 3 angles
        doa_array = np.array([doa1_est, doa2_est, doa3_est])
        x_meas, y_meas = estimate_position_from_angles(array_centers, doa_array)
        
        # Kalman: predict + update
        kf.predict()
        kf.update(np.array([x_meas, y_meas]))
        
        est_history.append(kf.x_est.copy())
        
        print(f"Iter={t}, Measured (x,y)=({x_meas:.2f}, {y_meas:.2f}) -> "
              f"KF=({kf.x_est[0]:.2f}, {kf.x_est[1]:.2f})")
    
    est_history = np.array(est_history)
    
    # -------------------------------------------------------------
    # 5) Plot results
    # -------------------------------------------------------------
    plt.figure(figsize=(8, 6))
    plt.plot(room_polygon[:, 0], room_polygon[:, 1], 'k-')
    plt.fill(room_polygon[:, 0], room_polygon[:, 1], facecolor='none', 
             edgecolor='k', alpha=0.3, label='Room Boundaries')
    
    # Array centers
    plt.plot(array_centers[:, 0], array_centers[:, 1], 'kx', markersize=10, 
             label='Array Centers')
    
    # True source
    plt.plot(src_position[0], src_position[1], 'r*', markersize=12, 
             label='True Source')
    
    # KF estimation path
    plt.plot(est_history[:, 0], est_history[:, 1], 'bo--', 
             label='KF Estimate')
    
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.title("Multiple Arrays + MUSIC -> (x,y) Measurement + Linear KF")
    plt.show()

if __name__ == "__main__":
    main()