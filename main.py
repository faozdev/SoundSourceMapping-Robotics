import numpy as np
import matplotlib.pyplot as plt

from geometry_utils import (
    generate_random_points_in_polygon, 
    line_intersection_2d
)
from array_utils import (
    generate_circular_array, 
    measure_covmat
)
from music_utils import (
    get_music_peaks
)

def main():
    # -------------------------
    # 1) Scenario Settings
    # -------------------------
    np.random.seed(6)
    L = 3           # Number of sources
    N = 16          # Number of array elements
    snr = 10        # SNR (linear scale)
    array_radius = 1.0
    
    # L-shaped room polygon
    room_polygon = np.array([
        [-15, -15],
        [ 15, -15],
        [ 15,   0],
        [ 10,   0],
        [ 10,  15],
        [-15,  15],
        [-15, -15]  
    ])
    
    # Generate L random source positions inside the polygon
    src_positions = generate_random_points_in_polygon(room_polygon, L)
    
    # Two array centers
    P1 = np.array([0.0, 0.0])     
    P2 = np.array([-10.0, -6.0]) 
    
    # Generate circular arrays
    array_2D_1 = generate_circular_array(P1, N, array_radius)
    array_2D_2 = generate_circular_array(P2, N, array_radius)
    
    # True DOAs (arctan2)
    Thetas_1 = np.arctan2(src_positions[:,1] - P1[1], src_positions[:,0] - P1[0])
    Thetas_2 = np.arctan2(src_positions[:,1] - P2[1], src_positions[:,0] - P2[0])
    
    # Random complex amplitudes
    Alphas = np.sqrt(0.5)*(np.random.randn(L) + 1j*np.random.randn(L))
    
    # -------------------------
    # 2) Covariance Measurements
    # -------------------------
    CovMat1 = measure_covmat(array_2D_1, Thetas_1, Alphas, snr)
    CovMat2 = measure_covmat(array_2D_2, Thetas_2, Alphas, snr)
    
    # -------------------------
    # 3) MUSIC-based DOA Estimation
    # -------------------------
    Angles = np.linspace(-np.pi, np.pi, 360)
    
    DoAs1, pidx1, pspectrum1 = get_music_peaks(CovMat1, L, N, array_2D_1, Angles)
    DoAs2, pidx2, pspectrum2 = get_music_peaks(CovMat2, L, N, array_2D_2, Angles)
    
    # -------------------------
    # 4) Position Estimation via Line Intersection
    # -------------------------
    estimated_positions = []
    for i in range(L):
        if i < len(DoAs1) and i < len(DoAs2):
            intersec = line_intersection_2d(P1, DoAs1[i], P2, DoAs2[i])
            if intersec is not None:
                estimated_positions.append(intersec)
    estimated_positions = np.array(estimated_positions)
    
    print("\n--- True Source Positions ---")
    for i, (xs, ys) in enumerate(src_positions):
        print(f"  Source {i+1}: ({xs:.2f}, {ys:.2f})")

    print("\n--- Estimated Source Positions (Line Intersection) ---")
    if len(estimated_positions) > 0:
        for i, (xe, ye) in enumerate(estimated_positions):
            print(f"  Estimate {i+1}: ({xe:.2f}, {ye:.2f})")
    else:
        print("  No intersection found.")
    
    # -------------------------
    # 5) Plotting
    # -------------------------

    # (a) MUSIC Spectra
    plt.figure(figsize=(14,5))
    
    # Measurement 1
    plt.subplot(121)
    psindB1 = np.log10(10 * pspectrum1 / pspectrum1.min())
    plt.plot(Angles, psindB1, 'b-', linewidth=2, label="Spectrum")
    for idx in pidx1:
        plt.plot(Angles[idx], psindB1[idx], 'ro', markersize=8, 
                 label="Peak" if idx == pidx1[0] else None)
    plt.title('MUSIC Spectrum - Measurement 1 (Center (0,0))')
    plt.xlabel('Angle (rad)', fontsize=14)
    plt.ylabel('Power (dB)', fontsize=14)
    plt.legend(fontsize=14)
    
    # Measurement 2
    plt.subplot(122)
    psindB2 = np.log10(10 * pspectrum2 / pspectrum2.min())
    plt.plot(Angles, psindB2, 'b-', linewidth=2, label="Spectrum")
    for idx in pidx2:
        plt.plot(Angles[idx], psindB2[idx], 'ro', markersize=8, 
                 label="Peak" if idx == pidx2[0] else None)
    plt.title('MUSIC Spectrum - Measurement 2 (Center (-10,-6))')
    plt.xlabel('Angle (rad)', fontsize=14)
    plt.ylabel('Power (dB)', fontsize=14)
    plt.legend(fontsize=14)
    
    plt.tight_layout()
    plt.show()
    
    # Room Layout and Positions
    plt.figure(figsize=(8,6))
    
    # Plot the room polygon
    plt.plot(room_polygon[:,0], room_polygon[:,1], 'k-', linewidth=2)
    plt.fill(room_polygon[:,0], room_polygon[:,1], facecolor='none', edgecolor='k', linewidth=2)
    plt.title('Microphone Arrays and Source Positioning')
    plt.axis('equal')
    plt.grid(True)

    # Mark array centers
    plt.plot(P1[0], P1[1], 'kx', markersize=12, linewidth=2, label='Center 1')
    plt.plot(P2[0], P2[1], 'kx', markersize=12, linewidth=2, label='Center 2')
    
    # Plot array elements
    plt.plot(array_2D_1[:,0], array_2D_1[:,1], 'bo-', linewidth=2, label='Array 1')
    plt.plot(array_2D_2[:,0], array_2D_2[:,1], 'mo-', linewidth=2, label='Array 2')
    
    # True source positions
    plt.plot(src_positions[:,0], src_positions[:,1], 'r*', markersize=14, label='True Sources')
    for (xs, ys) in src_positions:
        plt.text(xs, ys + 0.4, f"({xs:.2f}, {ys:.2f})", color='red')
    
    # Estimated positions
    if len(estimated_positions) > 0:
        plt.plot(estimated_positions[:,0], estimated_positions[:,1], 'gx', 
                 markersize=12, linewidth=2, label='Estimates (2 Measurements)')
        for (xe, ye) in estimated_positions:
            plt.text(xe, ye - 0.4, f"({xe:.2f}, {ye:.2f})", color='green')
    
    # Draw arrows from array centers to sources
    for px, py in [P1, P2]:
        for (xs, ys) in src_positions:
            dx = xs - px
            dy = ys - py
            plt.arrow(px, py, dx, dy, 
                      length_includes_head=True,
                      head_width=0.3, 
                      head_length=0.5,
                      linestyle='--', 
                      linewidth=1.5,
                      color='gray',
                      alpha=0.6)
    
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
