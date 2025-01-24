import numpy as np
import scipy.linalg as LA

def generate_circular_array(center, N, radius):
    """
    Generate a 2D circular array (N elements) around center with given radius.
    Returns the array of (x, y) coordinates.
    """
    arr = []
    for i in range(N):
        angle = 2.0 * np.pi * i / N
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        arr.append([x, y])
    return np.array(arr)

def array_response_vector_circular(array_2D, theta):
    """
    Compute the array response (manifold) vector for a circular array in 2D.
    array_2D.shape[0] = number of elements in the array.
    """
    N = array_2D.shape[0]
    x = array_2D[:,0]
    y = array_2D[:,1]
    kx = np.cos(theta)
    ky = np.sin(theta)
    # Here, 2*pi is used as the wavenumber factor.
    v = np.exp(1j * 2*np.pi * (x*kx + y*ky))
    return v / np.sqrt(N)

def measure_covmat(array_2D, Thetas, Alphas, snr, num_snapshot=100):
    """
    Simulate array measurements and compute the covariance matrix.
    
    Parameters:
      array_2D: N x 2 array of element positions
      Thetas: angles of incoming signals
      Alphas: random complex amplitudes
      snr: signal-to-noise ratio (linear scale)
      num_snapshot: number of snapshots for covariance estimation
    """
    N = array_2D.shape[0]
    L = len(Thetas)
    H = np.zeros((N, num_snapshot), dtype=complex)
    for it in range(num_snapshot):
        htmp = np.zeros(N, dtype=complex)
        for i in range(L):
            # random phase
            pha = np.exp(1j * 2*np.pi * np.random.rand())
            htmp += pha * Alphas[i] * array_response_vector_circular(array_2D, Thetas[i])
        # add Gaussian noise
        htmp += np.sqrt(0.5/snr)*(np.random.randn(N) + 1j*np.random.randn(N))
        H[:, it] = htmp
    
    return H @ H.conj().T

def music(CovMat, L, N, array_2D, Angles):
    """
    CovMat: Covariance matrix
    L: number of sources
    N: number of array elements
    array_2D: (N x 2) sensor array
    Angles: 1D array of angles (radians) to scan over
    Returns the MUSIC pseudo-spectrum.
    """
    # Eigen decomposition of CovMat
    _, V = LA.eig(CovMat)
    # Noise subspace (columns for the smallest eigenvalues)
    Qn = V[:, L:N]

    pspectrum = np.zeros_like(Angles)
    for i, angle in enumerate(Angles):
        av = array_response_vector_circular(array_2D, angle)
        # 1 / || Qn^H * a(θ) ||
        pspectrum[i] = 1.0 / LA.norm(Qn.conj().T @ av)
    return pspectrum

def get_music_peak(CovMat, array_2D, Angles):
    """
    For a single source, find the angle where the MUSIC spectrum is maximized.
    Returns (doa_est, pspectrum, peak_index).
    """
    N = array_2D.shape[0]
    pspectrum = music(CovMat, L=1, N=N, array_2D=array_2D, Angles=Angles)
    # Convert to dB-like scale for easier peak detection
    psindB = np.log10(10 * pspectrum / pspectrum.min())
    peak_idx = np.argmax(psindB)
    doa_est = Angles[peak_idx]
    return doa_est, pspectrum, peak_idx

def true_angle(src, center):
    """
    Given the coordinates of a source (src) and center (center), returns an angle in [rad]. (atan2)
    """
    return np.arctan2(src[1] - center[1], src[0] - center[0])

def get_music_peak_KL(CovMat, array_2D, Angles, L=1):
    """
    MUSIC spektrumunda en büyük zirveyi (peak) bularak
    DoA (direction-of-arrival) açısını tahmin eder.
    """
    N = array_2D.shape[0]
    pspectrum = music(CovMat, L=L, N=N, array_2D=array_2D, Angles=Angles)
    
    # dB ölçeğine almak yerine basitçe zirveyi bulabiliriz;
    # istenirse log alınır vs.
    idx = np.argmax(pspectrum)
    doa_est = Angles[idx]
    return doa_est