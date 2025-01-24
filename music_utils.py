import numpy as np
import scipy.linalg as LA
import scipy.signal as ss
from array_utils import array_response_vector_circular

def music(CovMat, L, N, array_2D, Angles):
    """
    Compute the MUSIC pseudospectrum for a given covariance matrix.
    
    Parameters:
      CovMat: NxN covariance matrix
      L: number of sources
      N: number of elements in the array
      array_2D: positions of array elements
      Angles: array of angles to be scanned
    
    Returns:
      peaks: indices of found peaks
      pspectrum: pseudospectrum values at each angle
    """
    # Eigenvalue decomposition
    _, V = LA.eig(CovMat)
    Qn = V[:, L:N]  # Noise subspace 
    
    pspectrum = np.zeros_like(Angles, dtype=float)
    for i, angle in enumerate(Angles):
        av = array_response_vector_circular(array_2D, angle)
        pspectrum[i] = 1.0 / LA.norm(Qn.conj().T @ av)
    
    # convert to dB scale
    psindB = np.log10(10 * pspectrum / pspectrum.min())
    # find peaks
    peaks, _ = ss.find_peaks(psindB, distance=1.5, height=1.35)
    
    return peaks, pspectrum

def get_music_peaks(CovMat, L, N, array_2D, Angles):
    """
    A convenience function to extract the sorted DOA peaks from the MUSIC algorithm.
    """
    pidx, pspectrum = music(CovMat, L, N, array_2D, Angles)
    doa_candidates = Angles[pidx]
    # sort and pick the first L peaks
    doa_sorted = np.sort(doa_candidates)[:L]
    return doa_sorted, pidx, pspectrum

def measure_covmat(array_2D, theta_source, alpha_source, snr, num_snapshot=200):
    """
    It produces a covariance matrix by adding Gaussian noise (num_snapshot count) to a 
    signal measured from a single source at a given angle and amplitude.
    """
    N = array_2D.shape[0]
    H = np.zeros((N, num_snapshot), dtype=complex)
    
    for it in range(num_snapshot):
        # Rastgele bir faz
        pha = np.exp(1j * 2*np.pi * np.random.rand())
        # Kaynak sinyali
        htmp = pha * alpha_source * array_response_vector_circular(array_2D, theta_source)
        # Gauss gürültü ekle
        noise_scale = np.sqrt(0.5 / snr)
        htmp += noise_scale * (np.random.randn(N) + 1j*np.random.randn(N))
        H[:, it] = htmp
    
    CovMat = H @ H.conj().T
    return CovMat
