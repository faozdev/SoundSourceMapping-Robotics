import numpy as np

def estimate_position_from_angles(positions, angles):
    """
    positions: shape (M,2) -> Centers of the arrays
    angles:    shape (M,)  -> Measured angle from each array (radians)
    
    Calculates least squares intersection point using "Lines" approach.
    
    line_i:  P_i + t * [cos(theta_i), sin(theta_i)]
    Objective: ArgMin_{x,y} sum_i( distance( (x,y), line_i )^2 )
    
    Using analytical method:
    line_i normal form: 
        n_i = [ -sin(theta_i), cos(theta_i) ]  
        For any point on line_i: 
            (X - p_ix)*n_ix + (Y - p_iy)*n_iy = 0
        Which means n_i · (X - p_i) = 0 
        => n_i · X = n_i · p_i
    
    Then solve using least squares:
      A = n,  b = c,  
      A * X = b  => (A^T A) X = A^T b
      X = pinv(A^T A) A^T b
    """
    M = len(angles)
    n = np.zeros((M, 2))
    c = np.zeros(M)
    
    for i in range(M):
        th = angles[i]
        # Normal vector components
        nx = -np.sin(th)
        ny =  np.cos(th)
        pix, piy = positions[i]
        # c_i = n_i · p_i
        c[i] = nx * pix + ny * piy
        n[i, 0] = nx
        n[i, 1] = ny
    
    A = n
    b = c
    # Solve X = inv(A^T A) A^T b
    ATA = A.T @ A
    ATb = A.T @ b
    X = np.linalg.pinv(ATA) @ ATb  # pseudo-inverse
    return X[0], X[1]