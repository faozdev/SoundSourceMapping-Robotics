import numpy as np

def is_point_in_polygon(point, polygon):
    """
    Check if a point is inside a polygon using the ray-casting method.
    """
    x, y = point
    inside = False
    n = len(polygon)
    
    for i in range(n - 1):
        x1, y1 = polygon[i]
        x2, y2 = polygon[i + 1]
        
        if ((y1 > y) != (y2 > y)):
            x_intersect = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x_intersect > x:
                inside = not inside
    return inside

def generate_random_points_in_polygon(poly, L):
    """
    Generate L random points inside the given polygon.
    """
    minx = np.min(poly[:,0])
    maxx = np.max(poly[:,0])
    miny = np.min(poly[:,1])
    maxy = np.max(poly[:,1])
    
    points = []
    while len(points) < L:
        xr = np.random.uniform(minx, maxx)
        yr = np.random.uniform(miny, maxy)
        if is_point_in_polygon((xr, yr), poly):
            points.append([xr, yr])
    return np.array(points)

def line_intersection_2d(p0, theta0, p1, theta1):
    """
    Find the 2D intersection point of two lines defined by:
    
      line0: p0 + t*v0,  where v0 = [cos(theta0), sin(theta0)]
      line1: p1 + s*v1,  where v1 = [cos(theta1), sin(theta1)]
    
    Returns the intersection point (x, y), or None if parallel.
    
    Parameters:
      p0: (x0, y0) center of array 1
      theta0: angle from measurement 1
      p1: (x1, y1) center of array 2
      theta1: angle from measurement 2
    """
    v0 = np.array([np.cos(theta0), np.sin(theta0)])
    v1 = np.array([np.cos(theta1), np.sin(theta1)])
    
    A = np.array([[v0[0], -v1[0]],
                  [v0[1], -v1[1]]], dtype=float)
    b = np.array([p1[0] - p0[0],
                  p1[1] - p0[1]], dtype=float)
    
    detA = np.linalg.det(A)
    if abs(detA) < 1e-10:
        return None 
    
    ts = np.linalg.inv(A) @ b
    t = ts[0]
    # s = ts[1]  
    
    intersec = p0 + t * v0
    return intersec
