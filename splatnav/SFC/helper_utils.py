import numpy as np
import scipy
import torch
import open3d as o3d

from splatnav.polytopes.polytopes_utils import find_interior

def shrink_polytope(bs, box_bs, r):
    # NOTE: IMPORTANT! This function assumes that the hyperplane normals are already normalized!
    # As: M x 3
    # bs: M
 
    # this only shrinks the corridor, and does not pivot it
    bs = torch.cat([bs, box_bs])

    # find norm of each plane
    bs_new = bs - r

    return bs_new

def find_ellipsoid(line_segment, point_cloud):
    # line_segment: 2 x 3
    # point_cloud: M x 3

    # Frame1: Local frame with x-axis along the line segment, y-z are aligned with world
    # Frame2: After first iteration, x-y plane is aligned with p* and z is perpendicular to x-y plane (y-z are no longer aligned with frame1)

    # returns the ellipsoid dictionary {'R': R, 'S': S}, R is the rotation matrix from local frame to world frame. S is the scaling matrix.
    # So E = R @ S @ S.T @ R.T (this is flipped from the paper, but follows the Gaussian Splatting decomposition). 
    # For convenience, R is a matrix, but S is a vector that parametrized the diagonal scaling matrix.

    # find the ellipsoid center       
    p0, p1 = line_segment[0], line_segment[1]
    d = (p0 + p1) / 2
    # find length of segment
    direction = p1 - p0
    L = torch.norm(direction)

    # create initial rotation matrix
    world_x_axis = torch.tensor([1.0, 0.0, 0.0], dtype=direction.dtype, device=direction.device)
    R = rotation_matrix_from_vectors(direction, world_x_axis)

    # initialize scale as half the length of the line segment
    S_orig = torch.tensor([L/2, L/2, L/2], dtype=direction.dtype, device=direction.device)

    # transform points to local frame
    shifted_points = point_cloud - d[None, :]
    rotated_points = shifted_points @ R.T 
    scaled_points = rotated_points / S_orig
    S = S_orig / S_orig # scale to unity

    # grab points inside the spheroid 
    dists = torch.sum(scaled_points**2, axis=1) 
    inside = dists < 1.0 - 1e-6

    closest_point = None
    
    while inside.any():     
    
        # grab closest point
        closest_id = torch.argmin(dists[inside])
        scaled_points_inside = scaled_points[inside]
        closest_point = scaled_points_inside[closest_id]
        # find scale factor for y-z axes
        distance_yz = closest_point[1]**2 + closest_point[2]**2
        denominator = 1.0 - closest_point[0]**2
        scale_factor = torch.sqrt(distance_yz / denominator)
        # update scales
        S[1] = scale_factor
        S[2] = scale_factor
        
        # redefine sphere using new scales
        scaled_points_norm = scaled_points / S
        
        # find any points inside new sphere
        dists = torch.sum(scaled_points_norm**2, axis=1)
        inside = dists < 1.0 - 1e-6

    if closest_point is not None: 
        p_star_xy = closest_point
        # remove p* from list of points
        scaled_points = scaled_points[torch.norm(scaled_points - p_star_xy[None, :], dim=1) > 0]

        # project p* onto yz plane
        p_yz = p_star_xy - p_star_xy[0] * torch.tensor([1, 0, 0], dtype=torch.float32, device=p_star_xy.device)
        p_yz_norm = p_yz / torch.norm(p_yz)
        
        # define current y-axis
        y_axis_R = torch.tensor([0, 1, 0], dtype=torch.float32, device=p_star_xy.device)
        

        cos_theta = torch.dot(y_axis_R, p_yz_norm)
        sin_theta = torch.sqrt(1 - cos_theta**2)  
        # rotation from original line segment align to final 
        R_2_1 = torch.tensor([
            [1, 0, 0],
            [0, cos_theta, -sin_theta],
            [0, sin_theta, cos_theta]
        ], dtype=torch.float32, device=p_star_xy.device)

        scaled_points = scaled_points @ R_2_1   
        # get p_star in world frame
        p_star_xy = ((p_star_xy * S_orig) @ R) + d

    else:  # no collision found
        # set p_star on ellipsoid boundary
        p_star_xy = torch.tensor([0, 0, 0], dtype=torch.float32, device=d.device)
        R_2_1 = torch.eye(3, dtype=torch.float32, device=d.device)

    # prep variables for z-axis shrink
    S[2] = 1 # reset z-axis scale to x-axis scale
    S_temp = S.clone()
    scaled_points = (scaled_points) / S_temp
    
    dists = torch.sum(scaled_points**2, axis=1) 
    inside = dists < 1.0 - 1e-6
    closest_point = None # reset closest point

    while inside.any():

        # Grab closest point
        closest_id = torch.argmin(dists[inside])
        scaled_points_inside = scaled_points[inside]
        closest_point = scaled_points_inside[closest_id]

        # find scale factor for z-axis
        distance_z = closest_point[2]**2
        denominator = 1.0 - (closest_point[0]**2) - (closest_point[1]**2)
        scale_factor = torch.sqrt(distance_z / denominator)

        # update scales
        S[2] = scale_factor
        # update points inside
        scaled_points_norm = scaled_points / S
        dists = torch.sum(scaled_points_norm**2, axis=1)
        inside = dists < 1.0 - 1e-6

    if closest_point is not None: 
        p_star_z = closest_point
        # find p* in world frame
        p_star_z = ((p_star_z * S_temp * S_orig) @ R_2_1.T @ R) + d
    else:  # no collision found
        # set p_star on ellipsoid boundary
        p_star_z = torch.tensor([0, 0, 0], dtype=torch.float32, device=d.device)
    
    # return final values in world frame
    R_final = R.T @ R_2_1
    S_final = S * S_orig
    p_stars = torch.stack([p_star_xy, p_star_z], axis=0) 

    center = d  # in world frame 
    return R_final, S_final, center, p_stars

def find_polyhedron(point_cloud, midpoint, R, S):
    # point_cloud: M x 3
    # midpoint: 3
    
    # We basically want to compute the Mahalanobis distance of each point from the ellipsoid. And sort them.
    # If it is less than 1, it is inside the ellipsoid.

    # However, it is actually faster (when doing this iteratively) to transform the entire coordinate frame
    # into the local frame of the ellipsoid, scale it so that it becomes a unit sphere, and then do the checking. This is
    # without loss of generality because intersections are preserved under linear transformations.

    # Shift the coordinate system to the midpoint
    shifted_points = point_cloud - midpoint[None, :]    # M x 3
    rotated_points = shifted_points @ R    # M x 3
    scaled_points = rotated_points / S    # M x 3
    # NOTE: Do we need a sanity check to make sure the ellipsoid is collision-free?
    #assert (torch.linalg.norm(scaled_points, dim=-1) > 1 - 1e-3).all()
    As = []
    bs = []
    ps_star = []
    pivot_indices = []
    point_cloud_indices = torch.arange(len(scaled_points), device=scaled_points.device)

    while len(scaled_points) > 0:
        # Calculate the distance to the origin
        dists = torch.linalg.norm(scaled_points, dim=-1) # M
        # Find the closest distance and which point it corresponds to
        where_min = torch.argmin(dists)
        min_val = dists[where_min] # this is distance squared
        p_star = scaled_points[where_min]
        p_star_norm = p_star / torch.linalg.norm(p_star)

        # After finding the closest point, we compute the tangent hyperplane and delete those points that are outside this hyperplane
        # For spheres it is easy, the normal is just the point, and the intercept is the squared distance.
        # convert p_star to world frame and set half-space constraints
        As.append(p_star)
        bs.append(min_val**2)
        ps_star.append(p_star)
        pivot_indices.append(point_cloud_indices[where_min])

        # We now find and delete all points that are on or outside this hyperplane
        mask = torch.sqrt(torch.sum(scaled_points * p_star, dim=-1)) < min_val  # M_i, true if on the right side of the hyperplane
        scaled_points = scaled_points[mask] 
        point_cloud_indices = point_cloud_indices[mask]

    # If we only performed one iteration, we can just return the hyperplane
    if len(As) == 1:
        As = As[0][None]        # 1 x 3
        bs = bs[0].reshape(1,)    # 1
        ps_star = ps_star[0].reshape(1, -1)
        pivot_indices = pivot_indices[0].reshape(1,)

    elif len(As) > 1:
        # otherwise we stack them
        As = torch.stack(As, dim=0)
        bs = torch.stack(bs)
        ps_star = torch.stack(ps_star, dim=0)
        pivot_indices = torch.stack(pivot_indices)

    else:
        raise ValueError("No points in the point cloud")

    # Now we need to transform the hyperplanes back to the world frame
    As = As * (1. / S)[None, :] @ R.T    # M x 3
    bs = bs + torch.sum(midpoint[None, :] * As, dim=-1)    # M

    return As, bs, ps_star, pivot_indices

def save_polytope(polytopes, save_path):
    # polytopes: list of tuples (A, b)
    # save_path: str

    # Initialize mesh object
    mesh = o3d.geometry.TriangleMesh()

    for (A, b) in polytopes:
        # Transfer all tensors to numpy
        A = A.cpu().numpy()
        b = b.cpu().numpy()

        pt = find_interior(A, b)

        halfspaces = np.concatenate([A, -b[..., None]], axis=-1)
        hs = scipy.spatial.HalfspaceIntersection(halfspaces, pt, incremental=False, qhull_options=None)
        qhull_pts = hs.intersections

        pcd_object = o3d.geometry.PointCloud()
        pcd_object.points = o3d.utility.Vector3dVector(qhull_pts)
        bb_mesh, qhull_indices = pcd_object.compute_convex_hull()
        mesh += bb_mesh
    
    success = o3d.io.write_triangle_mesh(save_path, mesh, print_progress=True)

    return success

def rotation_matrix_from_vectors(vec1, vec2):
    # Reference: https://stackoverflow.com/questions/45142959/calculate-rotation-matrix-to-align-two-vectors-in-3d-space
    dim = vec1.shape[0]

    if dim == 2:
        # 2D rotation matrix
        theta = torch.atan2(vec2[1], vec2[0]) - torch.atan2(vec1[1], vec1[0])
        rotation_matrix = torch.tensor([[torch.cos(theta), -torch.sin(theta)], [torch.sin(theta), torch.cos(theta)]], device=vec1.device)
        return rotation_matrix
    a, b = (vec1 / torch.norm(vec1)).reshape(3), (vec2 / torch.norm(vec2)).reshape(3)
    v = torch.cross(a, b)
    c = torch.dot(a, b)
    s = torch.norm(v)
    # if parallel vectors
    if s == 0:
        return torch.eye(3, device=vec1.device) if c > 0 else torch.eye(3, device=vec1.device) * -1  
    
    kmat = torch.tensor([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], device=vec1.device)
    rotation_matrix = torch.eye(3, device=vec1.device) + kmat + kmat.mm(kmat) * ((1 - c) / (s ** 2))
    
    return rotation_matrix


def find_closest_hyperplane(pivots, As, point, rad):
    # Tries to find the closest hyperplane to the current one that is anchored at the pivot
    # and is some R away from the point. 
    # In fact, it is equivalent to finding the hyperplane that is normal to the sphere of radius R
    # centered about the pivot point and passes through our desired point.
    # pivot: 3
    # As: M x 3

    # As is typically outward facing (facing away from the line segment). For this algorithm,
    # we've assumed it is inward facing. We will flip it back at the end.
    As = -As

    # We need to make sure that the polytope constraints are normalized. We also want it
    # to have length R.
    As = rad * As / torch.linalg.norm(As, dim=-1, keepdim=True)    # M x 3

    # Preliminary variables
    delta = point[None, :] - pivots    # M x 3
    delta_norm = torch.linalg.norm(delta, dim=-1)    # M
    a0 = As
    a0_delta = torch.sum( a0 * delta, dim=-1)    # M

    # Coefficients of the quadratic equation
    coeff1 = (delta_norm**2 - delta_norm**4 / (rad**2) ) / 4      # M
    coeff2 = - a0_delta * (1 - ( delta_norm / rad ) **2  ) 
    coeff3 = torch.sum( As**2, dim=-1 ) - (a0_delta / rad )**2

    # Solve the quadratic equation
    # first find if it is even solvable
    discriminant = coeff2**2 - 4 * coeff1 * coeff3

    # If the discriminant is negative, there are no solutions
    if torch.any(discriminant < 0):
        return None, None, None
    
    # Otherwise, we find the two solutions
    l1_1 = (-coeff2 + torch.sqrt(discriminant)) / (2 * coeff1)      # M
    l1_2 = (-coeff2 - torch.sqrt(discriminant)) / (2 * coeff1)      # M

    l2_1 = (l1_1 * delta_norm**2 / 2 - a0_delta) / rad**2 - 1
    l2_2 = (l1_2 * delta_norm**2 / 2 - a0_delta) / rad**2 - 1

    # We now compute the closest hyperplane candidates
    a_1 = ( As - l1_1 * delta / 2 ) / (1 + l2_1)        # M x 3
    a_2 = ( As - l1_2 * delta / 2 ) / (1 + l2_2)        # M x 3

    # Find the closest hyperplane by plugging back into objctive
    obj1 = torch.sum( (a_1 - As)**2, dim=-1)
    obj2 = torch.sum( (a_2 - As)**2, dim=-1)

    objs = torch.stack([obj1, obj2], dim=-1)    # M x 2

    new_pivot1 = a_1 + pivots
    new_pivot2 = a_2 + pivots

    new_pivots = torch.stack([new_pivot1, new_pivot2], dim=1)    # M x 2 x 3

    # Recall that we flipped the As to point inward, so we need to flip it back
    closest_A = -torch.stack([a_1, a_2], dim=-2) # M x 2 x 3
    closest_b = torch.sum(closest_A * new_pivots, dim=-1)    # M x 2

    return closest_A, closest_b, objs

