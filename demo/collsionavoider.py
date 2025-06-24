import numpy as np
import open3d as o3d
import cvxpy as cp
from demo.data_utils import pose_to_fli_ocv
from threading import Lock

class CollisionAvoider:
    def __init__(self,
                robot_radius=0.3,
                search_radius=1.5,
                conf_thres=35,
                voxel_size=0.1,
                nb_neighbors=20,
                std_ratio=2.0,
                min_depth=0.1,
                max_depth=5.0,
                n_memory=10,
                violation_cost=1e5,
                fx=365.4914,
                fy=365.4914,
                cx=314.6048,
                cy=185.9464,
                hand_eye=np.identity(4)):

        self.pcloud = o3d.geometry.PointCloud()  # o3d point cloud
        self.memory = []
        self.lock = Lock()

        self.robot_radius = robot_radius
        self.search_radius = search_radius
        self.conf_thres = conf_thres
        self.voxel_size = voxel_size
        self.nb_neighbors = nb_neighbors
        self.std_ratio = std_ratio
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.n_memory = n_memory
        self.violation_cost = violation_cost

        self.fx, self.fy = fx, fy
        self.cx, self.cy = cx, cy
        self.hand_eye = hand_eye
        self.latest_A = None
        self.latest_b = None
        self.latest_pos = None

    def safeguard(self, position, candidate_point):
        points = np.asarray(self.pcloud.points)
        lpc = points[np.linalg.norm(position - points, axis=-1) < self.search_radius]
        if len(lpc) != 0:
            diff = position - lpc
            A = diff/np.linalg.norm(diff, axis=-1, keepdims=True)
            b = np.ones(len(A))*self.robot_radius + np.sum(A*lpc,axis=-1)
            x = cp.Variable(3)
            s = cp.Variable(len(lpc))
            cons = [A @ x  >= b - s,
                    s >= 0]
            obj = cp.Minimize(cp.quad_form(x-candidate_point, np.identity(3)) + self.violation_cost*cp.sum(s))
            prob = cp.Problem(obj, cons)
            prob.solve(solver="CLARABEL")
            wayp = x.value
            nudgedist = np.linalg.norm(wayp - candidate_point)
        else:
            wayp = candidate_point[:]
            nudgedist = 0.0
        return wayp, nudgedist
    

    def safeguard_min(self, position, candidate_point):
        points = np.asarray(self.pcloud.points)
        cand_dist = np.linalg.norm(position - candidate_point)
        diffs = position - points
        dists = np.linalg.norm(diffs, axis=-1)
        dist_index = dists < cand_dist + self.robot_radius
        if np.sum(dist_index) != 0:
            lpc = points[dist_index]
            dirs = diffs[dist_index]/dists[dist_index, None]
            adj_lpc = lpc + self.robot_radius*dirs
            A,b = self.minimal_polytope(position, adj_lpc)
            x = cp.Variable(3)
            s = cp.Variable(len(b))
            cons = [A @ x <= b + s,
                    s >= 0]
            obj = cp.Minimize(cp.quad_form(x-candidate_point, np.identity(3)) + self.violation_cost*cp.sum(s))
            prob = cp.Problem(obj, cons)
            prob.solve(solver="CLARABEL")
            wayp = x.value
            nudgedist = np.linalg.norm(wayp - candidate_point)
            self.latest_A = A
            self.latest_b = b
        else:
            wayp = candidate_point
            nudgedist = 0.0
            self.latest_A = None
            self.latest_b = None

        self.latest_pos = position
        return wayp, nudgedist

    def minimal_polytope(self, position, points):
        ppoints = points[:]
        essential_points = []
        eps = 1e-9
        As = []
        bs = []
        while len(ppoints) != 0:
            diffs = np.linalg.norm(ppoints - position, axis=-1)
            essential_points.append(ppoints[np.argmin(diffs)])
            a = (ppoints[np.argmin(diffs)] - position)/diffs[np.argmin(diffs)]
            b = a.T @ ppoints[np.argmin(diffs)]
            ppoints = ppoints[ppoints @ a < b - eps]
            As.append(a)
            bs.append(b)
           
        As = np.stack(As)
        bs = np.stack(bs)
        return As, bs
            
    def slide_pcloud(self, new_pcloud):
        if len(self.memory) >= self.n_memory:
            self.memory.pop(0)
        self.memory.append(new_pcloud)
        self.pcloud = sum(self.memory[1:], self.memory[0])
        self.pcloud = self.filter_pcloud(self.pcloud)
        #np.save("pcloud", np.asarray(self.pcloud.points))

    def filter_pcloud(self, pcloud):
        pcloud = pcloud.voxel_down_sample(self.voxel_size)
        #pcloud, ind = pcloud.remove_statistical_outlier(nb_neighbors=self.nb_neighbors, std_ratio=self.std_ratio)
        pcloud, ind = pcloud.remove_radius_outlier(nb_points=6, radius=0.25)
        return pcloud

    def project_pcloud(self, depth_image):
        h, w = depth_image.shape
        i, j = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')
        valid = np.isfinite(depth_image)
        z = depth_image[valid]
        x = (i[valid] - self.cx) * z / self.fx
        y = (j[valid] - self.cy) * z / self.fy
        return np.stack((x, y, z), axis=1)
    
    def step(self, depth_image, conf_image, position, quaternion):
        depth_image = self.filter_depthimage(depth_image, conf_image)
        new_points = self.project_pcloud(depth_image)
        T = pose_to_fli_ocv(position, quaternion, hand_eye=self.hand_eye)
        points_world = self.transform_point_cloud(new_points, T)
        new_pcloud = o3d.geometry.PointCloud()
        new_pcloud.points = o3d.utility.Vector3dVector(points_world)
        new_pcloud = self.filter_pcloud(new_pcloud)
        with self.lock:
            self.slide_pcloud(new_pcloud)
        #print(f"POINT CLOUD WITH {len(self.pcloud.points)} POINTS")
    
    def filter_depthimage(self, depth_image, conf_image):
        valid = np.isfinite(depth_image) & (depth_image > self.min_depth) & (depth_image < self.max_depth) & (conf_image < self.conf_thres)
        depth_image[~valid] = np.nan
        return depth_image
    
    def transform_point_cloud(self, points_cam, T_cam_in_world):
        points_hom = np.hstack([points_cam, np.ones((points_cam.shape[0], 1))])
        points_world = (T_cam_in_world @ points_hom.T).T[:, :3]
        return points_world
