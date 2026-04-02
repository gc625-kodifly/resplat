import numpy as np
import os
import torch
try:
    import open3d as o3d
except:
    pass


def export_to_point_cloud(point_cloud, colors,
    save_path='point_cloud.ply',
    denoise_cloud=False,
    denoise_nb_points=10,
    denoise_radius=0.03,
    ):
    # point_cloud: [N, 3]
    # colors: [N, 3]

    # point_cloud = point_cloud / 10.

    point_cloud = point_cloud - np.median(point_cloud, axis=0, keepdims=True)

    # Ensure colors are in the range [0, 1]
    colors = np.clip(colors, 0, 1)

    # Create an Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    save_dir = os.path.dirname(save_path)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Export the point cloud to a PLY file
    o3d.io.write_point_cloud(save_path, pcd)

    if denoise_cloud:
        print("denoise point cloud...")
        cl, ind = pcd.remove_radius_outlier(nb_points=denoise_nb_points, radius=denoise_radius)
        inlier_cloud = pcd.select_by_index(ind)
        o3d.io.write_point_cloud(save_path[:-4] + '_denoise.ply', inlier_cloud)


def transform_points(world_points, cam_to_world):
    """
    Transforms world 3D points to camera coordinates.
    
    Args:
        world_points (torch.Tensor): Nx3 tensor of 3D points in world coordinates.
        cam_to_world (torch.Tensor): 4x4 tensor of camera-to-world extrinsics.
    
    Returns:
        torch.Tensor: Nx3 tensor of 3D points in camera coordinates.
    """
    # Convert world points to homogeneous coordinates (Nx4)
    N = world_points.shape[0]
    ones = torch.ones((N, 1), device=world_points.device)
    world_points_h = torch.cat([world_points, ones], dim=1)  # Nx4
    
    # Compute the inverse of the extrinsics (world-to-camera transformation)
    world_to_cam = torch.inverse(cam_to_world)
    
    # Apply transformation
    camera_points_h = (world_to_cam @ world_points_h.T).T  # Nx4
    
    # Convert back to 3D coordinates (drop the homogeneous coordinate)
    camera_points = camera_points_h[:, :3]  # Nx3
    
    return camera_points

