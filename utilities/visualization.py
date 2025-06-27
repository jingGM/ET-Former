import copy
import open3d as o3d
import torch
import numpy as np
from mayavi import mlab

from utilities.functions import get_grid_coords

SemanticColors = [[0, 0, 0, 255], [0, 0, 255, 255], [31, 119, 180, 255], [255, 127, 14, 255], [44, 160, 44, 255],
                  [214, 39, 40, 255], [148, 103, 189, 255], [140, 86, 75, 255], [227, 119, 194, 255],
                  [189, 189, 34, 255], [245, 150, 100, 255], [245, 230, 100, 255], [255, 187, 120, 255],
                  [250, 80, 100, 255], [23, 190, 207, 255], [150, 60, 30, 255], [255, 0, 0, 255], [0, 0, 0, 255],
                  [180, 30, 80, 255], [0, 0, 0, 255], [255, 255, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255],
                  [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255],
                  [0, 0, 0, 255], [30, 30, 255, 255], [200, 40, 255, 255], [90, 30, 150, 255], [0, 0, 0, 255],
                  [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255],
                  [255, 0, 255, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [255, 150, 255, 255],
                  [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [75, 0, 75, 255], [75, 0, 175, 255],
                  [0, 200, 255, 255], [50, 120, 255, 255], [0, 150, 255, 255], [0, 0, 0, 255], [0, 0, 0, 255],
                  [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [170, 255, 150, 255],
                  [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255],
                  [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 175, 0, 255], [0, 60, 135, 255],
                  [80, 240, 150, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255],
                  [0, 0, 0, 255], [0, 0, 0, 255], [150, 240, 255, 255], [0, 0, 255, 255], [0, 0, 0, 255],
                  [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255],
                  [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255],
                  [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [255, 255, 50, 255], [0, 0, 0, 255],
                  [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255],
                  [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255],
                  [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255],
                  [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255],
                  [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255],
                  [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255],
                  [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255],
                  [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255],
                  [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255],
                  [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255],
                  [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255],
                  [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255],
                  [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255],
                  [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255],
                  [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255],
                  [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255],
                  [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255],
                  [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255],
                  [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255],
                  [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255],
                  [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255],
                  [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255],
                  [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255],
                  [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255],
                  [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255],
                  [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 255], [255, 255, 255, 255]]


def display(points_list, point_colors=None, point_size=5, edges=None, edge_size=2, edge_colors=None,
            frustum_size=None, P=None, T_velo_2_cam=None, frustum_color=[0, 1, 0, 1], frustum_line_width=5,
            camera_center=[0, 0, 0]):
    """
    Args:
        points_list: [pts, pts, ...]
        colors: [(4,), (4,), ...]
        point_size: [int, int, ...]
        edges: []
        edge_size: int
        frustum_size: W,H,D
        P: Intrinsic
        T_velo_2_cam: extrinsic
        frustum_color: (4, )
        frustum_line_width: float
    Returns:
    """
    app = o3d.visualization.gui.Application.instance
    app.initialize()
    window = app.create_window("Open3d", 1024, 768)
    widget3d = o3d.visualization.gui.SceneWidget()
    widget3d.scene = o3d.visualization.rendering.Open3DScene(window.renderer)
    widget3d.scene.set_background([0, 0, 0, 1.0])
    window.add_child(widget3d)

    if point_colors is None:
        point_colors = np.array([[1, 0, 0, 1] for i in range(len(points_list))])
    for pts_idx in range(len(points_list)):
        color = point_colors[pts_idx]
        pts = points_list[pts_idx]
        pts_size = point_size if type(point_size) is int else point_size[pts_idx]
        add_cloud(widget3d.scene, "p_" + str(pts_idx), pts, color, pts_size)

    if edges is not None:
        if edge_colors is None:
            edge_colors = np.array([[0, 0.3, 0, 1] for i in range(len(edges))])
        for edge_index in range(len(edges)):
            edge = edges[edge_index]
            e_size = edge_size if type(edge_size) is int else edge_size[edge_index]
            add_edges(widget3d.scene, "e_" + str(edge_index), edge[0], edge[1], edge_colors[edge_index], e_size)

    if frustum_size is not None:
        add_camera_frame(widget3d.scene, frustum_size, P, T_velo_2_cam, color=frustum_color, size=frustum_line_width)

    if isinstance(P, torch.Tensor):
        try:
            P = P.cpu().numpy()
        except:
            P = P.detach().cpu().numpy()
    if isinstance(T_velo_2_cam, torch.Tensor):
        try:
            T_velo_2_cam = T_velo_2_cam.cpu().numpy()
        except:
            T_velo_2_cam = T_velo_2_cam.detach().cpu().numpy()
    if T_velo_2_cam is not None:
        T_velo_2_cam = T_velo_2_cam.astype(float)
        widget3d.setup_camera(P, T_velo_2_cam,
                              frustum_size[0], frustum_size[1],
                              widget3d.scene.bounding_box)
    app.run()


def display_single_points(pts, color=None, point_size=5, frame_size=0.2, frame_coordinate=None, center=[0, 0, 0]):
    app = o3d.visualization.gui.Application.instance
    app.initialize()
    window = app.create_window("Open3d", 1024, 768)
    widget3d = o3d.visualization.gui.SceneWidget()
    widget3d.scene = o3d.visualization.rendering.Open3DScene(window.renderer)
    widget3d.scene.set_background([1.0, 1.0, 1.0, 1.0])
    window.add_child(widget3d)
    if color is None:
        color = [1, 0, 0, 1]
    add_cloud(widget3d.scene, "ref", pts, color, point_size)

    if frame_coordinate is None:
        pose = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    elif frame_coordinate.shape[0] == 3 and len(frame_coordinate.shape) == 1:
        pose = np.array([[1, 0, 0, frame_coordinate[0]], [0, 1, 0, frame_coordinate[1]],
                         [0, 0, 1, frame_coordinate[2]], [0, 0, 0, 1]])
    elif frame_coordinate.shape == (3, 4):
        pose = np.eye(4)
        pose[:3, :] = frame_coordinate
    else:
        pose = frame_coordinate
    add_frame(scene=widget3d.scene, name="ref_frame", frame_size=frame_size, frame=pose)

    widget3d.setup_camera(60, widget3d.scene.bounding_box, center)
    app.run()


def add_camera_frame(scene, frustrum_size, P, T_velo_2_cam, color, size):
    W, H, D = frustrum_size
    x = D * W / (2 * P[0, 0])
    y = D * H / (2 * P[1, 1])
    vertices = np.array([
        [0, 0, 0, 1],
        [x, y, D, 1],
        [-x, y, D, 1],
        [-x, -y, D, 1],
        [x, -y, D, 1]])
    vertices = (np.linalg.inv(T_velo_2_cam) @ vertices.T).T[:, :3]
    # vox_origin = np.array([0, -25.6, -2])
    # vertices -= vox_origin
    edges = np.array(
        [[vertices[0], vertices[0], vertices[0], vertices[0], vertices[1], vertices[2], vertices[3], vertices[4]],
         [vertices[1], vertices[2], vertices[3], vertices[4], vertices[2], vertices[3], vertices[4], vertices[1]]])
    add_edges(scene, "camera", edges[0], edges[1], color, size)


def add_frame(scene, name, frame, frame_size):
    if isinstance(frame, torch.Tensor):
        try:
            frame = frame.cpu().numpy()
        except:
            frame = frame.detach().cpu().numpy()
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size, origin=np.array([0.0, 0.0, 0.0]))
    mesh = copy.deepcopy(mesh)
    mesh.rotate(frame[:3, :3], center=(0, 0, 0))
    mesh.translate((frame[0, 3], frame[1, 3], frame[2, 3]))

    mtl = o3d.visualization.rendering.MaterialRecord()
    mtl.base_color = [1.0, 1.0, 1.0, 1.0]
    mtl.shader = "defaultUnlit"
    scene.add_geometry(name, mesh, mtl)


def add_cloud(scene, name, pts, color, size):
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultUnlit"
    material.point_size = size

    if isinstance(pts, torch.Tensor):
        try:
            pts = pts.cpu().numpy()
        except:
            pts = pts.detach().cpu().numpy()
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pts)
    if len(np.asarray(color).shape) > 1:
        cloud.colors = o3d.utility.Vector3dVector(color[:, :3])
    else:
        material.base_color = color
    scene.add_geometry(name, cloud, material)


def add_edges(scene, name, pt0, pt1, color, size):
    if isinstance(pt0, torch.Tensor):
        pt0 = pt0.cpu().numpy()
    if isinstance(pt1, torch.Tensor):
        pt1 = pt1.cpu().numpy()
    assert pt0.shape[0] == pt1.shape[0], Exception("the two poins have different sizes")
    lns = np.array([[i, i + len(pt0)] for i in range(len(pt0))])
    if isinstance(lns, torch.Tensor):
        lns = lns.cpu().numpy()
    assert lns.shape[1] == 2, Exception("the lines shape is not correct")
    lines = o3d.geometry.LineSet()
    lines.points = o3d.utility.Vector3dVector(np.concatenate((pt0, pt1), axis=0)[:, :3])
    lines.lines = o3d.utility.Vector2iVector(lns)
    # ln_c = np.zeros((lns.shape[0], 3))
    # ln_c[:, 1] += 1

    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultUnlit"

    color = np.array(color)
    if len(color.shape) == 1:
        material.base_color = color
    else:
        lines.colors = o3d.utility.Vector3dVector(color)
    material.point_size = size
    scene.add_geometry(name, lines, material)


def maya_draw_uncertainty(voxels, T_velo_2_cam, vox_dim,
                          img_size, vox_origin,
                          intrinsic,
                          color_levels=100,
                          voxel_size=0.2,
                          d=7,  # 7m - determine the size of the mesh representing the camera
                          factor=0.05,
                          file_name=None):
    # figure = mlab.figure(size=(1400, 1400), bgcolor=(1, 1, 1))
    figure = mlab.figure(size=(1600, 900), bgcolor=(1, 1, 1))

    # Compute the coordinates of the mesh representing camera
    xmax = d * (img_size[0] - 1 - intrinsic[0, 2]) / intrinsic[0, 0]
    ymax = d * (img_size[1] - 1 - intrinsic[1, 2]) / intrinsic[1, 1]
    xmin = d * (-intrinsic[0, 2]) / intrinsic[0, 0]
    ymin = d * (-intrinsic[1, 2]) / intrinsic[1, 1]
    tri_points = np.array(
        [
            [0, 0, 0],
            [xmax, ymax, d],
            [xmin, ymax, d],
            [xmin, ymin, d],
            [xmax, ymin, d]
        ]
    )
    # tri_points = np.matmul(tri_points, np.linalg.inv(T_velo_2_cam)[:3, :3].transpose(-1, -2)) + np.linalg.inv(T_velo_2_cam)[None, :3, 3]
    tri_points = np.hstack([tri_points, np.ones((5, 1))])
    tri_points = (np.linalg.inv(T_velo_2_cam) @ tri_points.T).T
    x = tri_points[:, 0]  # - vox_origin[0]
    y = tri_points[:, 1]  # - vox_origin[1]
    z = tri_points[:, 2]  # - vox_origin[2]
    triangles = [
        (0, 1, 2),
        (0, 1, 4),
        (0, 3, 4),
        (0, 2, 3),
    ]
    # Draw the camera
    mlab.triangular_mesh(
        x, y, z, triangles, representation="wireframe", color=(0, 0, 0), line_width=5
    )

    grid_coords = get_grid_coords(
        vox_dim, voxel_size
    ) + vox_origin[None, :]

    lut = np.zeros((color_levels, 4))
    lut[0, :] = [0, 0, 255, 255]
    for row in range(1, color_levels, 1):
        f = (row / color_levels)
        lut[row, :] = [255 * f, 0, 255 * (1 - f), 255]

    indicies = torch.where(voxels > 0)

    plt_plot_fov = mlab.points3d(
        grid_coords[indicies][:, 0],
        grid_coords[indicies][:, 1],
        grid_coords[indicies][:, 2],
        voxels[indicies],
        scale_mode="none",
        scale_factor=voxel_size - factor * voxel_size,
        mode="cube",
        opacity=1.0,
    )
    plt_plot_fov.module_manager.scalar_lut_manager.lut.number_of_colors = color_levels
    plt_plot_fov.module_manager.scalar_lut_manager.lut.table = lut

    mlab.view(azimuth=0, elevation=-45, distance=90, focalpoint=[20, 0, 0])
    if file_name is None:
        mlab.show()
    else:
        mlab.savefig(file_name)  # Save as a PNG file
        mlab.close()


def maya_draw(
        voxels,
        T_velo_2_cam,
        vox_origin,
        fov_mask,
        img_size,
        intrinsic,
        voxel_size=0.2,
        d=7,  # 7m - determine the size of the mesh representing the camera
        factor=0.05,
        file_name=None
):
    # figure = mlab.figure(size=(1400, 1400), bgcolor=(1, 1, 1))
    figure = mlab.figure(size=(1600, 900), bgcolor=(1, 1, 1))

    # Compute the coordinates of the mesh representing camera
    xmax = d * (img_size[0] - 1 - intrinsic[0, 2]) / intrinsic[0, 0]
    ymax = d * (img_size[1] - 1 - intrinsic[1, 2]) / intrinsic[1, 1]
    xmin = d * (-intrinsic[0, 2]) / intrinsic[0, 0]
    ymin = d * (-intrinsic[1, 2]) / intrinsic[1, 1]
    tri_points = np.array(
        [
            [0, 0, 0],
            [xmax, ymax, d],
            [xmin, ymax, d],
            [xmin, ymin, d],
            [xmax, ymin, d]
        ]
    )
    # tri_points = np.matmul(tri_points, np.linalg.inv(T_velo_2_cam)[:3, :3].transpose(-1, -2)) + np.linalg.inv(T_velo_2_cam)[None, :3, 3]
    tri_points = np.hstack([tri_points, np.ones((5, 1))])
    tri_points = (np.linalg.inv(T_velo_2_cam) @ tri_points.T).T
    x = tri_points[:, 0]  # - vox_origin[0]
    y = tri_points[:, 1]  # - vox_origin[1]
    z = tri_points[:, 2]  # - vox_origin[2]
    triangles = [
        (0, 1, 2),
        (0, 1, 4),
        (0, 3, 4),
        (0, 2, 3),
    ]
    # Draw the camera
    mlab.triangular_mesh(
        x, y, z, triangles, representation="wireframe", color=(0, 0, 0), line_width=5
    )

    # Compute the voxels coordinates
    grid_coords = get_grid_coords(
        [voxels.shape[0], voxels.shape[1], voxels.shape[2]], voxel_size
    ) + vox_origin[None, :]

    # Attach the predicted class to every voxel
    grid_coords = np.vstack([grid_coords.T, voxels.reshape(-1)]).T

    # Get the voxels inside FOV
    fov_grid_coords = grid_coords[fov_mask, :]

    # Get the voxels outside FOV
    outfov_grid_coords = grid_coords[~fov_mask, :]

    # Remove empty and unknown voxels
    fov_voxels = fov_grid_coords[(fov_grid_coords[:, 3] > 0) & (fov_grid_coords[:, 3] < 255)]
    outfov_voxels = outfov_grid_coords[(outfov_grid_coords[:, 3] > 0) & (outfov_grid_coords[:, 3] < 255)]

    # Draw occupied inside FOV voxels
    plt_plot_fov = mlab.points3d(
        fov_voxels[:, 0],
        fov_voxels[:, 1],
        fov_voxels[:, 2],
        fov_voxels[:, 3],
        colormap="viridis",
        scale_factor=voxel_size - factor * voxel_size,
        mode="cube",
        opacity=1.0,
        vmin=1,
        vmax=19,
    )

    # Draw occupied outside FOV voxels
    plt_plot_outfov = mlab.points3d(
        outfov_voxels[:, 0],
        outfov_voxels[:, 1],
        outfov_voxels[:, 2],
        outfov_voxels[:, 3],
        colormap="viridis",
        scale_factor=voxel_size - factor * voxel_size,
        mode="cube",
        opacity=1.0,
        vmin=1,
        vmax=19,
    )

    colors = np.array(
        [
            [100, 150, 245, 255],
            [100, 230, 245, 255],
            [30, 60, 150, 255],
            [80, 30, 180, 255],
            [100, 80, 250, 255],
            [255, 30, 30, 255],
            [255, 40, 200, 255],
            [150, 30, 90, 255],
            [255, 0, 255, 255],
            [255, 150, 255, 255],
            [75, 0, 75, 255],
            [175, 0, 75, 255],
            [255, 200, 0, 255],
            [255, 120, 50, 255],
            [0, 175, 0, 255],
            [135, 60, 0, 255],
            [150, 240, 80, 255],
            [255, 240, 150, 255],
            [255, 0, 0, 255],
        ]
    ).astype(np.uint8)

    plt_plot_fov.glyph.scale_mode = "scale_by_vector"
    plt_plot_outfov.glyph.scale_mode = "scale_by_vector"

    plt_plot_fov.module_manager.scalar_lut_manager.lut.table = colors

    outfov_colors = colors
    outfov_colors[:, :3] = outfov_colors[:, :3] // 3 * 2
    plt_plot_outfov.module_manager.scalar_lut_manager.lut.table = outfov_colors

    mlab.view(azimuth=0, elevation=-45, distance=90, focalpoint=[20, 0, 0])
    # mlab.show()
    if file_name is None:
        mlab.show()
    else:
        mlab.savefig(file_name)  # Save as a PNG file
        mlab.close()


def maya_draw_ori(
        voxels,
        T_velo_2_cam,
        vox_origin,
        fov_mask,
        img_size,
        f,
        voxel_size=0.2,
        d=7,  # 7m - determine the size of the mesh representing the camera
        factor=0.05
):
    # figure = mlab.figure(size=(1400, 1400), bgcolor=(1, 1, 1))
    figure = mlab.figure(size=(1600, 900), bgcolor=(1, 1, 1))

    # Compute the coordinates of the mesh representing camera
    x = d * img_size[0] / (2 * f)
    y = d * img_size[1] / (2 * f)
    tri_points = np.array(
        [
            [0, 0, 0],
            [x, y, d],
            [-x, y, d],
            [-x, -y, d],
            [x, -y, d],
        ]
    )
    tri_points = np.hstack([tri_points, np.ones((5, 1))])
    tri_points = (np.linalg.inv(T_velo_2_cam) @ tri_points.T).T
    x = tri_points[:, 0] - vox_origin[0]
    y = tri_points[:, 1] - vox_origin[1]
    z = tri_points[:, 2] - vox_origin[2]
    triangles = [
        (0, 1, 2),
        (0, 1, 4),
        (0, 3, 4),
        (0, 2, 3),
    ]
    # Draw the camera
    mlab.triangular_mesh(
        x, y, z, triangles, representation="wireframe", color=(0, 0, 0), line_width=5
    )

    # Compute the voxels coordinates
    grid_coords = get_grid_coords(
        [voxels.shape[0], voxels.shape[1], voxels.shape[2]], voxel_size
    )

    # Attach the predicted class to every voxel
    grid_coords = np.vstack([grid_coords.T, voxels.reshape(-1)]).T

    # Get the voxels inside FOV
    fov_grid_coords = grid_coords[fov_mask, :]

    # Get the voxels outside FOV
    outfov_grid_coords = grid_coords[~fov_mask, :]

    # Remove empty and unknown voxels
    fov_voxels = fov_grid_coords[(fov_grid_coords[:, 3] > 0) & (fov_grid_coords[:, 3] < 255)]
    outfov_voxels = outfov_grid_coords[(outfov_grid_coords[:, 3] > 0) & (outfov_grid_coords[:, 3] < 255)]

    # Draw occupied inside FOV voxels
    plt_plot_fov = mlab.points3d(
        fov_voxels[:, 0],
        fov_voxels[:, 1],
        fov_voxels[:, 2],
        fov_voxels[:, 3],
        colormap="viridis",
        scale_factor=voxel_size - factor * voxel_size,
        mode="cube",
        opacity=1.0,
        vmin=1,
        vmax=19,
    )

    # Draw occupied outside FOV voxels
    plt_plot_outfov = mlab.points3d(
        outfov_voxels[:, 0],
        outfov_voxels[:, 1],
        outfov_voxels[:, 2],
        outfov_voxels[:, 3],
        colormap="viridis",
        scale_factor=voxel_size - factor * voxel_size,
        mode="cube",
        opacity=1.0,
        vmin=1,
        vmax=19,
    )

    colors = np.array(
        [
            [100, 150, 245, 255],
            [100, 230, 245, 255],
            [30, 60, 150, 255],
            [80, 30, 180, 255],
            [100, 80, 250, 255],
            [255, 30, 30, 255],
            [255, 40, 200, 255],
            [150, 30, 90, 255],
            [255, 0, 255, 255],
            [255, 150, 255, 255],
            [75, 0, 75, 255],
            [175, 0, 75, 255],
            [255, 200, 0, 255],
            [255, 120, 50, 255],
            [0, 175, 0, 255],
            [135, 60, 0, 255],
            [150, 240, 80, 255],
            [255, 240, 150, 255],
            [255, 0, 0, 255],
        ]
    ).astype(np.uint8)

    plt_plot_fov.glyph.scale_mode = "scale_by_vector"
    plt_plot_outfov.glyph.scale_mode = "scale_by_vector"

    plt_plot_fov.module_manager.scalar_lut_manager.lut.table = colors

    outfov_colors = colors
    outfov_colors[:, :3] = outfov_colors[:, :3] // 3 * 2
    plt_plot_outfov.module_manager.scalar_lut_manager.lut.table = outfov_colors

    mlab.show()
