import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree
import trimesh
import tempfile
import os

def convert_obj_to_ply(obj_path):
    """Chuyển đổi file OBJ sang PLY"""
    try:
        mesh = trimesh.load(obj_path)
        with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
            ply_path = tmp_file.name
        mesh.export(ply_path)
        return ply_path
    except Exception as e:
        print(f"Lỗi khi chuyển đổi file {obj_path}: {str(e)}")
        return None

def normalize_points(points):
    """Chuẩn hóa point cloud về center và scale"""
    # Di chuyển về center
    centroid = np.mean(points, axis=0)
    points = points - centroid
    
    # Chuẩn hóa scale
    max_dist = np.max(np.sqrt(np.sum(points**2, axis=1)))
    points = points / max_dist
    
    return points

def load_point_cloud_from_mesh(file_path, num_points=4096):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File không tồn tại: {file_path}")

    try:
        # Chuyển OBJ sang PLY nếu cần
        if file_path.lower().endswith('.obj'):
            print(f"Chuyển đổi {file_path} sang PLY...")
            ply_path = convert_obj_to_ply(file_path)
            if ply_path is None:
                raise ValueError(f"Không thể chuyển đổi file {file_path}")
            file_to_process = ply_path
        else:
            file_to_process = file_path

        # Đọc và xử lý mesh
        mesh = o3d.io.read_triangle_mesh(file_to_process)
        if not mesh.has_vertices() or len(mesh.vertices) == 0:
            raise ValueError(f"Mesh không có vertices hợp lệ")

        # Tiền xử lý mesh
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles()
        mesh.compute_vertex_normals()

        # Tăng số lượng điểm sample để có kết quả chính xác hơn
        pcd = mesh.sample_points_uniformly(number_of_points=num_points)
        points = np.asarray(pcd.points)
        
        # Chuẩn hóa point cloud
        points = normalize_points(points)

        # Xóa file tạm nếu có
        if file_path.lower().endswith('.obj'):
            os.unlink(file_to_process)

        return points

    except Exception as e:
        print(f"Lỗi khi xử lý file {file_path}: {str(e)}")
        raise

def chamfer_distance(pc1, pc2):
    sum_p = 0.0
    for p in pc1:
        min_dist = np.min(np.sum((pc2 - p) ** 2, axis=1))
        sum_p += min_dist
    term1 = sum_p / len(pc1)

    sum_q = 0.0
    for q in pc2:
        min_dist = np.min(np.sum((pc1 - q) ** 2, axis=1))
        sum_q += min_dist
    term2 = sum_q / len(pc2)

    return term1 + term2


def chamfer_distance(pc1, pc2):
    sum_p = 0.0
    for p in pc1:
        min_dist = np.min(np.sqrt(np.sum((pc2 - p) ** 2, axis=1)))
        sum_p += min_dist
    term1 = sum_p / len(pc1)

    sum_q = 0.0
    for q in pc2:
        min_dist = np.min(np.sqrt(np.sum((pc1 - q) ** 2, axis=1)))
        sum_q += min_dist
    term2 = sum_q / len(pc2)

    return term1 + term2

def visualize_point_clouds(pc1, pc2, file1_name, file2_name):
    pcd1 = o3d.geometry.PointCloud()
    pcd2 = o3d.geometry.PointCloud()
    
    pcd1.points = o3d.utility.Vector3dVector(pc1)
    pcd2.points = o3d.utility.Vector3dVector(pc2)
    pcd1.paint_uniform_color([1, 0, 0])  
    pcd2.paint_uniform_color([0, 0, 1])
    
    print(f"\nHiển thị point cloud:")
    print(f"Màu đỏ: {file1_name}")
    print(f"Màu xanh: {file2_name}")
    o3d.visualization.draw_geometries([pcd1, pcd2])


file1 = "result_0de9f2815a47550578e024681b10032a_256.obj"
file2 = "base1.obj"

try:
    print("\nĐang xử lý file 1...")
    pc1 = load_point_cloud_from_mesh(file1)
    print("\nĐang xử lý file 2...")
    pc2 = load_point_cloud_from_mesh(file2)

  
    dist = chamfer_distance(pc1, pc2)
    print(f"\nChamfer Distance: {dist:.6f}")
    

    visualize_point_clouds(pc1, pc2, file1, file2)

except Exception as e:
    print(f"\nLỗi: {str(e)}") 