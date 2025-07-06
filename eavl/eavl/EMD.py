import open3d as o3d
import numpy as np
import ot  # from POT: Python Optimal Transport
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

def load_point_cloud_from_mesh(file_path, num_points=1024):
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

        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles()
        
        # Tạo point cloud
        pcd = mesh.sample_points_uniformly(number_of_points=num_points)
        points = np.asarray(pcd.points)

        # Xóa file tạm nếu có
        if file_path.lower().endswith('.obj'):
            os.unlink(file_to_process)

        return points
    except Exception as e:
        print(f"Lỗi khi xử lý file {file_path}: {str(e)}")
        raise

def earth_movers_distance(pc1, pc2):
    n = min(len(pc1), len(pc2))
    pc1 = pc1[:n]
    pc2 = pc2[:n]

    M = ot.dist(pc1, pc2, metric='euclidean')


    a = np.ones((n,)) / n
    b = np.ones((n,)) / n

    emd_value = ot.emd2(a, b, M)
    return emd_value

file1 = "result_0de9f2815a47550578e024681b10032a_256.obj"
file2 = "base1.obj"

pc1 = load_point_cloud_from_mesh(file1)
pc2 = load_point_cloud_from_mesh(file2)
emd = earth_movers_distance(pc1, pc2)
print(f"Earth Mover's Distance (EMD): {emd:.6f}")
