import open3d as o3d
import trimesh
import tempfile
import os
import numpy as np

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

def load_mesh_as_voxel(file_path, voxel_size=0.01):
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
        mesh.compute_vertex_normals()

        # Tạo voxel grid
        voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size=voxel_size)

        # Xóa file tạm nếu có
        if file_path.lower().endswith('.obj'):
            os.unlink(file_to_process)

        return voxel_grid
    except Exception as e:
        print(f"Lỗi khi xử lý file {file_path}: {str(e)}")
        raise

def compute_voxel_iou(voxel1, voxel2):
    voxels1 = set([tuple(voxel.grid_index) for voxel in voxel1.get_voxels()])
    voxels2 = set([tuple(voxel.grid_index) for voxel in voxel2.get_voxels()])
    
    intersection = voxels1 & voxels2
    union = voxels1 | voxels2
    
    if len(union) == 0:
        return 0.0
    iou = len(intersection) / len(union)
    return iou

# Test với các file
file1 = "0de9f2815a47550578e024681b10032a_remesh.obj"
file2 = "base1.obj"

try:
    print("\nĐang xử lý file 1...")
    voxel1 = load_mesh_as_voxel(file1, voxel_size=0.03)
    print("\nĐang xử lý file 2...")
    voxel2 = load_mesh_as_voxel(file2, voxel_size=0.03)

    iou_score = compute_voxel_iou(voxel1, voxel2)
    print(f"\nIoU (Voxel-based): {iou_score:.4f}")
except Exception as e:
    print(f"\nLỗi: {str(e)}")
