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

def normalize_mesh(mesh):
    """Chuẩn hóa mesh về center và scale"""
    # Lấy vertices
    vertices = np.asarray(mesh.vertices)
    
    # Di chuyển về center
    centroid = np.mean(vertices, axis=0)
    vertices = vertices - centroid
    
    # Chuẩn hóa scale
    max_dist = np.max(np.sqrt(np.sum(vertices**2, axis=1)))
    vertices = vertices / max_dist
    
    # Cập nhật vertices cho mesh
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    return mesh

def load_mesh_as_voxel(file_path, voxel_size=0.05):
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
        
        # Chuẩn hóa mesh
        mesh = normalize_mesh(mesh)
        
        print(f"Thông tin mesh:")
        print(f"- Số vertices: {len(mesh.vertices)}")
        print(f"- Số triangles: {len(mesh.triangles)}")

        # Tạo voxel grid với kích thước voxel lớn hơn
        voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size=voxel_size)
        
        # In thông tin về voxel grid
        voxels = voxel_grid.get_voxels()
        print(f"- Số voxels: {len(voxels)}")

        # Xóa file tạm nếu có
        if file_path.lower().endswith('.obj'):
            os.unlink(file_to_process)

        return voxel_grid
    except Exception as e:
        print(f"Lỗi khi xử lý file {file_path}: {str(e)}")
        raise

def compute_voxel_iou(voxel1, voxel2):
    """Tính IoU giữa hai voxel grid"""
    voxels1 = set([tuple(voxel.grid_index) for voxel in voxel1.get_voxels()])
    voxels2 = set([tuple(voxel.grid_index) for voxel in voxel2.get_voxels()])
    
    intersection = voxels1 & voxels2
    union = voxels1 | voxels2
    
    print(f"\nThông tin IoU:")
    print(f"- Số voxel trong mesh 1: {len(voxels1)}")
    print(f"- Số voxel trong mesh 2: {len(voxels2)}")
    print(f"- Số voxel trong intersection: {len(intersection)}")
    print(f"- Số voxel trong union: {len(union)}")
    
    if len(union) == 0:
        return 0.0
    iou = len(intersection) / len(union)
    return iou

def visualize_voxel_grids(voxel1, voxel2, file1_name, file2_name):
    """Hiển thị hai voxel grid để so sánh"""
    # Tạo màu khác nhau cho hai voxel grid
    voxels1 = voxel1.get_voxels()
    voxels2 = voxel2.get_voxels()
    
    # Tạo point cloud từ voxel centers
    pcd1 = o3d.geometry.PointCloud()
    pcd2 = o3d.geometry.PointCloud()
    
    if voxels1:
        centers1 = np.array([voxel.grid_index for voxel in voxels1])
        pcd1.points = o3d.utility.Vector3dVector(centers1)
        pcd1.paint_uniform_color([1, 0, 0])  # Đỏ cho voxel grid 1
    
    if voxels2:
        centers2 = np.array([voxel.grid_index for voxel in voxels2])
        pcd2.points = o3d.utility.Vector3dVector(centers2)
        pcd2.paint_uniform_color([0, 0, 1])  # Xanh cho voxel grid 2
    
    print(f"\nHiển thị voxel grids:")
    print(f"Màu đỏ: {file1_name}")
    print(f"Màu xanh: {file2_name}")
    o3d.visualization.draw_geometries([pcd1, pcd2])

# Test với các file
file1 = "result_0de9f2815a47550578e024681b10032a_256.obj"
file2 = "base1.obj"

try:
    # Thử các giá trị voxel_size khác nhau
    voxel_sizes = [0.05, 0.1, 0.15]
    
    for vsize in voxel_sizes:
        print(f"\n=== Testing with voxel_size = {vsize} ===")
        print("\nĐang xử lý file 1...")
        voxel1 = load_mesh_as_voxel(file1, voxel_size=vsize)
        print("\nĐang xử lý file 2...")
        voxel2 = load_mesh_as_voxel(file2, voxel_size=vsize)

        iou_score = compute_voxel_iou(voxel1, voxel2)
        print(f"\nIoU (Voxel-based) với voxel_size={vsize}: {iou_score:.4f}")
        
        # Hiển thị visualization cho voxel_size tốt nhất
        if vsize == 0.1:  # Giá trị trung bình
            visualize_voxel_grids(voxel1, voxel2, file1, file2)

except Exception as e:
    print(f"\nLỗi: {str(e)}") 