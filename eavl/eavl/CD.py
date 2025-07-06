# import open3d as o3d
# import numpy as np
# from scipy.spatial import cKDTree
# import os
# import trimesh
# import tempfile

# def convert_obj_to_ply(obj_path):
#     """Chuyển đổi file OBJ sang PLY sử dụng trimesh"""
#     try:
#         # Đọc mesh với trimesh
#         mesh = trimesh.load(obj_path)
#         # Tạo temporary file với đuôi .ply
#         with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
#             ply_path = tmp_file.name
#         # Lưu dưới dạng PLY
#         mesh.export(ply_path)
#         return ply_path
#     except Exception as e:
#         print(f"Lỗi khi chuyển đổi file {obj_path}: {str(e)}")
#         return None

# def load_point_cloud_from_mesh(file_path, num_points=2048):
#     if not os.path.exists(file_path):
#         raise FileNotFoundError(f"File không tồn tại: {file_path}")

#     try:
#         # Nếu là file OBJ, chuyển sang PLY trước
#         if file_path.lower().endswith('.obj'):
#             print(f"Chuyển đổi {file_path} sang PLY...")
#             ply_path = convert_obj_to_ply(file_path)
#             if ply_path is None:
#                 raise ValueError(f"Không thể chuyển đổi file {file_path} sang PLY")
#             file_to_process = ply_path
#         else:
#             file_to_process = file_path

#         # Đọc mesh
#         mesh = o3d.io.read_triangle_mesh(file_to_process)
        
#         # Kiểm tra và sửa mesh
#         if not mesh.has_vertices() or len(mesh.vertices) == 0:
#             raise ValueError(f"Mesh không có vertices hợp lệ")
        
#         mesh.remove_degenerate_triangles()
#         mesh.remove_duplicated_vertices()
#         mesh.remove_duplicated_triangles()
#         mesh.compute_vertex_normals()
        
#         print(f"Thông tin mesh:")
#         print(f"- Số vertices: {len(mesh.vertices)}")
#         print(f"- Số triangles: {len(mesh.triangles)}")
        
#         # Convert mesh -> point cloud
#         pcd = mesh.sample_points_uniformly(number_of_points=num_points)
#         points = np.asarray(pcd.points)
#         print(f"- Đã tạo point cloud với {len(points)} điểm")

#         # Xóa file PLY tạm nếu đã tạo
#         if file_path.lower().endswith('.obj'):
#             os.unlink(file_to_process)
            
#         return points
        
#     except Exception as e:
#         print(f"Lỗi khi xử lý file {file_path}: {str(e)}")
#         raise

# def chamfer_distance(pc1, pc2):
#     tree1 = cKDTree(pc1)
#     tree2 = cKDTree(pc2)

#     dist1, _ = tree1.query(pc2)
#     dist2, _ = tree2.query(pc1)

#     chamfer = np.mean(dist1**2) + np.mean(dist2**2)
#     return chamfer

# # Test với các file
# file1 = "base.obj"
# file2 = "result.obj"

# try:
#     print("\nĐang xử lý file 1...")
#     pc1 = load_point_cloud_from_mesh(file1)
#     print("\nĐang xử lý file 2...")
#     pc2 = load_point_cloud_from_mesh(file2)

#     dist = chamfer_distance(pc1, pc2)
#     print(f"\nChamfer Distance: {dist:.6f}")
# except Exception as e:
#     print(f"\nLỗi: {str(e)}")


import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

# =============================
# 1. Dữ liệu mẫu: P và Q
# =============================
# Tạo 10 điểm ngẫu nhiên cho P (Ground truth) trong vùng [0, 1] x [0, 1]
np.random.seed(42)
P = np.random.rand(10, 2)

# Tạo 10 điểm ngẫu nhiên cho Q (Reconstructed) trong vùng [0.5, 1.5] x [0.5, 1.5]
Q = 0.5 + np.random.rand(10, 2)

# =============================
# 2. Tạo KD-Tree
# =============================
tree_P = cKDTree(P)
tree_Q = cKDTree(Q)

# =============================
# 3. Tìm NN: P -> Q và Q -> P
# =============================
dist_PQ, idx_PQ = tree_Q.query(P)  # Mỗi P tìm điểm gần nhất trong Q
dist_QP, idx_QP = tree_P.query(Q)  # Mỗi Q tìm điểm gần nhất trong P

print(f"P -> Q distances: {dist_PQ}, indices: {idx_PQ}")
print(f"Q -> P distances: {dist_QP}, indices: {idx_QP}")

# =============================
# 4. Tính Chamfer Distance
# =============================
CD = np.mean(dist_PQ ** 2) + np.mean(dist_QP ** 2)
print(f"Chamfer Distance: {CD}")

# =============================
# 5. Vẽ minh hoạ
# =============================
plt.figure(figsize=(6, 6))
plt.scatter(P[:,0], P[:,1], c='blue', label='P', s=100)
plt.scatter(Q[:,0], Q[:,1], c='red', label='Q', s=100)

# Vẽ vector P -> Q
for i, p in enumerate(P):
    q_idx = idx_PQ[i]
    q = Q[q_idx]
    plt.plot([p[0], q[0]], [p[1], q[1]], 'k--', alpha=0.7)

# Vẽ vector Q -> P
for i, q in enumerate(Q):
    p_idx = idx_QP[i]
    p = P[p_idx]
    plt.plot([q[0], p[0]], [q[1], p[1]], 'g--', alpha=0.7)

# Trang trí
plt.title('Chamfer Distance: P <-> Q (Nearest Neighbors)')
plt.legend()
plt.grid(True)
plt.xlim(-0.5, 1.5)
plt.ylim(-0.5, 1.5)
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
