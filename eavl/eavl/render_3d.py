import open3d as o3d
import numpy as np
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

def load_and_process_mesh(file_path):
    """Load và xử lý mesh"""
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

        # Đọc mesh
        mesh = o3d.io.read_triangle_mesh(file_to_process)
        if not mesh.has_vertices():
            raise ValueError(f"Mesh không có vertices hợp lệ")

        # Xử lý mesh
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()
        
        # Xóa file tạm nếu có
        if file_path.lower().endswith('.obj'):
            os.unlink(file_to_process)

        return mesh
    except Exception as e:
        print(f"Lỗi khi xử lý file {file_path}: {str(e)}")
        raise

def render_mesh(mesh, output_path, width=1920, height=1080):
    """Render mesh với các góc nhìn khác nhau"""
    # Tạo visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=False)
    
    # Thêm mesh vào scene
    vis.add_geometry(mesh)
    
    # Thiết lập render options
    opt = vis.get_render_option()
    opt.background_color = np.asarray([1, 1, 1])  # Nền trắng
    opt.mesh_show_back_face = True
    opt.mesh_show_wireframe = False
    opt.point_size = 1.0
    
    # Thiết lập camera
    ctr = vis.get_view_control()
    
    # Các góc nhìn khác nhau
    views = [
        {"front": [0, 0, 0], "lookat": [0, 0, 0], "up": [0, 1, 0]},  # Mặt trước
        {"front": [0.8, 0, 0.3], "lookat": [0, 0, 0], "up": [0, 1, 0]},  # Góc 45 độ
        {"front": [0, 0.8, 0.3], "lookat": [0, 0, 0], "up": [0, 0, 1]},  # Góc trên
    ]
    
    for i, view in enumerate(views):
        # Thiết lập góc nhìn
        ctr.set_front(view["front"])
        ctr.set_lookat(view["lookat"])
        ctr.set_up(view["up"])
        
        # Cập nhật camera
        vis.poll_events()
        vis.update_renderer()
        
        # Lưu ảnh
        output_file = f"{output_path}_view{i+1}.png"
        vis.capture_screen_image(output_file, do_render=True)
        print(f"Đã lưu ảnh: {output_file}")
    
    vis.destroy_window()

def render_comparison(file1, file2, output_prefix="render"):
    """Render và so sánh hai mesh"""
    try:
        # Load và xử lý các mesh
        print("\nĐang xử lý mesh 1...")
        mesh1 = load_and_process_mesh(file1)
        print("\nĐang xử lý mesh 2...")
        mesh2 = load_and_process_mesh(file2)
        
        # Render từng mesh
        print("\nĐang render mesh 1...")
        render_mesh(mesh1, f"{output_prefix}_mesh1")
        print("\nĐang render mesh 2...")
        render_mesh(mesh2, f"{output_prefix}_mesh2")
        
        print("\nHoàn thành render!")
        
    except Exception as e:
        print(f"\nLỗi: {str(e)}")

# Test với các file
file1 = "result_0de9f2815a47550578e024681b10032a_256.obj"
file2 = "base1.obj"

# Render các mesh
render_comparison(file1, file2) 