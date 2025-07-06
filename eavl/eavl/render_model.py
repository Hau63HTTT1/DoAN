import open3d as o3d
import numpy as np

def render_model(obj_file, output_prefix="render"):
    """Render mô hình 3D từ nhiều góc nhìn"""
    try:
        # Đọc mesh trực tiếp từ file OBJ
        print(f"\nĐang đọc file {obj_file}...")
        mesh = o3d.io.read_triangle_mesh(obj_file)
        
        if not mesh.has_vertices():
            raise ValueError(f"Không thể đọc file {obj_file}")
        
        # Xử lý mesh
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()
        
        # Tạo visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=800, height=800, visible=False)
        vis.add_geometry(mesh)
        
        # Thiết lập render options
        opt = vis.get_render_option()
        opt.background_color = np.asarray([1, 1, 1])  # Nền trắng
        opt.mesh_show_back_face = True
        opt.mesh_show_wireframe = False
        opt.point_size = 1.0
        
        # Thiết lập camera
        ctr = vis.get_view_control()
        
        # Các góc nhìn khác nhau (tương tự như trong hình)
        views = [
            {   # Góc nhìn mặt trước
                "front": [0, 0, -1],
                "lookat": [0, 0, 0],
                "up": [0, 1, 0],
                "zoom": 0.7
            },
            {   # Góc nhìn 45 độ bên phải
                "front": [0.7, 0, -0.7],
                "lookat": [0, 0, 0],
                "up": [0, 1, 0],
                "zoom": 0.7
            },
            {   # Góc nhìn từ bên phải
                "front": [1, 0, 0],
                "lookat": [0, 0, 0],
                "up": [0, 1, 0],
                "zoom": 0.7
            },
            {   # Góc nhìn 45 độ phía sau
                "front": [0.7, 0, 0.7],
                "lookat": [0, 0, 0],
                "up": [0, 1, 0],
                "zoom": 0.7
            },
            {   # Góc nhìn từ phía sau
                "front": [0, 0, 1],
                "lookat": [0, 0, 0],
                "up": [0, 1, 0],
                "zoom": 0.7
            },
            {   # Góc nhìn 45 độ bên trái phía sau
                "front": [-0.7, 0, 0.7],
                "lookat": [0, 0, 0],
                "up": [0, 1, 0],
                "zoom": 0.7
            }
        ]
        
        # Render từng góc nhìn
        for i, view in enumerate(views):
            # Thiết lập góc nhìn
            ctr.set_front(view["front"])
            ctr.set_lookat(view["lookat"])
            ctr.set_up(view["up"])
            ctr.set_zoom(view["zoom"])
            
            # Cập nhật camera
            vis.poll_events()
            vis.update_renderer()
            
            # Lưu ảnh
            output_file = f"{output_prefix}_view{i+1}.png"
            vis.capture_screen_image(output_file, do_render=True)
            print(f"Đã lưu ảnh: {output_file}")
        
        vis.destroy_window()
        print("\nHoàn thành render!")
        
    except Exception as e:
        print(f"\nLỗi: {str(e)}")

# Test với file obj
obj_file = "result_0de9f2815a47550578e024681b10032a_256.obj"
render_model(obj_file) 