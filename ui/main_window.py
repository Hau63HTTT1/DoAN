import os
from PyQt5.QtWidgets import (QMainWindow, QWidget, QHBoxLayout, 
                            QVBoxLayout, QLabel, QPushButton, QMessageBox)
from PyQt5.QtCore import Qt, QSize, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QIcon, QFont

from ui.sidebar import Sidebar
from ui.preview_area import PreviewArea
from ui.header_bar import HeaderBar
from utils.animations import FadeTransition

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
   
        self.setWindowTitle("3D Character Creator")
        self.setMinimumSize(1200, 800)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1a1a1a;
            }
        """)
        
     
        self.showMaximized()
        
       
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        
       
        self.header = HeaderBar()
        
      
        self.sidebar = Sidebar()
        
     
        self.preview_area = PreviewArea()
        
 
        self.sidebar_container = QWidget()
        self.sidebar_layout = QVBoxLayout(self.sidebar_container)
        self.sidebar_layout.setContentsMargins(0, 0, 0, 0)
        self.sidebar_layout.setSpacing(0)
        self.sidebar_layout.addWidget(self.header)
        self.sidebar_layout.addWidget(self.sidebar)
        
        
        self.main_layout.addWidget(self.sidebar_container, 1)
        self.main_layout.addWidget(self.preview_area, 3)
        
       
        self.setup_connections()
        self.setup_animations()
        
    def setup_connections(self):
        self.sidebar.model_uploaded.connect(self.preview_area.load_3d_model)
        self.sidebar.download_button_clicked.connect(self.on_download_model)
        self.sidebar.delete_button_clicked.connect(self.on_delete_clicked)
        self.sidebar.create_button_clicked.connect(self.on_create_clicked)
        
        
        if hasattr(self.preview_area, 'model_viewer'):
            self.preview_area.model_viewer.mesh_loaded.connect(self.on_model_loaded)
        
    def on_model_loaded(self, success):
        self.sidebar.enable_model_actions(success)
        
    def on_delete_clicked(self):
        self.preview_area.delete_model()
        
        
        self.sidebar.upload_area.clear_image()
        
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
      
        if self.sidebar.current_image_path:
            image_name = os.path.basename(self.sidebar.current_image_path)
            image_name_no_ext = os.path.splitext(image_name)[0]
            

            singleview_dir = os.path.join(project_root, 'model', 'PiFu-singleview')
            singleview_sample_dir = os.path.join(singleview_dir, 'sample_images')
            singleview_results_dir = os.path.join(singleview_dir, 'results', 'pifuhd_final', 'recon')
            
            singleview_files = [
                os.path.join(singleview_results_dir, f'result_{image_name_no_ext}_256.obj'),
                os.path.join(singleview_results_dir, f'result_{image_name_no_ext}_256.png'),
                os.path.join(singleview_sample_dir, f'{image_name_no_ext}_rect.txt'),
            ]
            for ext in ['.jpg', '.jpeg', '.png', '.webp']:
                singleview_files.append(os.path.join(singleview_sample_dir, f'{image_name_no_ext}{ext}'))
            
    
            multiview_dir = os.path.join(project_root, 'model', 'PIFu-multiview')
            multiview_sample_dir = os.path.join(multiview_dir, 'sample_images', 'rp_Man')
            multiview_results_dir = os.path.join(multiview_dir, 'results')
            
            multiview_files = []
            for i in range(4):
                angles = ['0_0_00', '90_0_00', '180_0_00', '270_0_00']
                multiview_files.extend([
                    os.path.join(multiview_sample_dir, f'{angles[i]}.png'),
                    os.path.join(multiview_sample_dir, f'{angles[i]}_mask.png'),
                ])
      
            import glob
            for results_dir in [singleview_results_dir, multiview_results_dir]:
                if os.path.exists(results_dir):
                    obj_files = glob.glob(os.path.join(results_dir, '*.obj'))
                    multiview_files.extend(obj_files)
            
        
            all_files = singleview_files + multiview_files
            for file_path in all_files:
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Error deleting file {file_path}: {e}")
        

        self.sidebar.enable_model_actions(False)
        
    def setup_animations(self):
        """Setup animations for the UI"""
        self.fade_transition = FadeTransition(self.central_widget)
        
    def on_create_clicked(self, obj_file_path):
        """Handle create button click"""
        self.fade_transition.start()
        self.preview_area.load_3d_model(obj_file_path)

        self.sidebar.current_obj_path = obj_file_path

    def on_download_model(self):
        import shutil
        import glob
        file_name = getattr(self.sidebar, 'selected_download_name', '').strip()
        output_dir = getattr(self.sidebar, 'selected_download_dir', '').strip()
        if not file_name or not output_dir:
            QMessageBox.warning(self, "Thiếu thông tin", "Không tìm thấy tên file hoặc thư mục đầu ra.")
            return

        obj_path = getattr(self.sidebar, 'current_obj_path', None)
        if obj_path and os.path.exists(obj_path):
            pass
        else:
    
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            possible_paths = []
            singleview_dir = os.path.join(project_root, 'model', 'PiFu-singleview')
            singleview_results = os.path.join(singleview_dir, 'results', 'pifuhd_final', 'recon')
            possible_paths.extend([
                os.path.join(singleview_results, '*.obj'),
                os.path.join(singleview_dir, 'results', '*.obj'),
                os.path.join(singleview_dir, '*.obj')
            ])
            multiview_dir = os.path.join(project_root, 'model', 'PIFu-multiview')
            multiview_results = os.path.join(multiview_dir, 'results', 'pifu_demo', 'rp_Man')
            possible_paths.extend([
                os.path.join(multiview_results, '*.obj'),
                os.path.join(multiview_dir, '*.obj')
            ])
            for pattern in possible_paths:
                matching_files = glob.glob(pattern)
                if matching_files:
                    obj_path = matching_files[0]
                    break
        if not obj_path or not os.path.exists(obj_path):
            QMessageBox.warning(self, "Không tìm thấy mô hình", "Bạn cần tạo mô hình trước khi tải xuống.")
            return
        if not file_name.lower().endswith('.obj'):
            file_name += '.obj'
        dest_path = os.path.join(output_dir, file_name)
        try:
            shutil.copyfile(obj_path, dest_path)
            QMessageBox.information(self, "Thành công", f"Đã tải xuống mô hình 3D thành công:\n{dest_path}\n\nNguồn: {obj_path}")
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Không thể tải xuống file: {str(e)}")