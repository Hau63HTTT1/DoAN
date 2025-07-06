from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                            QLabel, QLineEdit, QComboBox, QSlider, QCheckBox,
                            QRadioButton, QButtonGroup, QFrame, QSpacerItem,
                            QSizePolicy, QFileDialog, QMessageBox, QDialog, QProgressBar,
                            QInputDialog,QGridLayout)
from PyQt5.QtCore import Qt, QSize, pyqtSignal, QPropertyAnimation, QEasingCurve, QRect, QTimer
from PyQt5.QtGui import QIcon, QFont, QColor, QPainter, QPainterPath, QLinearGradient, QMovie
import os
import subprocess
import sys
from PIL import Image

from ui.custom_widgets import (GradientButton, ToggleSwitch, InfoButton, 
                              AnimatedUploadArea, CustomComboBox, GlowEffect)
from ui.loading_dialog import LoadingDialog
from model.process_upload import image_processor

class Sidebar(QWidget):
 
    upload_button_clicked = pyqtSignal()
    create_button_clicked = pyqtSignal(str) 
    model_uploaded = pyqtSignal(str) 
    download_button_clicked = pyqtSignal()  
    delete_button_clicked = pyqtSignal()   
    image_processed = pyqtSignal(str, str)  
    
    def __init__(self):
        super().__init__()
        self.setObjectName("sidebar")
        self.setMinimumWidth(300)
        self.setMaximumWidth(350)
        self.current_image_path = None
        self.current_rect_path = None
        self.init_ui()
        
    def init_ui(self):
       
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(16, 16, 16, 16)
        self.main_layout.setSpacing(8)  
        
        
        self.setup_new_template_section()
        
        
        self.add_separator()
        
       
        self.setup_image_type_section()
        
       
        self.setup_image_upload_section()
        
       
        self.setup_name_section()
        
      
        self.setup_ai_model_section()
        
      
        self.setup_animation_section()
        
       
        self.main_layout.addSpacerItem(QSpacerItem(0, 10, QSizePolicy.Minimum, QSizePolicy.Expanding))
        
       
        self.setup_create_button()
        
       
        self.upload_button.clicked.connect(self.open_model_file_dialog)
    
        self.create_button.setObjectName("create_button")
        self.upload_area.setObjectName("upload_area")
        self.image_button.setObjectName("image_button")
        

        self.setStyleSheet("""
            #sidebar {
                background-color: #23272f;
                border-right: 1px solid #333333;
            }
            QLabel {
                color: #e0e0e0;
                font-size: 13px;
            }
            QLabel#section_header {
                font-size: 15px;
                font-weight: bold;
                color: #aee571;
                margin-bottom: 4px;
                letter-spacing: 0.5px;
            }
            QLineEdit {
                background-color: #262b34;
                border: 1.5px solid #3a3f4b;
                border-radius: 8px;
                padding: 0 16px;
                color: #ffffff;
                font-size: 16px;
                margin-bottom: 8px;
                min-height: 44px;
                max-height: 44px;
                transition: border 0.2s;
            }
            QLineEdit:focus {
                border: 1.5px solid #8ACA39;
                background-color: #23272f;
            }
            QPushButton {
                padding: 10px;
                font-size: 14px;
                border-radius: 8px;
                font-weight: 500;
                transition: background 0.2s;
            }
            QPushButton#image_button, QPushButton:checked {
                background-color: #8ACA39;
                color: #232323;
                font-weight: bold;
            }
            QPushButton#image_button:hover, QPushButton:checked:hover {
                background-color: #aee571;
            }
            QPushButton#create_button {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #8ACA39, stop:1 #ffb199);
                color: #232323;
                font-weight: bold;
                border: none;
                margin-top: 10px;
            }
            QPushButton#create_button:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #aee571, stop:1 #ffb199);
                color: #232323;
            }
            QFrame {
                background-color: transparent;
                border: none;
                border-bottom: 1px solid #353b45;
                margin: 10px 0;
            }
            QWidget#upload_area {
                border: 2px dashed #8ACA39;
                border-radius: 10px;
                background-color: #23272f;
                margin-bottom: 10px;
            }
        """)
        
    def setup_new_template_section(self):
        """Setup the new template section"""
      
        section = QWidget()
        section_layout = QVBoxLayout(section)
        section_layout.setContentsMargins(0, 0, 0, 0)  
        section_layout.setSpacing(6)
        
        
        label = QLabel("Mẫu mới")
        label.setObjectName("section_header")
        section_layout.addWidget(label)
        

        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(8)
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        
   
        self.upload_button = QPushButton("  Tải lên")
        self.upload_button.setIcon(QIcon("resources/icons/upload.svg"))
        self.upload_button.setFixedHeight(36)
        self.upload_button.setCursor(Qt.PointingHandCursor)
        self.upload_button.setStyleSheet("""
            QPushButton {
                background-color: #8ACA39;
                color: #000000;
                border-radius: 4px;
                font-weight: bold;
                padding-left: 15px;
                text-align: left;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #9BD94A;
            }
            QPushButton:pressed {
                background-color: #7AB929;
            }
        """)
        self.upload_button.clicked.connect(self.upload_button_clicked.emit)
        buttons_layout.addWidget(self.upload_button)
        
 
        self.download_button = QPushButton("  Tải xuống")
        self.download_button.setIcon(QIcon("resources/icons/download.svg"))
        self.download_button.setFixedHeight(36)
        self.download_button.setCursor(Qt.PointingHandCursor)
        self.download_button.setStyleSheet("""
            QPushButton {
                background-color: #2D2D2D;
                color: #FFFFFF;
                border-radius: 4px;
                font-weight: bold;
                padding-left: 15px;
                text-align: left;
                font-size: 13px;
                border: 1px solid #444444;
            }
            QPushButton:hover {
                background-color: #006400;
                border: 1px solid #008000;
            }
            QPushButton:pressed {
                background-color: #004D00;
            }
            QPushButton:disabled {
                background-color: #1D1D1D;
                color: #666666;
                border: 1px solid #333333;
            }
        """)
        try:
            self.download_button.clicked.disconnect(self.download_button_clicked.emit)
        except Exception:
            pass
        self.download_button.clicked.connect(self.on_download_clicked)
        self.download_button.setEnabled(False)
        buttons_layout.addWidget(self.download_button)
        
      
        self.delete_button = QPushButton("  Xóa")
        self.delete_button.setIcon(QIcon("resources/icons/delete.svg"))
        self.delete_button.setFixedHeight(36)
        self.delete_button.setCursor(Qt.PointingHandCursor)
        self.delete_button.setStyleSheet("""
            QPushButton {
                background-color: #2D2D2D;
                color: #FFFFFF;
                border-radius: 4px;
                font-weight: bold;
                padding-left: 15px;
                text-align: left;
                font-size: 13px;
                border: 1px solid #444444;
            }
            QPushButton:hover {
                background-color: #8B0000;
                border: 1px solid #A00000;
            }
            QPushButton:pressed {
                background-color: #6B0000;
            }
            QPushButton:disabled {
                background-color: #1D1D1D;
                color: #666666;
                border: 1px solid #333333;
            }
        """)
        self.delete_button.clicked.connect(self.on_delete_clicked)
        self.delete_button.setEnabled(False)  # Initially disabled
        buttons_layout.addWidget(self.delete_button)
        
        section_layout.addLayout(buttons_layout)
        self.main_layout.addWidget(section)
    
    def setup_image_type_section(self):

        section = QWidget()
        section_layout = QVBoxLayout(section)
        section_layout.setContentsMargins(0, 0, 0, 8)
        section_layout.setSpacing(6)
        

        type_layout = QHBoxLayout()
        type_layout.setSpacing(6)
        type_layout.setContentsMargins(0, 0, 0, 0)
        
        button_style = """
            QPushButton {
                background-color: #333333;
                color: #e0e0e0;
                border-radius: 4px;
                padding: 8px;
                font-size: 13px;
                font-weight: normal;
            }
            QPushButton:hover {
                background-color: #444444;
            }
            QPushButton:checked {
                background-color: #8ACA39;
                color: #000000;
                font-weight: bold;
            }
        """
   
        self.image_button = QPushButton("Hình ảnh thành 3D")
        self.image_button.setFixedHeight(32)
        self.image_button.setStyleSheet(button_style)
        self.image_button.setCheckable(True)
        self.image_button.setChecked(True)
        
        self.multi_image_button = QPushButton("Đa hình ảnh")
        self.multi_image_button.setFixedHeight(32)
        self.multi_image_button.setStyleSheet(button_style)
        self.multi_image_button.setCheckable(True)
        self.multi_image_button.setChecked(False)
        type_layout.addWidget(self.image_button)
        type_layout.addWidget(self.multi_image_button)
        section_layout.addLayout(type_layout)
        
        self.main_layout.addWidget(section)
        self.image_button.clicked.connect(lambda: self.multi_image_button.setChecked(False))
        self.multi_image_button.clicked.connect(lambda: self.image_button.setChecked(False))
        
    def setup_image_upload_section(self):
        self.upload_section = QWidget()
        self.upload_section_layout = QVBoxLayout(self.upload_section)
        self.upload_section_layout.setContentsMargins(0, 0, 0, 8)
        self.upload_section_layout.setSpacing(6)

        self.upload_area = AnimatedUploadArea()
        self.upload_area.setMinimumHeight(120)
        self.upload_area.setStyleSheet("""
            QWidget {
                border: 2px dashed #444444;
                border-radius: 4px;
                background-color: #2A2A2A;
            }
        """)
        self.upload_area.clicked.connect(self.open_image_dialog)
        self.upload_section_layout.addWidget(self.upload_area)

        self.multi_upload_widget = QWidget()
        multi_grid = QGridLayout(self.multi_upload_widget)
        multi_grid.setSpacing(8)
        multi_grid.setContentsMargins(0, 0, 0, 0)
        self.multi_image_areas = []
        for i in range(4):
            area = AnimatedUploadArea()
            area.setMinimumHeight(100)
            area.setMaximumHeight(120)
            area.title_label.setStyleSheet("""
                color: #ffffff;
                font-size: 12px;
                font-weight: bold;
                background-color: transparent;
                border: none;
                padding: 0px;
            """)
            area.icon_label.hide()
            area.title_label.setAlignment(Qt.AlignCenter)
            area.title_label.setWordWrap(True)
            area.subtitle_label.hide()
            area.support_label.hide()
            area.size_label.hide()
            area.clicked.connect(lambda checked=False, idx=i: self.select_multi_image(idx))
            multi_grid.addWidget(area, i // 2, i % 2)
            self.multi_image_areas.append(area)
        self.multi_upload_widget.setVisible(False)
        self.upload_section_layout.addWidget(self.multi_upload_widget)

        self.multi_upload_button = QPushButton("Tải lên nhiều ảnh")
        self.multi_upload_button.setFixedHeight(36)
        self.multi_upload_button.setStyleSheet("""
            QPushButton {
                background-color: #8ACA39;
                color: #000000;
                border-radius: 4px;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #9BD94A;
            }
        """)
        self.multi_upload_button.setVisible(False)
        self.upload_section_layout.addWidget(self.multi_upload_button)

        self.main_layout.addWidget(self.upload_section)
        self.add_separator()

        self.image_button.clicked.connect(self.show_single_upload)
        self.multi_image_button.clicked.connect(self.show_multi_upload)
        self.multi_upload_button.clicked.connect(self.check_multi_images)

        # self.upload_area.clicked.connect(self.open_image_dialog)
    
    def select_multi_image(self, idx):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Chọn ảnh",
            "",
            "Image Files (*.png *.jpg *.jpeg *.webp);;All Files (*.*)"
        )
        if file_path:
            self.multi_image_areas[idx].set_image(file_path)

    def show_single_upload(self):
        self.upload_area.setVisible(True)
        self.multi_upload_widget.setVisible(False)
        self.multi_upload_button.setVisible(False)

    def show_multi_upload(self):
        self.upload_area.setVisible(False)
        self.multi_upload_widget.setVisible(True)
        self.multi_upload_button.setVisible(True)
    def check_multi_images(self):
        import os
        save_dir = os.path.join("model", "PIFu-multiview", "sample_images", "rp_Man")
        os.makedirs(save_dir, exist_ok=True)
        names = ["0_0_00.png", "90_0_00.png", "180_0_00.png", "270_0_00.png"]
        mask_names = ["0_0_00_mask.png", "90_0_00_mask.png", "180_0_00_mask.png", "270_0_00_mask.png"]
        image_paths = []

        for i, area in enumerate(self.multi_image_areas):
            if not area.current_image:
                QMessageBox.warning(self, "Thiếu ảnh", f"Vui lòng chọn đủ 4 ảnh (thiếu ở vị trí {i+1})!")
                return
            with Image.open(area.current_image) as img:
                img = img.convert("RGBA")
                img = resize_and_pad(img, (512, 512))
                save_path = os.path.join(save_dir, names[i])
                img.save(save_path)
                image_paths.append(save_path)

        for i, img_path in enumerate(image_paths):
            mask_path = os.path.join(save_dir, mask_names[i])
            subprocess.run([
                sys.executable, "model/PIFu-multiview/mask.py",
                "--input", img_path,
                "--output", mask_path
            ])

        QMessageBox.information(self, "Thành công", "Upload ảnh thành công!")
    def setup_name_section(self):
        section = QWidget()
        section_layout = QVBoxLayout(section)
        section_layout.setContentsMargins(0, 0, 0, 8)
        section_layout.setSpacing(6)
        
        name_layout = QHBoxLayout()
        name_layout.setSpacing(6)
        name_layout.setContentsMargins(0, 0, 0, 0)
        name_label = QLabel("Tên")
        name_label.setObjectName("section_header")
        name_layout.addWidget(name_label)
        name_layout.addStretch()
        section_layout.addLayout(name_layout)
        
        self.name_input = QLineEdit()
        self.name_input.setFixedHeight(44)
        self.name_input.setPlaceholderText("Đặt tên cho sản phẩm của bạn")
        section_layout.addWidget(self.name_input)
        
        self.main_layout.addWidget(section)
        self.add_separator()
    
    def setup_ai_model_section(self):
        section = QWidget()
        section_layout = QVBoxLayout(section)
        section_layout.setContentsMargins(0, 0, 0, 8)
        section_layout.setSpacing(6)
        
        ai_layout = QHBoxLayout()
        ai_layout.setSpacing(6)
        ai_layout.setContentsMargins(0, 0, 0, 0)
        ai_label = QLabel("Mô hình AI")
        ai_label.setObjectName("section_header")
        
        model_info = """Thông tin chi tiết về các mô hình:

• PiFu: Tối ưu cho các nhân vật hoạt hình cartoon, cho kết quả tốt với các mô hình đơn giản, phong cách hoạt hình cơ bản."""
        self.ai_info = InfoButton(model_info)
        ai_layout.addWidget(ai_label)
        ai_layout.addStretch()
        ai_layout.addWidget(self.ai_info)
        section_layout.addLayout(ai_layout)
        
        self.ai_model_label = QLabel("PiFu")
        self.ai_model_label.setAlignment(Qt.AlignCenter)
        self.ai_model_label.setFixedWidth(200)
        self.ai_model_label.setStyleSheet("""
            background-color: #262b34;
            border: 1.5px solid #3a3f4b;
            border-radius: 8px;
            padding: 0 16px;
            color: #ffffff;
            font-size: 16px;
            min-height: 30px;
            max-height: 30px;
        """)


        selector_container = QHBoxLayout()
        selector_container.setContentsMargins(0, 0, 0, 0)
        selector_container.setSpacing(0)
        selector_container.addStretch()
        selector_container.addWidget(self.ai_model_label)
        selector_container.addStretch()
        
        section_layout.addLayout(selector_container)
        
        self.main_layout.addWidget(section)
        self.add_separator()
    
    def setup_animation_section(self):
        section = QWidget()
        section_layout = QVBoxLayout(section)
        section_layout.setContentsMargins(0, 0, 0, 8)
        section_layout.setSpacing(6)
        
        animation_layout = QHBoxLayout()
        animation_layout.setSpacing(6)
        animation_layout.setContentsMargins(0, 0, 0, 0)
        animation_label = QLabel("Đường dẫn đầu ra")
        animation_label.setObjectName("section_header")
        animation_layout.addWidget(animation_label)
        animation_layout.addStretch()
        section_layout.addLayout(animation_layout)
        
        self.output_path = QLineEdit()
        self.output_path.setFixedHeight(44)
        self.output_path.setPlaceholderText("Nhập đường dẫn đến thư mục đầu ra...")
        self.output_path.setStyleSheet("""
            QLineEdit {
                background-color: #333333;
                border: 1px solid #444444;
                border-radius: 4px;
                padding: 0 10px;
                color: #ffffff;
                font-size: 13px;
            }
            QLineEdit:focus {
                border: 1px solid #4a9fff;
            }
            QLineEdit::placeholder {
                color: #888888;
            }
        """)
        section_layout.addWidget(self.output_path)
        
        self.main_layout.addWidget(section)
        self.add_separator()
    
    
    def setup_create_button(self):
        """Setup the create button"""
        self.create_button = GradientButton("Tạo ra")
        self.create_button.setFixedHeight(40)
        self.create_button.clicked.connect(self.on_create_clicked)
        self.main_layout.addWidget(self.create_button)
        
    def add_separator(self):
        """Add a separator line"""
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setStyleSheet("""
            background-color: transparent;
        """)
        separator.setFixedHeight(16)
        separator.setContentsMargins(0, 4, 0, 4)
        self.main_layout.addWidget(separator)

    def open_model_file_dialog(self):
        """Open file dialog for 3D model selection"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open 3D Model",
            "",
            "3D Models (*.obj *.stl *.ply *.fbx);;All Files (*.*)"
        )
        
        if file_path:
            self.model_uploaded.emit(file_path)

    def on_create_clicked(self):
        """Handle create button click"""
        loading_dialog = None 
        if self.multi_image_button.isChecked():
            try:
                import time
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                detect_dir = os.path.join(project_root, 'model', 'PIFu-multiview')
                detect_py = os.path.join(detect_dir, 'detect.py')
                loading_dialog = LoadingDialog(self)
                loading_dialog.show()
                loading_dialog.update_step(1, "Đang tạo mô hình 3D từ đa hình ảnh...")

                process = subprocess.Popen([
                    sys.executable, detect_py
                ], cwd=detect_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                stdout, stderr = process.communicate()
                loading_dialog.complete()
                time.sleep(1.5)
                loading_dialog.close()
                output_dir = os.path.join(detect_dir, 'results','pifu_demo','rp_Man')
                found_obj = None

                if os.path.exists(output_dir):
                    for root, dirs, files in os.walk(output_dir):
                        for f in files:
                            if f.endswith('.obj'):
                                found_obj = os.path.join(root, f)
                                break
                        if found_obj:
                            break

                if not found_obj and os.path.exists(output_dir):
                    print(f"Debug: Checking directory {output_dir}")
                    for root, dirs, files in os.walk(output_dir):
                        print(f"Debug: {root} -> {files}")
                
                if process.returncode != 0:
                    QMessageBox.critical(self, "Lỗi", f"Tạo mô hình 3D thất bại:\n{stderr}")
                elif found_obj:
                    QMessageBox.information(self, "Thành công", f"Đã tạo mô hình 3D thành công!")
                    self.create_button_clicked.emit(found_obj)
                else:
                    QMessageBox.warning(self, "Cảnh báo", f"Không tìm thấy file mô hình 3D (.obj) trong thư mục output!\nĐã kiểm tra: {output_dir}")
            except Exception as e:
                QMessageBox.critical(self, "Lỗi", f"Lỗi khi tạo: {str(e)}")
            return

        if not self.current_image_path or not self.current_rect_path:
            QMessageBox.warning(
                self,
                "Warning",
                "Please upload and process an image first!"
            )
            return

        try:

            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

            pifuhd_dir = os.path.join(project_root, 'model', 'PiFu-singleview')
            sample_images_dir = os.path.join(pifuhd_dir, 'sample_images')
            results_dir = os.path.join(pifuhd_dir, 'results')
            final_output_dir = os.path.join(results_dir, 'pifuhd_final', 'recon')

            print(f"Debug: Project root: {project_root}")
            print(f"Debug: PiFu-singleview dir: {pifuhd_dir}")
            print(f"Debug: Sample images dir: {sample_images_dir}")
            print(f"Debug: Results dir: {results_dir}")
            print(f"Debug: Final output dir: {final_output_dir}")
            
 
            if not os.path.exists(pifuhd_dir):
                raise Exception(f"PiFu-singleview directory not found: {pifuhd_dir}")
            if not os.path.exists(sample_images_dir):
                print(f"Debug: Creating sample_images directory: {sample_images_dir}")
            if not os.path.exists(results_dir):
                print(f"Debug: Creating results directory: {results_dir}")
            if not os.path.exists(final_output_dir):
                print(f"Debug: Creating final_output directory: {final_output_dir}")

            os.makedirs(sample_images_dir, exist_ok=True)
            os.makedirs(results_dir, exist_ok=True)
            os.makedirs(final_output_dir, exist_ok=True)
            
            loading_dialog = LoadingDialog(self)
            loading_dialog.show()

            loading_dialog.update_step(1, "Preparing environment...")

            current_pid = os.getpid()
            
            try:
                import psutil
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    try:
                        if proc.pid == current_pid:
                            continue
                        if proc.name().lower().startswith('python'):
                            cmdline = proc.cmdline()
                            if any('pifuhd' in cmd.lower() for cmd in cmdline):
                                proc.kill()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
            except Exception as e:
                print(f"Warning: Could not clean up processes: {e}")
            
            try:
                os.chdir(pifuhd_dir)

                os.environ['CUDA_VISIBLE_DEVICES'] = '0'
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
                

                loading_dialog.update_step(2, "Creating 3D model...")

                cmd = [
                    sys.executable,
                    '-m', 'apps.simple_test',
                    '-r', '256',
                    '--use_rect',
                    '-i', sample_images_dir,
                    '-o', results_dir,
                    '-c', os.path.join('checkpoints', 'pifuhd.pt')
                ]

                print(f"Debug: Running command: {' '.join(cmd)}")
                print(f"Debug: Working directory: {os.getcwd()}")
                print(f"Debug: Sample images dir: {sample_images_dir}")
                print(f"Debug: Results dir: {results_dir}")

                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                stdout, stderr = process.communicate()
                

                print(f"Debug: PiFuHD stdout: {stdout}")
                print(f"Debug: PiFuHD stderr: {stderr}")
                print(f"Debug: PiFuHD return code: {process.returncode}")
                
                if process.returncode != 0:
                    loading_dialog.error("Failed to create 3D model")
                    raise Exception(f"PiFu-singleview failed: {stderr}")

                loading_dialog.update_step(3, "Optimizing 3D model...")
                clean_cmd = [
                    sys.executable,
                    'apps/clean_mesh.py',
                    '-f', os.path.join(results_dir, 'pifuhd_final', 'recon')
                ]

                print(f"Debug: Running clean mesh command: {' '.join(clean_cmd)}")
                
                process = subprocess.Popen(
                    clean_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                stdout, stderr = process.communicate()
                
                print(f"Debug: Clean mesh stdout: {stdout}")
                print(f"Debug: Clean mesh stderr: {stderr}")
                print(f"Debug: Clean mesh return code: {process.returncode}")
                
                if process.returncode != 0:
                    loading_dialog.error("Failed to optimize 3D model")
                    raise Exception(f"Clean mesh failed: {stderr}")
                
                image_name = os.path.basename(self.current_image_path)
                image_name_no_ext = os.path.splitext(image_name)[0]

                possible_obj_files = [
                    os.path.join(final_output_dir, f'result_{image_name_no_ext}_256.obj'),
                    os.path.join(final_output_dir, f'result_{image_name_no_ext}_128.obj'),
                    os.path.join(final_output_dir, f'result_{image_name_no_ext}.obj'),
                    os.path.join(results_dir, 'pifuhd_final', 'recon', f'result_{image_name_no_ext}_256.obj'),
                    os.path.join(results_dir, 'pifuhd_final', 'recon', f'result_{image_name_no_ext}_128.obj'),
                    os.path.join(results_dir, 'pifuhd_final', 'recon', f'result_{image_name_no_ext}.obj'),
                 
                    os.path.join(final_output_dir, f'{image_name_no_ext}_256.obj'),
                    os.path.join(final_output_dir, f'{image_name_no_ext}_128.obj'),
                    os.path.join(final_output_dir, f'{image_name_no_ext}.obj'),
                    os.path.join(results_dir, 'pifuhd_final', 'recon', f'{image_name_no_ext}_256.obj'),
                    os.path.join(results_dir, 'pifuhd_final', 'recon', f'{image_name_no_ext}_128.obj'),
                    os.path.join(results_dir, 'pifuhd_final', 'recon', f'{image_name_no_ext}.obj'),
               
                    os.path.join(final_output_dir, 'result_*_256.obj'),
                    os.path.join(results_dir, 'pifuhd_final', 'recon', 'result_*_256.obj'),
                ]
                
                obj_file = None
                
              
                for root, dirs, files in os.walk(results_dir):
                    for file in files:
                        if file.endswith('.obj'):
                            obj_file = os.path.join(root, file)
                            print(f"Debug: Found obj file: {obj_file}")
                            break
                    if obj_file:
                        break
                
               
                if not obj_file:
                    import glob
                    for possible_file in possible_obj_files:
                     
                        if '*' in possible_file:
                            try:
                                matching_files = glob.glob(possible_file)
                                if matching_files:
                                    obj_file = matching_files[0]  
                                    print(f"Debug: Found obj file with pattern matching: {obj_file}")
                                    break
                            except Exception as e:
                                print(f"Debug: Pattern matching failed for {possible_file}: {e}")
                        elif os.path.exists(possible_file):
                            obj_file = possible_file
                            print(f"Debug: Found obj file in possible locations: {obj_file}")
                            break
                
            
                if not obj_file:
                    pifuhd_root = os.path.dirname(results_dir)
                    print(f"Debug: Searching entire PiFu-singleview directory: {pifuhd_root}")
                    for root, dirs, files in os.walk(pifuhd_root):
                        for file in files:
                            if file.endswith('.obj'):
                                obj_file = os.path.join(root, file)
                                print(f"Debug: Found obj file in PiFu-singleview directory: {obj_file}")
                                break
                        if obj_file:
                            break
                
        
                if not obj_file:
                    recon_dir = os.path.join(results_dir, 'pifuhd_final', 'recon')
                    if os.path.exists(recon_dir):
                        print(f"Debug: Searching recon directory: {recon_dir}")
                        for file in os.listdir(recon_dir):
                            if file.endswith('.obj'):
                                obj_file = os.path.join(recon_dir, file)
                                print(f"Debug: Found obj file in recon directory: {obj_file}")
                                break
                
                if obj_file and os.path.exists(obj_file):
                    try:
                      
                        with open(obj_file, 'rb') as f:
                            f.read(1)
                        
                        
                        loading_dialog.complete()
                        import time
                        time.sleep(1.5)  
                        loading_dialog.close()
                        
                        self.create_button_clicked.emit(obj_file)
                        
                        QMessageBox.information(
                            self,
                            "Success",
                            f"Tạo mô hình 3D thành công!"
                        )
                    except Exception as e:
                        loading_dialog.error("File access error")
                        raise Exception(f"Cannot access output file: {str(e)}")
                else:
                    
                    print(f"Debug: Checking directory {final_output_dir}")
                    if os.path.exists(final_output_dir):
                        print(f"Debug: Files in {final_output_dir}: {os.listdir(final_output_dir)}")
                    
                    print(f"Debug: Checking directory {results_dir}")
                    if os.path.exists(results_dir):
                        for root, dirs, files in os.walk(results_dir):
                            print(f"Debug: {root} -> {files}")
                    
                    loading_dialog.error("No 3D model files found")
                    raise Exception(f"No .obj files found in output directories. Checked: {final_output_dir}, {results_dir}")
                
            except Exception as e:
                raise e
            finally:
          
                os.chdir(project_root)
                
        except Exception as e:
            if 'loading_dialog' in locals():
                loading_dialog.error(str(e))
                import time
                time.sleep(2)  
                loading_dialog.close()
            
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to create 3D model: {str(e)}"
            )

    def open_image_dialog(self):
        """Open file dialog for image selection"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            "",
            "Image Files (*.png *.jpg *.jpeg *.webp);;All Files (*.*)"
        )
        
        if file_path:
            try:
               
                if self.upload_area.set_image(file_path):
                   
                    image_path, rect_path = image_processor.process_image(file_path)
                    
                
                    self.current_image_path = image_path
                    self.current_rect_path = rect_path
                    
                   
                    self.image_processed.emit(image_path, rect_path)
                    
                
                    QMessageBox.information(
                        self,
                        "Success",
                        f"Upload ảnh thành công!"
                    )
                    
                    
                    self.enable_model_actions(True)
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error",
                    f"Failed to process image: {str(e)}"
                )

    def enable_model_actions(self, enable=True):
        self.download_button.setEnabled(enable)
        self.delete_button.setEnabled(enable)

    def on_download_clicked(self):
        name = self.name_input.text().strip() if hasattr(self, 'name_input') else ''
        output_dir = self.output_path.text().strip() if hasattr(self, 'output_path') else ''
        if not name or not output_dir:
            QMessageBox.warning(self, "Thiếu thông tin", "Bạn phải nhập đầy đủ Tên và Đường dẫn đầu ra trước khi tải xuống.")
            return
        self.selected_download_name = name
        self.selected_download_dir = output_dir
        self.download_button_clicked.emit()

    def on_delete_clicked(self):
        reply = QMessageBox.question(
            self,
            "Xác nhận xóa",
            "Bạn có chắc muốn xóa không?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            QMessageBox.information(self, "Thành công", "Xóa thành công!")
            self.delete_button_clicked.emit()

            if self.multi_image_button.isChecked():
                for area in self.multi_image_areas:
                    area.clear_image()

            if self.image_button.isChecked():
                self.upload_area.clear_image()

def resize_and_pad(img, size=(512, 512), color=(0, 0, 0, 0)):
        img.thumbnail(size, Image.LANCZOS)
        new_img = Image.new("RGBA", size, color)
        left = (size[0] - img.width) // 2
        top = (size[1] - img.height) // 2
        new_img.paste(img, (left, top))
        return new_img