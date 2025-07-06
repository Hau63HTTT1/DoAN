from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap

from ui.model_viewer import ModelViewer

class PreviewArea(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(20, 20, 20, 20)
        self.layout.setSpacing(0)
        
     
        self.model_container = QWidget()
        model_layout = QVBoxLayout(self.model_container)
        model_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create message label
        self.message_label = QLabel("Chọn một mô hình để chỉnh sửa")
        self.message_label.setAlignment(Qt.AlignCenter)
        self.message_label.setStyleSheet("""
            color: rgba(255, 255, 255, 0.5);
            font-size: 16px;
            font-weight: 400;
        """)
        
     
        self.model_viewer = ModelViewer()
        self.model_viewer.hide()
        
        model_layout.addWidget(self.message_label)
        model_layout.addWidget(self.model_viewer)
        
      
        self.layout.addWidget(self.model_container)
        
        self.current_model_path = None
      
        self.setStyleSheet("""
            QWidget {
                background-color: #2A2A2A;
            }
            QLabel {
                color: #FFFFFF;
            }
        """)

    def load_3d_model(self, file_path):
        if file_path:
            self.current_model_path = file_path
            self.message_label.hide()
            self.model_viewer.show()
            self.model_viewer.load_model(file_path)

    def delete_model(self):
        if self.current_model_path:
            self.model_viewer.clear_model()
            self.model_viewer.hide()
            self.message_label.show()
            self.current_model_path = None
            
        
            if hasattr(self, 'current_image'):
                self.current_image = None
                self.update()

    def download_model(self):
        if self.current_model_path:
            pass

def validate_obj_file(file_path):
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        print(f"Total lines in OBJ file: {len(lines)}")
        print("First 10 lines:")
        for i, line in enumerate(lines[:10]):
            print(f"{i+1}: {line.strip()}")
    except Exception as e:
        print(f"Error reading OBJ file: {str(e)}")