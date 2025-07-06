from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QLabel, 
                            QProgressBar, QWidget, QApplication, QSizePolicy)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QMovie, QPainter, QColor

class LoadingDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Creating 3D Model")
        self.setFixedSize(400, 250)
        self.setWindowFlags(Qt.Dialog | Qt.CustomizeWindowHint | Qt.WindowTitleHint)
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
       
        self.title_label = QLabel("Creating Your 3D Model")
        self.title_label.setStyleSheet("""
            QLabel {
                color: #ffffff;
                font-size: 18px;
                font-weight: bold;
            }
        """)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setWordWrap(True)
        self.title_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        layout.addWidget(self.title_label)
        

        self.spinner_label = QLabel()
        self.spinner = QMovie("resources/icons/spinner.gif")
        self.spinner.setScaledSize(QSize(64, 64))
        self.spinner_label.setMovie(self.spinner)
        self.spinner.start()
        layout.addWidget(self.spinner_label, alignment=Qt.AlignCenter)
        
       
        self.status_label = QLabel("Initializing...")
        self.status_label.setStyleSheet("""
            QLabel {
                color: #e0e0e0;
                font-size: 14px;
            }
        """)
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setWordWrap(True)
        self.status_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        layout.addWidget(self.status_label)
        
       
        self.step_label = QLabel("Step 0/3")
        self.step_label.setStyleSheet("""
            QLabel {
                color: #8ACA39;
                font-size: 12px;
                font-weight: bold;
            }
        """)
        self.step_label.setAlignment(Qt.AlignCenter)
        self.step_label.setWordWrap(True)
        self.step_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        layout.addWidget(self.step_label)
        
      
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #444444;
                border-radius: 5px;
                text-align: center;
                background-color: #2A2A2A;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #8ACA39;
                border-radius: 3px;
            }
        """)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        layout.addWidget(self.progress_bar)
        
    
        self.setStyleSheet("""
            QDialog {
                background-color: #232323;
                border: 1px solid #444444;
                border-radius: 10px;
            }
        """)
        
      
        self.current_step = 0
        self.total_steps = 3
        
    def update_step(self, step_number, status_text):
        """Update the current step and status"""
        self.current_step = step_number
        self.step_label.setText(f"Step {step_number}/{self.total_steps}")
        self.status_label.setText(status_text)
        self.progress_bar.setValue(int((step_number / self.total_steps) * 100))
        QApplication.processEvents()  

    def complete(self):
        """Show completion state"""
        self.title_label.setText("3D Model Created Successfully!")
        self.status_label.setText("Your 3D model is ready!")
        self.step_label.setText("Completed")
        self.progress_bar.setValue(100)
        self.spinner.stop()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #444444;
                border-radius: 5px;
                text-align: center;
                background-color: #2A2A2A;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #8ACA39;
                border-radius: 3px;
            }
        """)
        QApplication.processEvents()  

    def error(self, message):
        """Show error state"""
        self.title_label.setText("Error Occurred")
        self.status_label.setText(message)
        self.spinner.stop()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #444444;
                border-radius: 5px;
                text-align: center;
                background-color: #2A2A2A;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #FF4444;
                border-radius: 3px;
            }
        """)
        QApplication.processEvents()