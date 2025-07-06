from PyQt5.QtWidgets import (QWidget, QHBoxLayout, QPushButton, QLabel)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QIcon, QFont

class HeaderBar(QWidget):
    theme_toggled = pyqtSignal(bool)
    
    def __init__(self):
        super().__init__()
        self.setFixedHeight(50)
        self.setObjectName("header_bar")
        

        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(16, 0, 16, 0)
        

        self.title = QLabel("Image Editor")
        self.title.setStyleSheet("""
            font-size: 18px;
            font-weight: bold;
            color: #ffffff;
        """)
        
 
        self.layout.addWidget(self.title)
        self.layout.addStretch()
        
        self.setStyleSheet("""
            #header_bar {
                background-color: #232323;
                border-bottom: 1px solid #333333;
            }
        """)
        
    def toggle_theme(self):

        if self.theme_button.property("is_dark") or self.theme_button.property("is_dark") is None:
            self.theme_button.setIcon(QIcon("resources/icons/sun.svg"))
            self.theme_button.setProperty("is_dark", False)
            self.theme_toggled.emit(False)
        else:
            self.theme_button.setIcon(QIcon("resources/icons/moon.svg"))
            self.theme_button.setProperty("is_dark", True)
            self.theme_toggled.emit(True)
