import os
import sys
from PyQt5.QtGui import QFontDatabase, QFont
from PyQt5.QtCore import QDir

def setup_resources(app):
    """Setup application resources and styles"""
    resources_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'resources')
    icons_dir = os.path.join(resources_dir, 'icons')
    fonts_dir = os.path.join(resources_dir, 'fonts')
    
    for directory in [resources_dir, icons_dir, fonts_dir]:
        os.makedirs(directory, exist_ok=True)
    
    create_basic_icons(icons_dir)
    
    style_path = os.path.join(resources_dir, 'style.qss')
    create_stylesheet(style_path)
    
    with open(style_path, 'r') as file:
        app.setStyleSheet(file.read())
    
    app.setFont(QFont("Segoe UI", 10))

def create_basic_icons(icons_dir):
    """Create basic SVG icon files"""
    upload_icon = '''
    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
      <polyline points="17 8 12 3 7 8"></polyline>
      <line x1="12" y1="3" x2="12" y2="15"></line>
    </svg>
    '''
    
    magic_icon = '''
    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <path d="M15 4V2"></path>
      <path d="M15 16v-2"></path>
      <path d="M8 9h2"></path>
      <path d="M20 9h2"></path>
      <path d="M17.8 11.8L19 13"></path>
      <path d="M15 9h0"></path>
      <path d="M17.8 6.2L19 5"></path>
      <path d="M3 21l9-9"></path>
      <path d="M12.2 6.2L11 5"></path>
    </svg>
    '''
    
    moon_icon = '''
    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>
    </svg>
    '''
    
    sun_icon = '''
    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <circle cx="12" cy="12" r="5"></circle>
      <line x1="12" y1="1" x2="12" y2="3"></line>
      <line x1="12" y1="21" x2="12" y2="23"></line>
      <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line>
      <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line>
      <line x1="1" y1="12" x2="3" y2="12"></line>
      <line x1="21" y1="12" x2="23" y2="12"></line>
      <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line>
      <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line>
    </svg>
    '''
    
    icons = {
        'upload.svg': upload_icon,
        'magic.svg': magic_icon,
        'moon.svg': moon_icon,
        'sun.svg': sun_icon
    }
    
    for filename, content in icons.items():
        with open(os.path.join(icons_dir, filename), 'w') as file:
            file.write(content)

def create_stylesheet(style_path):
    """Create application stylesheet"""
    stylesheet = '''
    /* Global styles */
    QWidget {
        font-family: "Segoe UI", "Roboto", sans-serif;
        color: #e0e0e0;
    }
    
    QMainWindow {
        background-color: #1a1a1a;
    }
    
    /* Button styles */
    QPushButton {
        background-color: #333333;
        border: none;
        border-radius: 4px;
        padding: 8px 16px;
        color: #e0e0e0;
        font-weight: 500;
    }
    
    QPushButton:hover {
        background-color: #444444;
    }
    
    QPushButton:pressed {
        background-color: #2a2a2a;
    }
    
    /* Input field styles */
    QLineEdit {
        background-color: #333333;
        border: 1px solid #444444;
        border-radius: 4px;
        padding: 8px;
        color: #ffffff;
        selection-background-color: #4a9fff;
    }
    
    QLineEdit:focus {
        border: 1px solid #4a9fff;
    }
    
    /* Combo box styles */
    QComboBox {
        background-color: #333333;
        border: 1px solid #444444;
        border-radius: 4px;
        padding: 8px;
        color: #ffffff;
        min-height: 20px;
    }
    
    QComboBox::drop-down {
        subcontrol-origin: padding;
        subcontrol-position: top right;
        width: 20px;
        border-left-width: 1px;
        border-left-color: #444444;
        border-left-style: solid;
        border-top-right-radius: 4px;
        border-bottom-right-radius: 4px;
    }
    
    QComboBox::down-arrow {
        image: url(resources/icons/down-arrow.png);
    }
    
    QComboBox QAbstractItemView {
        background-color: #333333;
        border: 1px solid #444444;
        selection-background-color: #4a9fff;
    }
    
    /* Checkbox styles */
    QCheckBox {
        spacing: 8px;
    }
    
    QCheckBox::indicator {
        width: 18px;
        height: 18px;
        border-radius: 3px;
        border: 1px solid #444444;
    }
    
    QCheckBox::indicator:unchecked {
        background-color: #333333;
    }
    
    QCheckBox::indicator:checked {
        background-color: #4a9fff;
    }
    
    /* Slider styles */
    QSlider::groove:horizontal {
        border: none;
        height: 6px;
        background-color: #333333;
        border-radius: 3px;
    }
    
    QSlider::handle:horizontal {
        background-color: #8ACA39;
        border: none;
        width: 18px;
        height: 18px;
        margin: -6px 0;
        border-radius: 9px;
    }
    
    QSlider::handle:horizontal:hover {
        background-color: #9BD94A;
    }
    
    /* Dialog styles */
    QDialog {
        background-color: #232323;
    }
    
    QDialog QLabel {
        color: #e0e0e0;
    }
    
    /* File dialog styles */
    QFileDialog {
        background-color: #232323;
    }
    
    QFileDialog QTreeView, QFileDialog QListView {
        background-color: #2a2a2a;
        border: 1px solid #333333;
    }
    
    QFileDialog QTreeView::item:selected, QFileDialog QListView::item:selected {
        background-color: #4a9fff;
    }
    
    /* Menu styles */
    QMenu {
        background-color: #2a2a2a;
        border: 1px solid #333333;
    }
    
    QMenu::item {
        padding: 6px 25px 6px 20px;
    }
    
    QMenu::item:selected {
        background-color: #4a9fff;
    }
    
    /* Tooltip styles */
    QToolTip {
        background-color: #333333;
        border: 1px solid #444444;
        color: #e0e0e0;
        padding: 4px;
    }
    
    /* Scroll bar styles */
    QScrollBar:vertical {
        border: none;
        background-color: #2a2a2a;
        width: 12px;
        margin: 0px;
    }
    
    QScrollBar::handle:vertical {
        background-color: #444444;
        min-height: 20px;
        border-radius: 6px;
    }
    
    QScrollBar::handle:vertical:hover {
        background-color: #555555;
    }
    
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
        border: none;
        background: none;
    }
    
    QScrollBar:horizontal {
        border: none;
        background-color: #2a2a2a;
        height: 12px;
        margin: 0px;
    }
    
    QScrollBar::handle:horizontal {
        background-color: #444444;
        min-width: 20px;
        border-radius: 6px;
    }
    
    QScrollBar::handle:horizontal:hover {
        background-color: #555555;
    }
    
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
        border: none;
        background: none;
    }
    '''
    
    with open(style_path, 'w') as file:
        file.write(stylesheet)