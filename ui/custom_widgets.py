from PyQt5.QtWidgets import (QPushButton, QWidget, QVBoxLayout, QHBoxLayout,
                            QLabel, QSlider, QGraphicsOpacityEffect, QGraphicsDropShadowEffect,
                            QGraphicsEffect)
from PyQt5.QtCore import (Qt, QSize, QPropertyAnimation, QEasingCurve, 
                         QRect, QPoint, pyqtSignal, pyqtProperty, QTimer, QRectF)
from PyQt5.QtGui import (QIcon, QFont, QColor, QPainter, QPen, QBrush, 
                        QLinearGradient, QPainterPath, QPixmap, QImage)
from PyQt5.QtWidgets import QApplication

class GradientButton(QPushButton):
    """Button with gradient background and hover animations"""
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setCursor(Qt.PointingHandCursor)
        self._background_position = 0
        self._animation = QPropertyAnimation(self, b"background_position")
        self._animation.setDuration(300)
        self._animation.setEasingCurve(QEasingCurve.OutCubic)
        
        self.setStyleSheet("""
            QPushButton {
                border: none;
                border-radius: 6px;
                color: #ffffff;
                font-weight: bold;
                font-size: 16px;
                padding-left: 15px;
                text-align: left;
            }
        """)
    
    def enterEvent(self, event):
        self._animation.setStartValue(self._background_position)
        self._animation.setEndValue(100)
        self._animation.start()
        super().enterEvent(event)
    
    def leaveEvent(self, event):
        self._animation.setStartValue(self._background_position)
        self._animation.setEndValue(0)
        self._animation.start()
        super().leaveEvent(event)
    
    def get_background_position(self):
        return self._background_position
    
    def set_background_position(self, position):
        self._background_position = position
        self.update()
    
    background_position = pyqtProperty(int, get_background_position, set_background_position)
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
       
        gradient = QLinearGradient(0, 0, self.width(), 0)
        gradient.setColorAt(0, QColor(211, 255, 124))  
        gradient.setColorAt(1, QColor(255, 166, 158)) 
        
      
        offset = self._background_position / 100.0
        gradient.setColorAt(0.5 + offset * 0.5, QColor(138, 202, 57))  
        
       
        path = QPainterPath()
        path.addRoundedRect(QRectF(0, 0, self.width(), self.height()), 6, 6)
        painter.fillPath(path, QBrush(gradient))
        
       
        painter.setPen(QColor(255, 255, 255))
        
        
        if not self.icon().isNull():
            icon_size = self.iconSize()
            icon_rect = QRect(15, (self.height() - icon_size.height()) // 2, 
                             icon_size.width(), icon_size.height())
            self.icon().paint(painter, icon_rect)
            
            font = painter.font()
            font.setBold(True)
            painter.setFont(font)
            
            text_rect = self.rect().adjusted(icon_size.width() + 25, 0, -10, 0)
            painter.drawText(text_rect, Qt.AlignVCenter | Qt.AlignLeft, self.text())
        else:
            font = painter.font()
            font.setBold(True)
            painter.setFont(font)
            
            text_rect = self.rect().adjusted(15, 0, -10, 0)
            painter.drawText(text_rect, Qt.AlignVCenter | Qt.AlignLeft, self.text())

class ToggleSwitch(QWidget):
    """Custom toggle switch widget"""
    toggled = pyqtSignal(bool)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(50, 26)
        self._checked = True
        self._thumb_position = 26 
        
        
        self._animation = QPropertyAnimation(self, b"thumb_position")
        self._animation.setDuration(200)
        self._animation.setEasingCurve(QEasingCurve.InOutQuad) 
        
    
        self.setCursor(Qt.PointingHandCursor)
    
    def get_thumb_position(self):
        return self._thumb_position
    
    def set_thumb_position(self, position):
        self._thumb_position = position
        self.update()
    
    thumb_position = pyqtProperty(float, get_thumb_position, set_thumb_position)
    
    def is_checked(self):
        return self._checked
    
    def set_checked(self, checked):
        if self._checked != checked:
            self._checked = checked
            self._animation.setStartValue(self._thumb_position)
            if checked:
                self._animation.setEndValue(26)  
            else:
                self._animation.setEndValue(6)
            self._animation.start()
            self.toggled.emit(checked)
    
    checked = pyqtProperty(bool, is_checked, set_checked)
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.set_checked(not self._checked)
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
       
        path = QPainterPath()
        path.addRoundedRect(0, 0, self.width(), self.height(), 13, 13)
        
        if self._checked:
            bg_color = QColor(138, 202, 57)  
        else:
            bg_color = QColor(80, 80, 80)
        
        painter.fillPath(path, bg_color)
        
        
        thumb_rect = QRectF(self._thumb_position - 10, 3, 20, 20)
        thumb_path = QPainterPath()
        thumb_path.addEllipse(thumb_rect)
        
        
        shadow_offset = 1.5  
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(0, 0, 0, 30))  
        painter.drawEllipse(thumb_rect.adjusted(shadow_offset, shadow_offset, shadow_offset, shadow_offset))
        
        
        gradient = QLinearGradient(
            thumb_rect.topLeft(),
            thumb_rect.bottomLeft()
        )
        gradient.setColorAt(0, QColor(255, 255, 255))  
        gradient.setColorAt(1, QColor(240, 240, 240)) 
        
        painter.setBrush(gradient)
        painter.setPen(QPen(QColor(220, 220, 220), 0.5)) 
        painter.drawPath(thumb_path)

class InfoButton(QPushButton):
    """Custom info button with hover effect and popup"""
    def __init__(self, info_text="", parent=None):
        super().__init__(parent)
        self.setFixedSize(20, 20)
        self.setCursor(Qt.PointingHandCursor)
        self.info_text = info_text
        

        self.setup_popup()
        
        # Setup timer for delayed hiding
        self.popup_timer = QTimer(self)
        self.popup_timer.setSingleShot(True)
        self.popup_timer.timeout.connect(self.hide_popup)
        
    def setup_popup(self):
        """Setup the popup widget"""
        self.popup = QWidget(None, Qt.ToolTip | Qt.FramelessWindowHint)
        layout = QVBoxLayout(self.popup)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8) 
        
        
        text_label = QLabel(self.info_text)
        text_label.setWordWrap(True)
        text_label.setStyleSheet("""
            QLabel {
                color: #ffffff;
                font-size: 13px;
                line-height: 1.4;
                background-color: transparent;
                padding: 0px;
            }
        """)
        
        layout.addWidget(text_label)
        
        self.popup.setStyleSheet("""
            QWidget {
                background-color: #333333;
                border: 1px solid #444444;
                border-radius: 4px;
            }
        """)
        
        self.popup.setFixedWidth(300)
        
    def show_popup(self):
        """Show info popup"""
        if not self.popup.isVisible():
            pos = self.mapToGlobal(QPoint(0, self.height() + 5))
            
            screen = QApplication.primaryScreen().geometry()
            if pos.x() + self.popup.width() > screen.right():
                pos.setX(screen.right() - self.popup.width())
            if pos.y() + self.popup.height() > screen.bottom():
                pos.setY(self.mapToGlobal(QPoint(0, -self.popup.height() - 5)).y())
            
            self.popup.move(pos)
            self.popup.show()
        
        if self.popup_timer.isActive():
            self.popup_timer.stop()
    
    def hide_popup(self):
        """Hide the popup"""
        if self.popup and self.popup.isVisible():
            self.popup.hide()
    
    def enterEvent(self, event):
        """Show popup when mouse enters the button"""
        self.show_popup()
        super().enterEvent(event)
    
    def leaveEvent(self, event):
        """Start timer to hide popup when mouse leaves the button"""
        self.popup_timer.start(300) 
        super().leaveEvent(event)
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        

        if self.underMouse():
            bg_color = QColor(138, 202, 57)  
        else:
            bg_color = QColor(100, 100, 100) 
        
       
        painter.setPen(Qt.NoPen)
        painter.setBrush(bg_color)
        painter.drawEllipse(0, 0, 20, 20)
        
       
        painter.setPen(QColor(255, 255, 255))
        font = painter.font()
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(QRect(0, 0, 20, 20), Qt.AlignCenter, "i")

class AnimatedUploadArea(QWidget):
    """Custom animated upload area widget"""
    clicked = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(200)
        self.setCursor(Qt.PointingHandCursor)
        
        # Setup animation
        self._highlight = 0
        self._animation = QPropertyAnimation(self, b"highlight")
        self._animation.setDuration(1500)
        self._animation.setLoopCount(-1)
        self._animation.setStartValue(0)
        self._animation.setEndValue(100)
        self._animation.setEasingCurve(QEasingCurve.InOutQuad)
        self._animation.start()
        
        # Create container widget for content
        self.content_widget = QWidget()
        self.content_widget.setStyleSheet("""
            QWidget {
                background-color: transparent;
            }
        """)
        
        
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(10, 10, 10, 10)
        self.content_layout.setSpacing(4)
        self.content_layout.setAlignment(Qt.AlignCenter)
        
        
        self.default_widget = QWidget()
        default_layout = QVBoxLayout(self.default_widget)
        default_layout.setContentsMargins(0, 0, 0, 0)
        default_layout.setSpacing(4)
        default_layout.setAlignment(Qt.AlignCenter)
        
        
        self.icon_label = QLabel()
        self.icon_label.setFixedSize(32, 32)
        self.icon_label.setStyleSheet("background-color: transparent;")
        
      
        self.title_label = QLabel("Nhấn / Kéo & Thả")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("""
            color: #ffffff;
            font-size: 14px;
            font-weight: bold;
            background-color: transparent;
            border: none;
            padding: 0px;
        """)
        
        self.subtitle_label = QLabel("Dán Hình ảnh")
        self.subtitle_label.setAlignment(Qt.AlignCenter)
        self.subtitle_label.setStyleSheet("""
            color: #aaaaaa;
            font-size: 13px;
            background-color: transparent;
            border: none;
            padding: 0px;
        """)
        
        self.support_label = QLabel("Các định dạng được hỗ trợ: .png, .jpg, .jpeg, .webp")
        self.support_label.setAlignment(Qt.AlignCenter)
        self.support_label.setWordWrap(True)
        self.support_label.setStyleSheet("""
            color: #777777;
            font-size: 12px;
            background-color: transparent;
            border: none;
            padding: 0px;
        """)
        
        self.size_label = QLabel("Kích thước tối đa: 20MB")
        self.size_label.setAlignment(Qt.AlignCenter)
        self.size_label.setStyleSheet("""
            color: #777777;
            font-size: 12px;
            background-color: transparent;
            border: none;
            padding: 0px;
        """)
        
        
        default_layout.addWidget(self.icon_label)
        default_layout.addWidget(self.title_label)
        default_layout.addWidget(self.subtitle_label)
        default_layout.addSpacing(4)
        default_layout.addWidget(self.support_label)
        default_layout.addWidget(self.size_label)
        
       
        self.image_widget = QWidget()
        self.image_widget.hide()
        image_layout = QVBoxLayout(self.image_widget)
        image_layout.setContentsMargins(0, 0, 0, 0)
        image_layout.setSpacing(0)
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("""
            QLabel {
                background-color: transparent;
                border-radius: 4px;
            }
        """)
        
        image_layout.addWidget(self.image_label)
        
      
        self.content_layout.addWidget(self.default_widget)
        self.content_layout.addWidget(self.image_widget)
        
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        main_layout.addWidget(self.content_widget)
        
       
        self.setStyleSheet("""
            AnimatedUploadArea {
                background-color: transparent;
                border-radius: 4px;
            }
        """)
        

        self.current_image = None
        
    def set_image(self, image_path):
        """Set and display an image"""
        try:
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                
                scaled_pixmap = pixmap.scaled(
                    self.width() - 40, 
                    self.height() - 40,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                
                self.image_label.setPixmap(scaled_pixmap)
                self.current_image = image_path
                
               
                self.default_widget.hide()
                self.image_widget.show()
                return True
        except Exception as e:
            print(f"Error loading image: {e}")
        return False
        
    def clear_image(self):
        """Clear the current image"""
        self.image_label.clear()
        self.current_image = None
        self.image_widget.hide()
        self.default_widget.show()
        
    def resizeEvent(self, event):
        """Handle resize events"""
        super().resizeEvent(event)
        if self.current_image:
            self.set_image(self.current_image)  
            
    def get_highlight(self):
        return self._highlight
    
    def set_highlight(self, value):
        self._highlight = value
        self.update()
    
    highlight = pyqtProperty(float, get_highlight, set_highlight)
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        
        highlight_value = abs(self._highlight - 50) / 50.0  
        highlight_color = QColor(
            int(80 + 40 * highlight_value),  
            int(80 + 122 * highlight_value), 
            int(80 + 20 * highlight_value)    
        )
        

        pen = QPen(highlight_color)
        pen.setStyle(Qt.DashLine)
        pen.setWidth(2)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(4, 4, self.width() - 8, self.height() - 8, 4, 4)
        

        icon_rect = self.icon_label.geometry()
        upload_icon = QPixmap(32, 32)
        upload_icon.fill(Qt.transparent)
        
        icon_painter = QPainter(upload_icon)
        icon_painter.setRenderHint(QPainter.Antialiasing)
        
       
        icon_painter.setPen(QPen(highlight_color, 2))
        icon_painter.drawLine(16, 8, 16, 24)  
        icon_painter.drawLine(10, 14, 16, 8)  
        icon_painter.drawLine(22, 14, 16, 8)  
        
        icon_painter.end()
        
        self.icon_label.setPixmap(upload_icon)

class CustomComboBox(QWidget):
    """Custom styled combo box with animated dropdown"""
    selectionChanged = pyqtSignal(int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(40)
        
     
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 0, 10, 0)
        
  
        self.text_label = QLabel()
        self.text_label.setStyleSheet("""
            color: #ffffff;
            font-size: 14px;
        """)
       
        self.arrow_label = QLabel()
        self.arrow_label.setFixedSize(16, 16)
        
       
        layout.addWidget(self.text_label)
        layout.addStretch()
        layout.addWidget(self.arrow_label)
        
        
        self.popup_menu = None
        self.items = []
        self.current_index = 0
        self._animation = None
        
       
        self.setCursor(Qt.PointingHandCursor)
        
        
        self.setObjectName("custom_combo")
        self.setStyleSheet("""
            #custom_combo {
                background-color: #333333;
                border: 1px solid #444444;
                border-radius: 4px;
            }
            #custom_combo:hover {
                background-color: #3a3a3a;
                border: 1px solid #555555;
            }
        """)
    
    def addItem(self, text):
        """Add an item to the combo box"""
        self.items.append(text)
        
        if len(self.items) == 1:
            self.current_index = 0
            self.text_label.setText(text)
        
        self.update_arrow()
    
    def setCurrentIndex(self, index):
        """Set the current index"""
        if 0 <= index < len(self.items):
            self.current_index = index
            self.text_label.setText(self.items[index])
            self.selectionChanged.emit(index)
    
    def update_arrow(self):
        """Update the arrow indicator"""
        arrow = QPixmap(16, 16)
        arrow.fill(Qt.transparent)
        
        painter = QPainter(arrow)
        painter.setRenderHint(QPainter.Antialiasing)
        
       
        painter.setPen(QPen(QColor(200, 200, 200), 2))
        painter.drawLine(4, 6, 8, 10)
        painter.drawLine(8, 10, 12, 6)
        
        painter.end()
        
        self.arrow_label.setPixmap(arrow)
    
    def mousePressEvent(self, event):
        """Handle mouse press event"""
        if event.button() == Qt.LeftButton:
            
            from PyQt5.QtWidgets import QMenu, QAction
            
            menu = QMenu(self)
            menu.setWindowFlags(menu.windowFlags() | Qt.NoDropShadowWindowHint)
            menu.setAttribute(Qt.WA_TranslucentBackground)
            menu.setStyleSheet("""
                QMenu {
                    background-color: #333333;
                    border: 1px solid #444444;
                    border-radius: 4px;
                    padding: 5px;
                    margin-top: 1px;
                }
                QMenu::item {
                    background-color: transparent;
                    padding: 8px 20px;
                    border-radius: 3px;
                    color: #ffffff;
                    font-size: 13px;
                }
                QMenu::item:selected {
                    background-color: #4a9fff;
                }
            """)
            
            
            for i, item in enumerate(self.items):
                action = QAction(item, self)
                action.setData(i)
                menu.addAction(action)
            
            
            menu.triggered.connect(self.on_action_triggered)
            
            # Setup animation
            self._animation = QPropertyAnimation(menu, b"geometry", self)
            self._animation.setDuration(150)
            self._animation.setEasingCurve(QEasingCurve.OutQuad)
            
           
            pos = self.mapToGlobal(QPoint(0, self.height()))
            target_geometry = QRect(pos.x(), pos.y(), self.width(), len(self.items) * 40)
            start_geometry = QRect(pos.x(), pos.y(), self.width(), 0)
            
            
            menu.setGeometry(start_geometry)
            menu.show()
            
          
            self._animation.setStartValue(start_geometry)
            self._animation.setEndValue(target_geometry)
            self._animation.start()
            
           
            menu.exec_(pos)
    
    def on_action_triggered(self, action):
        """Handle action triggered"""
        index = action.data()
        self.setCurrentIndex(index)

class GridBackground(QWidget):
    """Grid background widget with perspective effect"""
    def __init__(self, parent=None):
        super().__init__(parent)
        
        
        self._layout = QVBoxLayout(self)
        self._layout.setAlignment(Qt.AlignCenter)
        
        
        self._offset_x = 0
        self._offset_y = 0
        self._animation_x = QPropertyAnimation(self, b"offset_x")
        self._animation_x.setDuration(60000) 
        self._animation_x.setStartValue(0)
        self._animation_x.setEndValue(40)  
        self._animation_x.setLoopCount(-1) 
        
        self._animation_y = QPropertyAnimation(self, b"offset_y")
        self._animation_y.setDuration(80000) 
        self._animation_y.setStartValue(0)
        self._animation_y.setEndValue(40)  
        self._animation_y.setLoopCount(-1) 
        
       
        self._animation_x.start()
        self._animation_y.start()
    
    def layout(self):
        """Return the layout"""
        return self._layout
    
    def get_offset_x(self):
        return self._offset_x
    
    def set_offset_x(self, value):
        self._offset_x = value
        self.update()
    
    offset_x = pyqtProperty(float, get_offset_x, set_offset_x)
    
    def get_offset_y(self):
        return self._offset_y
    
    def set_offset_y(self, value):
        self._offset_y = value
        self.update()
    
    offset_y = pyqtProperty(float, get_offset_y, set_offset_y)
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
       
        painter.fillRect(self.rect(), QColor(26, 26, 26))
        
      
        pen = QPen(QColor(50, 50, 50))
        pen.setWidth(1)
        painter.setPen(pen)
        
        
        cell_size = 40
        
      
        horizon = self.height() * 0.5  
        vanishing_x = self.width() * 0.5
        vanishing_y = horizon
        

        for i in range(-20, int(self.height() / cell_size) + 20):
            y = i * cell_size + self._offset_y % cell_size
            
            
            y_perspective = (y - vanishing_y) * 0.2 + vanishing_y
            
            painter.drawLine(0, y_perspective, self.width(), y_perspective)
        
        for i in range(-20, int(self.width() / cell_size) + 20):
            x = i * cell_size + self._offset_x % cell_size
            
            
            x_perspective = (x - vanishing_x) * 0.2 + vanishing_x
            
            painter.drawLine(x_perspective, 0, x_perspective, self.height())

class GlowEffect(QGraphicsEffect):
    """Custom glow effect for widgets"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._radius = 10
        self._color = QColor(138, 202, 57, 150) 
    
    def setRadius(self, radius):
        self._radius = radius
        self.update()
    
    def setColor(self, color):
        self._color = color
        self.update()
    
    def draw(self, painter):
        
        source = self.sourcePixmap()
        
       
        result = QPixmap(source.size() + QSize(self._radius * 2, self._radius * 2))
        result.fill(Qt.transparent)
        
        
        temp_painter = QPainter(result)
        temp_painter.setRenderHint(QPainter.Antialiasing)
        
        
        for i in range(self._radius, 0, -2):
            temp_color = QColor(self._color)
            temp_color.setAlpha(self._color.alpha() * i / self._radius)
            
            temp_painter.setPen(QPen(temp_color, i))
            temp_painter.setBrush(Qt.NoBrush)
            temp_painter.drawRoundedRect(
                self._radius - i/2, self._radius - i/2,
                source.width() + i, source.height() + i,
                5, 5
            )
        
       
        temp_painter.drawPixmap(self._radius, self._radius, source)
        temp_painter.end()
        
      
        painter.drawPixmap(
            -self._radius, -self._radius,
            result
        )