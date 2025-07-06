from PyQt5.QtWidgets import QGraphicsOpacityEffect
from PyQt5.QtCore import QPropertyAnimation, QEasingCurve

class FadeTransition:
    def __init__(self, target_widget, duration=300):
        self.target = target_widget
        self.duration = int(duration)
        
        self.opacity_effect = QGraphicsOpacityEffect(self.target)
        self.target.setGraphicsEffect(self.opacity_effect)
        self.opacity_effect.setOpacity(1.0)
        
        self.animation = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.animation.setDuration(self.duration)
        self.animation.setEasingCurve(QEasingCurve.InOutCubic)
        
    def start(self):
        original_opacity = self.opacity_effect.opacity()
        
        self.animation.setStartValue(original_opacity)
        self.animation.setEndValue(0.3)
        self.animation.setDuration(int(self.duration / 2))
        
        try:
            self.animation.finished.disconnect(self._fade_in)
        except:
            pass
        self.animation.finished.connect(self._fade_in)
        
        self.animation.start()
        
    def _fade_in(self):
        try:
            self.animation.finished.disconnect(self._fade_in)
        except:
            pass
        
        self.animation.setStartValue(self.opacity_effect.opacity())
        self.animation.setEndValue(1.0)
        self.animation.setDuration(int(self.duration / 2))
        
        self.animation.start() 