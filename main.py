import sys
from PyQt5.QtWidgets import QApplication
from ui.main_window import MainWindow
from utils.resource_loader import setup_resources

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
   
    setup_resources(app)
    
 
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())