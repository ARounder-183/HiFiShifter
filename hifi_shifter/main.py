import sys
from PyQt6.QtWidgets import QApplication
from .main_window import HifiShifterGUI

def main():
    app = QApplication(sys.argv)
    window = HifiShifterGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
