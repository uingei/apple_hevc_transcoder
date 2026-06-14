# main.py
import sys
from PySide6.QtWidgets import QApplication
from gui.mainwindow import MainWindow


def main():
    """Console script entry point (for pyproject.toml [project.scripts])"""
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
