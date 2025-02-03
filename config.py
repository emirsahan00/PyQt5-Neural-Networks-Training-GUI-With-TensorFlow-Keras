class Config:
    WINDOW_STYLE = """
        QWidget {
            background-color: #1e1e1e;
            font-family: Arial, sans-serif;
            color: #ffffff;
        }
        QLabel {
            font-size: 14px;
            color: #ffffff;
        }
        QLineEdit, QComboBox, QSpinBox {
            background-color: #2d2d2d;
            border: 1px solid #3c3c3c;
            border-radius: 5px;
            padding: 5px;
            color: #ffffff;
        }
        QPushButton {
            background-color: #3c3c3c;
            color: #ffffff;
            border: none;
            border-radius: 5px;
            padding: 10px;
            font-size: 14px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #5a5a5a;
        }
        QFrame {
            background-color: #2d2d2d;
            border: 1px solid #3c3c3c;
            border-radius: 10px;
            padding: 15px;
        }
    """
    
    PROGRESS_WINDOW_STYLE = """
        QDialog {
            background-color: #1e1e1e;
            color: #ffffff;
        }
        QLabel {
            color: #ffffff;
            font-size: 14px;
        }
        QProgressBar {
            border: 2px solid grey;
            border-radius: 5px;
            text-align: center;
            color: white;
        }
        QProgressBar::chunk {
            background-color: #3498db;
            width: 10px;
        }
    """
