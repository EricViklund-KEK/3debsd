from PySide6.QtWidgets import QWidget, QVBoxLayout, QPlainTextEdit
from PySide6.QtCore import Qt
import logging

class QTextEditLogger(logging.Handler):
    def __init__(self, parent):
        super().__init__()
        self.widget = QPlainTextEdit(parent)
        self.widget.setReadOnly(True)
        self.widget.setMaximumHeight(120)
        
        # Create a format for the logs
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.setFormatter(formatter)

    def emit(self, record):
        msg = self.format(record)
        self.widget.appendPlainText(msg)
        # Ensure the latest log message is visible
        self.widget.verticalScrollBar().setValue(
            self.widget.verticalScrollBar().maximum()
        )

class LogWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        
        # Create and add the log handler
        self.log_handler = QTextEditLogger(self)
        layout.addWidget(self.log_handler.widget)
        
        # Remove margins to make it more compact
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Configure the root logger
        logging.getLogger().addHandler(self.log_handler)
        logging.getLogger().setLevel(logging.INFO) 