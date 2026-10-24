"""Full-area translucent loading overlay for the interactive viewer."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QProgressBar, QVBoxLayout, QWidget


class LoadingOverlay(QWidget):
    """A semi-opaque panel that covers its parent while the window loads. Tracks the parent's size
    (call :meth:`cover` from the parent's ``resizeEvent``) and shows a message + a busy bar."""

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet("background: rgba(16,18,22,205);")
        lay = QVBoxLayout(self)
        lay.setAlignment(Qt.AlignCenter)
        self._label = QLabel("Loading…")
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setStyleSheet("color:#e8eaed; font-size:16px; background:transparent;")
        bar = QProgressBar()
        bar.setRange(0, 0)                                        # indeterminate (busy) bar
        bar.setFixedWidth(280)
        bar.setTextVisible(False)
        lay.addWidget(self._label)
        lay.addSpacing(12)
        lay.addWidget(bar, alignment=Qt.AlignCenter)
        self.cover()

    def set_message(self, text: str) -> None:
        self._label.setText(text)

    def cover(self) -> None:
        if self.parent() is not None:
            self.setGeometry(self.parent().rect())

    def show_over(self, message: str | None = None) -> None:
        if message:
            self.set_message(message)
        self.cover()
        self.raise_()
        self.show()
