"""A styled on/off toggle switch — an iOS-style sliding pill, not a flat checkbox."""

from __future__ import annotations

from PySide6.QtCore import Property, QEasingCurve, QPropertyAnimation, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QAbstractButton, QHBoxLayout, QLabel, QWidget

_OFF = QColor("#565b66")
_ON = QColor("#3b82f6")
_KNOB = QColor("#f5f6f8")


class ToggleSwitch(QAbstractButton):
    """Checkable pill switch with a sliding knob and an animated track-colour blend."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setFixedSize(46, 26)
        self.setCursor(Qt.PointingHandCursor)
        self._p = 0.0
        self._anim = QPropertyAnimation(self, b"knob", self)
        self._anim.setDuration(140)
        self._anim.setEasingCurve(QEasingCurve.InOutCubic)
        self.toggled.connect(self._animate)

    def _animate(self, on: bool) -> None:
        self._anim.stop()
        self._anim.setStartValue(self._p)
        self._anim.setEndValue(1.0 if on else 0.0)
        self._anim.start()

    def getKnob(self) -> float:
        return self._p

    def setKnob(self, v: float) -> None:
        self._p = float(v)
        self.update()

    knob = Property(float, getKnob, setKnob)

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        track = QColor(int(_OFF.red() + (_ON.red() - _OFF.red()) * self._p),
                       int(_OFF.green() + (_ON.green() - _OFF.green()) * self._p),
                       int(_OFF.blue() + (_ON.blue() - _OFF.blue()) * self._p))
        p.setPen(Qt.NoPen)
        p.setBrush(track)
        p.drawRoundedRect(QRectF(0, 0, w, h), h / 2, h / 2)
        m = 3.0
        d = h - 2 * m
        kx = m + self._p * (w - 2 * m - d)
        p.setBrush(_KNOB)
        p.drawEllipse(QRectF(kx, m, d, d))


class SegmentedToggle(QWidget):
    """A multi-position segmented switch (a rounded pill split into labelled segments, the active one
    highlighted). Used for the 3-way overlay selector: streamlets · none · confidence. Emits
    :attr:`changed` (the active index) when the selection changes."""

    changed = Signal(int)

    def __init__(self, labels, current: int = 0, seg_width: int = 82, parent=None):
        super().__init__(parent)
        self._labels = list(labels)
        self._current = int(current)
        self._seg = int(seg_width)
        self._m = 3.0
        self.setFixedHeight(26)
        self.setFixedWidth(int(self._seg * len(self._labels) + 2 * self._m))
        self.setCursor(Qt.PointingHandCursor)

    def currentIndex(self) -> int:
        return self._current

    def setCurrentIndex(self, i: int) -> None:
        i = max(0, min(len(self._labels) - 1, int(i)))
        if i != self._current:
            self._current = i
            self.update()
            self.changed.emit(i)

    def mousePressEvent(self, ev) -> None:
        i = int((ev.position().x() - self._m) // self._seg)
        self.setCurrentIndex(i)

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h, m = self.width(), self.height(), self._m
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#2b2f38"))
        p.drawRoundedRect(QRectF(0, 0, w, h), h / 2, h / 2)       # track
        seg = (w - 2 * m) / len(self._labels)
        p.setBrush(_ON)                                          # active-segment pill
        p.drawRoundedRect(QRectF(m + self._current * seg, m, seg, h - 2 * m),
                          (h - 2 * m) / 2, (h - 2 * m) / 2)
        f = p.font()
        f.setPointSize(9)
        f.setBold(True)
        p.setFont(f)
        for i, lab in enumerate(self._labels):
            p.setPen(QColor("#ffffff") if i == self._current else QColor("#9aa0ab"))
            p.drawText(QRectF(m + i * seg, 0, seg, h), Qt.AlignCenter, lab)


class LabelledToggle(QWidget):
    """A :class:`ToggleSwitch` with a caption; exposes ``toggle`` and forwards ``toggled``."""

    def __init__(self, text: str, checked: bool = False, parent=None):
        super().__init__(parent)
        self.toggle = ToggleSwitch()
        self.toggle.setChecked(checked)
        self.toggle.setKnob(1.0 if checked else 0.0)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        label = QLabel(text)
        label.setStyleSheet("color:#c9ccd2; font-size:12px;")
        lay.addWidget(self.toggle)
        lay.addWidget(label)
        self.toggled = self.toggle.toggled
