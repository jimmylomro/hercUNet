"""A big rounded colour swatch with a centred 'sheet <id>' label — the prominent hover/selection
read-out. Splits into two side-by-side cells when two sheets are selected."""

from __future__ import annotations

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget


class SheetSwatch(QWidget):
    """Shows a sheet's colour as a big rounded square with 'sheet <id>' centred on it. Transparent
    with 'hover a sheet' when idle; grey 'noise' for the noise cluster. When two sheets are selected
    the (same-sized) box splits horizontally into two cells, one per sheet."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(168, 88)
        self._colour = None                                      # QColor filled, or None = transparent (idle)
        self._text = "hover a sheet"
        self._pair = None                                        # [(cid, QColor), (cid, QColor)] when two selected

    def set_sheet(self, cid: int, hexcol: str) -> None:
        self._pair = None
        self._colour = QColor(hexcol) if hexcol else QColor("#8a8f99")
        self._text = f"sheet {cid}"
        self.update()

    def set_pair(self, cid1: int, hex1: str, cid2: int, hex2: str) -> None:
        self._colour = None
        self._pair = [(cid1, QColor(hex1) if hex1 else QColor("#8a8f99")),
                      (cid2, QColor(hex2) if hex2 else QColor("#8a8f99"))]
        self.update()

    def set_noise(self) -> None:
        self._pair = None
        self._colour = QColor("#8a8f99")
        self._text = "noise"
        self.update()

    def clear(self) -> None:
        self._pair = None
        self._colour = None
        self._text = "hover a sheet"
        self.update()

    def _draw_cell(self, p: QPainter, rect: QRectF, colour: QColor, text: str) -> None:
        p.setPen(QPen(QColor(0, 0, 0, 60), 1.5))
        p.setBrush(colour)
        p.drawRoundedRect(rect, 10, 10)
        lum = 0.299 * colour.red() + 0.587 * colour.green() + 0.114 * colour.blue()
        p.setPen(QColor("#101216") if lum > 140 else QColor("#f5f6f8"))
        f = p.font()
        f.setPointSize(12)
        f.setBold(True)
        p.setFont(f)
        p.drawText(rect, Qt.AlignCenter, text)

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect().adjusted(3, 3, -3, -3))
        if self._pair is not None:                               # two selected → split horizontally, box unchanged
            gap = 4.0
            w = (r.width() - gap) / 2.0
            left = QRectF(r.left(), r.top(), w, r.height())
            right = QRectF(r.left() + w + gap, r.top(), r.width() - w - gap, r.height())
            for rect, (cid, col) in zip((left, right), self._pair):
                self._draw_cell(p, rect, col, f"sheet {cid}")
            return
        if self._colour is None:                                 # idle placeholder
            p.setPen(QPen(QColor("#4a4f59"), 1.5, Qt.DashLine))
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(r, 14, 14)
            p.setPen(QColor("#9095a0"))
            f = p.font()
            f.setPointSize(14)
            f.setBold(True)
            p.setFont(f)
            p.drawText(r, Qt.AlignCenter, self._text)
            return
        p.setPen(QPen(QColor(0, 0, 0, 60), 1.5))                 # single sheet
        p.setBrush(self._colour)
        p.drawRoundedRect(r, 14, 14)
        lum = 0.299 * self._colour.red() + 0.587 * self._colour.green() + 0.114 * self._colour.blue()
        p.setPen(QColor("#101216") if lum > 140 else QColor("#f5f6f8"))
        f = p.font()
        f.setPointSize(14)
        f.setBold(True)
        p.setFont(f)
        p.drawText(r, Qt.AlignCenter, self._text)
