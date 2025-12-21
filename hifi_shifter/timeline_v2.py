import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QBrush, QPen
from PyQt6.QtWidgets import QGraphicsItem, QGraphicsRectItem, QGraphicsScene, QGraphicsView, QVBoxLayout, QWidget

class TrackPanel(QWidget):
    trackSelected = pyqtSignal(int)  # Signal for track selection
    # Signal for file drop
    filesDropped = pyqtSignal(list)
    # Signal for cursor movement
    cursorMoved = pyqtSignal(float)
    # Signal for track type changes
    trackTypeChanged = pyqtSignal(int, str)

    # Placeholder for ruler_plot to avoid AttributeError
    ruler_plot = None

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("轨道面板")
        self.layout = QVBoxLayout(self)

        # Create a QGraphicsView to display tracks
        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene, self)
        self.layout.addWidget(self.view)

        # Initialize ruler_plot as a valid object
        self.ruler_plot = pg.PlotItem()
        self.scene.addItem(self.ruler_plot)

        # Example: Add a few tracks
        for i in range(3):
            track = TrackItem(track_name=f"Track {i+1}", track_index=i)
            self.scene.addItem(track)
            track.setPos(0, i * 50)  # Stack tracks vertically

class TrackItem(QGraphicsRectItem):
    def __init__(self, track_name, track_index, width=800, height=40):
        super().__init__(0, 0, width, height)
        self.track_name = track_name
        self.track_index = track_index

        # Set default appearance
        self.setBrush(QBrush(QColor(50, 50, 50)))
        self.setPen(QPen(QColor(200, 200, 200)))

        # Add text label (optional)
        self.label = pg.TextItem(track_name, color="w")
        self.label.setParentItem(self)
        self.label.setPos(10, 10)  # Position inside the track

    def mousePressEvent(self, event):
        print(f"Track {self.track_index} clicked!")
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        print(f"Track {self.track_index} moved!")
        super().mouseMoveEvent(event)
