import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSignal, QRectF, QPointF, QSize, QEvent
from PyQt6.QtGui import QColor, QBrush, QPen, QFont, QPainter, QPainterPath, QLinearGradient, QDragEnterEvent, QDropEvent, QAction, QPicture
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                             QScrollArea, QSizePolicy, QMenu, QSplitter, QFrame, 
                             QCheckBox, QSlider, QGridLayout, QStyle, QApplication)
import numpy as np
from utils.i18n import i18n
from .widgets import BPMAxis, PlaybackCursorItem, MusicGridItem, PitchGridItem

# Constants
TRACK_HEIGHT_MIN = 60
TRACK_HEIGHT_MAX = 300
DEFAULT_TRACK_HEIGHT = 120
CONTROL_PANEL_WIDTH = 220

class AudioBlockItem(pg.GraphicsObject):
    """
    Visual representation of an audio clip on the timeline.
    Displays a rounded rectangle with a waveform inside.
    """
    sigClicked = pyqtSignal(object) # emits self
    sigPositionChanged = pyqtSignal(object) # emits self

    def __init__(self, track, hop_size, height):
        super().__init__()
        self.track = track
        self.hop_size = hop_size
        self.height = height
        self.setFlag(self.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(self.GraphicsItemFlag.ItemIsMovable, True) # Let Qt handle movement base
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
        
        self.waveform_path = None
        self._generate_waveform_path()

    def itemChange(self, change, value):
        if change == self.GraphicsItemChange.ItemPositionChange:
            # Constrain movement
            new_pos = value
            x = max(0, new_pos.x()) # Constrain X >= 0
            y = 0 # Constrain Y = 0
            
            # Snap to grid? (Optional, for now just raw)
            
            return QPointF(x, y)
        return super().itemChange(change, value)

    def _generate_waveform_path(self):
        self.waveform_path = QPainterPath()
        
        if self.track.audio is None or len(self.track.audio) == 0:
            return
            
        audio = self.track.audio
        num_frames = len(audio) / self.hop_size
        width = num_frames
        
        # Normalize
        mx = np.max(np.abs(audio))
        if mx > 0:
            audio = audio / mx
            
        # Downsample for visualization
        # Reduce point count to improve performance
        target_points = 1000 
        step = max(1, int(len(audio) / target_points))
        
        # Calculate min/max envelope
        # Reshape to (n_chunks, step)
        n_chunks = len(audio) // step
        if n_chunks > 0:
            reshaped = audio[:n_chunks*step].reshape(n_chunks, step)
            min_vals = np.min(reshaped, axis=1)
            max_vals = np.max(reshaped, axis=1)
            
            x_step = width / n_chunks
            
            # Build path: Top edge (maxs) then Bottom edge (mins) in reverse
            self.waveform_path.moveTo(0, 0) # Start center-ish (relative Y will be handled in paint)
            
            # Top points
            for i in range(n_chunks):
                x = i * x_step
                y = -max_vals[i] # Negative because Y grows down, but we'll scale/translate later
                self.waveform_path.lineTo(x, y)
                
            # Bottom points (reverse)
            for i in range(n_chunks - 1, -1, -1):
                x = i * x_step
                y = -min_vals[i]
                self.waveform_path.lineTo(x, y)
                
            self.waveform_path.closeSubpath()

    def paint(self, p, option, widget):
        # Calculate geometry in scene coords
        if self.track.audio is None:
            return
            
        num_frames = len(self.track.audio) / self.hop_size
        width = num_frames
        height = 100.0 # Fixed logical height to match ViewBox range
        
        # Get transform to calculate pixel-perfect radius
        transform = p.transform()
        scale_x = transform.m11()
        scale_y = transform.m22()
        
        # Desired radius in pixels
        radius_px = 12 # Increased radius
        # Use abs() to handle negative scaling (flipped axes)
        rx = radius_px / abs(scale_x) if abs(scale_x) > 0 else 0
        ry = radius_px / abs(scale_y) if abs(scale_y) > 0 else 0
        
        rect = QRectF(0, 0, width, height)
        
        # Colors
        if self.track.track_type == 'vocal':
            bg_color = QColor(40, 60, 100, 200)
            wave_color = QColor(100, 180, 255, 255)
            border_color = QColor(100, 150, 255)
        else:
            bg_color = QColor(40, 80, 40, 200)
            wave_color = QColor(150, 255, 100, 255)
            border_color = QColor(150, 255, 100)
            
        if self.isSelected():
            bg_color = bg_color.lighter(120)
            border_color = QColor(255, 255, 255)
            border_width = 2
        else:
            border_width = 1
            
        # Draw Background
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setBrush(QBrush(bg_color))
        
        # Adjust pen width to be constant in pixels
        pen_width_px = border_width
        # Use Cosmetic pen for constant width regardless of scale
        pen = QPen(border_color, pen_width_px)
        pen.setCosmetic(True)
        p.setPen(pen)
        
        p.drawRoundedRect(rect, rx, ry)
        
        # Draw Waveform
        if self.waveform_path:
            p.save()
            # Translate to center vertically and scale height
            p.translate(0, height / 2)
            p.scale(1, height * 0.4) # 80% height amplitude
            
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(wave_color))
            # Disable Antialiasing for waveform to improve performance
            p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
            p.drawPath(self.waveform_path)
            p.restore()
            
        # Draw Name Label
        # We want the text to be constant size and not stretched
        p.save()
        p.setPen(QColor(255, 255, 255))
        # Reset transform to draw text in screen coordinates
        # But we need the position in screen coordinates
        # map(QPointF) maps from Item to View (Screen)
        screen_pos = transform.map(QPointF(10, 5))
        p.resetTransform()
        p.setFont(QFont("Arial", 10))
        
        # Correct measure and beat labels to match actual beats
        # Assuming `self.track.name` contains the label, calculate based on actual beats
        # Dynamically calculate frames_per_beat
        hop_size = self.track.hop_size if hasattr(self.track, 'hop_size') else 512
        sr = self.track.sr if hasattr(self.track, 'sr') and self.track.sr is not None else 44100
        bpm = self.track.bpm if hasattr(self.track, 'bpm') else 120

        # Removed label drawing as requested
        
        p.restore()

    def boundingRect(self):
        if self.track.audio is None:
            return QRectF()
        width = len(self.track.audio) / self.hop_size
        # Adjust for pen width and potential anti-aliasing artifacts
        return QRectF(-5, -5, width + 10, 100.0 + 10)

    def setHeight(self, height):
        # self.prepareGeometryChange() # Geometry in coords doesn't change
        self.height = height
        # Waveform path is normalized, so we don't need to regenerate it
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.sigClicked.emit(self)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        # If ItemIsMovable is True, we don't need manual position update here
        # unless we want custom behavior not covered by itemChange.
        # But we need to emit sigPositionChanged
        super().mouseMoveEvent(event)
        if self.isSelected():
             self.track.start_frame = int(self.pos().x())
             self.sigPositionChanged.emit(self)


class TrackControlWidget(QWidget):
    """
    Left side control panel for a track.
    """
    mute_toggled = pyqtSignal(bool)
    solo_toggled = pyqtSignal(bool)
    bgm_toggled = pyqtSignal(bool)
    volume_changed = pyqtSignal(float)
    delete_requested = pyqtSignal()
    copy_pitch_requested = pyqtSignal()
    paste_pitch_requested = pyqtSignal()
    
    def __init__(self, track):
        super().__init__()
        self.track = track
        self.setFixedWidth(CONTROL_PANEL_WIDTH)
        self.setAutoFillBackground(True)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)
        
        layout = QGridLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)
        layout.setSpacing(5)
        
        # Row 1: Name
        self.name_label = QLabel(track.name)
        self.name_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        self.name_label.setStyleSheet("color: #ddd;")
        layout.addWidget(self.name_label, 0, 0, 1, 3)

        # Delete Button
        self.del_btn = QPushButton("×")
        self.del_btn.setFixedSize(20, 20)
        self.del_btn.setToolTip(i18n.get("track.delete"))
        self.del_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus) # Prevent Tab capture
        self.del_btn.setStyleSheet("""
            QPushButton { background-color: transparent; color: #888; border: none; font-weight: bold; font-size: 14px; }
            QPushButton:hover { color: #f44; }
        """)
        self.del_btn.clicked.connect(self.delete_requested.emit)
        layout.addWidget(self.del_btn, 0, 3)
        
        # Row 2: Buttons (M, S, BGM)
        self.mute_btn = QPushButton("M")
        self.mute_btn.setCheckable(True)
        self.mute_btn.setChecked(track.muted)
        self.mute_btn.setFixedSize(25, 25)
        self.mute_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus) # Prevent Spacebar toggle
        self.mute_btn.setStyleSheet("""
            QPushButton { background-color: #444; border: none; border-radius: 3px; }
            QPushButton:checked { background-color: #d44; color: white; }
        """)
        self.mute_btn.clicked.connect(self.on_mute)
        layout.addWidget(self.mute_btn, 1, 0)
        
        self.solo_btn = QPushButton("S")
        self.solo_btn.setCheckable(True)
        self.solo_btn.setChecked(track.solo)
        self.solo_btn.setFixedSize(25, 25)
        self.solo_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus) # Prevent Spacebar toggle
        self.solo_btn.setStyleSheet("""
            QPushButton { background-color: #444; border: none; border-radius: 3px; }
            QPushButton:checked { background-color: #dd4; color: black; }
        """)
        self.solo_btn.clicked.connect(self.on_solo)
        layout.addWidget(self.solo_btn, 1, 1)
        
        self.bgm_check = QCheckBox("BGM")
        self.bgm_check.setChecked(track.track_type == 'bgm')
        self.bgm_check.setFocusPolicy(Qt.FocusPolicy.NoFocus) # Prevent Spacebar toggle
        self.bgm_check.toggled.connect(self.on_bgm)
        layout.addWidget(self.bgm_check, 1, 2, 1, 2)
        
        # Row 3: Volume
        vol_label = QLabel(i18n.get("label.volume"))
        vol_label.setStyleSheet("color: #888;")
        layout.addWidget(vol_label, 2, 0)
        
        self.vol_slider = QSlider(Qt.Orientation.Horizontal)
        self.vol_slider.setRange(0, 150) # 0% to 150%
        self.vol_slider.setValue(int(track.volume * 100))
        self.vol_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus) # Prevent Spacebar toggle
        self.vol_slider.valueChanged.connect(self.on_volume)
        layout.addWidget(self.vol_slider, 2, 1, 1, 3)
        
        # Style
        self.update_style(False)

    def show_context_menu(self, pos):
        menu = QMenu(self)
        
        copy_action = menu.addAction(i18n.get("track.copy_pitch"))
        copy_action.triggered.connect(self.copy_pitch_requested.emit)
        
        paste_action = menu.addAction(i18n.get("track.paste_pitch"))
        paste_action.triggered.connect(self.paste_pitch_requested.emit)
        
        menu.addSeparator()
        
        delete_action = menu.addAction(i18n.get("track.delete"))
        delete_action.triggered.connect(self.delete_requested.emit)
        
        menu.exec(self.mapToGlobal(pos))

    def update_style(self, selected):
        if selected:
            bg = "#3d3d3d"
            border = "#666"
        else:
            bg = "#282828"
            border = "#333"
            
        self.setStyleSheet(f"""
            TrackControlWidget {{
                background-color: {bg};
                border-bottom: 1px solid {border};
                border-right: 1px solid {border};
            }}
            QLabel {{ color: #ccc; }}
            QCheckBox {{ color: #ccc; }}
        """)

    def on_mute(self, checked):
        self.track.muted = checked
        self.mute_toggled.emit(checked)

    def on_solo(self, checked):
        self.track.solo = checked
        self.solo_toggled.emit(checked)

    def on_bgm(self, checked):
        self.bgm_toggled.emit(checked)

    def on_volume(self, val):
        vol = val / 100.0
        self.track.volume = vol
        self.volume_changed.emit(vol)


class LockedViewBox(pg.ViewBox):
    def mouseDragEvent(self, ev, axis=None):
        ev.ignore()
    def mouseClickEvent(self, ev):
        ev.ignore()
    def wheelEvent(self, ev, axis=None):
        ev.ignore()

class TrackLaneWidget(pg.PlotWidget):
    """
    Right side audio area for a track.
    """
    def __init__(self, parent_gui):
        # Use custom ViewBox to lock interactions
        vb = LockedViewBox()
        super().__init__(viewBox=vb)
        self.parent_gui = parent_gui
        self.setBackground('#1e1e1e')
        self.setMouseEnabled(x=True, y=False)
        self.hideAxis('left')
        self.hideAxis('bottom')
        # Disable default grid
        self.showGrid(x=False, y=False)
        self.setMenuEnabled(False)
        
        # Add Pitch Grid (Background)
        self.pitch_grid = PitchGridItem()
        self.addItem(self.pitch_grid)
        
        # Add Music Grid (Vertical Lines)
        self.music_grid = MusicGridItem(parent_gui)
        self.addItem(self.music_grid)
        
        # Remove margins
        self.getPlotItem().setContentsMargins(0, 0, 0, 0)
        
        # Limits
        self.setLimits(xMin=0)
        self.setYRange(-10, 110) # Normalized Y range with padding to prevent clipping

    def wheelEvent(self, event):
        # Pass wheel events to parent for global zoom handling
        event.ignore()


class TrackWidget(QWidget):
    """
    Combined Control + Lane for a single track.
    """
    def __init__(self, track, track_index, hop_size, height, parent_gui):
        super().__init__()
        self.track = track
        self.track_index = track_index
        self.hop_size = hop_size
        self.height = height
        self.parent_gui = parent_gui
        
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        
        # Control
        self.control = TrackControlWidget(track)
        self.layout.addWidget(self.control)
        
        # Lane
        self.lane = TrackLaneWidget(parent_gui)
        self.lane.setFixedHeight(height)
        self.layout.addWidget(self.lane)
        
        # Audio Item
        self.item = AudioBlockItem(track, hop_size, height)
        self.item.setPos(track.start_frame, 0)
        self.item.setZValue(10) # Ensure it's on top
        self.lane.addItem(self.item)
        
        # Cursor (Vertical Line)
        self.cursor = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen('y', width=1))
        self.lane.addItem(self.cursor)

    def set_height(self, height):
        self.height = height
        self.lane.setFixedHeight(height)
        self.control.setFixedHeight(height)
        self.setFixedHeight(height)
        self.item.setHeight(height) 

    def set_selected(self, selected):
        self.control.update_style(selected)
        self.item.setSelected(selected)


class TimelinePanel(QWidget):
    """
    Main Timeline Panel.
    """
    trackSelected = pyqtSignal(int)
    filesDropped = pyqtSignal(list)
    cursorMoved = pyqtSignal(float)
    trackTypeChanged = pyqtSignal(object, str)
    
    def __init__(self, parent_gui):
        super().__init__(parent_gui)
        self.parent_gui = parent_gui
        self.hop_size = 512
        self.track_height = DEFAULT_TRACK_HEIGHT
        self.rows = []
        
        # Middle mouse drag state
        self._is_panning = False
        self._last_pan_pos = None

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        
        # 1. Ruler Area
        self.ruler_container = QWidget()
        self.ruler_layout = QHBoxLayout(self.ruler_container)
        self.ruler_layout.setContentsMargins(0, 0, 0, 0)
        self.ruler_layout.setSpacing(0)
        
        # Spacer for Control Panel
        self.ruler_spacer = QWidget()
        self.ruler_spacer.setFixedWidth(CONTROL_PANEL_WIDTH)
        self.ruler_spacer.setStyleSheet("background-color: #282828; border-bottom: 1px solid #444; border-right: 1px solid #333;")
        self.ruler_layout.addWidget(self.ruler_spacer)
        
        # Ruler Plot
        self.ruler_container.setFixedHeight(45)
        self.ruler_plot = pg.PlotWidget(axisItems={'top': BPMAxis(parent_gui, orientation='top')})
        self.ruler_plot.setFixedHeight(45)
        self.ruler_plot.setBackground('#1e1e1e')
        self.ruler_plot.hideAxis('left')
        self.ruler_plot.hideAxis('bottom')
        self.ruler_plot.showAxis('top')
        self.ruler_plot.setMouseEnabled(x=True, y=False)
        self.ruler_plot.setMenuEnabled(False)
        self.ruler_plot.getPlotItem().setContentsMargins(0, 0, 0, 0)
        
        # Set default limits (3 minutes)
        self.default_duration = 180 # seconds
        self.update_timeline_bounds(self.default_duration)
        
        # Set initial view range to 4 measures
        self.set_initial_view_range()
        
        # Ruler Cursor24
        self.ruler_cursor = PlaybackCursorItem()
        self.ruler_plot.addItem(self.ruler_cursor)
        
        self.ruler_layout.addWidget(self.ruler_plot)
        self.layout.addWidget(self.ruler_container)
        
        # 2. Tracks Area (Scrollable)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff) # We pan via plot
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        
        self.tracks_container = QWidget()
        self.tracks_layout = QVBoxLayout(self.tracks_container)
        self.tracks_layout.setContentsMargins(0, 0, 0, 0)
        self.tracks_layout.setSpacing(1) # 1px gap between tracks
        self.tracks_layout.addStretch()
        
        self.scroll_area.setWidget(self.tracks_container)
        self.layout.addWidget(self.scroll_area)
        
        # Install filter on scroll area viewport to catch wheel events in empty space
        self.scroll_area.viewport().installEventFilter(self)
        
        # Drag & Drop
        self.setAcceptDrops(True)
        
        # Sync Ruler View
        self.ruler_plot.sigRangeChanged.connect(self.on_ruler_range_changed)

    def set_initial_view_range(self):
        # Calculate frames for 4 measures
        hop_size = self.hop_size
        sr = 44100
        if hasattr(self.parent_gui, 'processor') and self.parent_gui.processor:
             sr = self.parent_gui.processor.config.get('audio_sample_rate', 44100)
        
        bpm = 120
        beats_per_measure = 4
        if hasattr(self.parent_gui, 'bpm_spin'):
            bpm = self.parent_gui.bpm_spin.value()
        if hasattr(self.parent_gui, 'beats_spin'):
            beats_per_measure = self.parent_gui.beats_spin.value()
            
        frames_per_sec = sr / hop_size
        frames_per_beat = frames_per_sec * (60.0 / bpm)
        frames_per_measure = frames_per_beat * beats_per_measure
        
        initial_range = 4 * frames_per_measure
        
        self.ruler_plot.setXRange(0, initial_range, padding=0)
        
        # Also set range for main plot widget since they are decoupled
        if hasattr(self.parent_gui, 'plot_widget'):
             self.parent_gui.plot_widget.setXRange(0, initial_range, padding=0)

    def update_timeline_bounds(self, duration_sec=None):
        if duration_sec is None:
            duration_sec = self.default_duration
            
        # Check if we have longer audio
        max_audio_len = 0
        if self.rows:
            for row in self.rows:
                if row.track.audio is not None:
                    # Calculate duration in seconds
                    sr = row.track.sr if row.track.sr is not None else 44100
                    dur = len(row.track.audio) / sr
                    if dur > max_audio_len:
                        max_audio_len = dur
        
        final_duration = max(duration_sec, max_audio_len)
        
        # Convert to frames
        sr = 44100
        if hasattr(self.parent_gui, 'processor') and self.parent_gui.processor:
             sr = self.parent_gui.processor.config.get('audio_sample_rate', 44100)
             
        frames = int(final_duration * sr / self.hop_size)
        
        # Update limits
        # Add some padding
        frames_padded = int(frames * 1.05)
        
        self.ruler_plot.setLimits(xMin=0, xMax=frames_padded)
        
        # Update all track lanes
        for row in self.rows:
            row.lane.setLimits(xMin=0, xMax=frames_padded)
            
        # Update main plot widget limits if available
        if hasattr(self.parent_gui, 'plot_widget'):
            self.parent_gui.plot_widget.setLimits(xMin=0, xMax=frames_padded)

    def refresh_tracks(self, tracks):
        # Save current view range to prevent jumping
        current_range = self.ruler_plot.viewRange()[0]
        
        # Clear
        while self.tracks_layout.count():
            item = self.tracks_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        self.rows = []
        
        for i, track in enumerate(tracks):
            row = TrackWidget(track, i, self.hop_size, self.track_height, self.parent_gui)
            
            # Connect Signals
            row.control.mute_toggled.connect(self.parent_gui.update_plot) # Just trigger update
            row.control.solo_toggled.connect(self.parent_gui.update_plot)
            row.control.volume_changed.connect(self.parent_gui.update_plot)
            row.control.bgm_toggled.connect(lambda c, t=track: self.on_bgm_toggled(t, c))
            
            row.control.delete_requested.connect(lambda idx=i: self.parent_gui.delete_track(idx))
            row.control.copy_pitch_requested.connect(lambda idx=i: self.parent_gui.copy_pitch(idx))
            row.control.paste_pitch_requested.connect(lambda idx=i: self.parent_gui.paste_pitch(idx))
            
            row.item.sigClicked.connect(lambda item, idx=i: self.select_track(idx))
            row.item.sigPositionChanged.connect(lambda item: self.parent_gui.update_plot())
            
            # Link View
            row.lane.setXLink(self.ruler_plot)
            
            # Forward Wheel Events from Lane to Panel
            row.lane.viewport().installEventFilter(self)
            row.control.installEventFilter(self)
            
            self.tracks_layout.insertWidget(i, row)
            self.rows.append(row)
            
        self.tracks_layout.addStretch()
        
        self.update_timeline_bounds()
        
        # Restore view range if it was valid (not 0-1)
        if current_range[1] > 1:
             self.ruler_plot.setXRange(current_range[0], current_range[1], padding=0)

    def on_bgm_toggled(self, track, is_bgm):
        new_type = 'bgm' if is_bgm else 'vocal'
        self.trackTypeChanged.emit(track, new_type)

    def select_track(self, index):
        if 0 <= index < len(self.rows):
            for i, row in enumerate(self.rows):
                row.set_selected(i == index)
            self.trackSelected.emit(index)

    def eventFilter(self, source, event):
        # 1. Wheel Zoom (Alt=Y, Ctrl=X)
        if event.type() == QEvent.Type.Wheel:
            # Use QApplication.keyboardModifiers() as it is sometimes more reliable 
            # across different widget contexts than event.modifiers()
            modifiers = QApplication.keyboardModifiers()
            angle = event.angleDelta()
            delta_y = angle.y()
            delta_x = angle.x()
            
            # print(f"Wheel: mod={modifiers}, angle={angle}, delta_y={delta_y}, delta_x={delta_x}")

            if modifiers & Qt.KeyboardModifier.ControlModifier:
                # X Zoom
                # Use Y delta primarily, but fallback to X if Y is 0 (some mice/trackpads)
                delta = delta_y if delta_y != 0 else delta_x
                
                # Use ruler's ViewBox for calculation to ensure consistency
                vb = self.ruler_plot.plotItem.vb
                
                # Map mouse position to view coordinates
                # We need to handle different source widgets
                if isinstance(source, QWidget):
                    # Map local pos to global, then to ruler widget
                    global_pos = source.mapToGlobal(event.position().toPoint())
                    ruler_widget_pos = self.ruler_plot.mapFromGlobal(global_pos)
                    ruler_scene_pos = self.ruler_plot.mapToScene(ruler_widget_pos)
                    center = vb.mapSceneToView(ruler_scene_pos)
                else:
                    center = None # Fallback to center of view

                # Increased zoom speed as requested (was 1.02)
                s = 1.15 ** (delta / 120.0)
                vb.scaleBy((1/s, 1), center)
                return True
                
            elif modifiers & Qt.KeyboardModifier.AltModifier:
                # Y Zoom
                # Some systems map Alt+Scroll to Horizontal Scroll
                delta = delta_y if delta_y != 0 else delta_x
                self.zoom_y(delta)
                return True

        # 2. Middle Mouse Pan (Global)
        if event.type() == QEvent.Type.MouseButtonPress:
            if event.button() == Qt.MouseButton.MiddleButton:
                self._is_panning = True
                self._last_pan_pos = event.globalPosition()
                return True
                
        elif event.type() == QEvent.Type.MouseMove:
            if self._is_panning and self._last_pan_pos:
                current_pos = event.globalPosition()
                delta = current_pos - self._last_pan_pos
                self._last_pan_pos = current_pos
                
                # Pan X (Time)
                # We need to translate pixel delta to view units
                vb = self.ruler_plot.plotItem.vb
                view_range = vb.viewRange()[0]
                view_width = view_range[1] - view_range[0]
                pixel_width = self.ruler_plot.width()
                
                if pixel_width > 0:
                    scale = view_width / pixel_width
                    dx = -delta.x() * scale
                    vb.translateBy(x=dx, y=0)
                    
                # Pan Y (Scroll Tracks)
                # ScrollArea scrollbar
                vbar = self.scroll_area.verticalScrollBar()
                vbar.setValue(vbar.value() - int(delta.y()))
                
                return True
                
        elif event.type() == QEvent.Type.MouseButtonRelease:
            if event.button() == Qt.MouseButton.MiddleButton:
                self._is_panning = False
                self._last_pan_pos = None
                return True
                
        return super().eventFilter(source, event)

    def zoom_y(self, delta):
        if delta == 0:
            return
            
        step = 10
        # Standard behavior: Scroll Up (Positive) -> Enlarge (Taller)
        # Scroll Down (Negative) -> Shrink (Shorter)
        if delta > 0:
            self.track_height = min(TRACK_HEIGHT_MAX, self.track_height + step)
        else:
            self.track_height = max(TRACK_HEIGHT_MIN, self.track_height - step)
            
        for row in self.rows:
            row.set_height(self.track_height)

    def set_cursor_position(self, x_frame):
        self.ruler_cursor.setValue(x_frame)
        for row in self.rows:
            row.cursor.setValue(x_frame)

    def on_ruler_range_changed(self, _, range):
        # Ensure all lanes follow (they are X-linked, so they should)
        # Also update cursor position if needed
        pass

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        valid_files = [f for f in files if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg'))]
        if valid_files:
            self.filesDropped.emit(valid_files)


