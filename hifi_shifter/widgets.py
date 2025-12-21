import numpy as np
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QPen, QColor, QPainter, QPainterPath, QPolygonF, QFont, QBrush
from PyQt6.QtCore import Qt, QRectF, QPointF
import pyqtgraph as pg

class PlaybackCursorItem(pg.GraphicsObject):
    def __init__(self):
        super().__init__()
        self.setZValue(1000) # Always on top
        self.pen = pg.mkPen(color='#FFD700', width=1) # Gold color, width 1
        self.pen.setCosmetic(True)
        self.brush = pg.mkBrush(color='#FFD700')
        
    def setValue(self, value):
        self.setPos(value, 0)
        
    def boundingRect(self):
        # Return a rect that covers the vertical line and the head
        # Since it's an infinite vertical line effectively in view, 
        # we just need to cover the X width. The ViewBox handles clipping.
        return QRectF(-10, -10000, 20, 20000)
        
    def paint(self, p, option, widget):
        # Draw vertical line
        p.setPen(self.pen)
        # Draw a very long line
        p.drawLine(0, -10000, 0, 10000)
        
        # Draw Head (Triangle)
        # We want the head to be at the top of the view
        # But this item is in data coordinates (X=time, Y=pitch)
        # We need to map from view coordinates to find the "top"
        
        # Actually, for a simple cursor, we can just draw the line.
        # If we want a head that stays at the top of the screen, we need to use the view transform.
        
        transform = p.transform()
        # Invert transform to map screen pixels to item coords
        inv_transform, _ = transform.inverted()
        
        # Get visible rect in item coords
        # This is hard because we don't have easy access to the view rect here without passing it.
        # However, we can just draw the head at a fixed Y if we want, but that's not right.
        
        # Better approach: Use Non-Scaling items or handle it in ViewBox.
        # But let's try to just draw a nice line for now, maybe with a glow?
        
        # Let's stick to a simple line but nicer color/width for now as requested "better looking"
        # The user asked for "better looking", maybe a triangle head at the top of the ruler?
        # The ruler has its own cursor. This is the piano roll cursor.
        pass

class MusicGridItem(pg.GraphicsObject):
    def __init__(self, parent_gui):
        super().__init__()
        self.parent_gui = parent_gui
        self.grid_resolution = 4 # 1/4 note (beat) by default
        self.setZValue(-10) # Behind everything
        
    def set_resolution(self, res):
        self.grid_resolution = res
        self.update()
        
    def boundingRect(self):
        # Return a very large rect to ensure we are always painted
        # The paint method handles the actual culling based on view
        return QRectF(0, -10000, 10000000, 20000)
        
    def paint(self, p, option, widget):
        view = self.getViewBox()
        if view is None:
            return
            
        rect = view.viewRect()
        left = rect.left()
        right = rect.right()
        top = rect.top()
        bottom = rect.bottom()
        
        # Defaults
        hop_size = 512
        sr = 44100
        
        if hasattr(self.parent_gui, 'processor') and \
           self.parent_gui.processor and \
           self.parent_gui.processor.config:
            hop_size = self.parent_gui.processor.config.get('hop_size', 512)
            sr = self.parent_gui.processor.config.get('audio_sample_rate', 44100)
            
        bpm = self.parent_gui.bpm_spin.value()
        beats_per_bar = self.parent_gui.beats_spin.value()
        
        # Calculate frames per beat
        frames_per_sec = sr / hop_size
        frames_per_beat = frames_per_sec * (60.0 / bpm)
        
        # Calculate grid interval in frames
        # grid_resolution is denominator: 4 = 1/4 note, 8 = 1/8 note
        # 1 beat = 1/4 note usually (if denominator is 4)
        # Let's assume time signature denominator is 4 for now
        
        # Interval in beats
        interval_beats = 4.0 / self.grid_resolution
        interval_frames = frames_per_beat * interval_beats
        
        if interval_frames <= 0:
            return
            
        # LOD Check: Don't draw if lines are too dense
        transform = p.transform()
        scale_x = transform.m11()
        pixels_per_interval = interval_frames * scale_x
        
        # If lines are closer than 5 pixels, increase interval
        if pixels_per_interval < 5:
            # Try to fallback to beats
            interval_beats = 1.0
            interval_frames = frames_per_beat * interval_beats
            pixels_per_interval = interval_frames * scale_x
            
            if pixels_per_interval < 5:
                # Fallback to bars
                interval_beats = float(beats_per_bar)
                interval_frames = frames_per_beat * interval_beats
                pixels_per_interval = interval_frames * scale_x
                
                if pixels_per_interval < 5:
                    # Fallback to 4 bars
                    interval_beats = float(beats_per_bar * 4)
                    interval_frames = frames_per_beat * interval_beats
                    
        # Find first line
        start_idx = int(left / interval_frames)
        end_idx = int(right / interval_frames) + 1
        
        # Safety check for loop count
        if (end_idx - start_idx) > 2000:
             # Still too many lines, just draw bars or nothing
             return
        
        # Pens
        bar_pen = QPen(QColor(100, 100, 100, 150), 1) # Match horizontal line width
        bar_pen.setCosmetic(True)
        beat_pen = QPen(QColor(60, 60, 60, 100), 1)   # Match horizontal line width
        beat_pen.setCosmetic(True)
        sub_pen = QPen(QColor(40, 40, 40, 80), 1) # Match horizontal line width
        sub_pen.setCosmetic(True)
        
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        
        for i in range(start_idx, end_idx):
            x = i * interval_frames
            
            # Determine line type
            # Calculate total beats from 0
            total_beats = i * interval_beats
            
            # Check if it's a bar line
            # Use a small epsilon for float comparison
            if abs(total_beats % beats_per_bar) < 0.001:
                p.setPen(bar_pen)
            elif abs(total_beats % 1.0) < 0.001:
                p.setPen(beat_pen)
            else:
                p.setPen(sub_pen)
                
            p.drawLine(QPointF(x, top), QPointF(x, bottom))

class PitchGridItem(pg.GraphicsObject):
    def __init__(self):
        super().__init__()
        self.setZValue(-20) # Behind everything, even MusicGridItem
        
    def boundingRect(self):
        return QRectF(0, -10000, 10000000, 20000)
        
    def paint(self, p, option, widget):
        view = self.getViewBox()
        if view is None:
            return
            
        rect = view.viewRect()
        left = rect.left()
        right = rect.right()
        top = rect.top()
        bottom = rect.bottom()
        
        # Draw horizontal lines for pitch
        # Assuming Y is semitones
        
        start_y = int(np.floor(bottom))
        end_y = int(np.ceil(top))
        
        # Pens
        white_key_pen = QPen(QColor(60, 60, 60, 50), 1)
        white_key_pen.setCosmetic(True)
        
        black_key_brush = QBrush(QColor(30, 30, 30, 100))
        
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        
        for y in range(start_y, end_y + 1):
            # Check if black key
            # MIDI note % 12
            # 0=C, 1=C#, 2=D, 3=D#, 4=E, 5=F, 6=F#, 7=G, 8=G#, 9=A, 10=A#, 11=B
            # Black keys: 1, 3, 6, 8, 10
            note_in_octave = y % 12
            is_black = note_in_octave in [1, 3, 6, 8, 10]
            
            if is_black:
                # Draw background strip for black key
                p.fillRect(QRectF(left, y, right - left, 1), black_key_brush)
                # Draw thinner line? Or just the background is enough distinction
                # User asked for "black key slightly thinner" - maybe visually thinner line or darker background
                # Let's use darker background as implemented above.
            
            # Draw grid line (bottom of the key)
            p.setPen(white_key_pen)
            p.drawLine(QPointF(left, y), QPointF(right, y))

class CustomViewBox(pg.ViewBox):
    def __init__(self, parent_gui):
        pg.ViewBox.__init__(self)
        self.parent_gui = parent_gui

    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.MiddleButton:
            ev.accept()
        elif ev.button() == Qt.MouseButton.LeftButton or ev.button() == Qt.MouseButton.RightButton:
            # Ignore Left/Right so they don't trigger ViewBox default drag (Pan/Zoom)
            # This allows the Scene to handle them via sigMouseMoved for drawing
            ev.ignore()
        else:
            super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if ev.buttons() & Qt.MouseButton.MiddleButton:
            ev.accept()
            # Manual Pan
            # mapToView maps from Item coordinates (pixels) to Data coordinates
            p1 = self.mapToView(ev.pos())
            p2 = self.mapToView(ev.lastPos())
            tr = p1 - p2
            self.translateBy(-tr)
        else:
            super().mouseMoveEvent(ev)

    def wheelEvent(self, ev, axis=None):
        modifiers = QApplication.keyboardModifiers()
        if modifiers == Qt.KeyboardModifier.ControlModifier:
            # Zoom X (axis 0)
            super().wheelEvent(ev, axis=0)
        elif modifiers == Qt.KeyboardModifier.AltModifier:
            # Zoom Y (axis 1)
            super().wheelEvent(ev, axis=1)
        else:
            super().wheelEvent(ev, axis=axis)
            
    def mouseClickEvent(self, ev):
        # Ignore Left/Right clicks so they propagate or don't trigger ViewBox menus
        if ev.button() == Qt.MouseButton.LeftButton or ev.button() == Qt.MouseButton.RightButton:
            ev.ignore()
        else:
            super().mouseClickEvent(ev)

class PianoRollAxis(pg.AxisItem):
    def __init__(self, orientation='left', **kwargs):
        super().__init__(orientation, **kwargs)

    def tickStrings(self, values, scale, spacing):
        strings = []
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        for v in values:
            # Only label if close to integer
            if abs(v - round(v)) > 0.1:
                strings.append("")
                continue
            
            try:
                # v is MIDI note number
                note_idx = int(round(v))
                octave = note_idx // 12 - 1
                name = note_names[note_idx % 12]
                strings.append(f"{name}{octave}")
            except:
                strings.append("")
        return strings
    
    def tickValues(self, minVal, maxVal, size):
        # Force integer ticks
        min_idx = int(np.ceil(minVal))
        max_idx = int(np.floor(maxVal))
        values = list(range(min_idx, max_idx + 1))
        return [(1.0, values)]

class BPMAxis(pg.AxisItem):
    def __init__(self, parent_gui, orientation='top', **kwargs):
        super().__init__(orientation, **kwargs)
        self.parent_gui = parent_gui
        self.setStyle(tickLength=0, showValues=False)
        self.setHeight(40)

    def tickValues(self, minVal, maxVal, size):
        return []

    def drawPicture(self, p, axisSpec, tickSpecs, textSpecs):
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        
        # Draw Axis Line
        rect = self.boundingRect()
        y = rect.bottom()
        p.setPen(QPen(QColor(100, 100, 100), 1))
        p.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
        
        vb = self.linkedView()
        if vb is None:
            return
            
        view_range = vb.viewRange()[0]
        left, right = view_range
        if right <= left:
            return

        # Get Grid Info
        hop_size = 512
        sr = 44100
        if hasattr(self.parent_gui, 'processor') and \
           self.parent_gui.processor and \
           self.parent_gui.processor.config:
            hop_size = self.parent_gui.processor.config.get('hop_size', 512)
            sr = self.parent_gui.processor.config.get('audio_sample_rate', 44100)
            
        bpm = self.parent_gui.bpm_spin.value()
        beats_per_bar = self.parent_gui.beats_spin.value()
        
        frames_per_sec = sr / hop_size
        frames_per_beat = frames_per_sec * (60.0 / bpm)
        
        width_pixels = rect.width()
        view_width = right - left
        
        # Calculate pixels per beat to determine density
        pixels_per_beat = (frames_per_beat / view_width) * width_pixels
        
        if pixels_per_beat > 100:
            interval_beats = 1
        elif pixels_per_beat > 25:
            interval_beats = beats_per_bar
        else:
            interval_beats = beats_per_bar * 4
            
        interval_frames = frames_per_beat * interval_beats
        if interval_frames <= 0: return

        start_idx = int(left / interval_frames)
        end_idx = int(right / interval_frames) + 1
        
        font_beat = QFont("Arial", 10)
        font_time = QFont("Arial", 8)
        
        p.setPen(QColor(200, 200, 200))
        
        for i in range(start_idx, end_idx):
            x = i * interval_frames
            
            # Map x to pixel
            x_pixel = (x - left) / view_width * width_pixels
            
            # Skip if out of bounds
            if x_pixel < 0 or x_pixel > width_pixels:
                continue

            # Calculate Labels
            total_beats = i * interval_beats
            bar = int(total_beats // beats_per_bar) + 1
            beat = int(total_beats % beats_per_bar) + 1
            
            label_beat = f"{bar}.{beat}"
            if interval_beats >= beats_per_bar:
                 label_beat = f"{bar}"
            
            seconds = (x * hop_size) / sr
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            label_time = f"{mins}:{secs:02d}"
            
            # Draw Beat
            p.setFont(font_beat)
            p.setPen(QColor(200, 200, 200))
            p.drawText(QRectF(x_pixel - 30, y - 35, 60, 15), Qt.AlignmentFlag.AlignCenter, label_beat)
            
            # Draw Time
            p.setFont(font_time)
            p.setPen(QColor(150, 150, 150))
            p.drawText(QRectF(x_pixel - 30, y - 18, 60, 12), Qt.AlignmentFlag.AlignCenter, label_time)
            
            # Draw small tick mark
            p.setPen(QColor(100, 100, 100))
            p.drawLine(QPointF(x_pixel, y), QPointF(x_pixel, y - 5))

    def mouseClickEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            ev.accept()
            vb = self.linkedView()
            if vb:
                # Map click to view coord
                x_pixel = ev.pos().x()
                rect = self.boundingRect()
                width_pixels = rect.width()
                
                view_range = vb.viewRange()[0]
                left, right = view_range
                view_width = right - left
                
                x = left + (x_pixel / width_pixels) * view_width
                
                self.parent_gui.set_playback_position(x)
