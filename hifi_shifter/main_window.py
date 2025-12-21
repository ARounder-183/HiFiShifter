import sys
import os
import time
import json
import pathlib
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wavfile
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QMessageBox, QComboBox, QDoubleSpinBox, QSpinBox,
                             QButtonGroup, QSplitter, QScrollBar)
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtCore import Qt, QTimer
import pyqtgraph as pg

# Import widgets
from .widgets import CustomViewBox, PianoRollAxis, BPMAxis, MusicGridItem, PlaybackCursorItem
from .timeline import TimelinePanel, CONTROL_PANEL_WIDTH
from .track import Track
# Import AudioProcessor
from .audio_processor import AudioProcessor
# Import Config Manager
from . import config_manager

class HifiShifterGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HifiShifter - 音高修正工具")
        self.resize(1200, 800)
        
        # Initialize Audio Processor
        self.processor = AudioProcessor()
        
        # Data for UI
        self.project_path = None
        self.model_path = None
        
        # Track Management
        self.tracks = []
        self.current_track_idx = -1
        
        # Tools
        self.tool_mode = 'draw' # 'draw' only now
        
        # Playback State
        self.is_playing = False
        self.current_playback_time = 0.0 # seconds
        self.playback_start_time = 0.0 # seconds (for return to start)
        self.last_wall_time = 0.0
        self.playback_timer = QTimer()
        self.playback_timer.setInterval(30) # 30ms update
        self.playback_timer.timeout.connect(self.update_cursor)
        
        # Undo/Redo Stacks (Global or per track? Per track is better but harder. Let's keep global for now, but it needs to track which track was edited)
        # Actually, let's make undo/redo per track, or clear it when switching tracks.
        # For simplicity, let's clear undo stack when switching tracks for now.
        self.undo_stack = []
        self.redo_stack = []
        
        # Clipboard
        self.pitch_clipboard = None
        
        # Interaction State
        self.is_drawing = False
        self.last_mouse_pos = None
        
        self.init_ui()
        self.load_default_model()
        
    @property
    def current_track(self):
        if 0 <= self.current_track_idx < len(self.tracks):
            return self.tracks[self.current_track_idx]
        return None
        
    def create_menu_bar(self):
        menu_bar = self.menuBar()
        
        # File Menu
        file_menu = menu_bar.addMenu("文件")
        
        open_action = QAction("打开工程", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self.open_project_dialog)
        file_menu.addAction(open_action)
        
        save_action = QAction("保存工程", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self.save_project)
        file_menu.addAction(save_action)
        
        save_as_action = QAction("另存为工程", self)
        save_as_action.setShortcut(QKeySequence.StandardKey.SaveAs)
        save_as_action.triggered.connect(self.save_project_as)
        file_menu.addAction(save_as_action)
        
        file_menu.addSeparator()

        load_model_action = QAction("加载模型", self)
        load_model_action.triggered.connect(self.load_model_dialog)
        file_menu.addAction(load_model_action)

        load_audio_action = QAction("加载音频", self)
        load_audio_action.triggered.connect(self.load_audio_dialog)
        file_menu.addAction(load_audio_action)
        
        file_menu.addSeparator()
        
        export_action = QAction("导出音频", self)
        export_action.triggered.connect(self.export_audio_dialog)
        file_menu.addAction(export_action)

        # Edit Menu
        edit_menu = menu_bar.addMenu("编辑")
        
        undo_action = QAction("撤销", self)
        undo_action.setShortcut(QKeySequence.StandardKey.Undo)
        undo_action.triggered.connect(self.undo)
        edit_menu.addAction(undo_action)

        redo_action = QAction("重做", self)
        redo_action.setShortcut(QKeySequence.StandardKey.Redo)
        redo_action.triggered.connect(self.redo)
        edit_menu.addAction(redo_action)

        # Playback Menu
        play_menu = menu_bar.addMenu("播放")
        
        play_orig_action = QAction("播放原音", self)
        play_orig_action.triggered.connect(self.play_original)
        play_menu.addAction(play_orig_action)
        
        synth_play_action = QAction("合成并播放", self)
        synth_play_action.triggered.connect(self.synthesize_and_play)
        play_menu.addAction(synth_play_action)
        
        stop_action = QAction("停止", self)
        stop_action.setShortcut(Qt.Key.Key_Escape)
        stop_action.triggered.connect(self.stop_audio)
        play_menu.addAction(stop_action)

        # Settings Menu
        settings_menu = menu_bar.addMenu("设置")
        
        set_default_model_action = QAction("设置默认模型", self)
        set_default_model_action.triggered.connect(self.set_default_model_dialog)
        settings_menu.addAction(set_default_model_action)

    def init_ui(self):
        self.create_menu_bar()
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Controls Bar
        controls_layout = QHBoxLayout()
        controls_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        controls_layout.setContentsMargins(10, 5, 10, 5)

        self.shift_spin = QDoubleSpinBox()
        self.shift_spin.setRange(-24, 24)
        self.shift_spin.setSingleStep(1)
        self.shift_spin.setPrefix("移调: ")
        self.shift_spin.setSuffix(" 半音")
        self.shift_spin.valueChanged.connect(self.apply_shift)
        
        self.bpm_spin = QDoubleSpinBox()
        self.bpm_spin.setRange(10, 300)
        self.bpm_spin.setValue(120)
        self.bpm_spin.setPrefix("BPM: ")
        self.bpm_spin.valueChanged.connect(self.on_bpm_changed)

        self.beats_spin = QSpinBox()
        self.beats_spin.setRange(1, 32)
        self.beats_spin.setValue(4)
        self.beats_spin.setPrefix("拍号: ")
        self.beats_spin.setSuffix(" / 4")
        self.beats_spin.valueChanged.connect(self.on_beats_changed)

        # Grid Resolution
        self.grid_combo = QComboBox()
        self.grid_combo.addItems(["1/4", "1/8", "1/16", "1/32"])
        self.grid_combo.setCurrentIndex(0) # Default 1/4
        self.grid_combo.currentIndexChanged.connect(self.on_grid_changed)

        controls_layout.addWidget(QLabel("参数设置:"))
        controls_layout.addWidget(self.shift_spin)
        controls_layout.addWidget(self.bpm_spin)
        controls_layout.addWidget(self.beats_spin)
        controls_layout.addWidget(QLabel("网格:"))
        controls_layout.addWidget(self.grid_combo)
        
        controls_layout.addStretch()
        
        layout.addLayout(controls_layout)
        
        # Main Content Area (Splitter: Timeline / Piano Roll)
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Timeline Panel
        # Initialize timeline panel with default track_index and hop_size
        hop_size = self.processor.config.get('hop_size', 512)  # Default to 512 if not set
        self.timeline_panel = TimelinePanel(parent_gui=self)
        self.timeline_panel.hop_size = hop_size
        self.timeline_panel.trackSelected.connect(self.on_track_selected)
        self.timeline_panel.filesDropped.connect(self.on_files_dropped)
        self.timeline_panel.cursorMoved.connect(self.on_timeline_cursor_moved)
        self.timeline_panel.trackTypeChanged.connect(self.convert_track_type)
        splitter.addWidget(self.timeline_panel)
        
        # Plot Area (Piano Roll) Container
        self.plot_container = QWidget()
        self.plot_layout = QHBoxLayout(self.plot_container)
        self.plot_layout.setContentsMargins(0, 0, 0, 0)
        self.plot_layout.setSpacing(0)

        self.plot_widget = pg.PlotWidget(
            viewBox=CustomViewBox(self), 
            axisItems={
                'left': PianoRollAxis(orientation='left'),
                'top': BPMAxis(self, orientation='top'),
                'bottom': pg.AxisItem(orientation='bottom') # Standard axis, will be hidden
            }
        )
        
        # Set fixed width for left axis to align with track controls
        # self.plot_widget.getAxis('left').setWidth(CONTROL_PANEL_WIDTH)
        
        # Link Timeline X axis to Plot Widget X axis
        # self.timeline_panel.ruler_plot.setXLink(self.plot_widget) # Decoupled as requested
        
        # Disable AutoRange to prevent crash on startup with infinite items
        self.plot_widget.plotItem.vb.disableAutoRange()
        self.timeline_panel.ruler_plot.plotItem.vb.disableAutoRange()
        
        self.plot_widget.setBackground('#2b2b2b')
        self.plot_widget.setLabel('left', '音高 (Note)')
        # Disable default X grid, keep Y grid
        self.plot_widget.showGrid(x=False, y=True, alpha=0.5)
        self.plot_widget.getAxis('left').setGrid(128)
        self.plot_widget.setMouseEnabled(x=True, y=True)
        
        # Add Custom Music Grid
        self.music_grid = MusicGridItem(self)
        self.plot_widget.addItem(self.music_grid)
        
        # Configure Axes
        self.plot_widget.showAxis('top')
        self.plot_widget.hideAxis('bottom')
        # self.plot_widget.getAxis('top').setLabel('小节-拍') # Removed label as requested
        
        # Limit Y range: Start from C0 (MIDI 12)
        # 12 octaves from C0 is plenty (12 + 144 = 156)
        self.plot_widget.setLimits(yMin=12, yMax=156)
        self.plot_widget.setYRange(60, 72, padding=0) # Initial view: C4 to C5

        # Scrollbar for Piano Roll
        self.plot_scrollbar = QScrollBar(Qt.Orientation.Vertical)
        self.plot_scrollbar.setRange(0, 100) # Will be updated dynamically
        self.plot_scrollbar.valueChanged.connect(self.on_plot_scroll)
        
        self.plot_layout.addWidget(self.plot_widget)
        self.plot_layout.addWidget(self.plot_scrollbar)
        
        splitter.addWidget(self.plot_container)
        splitter.setSizes([200, 600])
        
        layout.addWidget(splitter)
        
        # Set Limits
        self.plot_widget.setLimits(xMin=0)
        self.timeline_panel.ruler_plot.setLimits(xMin=0)
        
        # Connect ViewBox Y range change to scrollbar
        self.plot_widget.plotItem.vb.sigYRangeChanged.connect(self.update_plot_scrollbar)

        # Playback Cursor
        self.play_cursor = PlaybackCursorItem()
        self.plot_widget.addItem(self.play_cursor)
        
        # Waveform View
        self.waveform_view = pg.ViewBox()
        self.waveform_view.setMouseEnabled(x=False, y=False)
        self.waveform_view.setMenuEnabled(False)
        self.waveform_view.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.waveform_view.setXLink(self.plot_widget.plotItem.vb)
        self.waveform_view.setYRange(-1, 1)
        self.waveform_view.setZValue(-1)
        self.plot_widget.scene().addItem(self.waveform_view)
        
        self.plot_widget.plotItem.vb.sigResized.connect(self.update_views)

        # Custom Mouse Interaction
        self.plot_widget.scene().sigMouseMoved.connect(self.on_scene_mouse_move)
        self.plot_widget.scene().sigMouseClicked.connect(self.on_scene_mouse_click)
        
        # Curves
        self.waveform_curve = pg.PlotCurveItem(pen=pg.mkPen(color=(255, 255, 255, 30), width=1), name="Waveform")
        self.waveform_view.addItem(self.waveform_curve)
        
        self.f0_orig_curve_item = self.plot_widget.plot(pen=pg.mkPen(color=(255, 255, 255, 80), width=2, style=Qt.PenStyle.DashLine), name="Original F0")
        self.f0_curve_item = self.plot_widget.plot(pen=pg.mkPen('#00ff00', width=3), name="F0")
        
        # Instructions
        instructions = QLabel("使用说明: 加载模型 -> 加载音频 -> 左键绘制音高，右键还原音高，中键拖动，Ctrl+滚轮缩放X，Alt+滚轮缩放Y -> 合成")
        layout.addWidget(instructions)
        
        # Update timeline bounds to ensure limits are applied to plot_widget
        self.timeline_panel.update_timeline_bounds()
        
        # Ensure initial view range is set correctly after linking
        self.timeline_panel.set_initial_view_range()
        
        # Status
        self.status_label = QLabel("就绪")
        layout.addWidget(self.status_label)

    def on_grid_changed(self, index):
        resolutions = [4, 8, 16, 32]
        if index < len(resolutions):
            res = resolutions[index]
            self.music_grid.set_resolution(res)
            # Update all track grids
            if hasattr(self, 'timeline_panel'):
                for row in self.timeline_panel.rows:
                    row.lane.music_grid.set_resolution(res)

    def on_bpm_changed(self):
        self.plot_widget.getAxis('top').update()
        self.music_grid.update()
        # Update all track grids
        if hasattr(self, 'timeline_panel'):
            for row in self.timeline_panel.rows:
                row.lane.music_grid.update()

    def on_beats_changed(self):
        self.plot_widget.getAxis('top').update()
        self.music_grid.update()
        # Update all track grids
        if hasattr(self, 'timeline_panel'):
            for row in self.timeline_panel.rows:
                row.lane.music_grid.update()

    def update_plot_scrollbar(self, vb, range):
        # range is (minY, maxY)
        min_y, max_y = range
        view_height = max_y - min_y
        
        # Total range: 12 to 156
        total_min = 12
        total_max = 156
        total_height = total_max - total_min
        
        # Scrollbar represents the "Top" of the view relative to total range
        # Inverted: 0 is Top (156), Max is Bottom (12 + view_height)
        
        # Calculate scrollbar max
        # The scrollbar "value" usually corresponds to the top of the slider
        # If slider is at top (value=0), view top should be total_max
        # If slider is at bottom (value=max), view bottom should be total_min
        # i.e. view top should be total_min + view_height
        
        # Let's map scrollbar value (0..1000) to view top (total_max .. total_min + view_height)
        
        self.plot_scrollbar.blockSignals(True)
        
        # Update page step
        # Page step is proportional to view height
        # Let's use a fixed large range for scrollbar for smoothness
        sb_max = 1000
        self.plot_scrollbar.setRange(0, sb_max)
        self.plot_scrollbar.setPageStep(int(sb_max * (view_height / total_height)))
        
        # Calculate value
        # Ratio of (total_max - current_top) / (total_max - (total_min + view_height))
        # Wait, simpler:
        # Available scrollable height = total_height - view_height
        # Current scroll position (from top) = total_max - max_y
        
        scrollable_height = total_height - view_height
        if scrollable_height <= 0:
            self.plot_scrollbar.setValue(0)
        else:
            ratio = (total_max - max_y) / scrollable_height
            val = int(ratio * sb_max)
            self.plot_scrollbar.setValue(val)
            
        self.plot_scrollbar.blockSignals(False)

    def on_plot_scroll(self, value):
        # Calculate new top
        sb_max = self.plot_scrollbar.maximum()
        if sb_max == 0:
            return
            
        ratio = value / sb_max
        
        # Get current view height
        current_range = self.plot_widget.plotItem.vb.viewRange()[1]
        view_height = current_range[1] - current_range[0]
        
        total_min = 12
        total_max = 156
        total_height = total_max - total_min
        scrollable_height = total_height - view_height
        
        # New Top = Total Max - (Ratio * Scrollable Height)
        new_top = total_max - (ratio * scrollable_height)
        new_bottom = new_top - view_height
        
        self.plot_widget.plotItem.vb.setYRange(new_bottom, new_top, padding=0)

    def set_tool_mode(self, mode):
        self.tool_mode = mode
        if mode == 'draw':
            self.plot_widget.setCursor(Qt.CursorShape.CrossCursor)
            self.status_label.setText("工具: 绘制 (左键绘制音高, 右键擦除)")
        elif mode == 'move':
            self.plot_widget.setCursor(Qt.CursorShape.OpenHandCursor)
            self.status_label.setText("工具: 移动 (左键拖动音轨)")

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key.Key_Space:
            self.toggle_playback()
        elif ev.modifiers() & Qt.KeyboardModifier.ControlModifier:
            if ev.key() == Qt.Key.Key_Z:
                if ev.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                    self.redo()
                else:
                    self.undo()
            elif ev.key() == Qt.Key.Key_Y:
                self.redo()
        else:
            super().keyPressEvent(ev)

    def push_undo(self):
        track = self.current_track
        if not track or track.track_type != 'vocal' or track.f0_edited is None:
            return
            
        # Deep copy current state
        track.undo_stack.append(track.f0_edited.copy())
        # Limit stack size
        if len(track.undo_stack) > 16:
            track.undo_stack.pop(0)
        # Clear redo stack on new action
        track.redo_stack.clear()

    def undo(self):
        track = self.current_track
        if not track or track.track_type != 'vocal':
            return

        if not track.undo_stack:
            self.status_label.setText("没有可撤销的操作")
            return
            
        # Save current state to redo
        track.redo_stack.append(track.f0_edited.copy())
        
        # Restore from undo
        track.f0_edited = track.undo_stack.pop()
        
        # Mark dirty
        for state in track.segment_states:
            state['dirty'] = True
            
        self.update_plot()
        self.status_label.setText("撤销")

    def redo(self):
        track = self.current_track
        if not track or track.track_type != 'vocal':
            return

        if not track.redo_stack:
            self.status_label.setText("没有可重做的操作")
            return
            
        # Save current state to undo
        track.undo_stack.append(track.f0_edited.copy())
        
        # Restore from redo
        track.f0_edited = track.redo_stack.pop()
        
        # Mark dirty
        for state in track.segment_states:
            state['dirty'] = True
            
        self.update_plot()
        self.status_label.setText("重做")

    def toggle_playback(self):
        if self.is_playing:
            self.stop_playback()
        else:
            self.start_playback()

    def start_playback(self):
        # Auto-synthesize if needed
        self.synthesize_audio_only()

        try:
            if not self.tracks:
                return

            if not self.is_playing:
                self.playback_start_time = self.current_playback_time

            # Mix audio
            mixed_audio = self.mix_tracks()
            if mixed_audio is None:
                return
            
            sr = self.processor.config['audio_sample_rate'] if self.processor.config else 44100
            total_duration = len(mixed_audio) / sr
            
            if self.current_playback_time >= total_duration:
                self.current_playback_time = 0
                self.play_cursor.setValue(0)
            
            start_sample = int(self.current_playback_time * sr)
            audio_to_play = mixed_audio[start_sample:]
            
            if len(audio_to_play) == 0:
                self.current_playback_time = 0
                self.play_cursor.setValue(0)
                audio_to_play = mixed_audio
            
            sd.play(audio_to_play, sr)
            self.is_playing = True
            self.last_wall_time = time.time()
            self.playback_timer.start()
            self.status_label.setText("正在播放...")
            
        except Exception as e:
            print(f"Playback error: {e}")
            self.stop_playback()

    def synthesize_audio_only(self):
        """Helper to synthesize all dirty tracks"""
        try:
            self.status_label.setText("正在合成...")
            QApplication.processEvents()
            
            hop_size = self.processor.config['hop_size'] if self.processor.config else 512
            
            for track in self.tracks:
                if track.track_type == 'vocal':
                    # Check dirty segments
                    for i, state in enumerate(track.segment_states):
                        if state['dirty']:
                            track.synthesize_segment(self.processor, i)
                    
                    # Update full audio buffer for the track
                    track.update_full_audio(hop_size)
            
            self.status_label.setText("合成完成")
        except Exception as e:
            print(f"Auto-synthesis failed: {e}")
            import traceback
            traceback.print_exc()
            self.status_label.setText("自动合成失败")

    def pause_playback(self):
        if not self.is_playing: return
        
        sd.stop()
        self.is_playing = False
        self.playback_timer.stop()
        
        # Update current time one last time
        now = time.time()
        self.current_playback_time += now - self.last_wall_time
        self.status_label.setText("暂停")

    def stop_playback(self, reset=False):
        sd.stop()
        self.is_playing = False
        self.playback_timer.stop()
        
        if reset:
            self.current_playback_time = 0
            self.play_cursor.setValue(0)
            self.playback_start_time = 0
        else:
            # Return to start position
            self.current_playback_time = self.playback_start_time
            if self.processor.config:
                hop_size = self.processor.config['hop_size']
                sr = self.processor.config.get('audio_sample_rate', 44100)
                self.play_cursor.setValue(self.current_playback_time * sr / hop_size)
                self.timeline_panel.set_cursor_position(self.current_playback_time * sr / hop_size)
                
        self.status_label.setText("停止")

    def set_playback_position(self, x_frame):
        if x_frame < 0: x_frame = 0
        
        # Update cursor visual
        self.play_cursor.setValue(x_frame)
        self.timeline_panel.set_cursor_position(x_frame)
        
        # Update internal time
        if self.processor.config:
            hop_size = self.processor.config['hop_size']
            sr = self.processor.config.get('audio_sample_rate', 44100)
            self.current_playback_time = x_frame * hop_size / sr
            self.playback_start_time = self.current_playback_time # Update start time on seek
            
            if self.is_playing:
                # Restart playback from new position
                sd.stop()
                self.playback_timer.stop()
                self.start_playback()

    def update_cursor(self):
        if not self.is_playing: return
        
        now = time.time()
        dt = now - self.last_wall_time
        self.last_wall_time = now
        
        self.current_playback_time += dt
        
        # Convert time to x (frames)
        # x = time * sr / hop_size
        if self.processor.config:
            hop_size = self.processor.config['hop_size']
            sr = self.processor.config.get('audio_sample_rate', 44100)
            x = self.current_playback_time * sr / hop_size
            self.play_cursor.setValue(x)
            self.timeline_panel.set_cursor_position(x)
            
            # Auto scroll if cursor goes out of view?
            # view_range = self.plot_widget.viewRange()[0]
            # if x > view_range[1]:
            #     self.plot_widget.plotItem.vb.translateBy(x - view_range[0])

        # Check if finished
        # Note: synthesized_audio is now per track, but we might have a mixed buffer?
        # Actually, start_playback mixes audio. We don't store mixed audio in self.synthesized_audio anymore?
        # Wait, start_playback plays directly.
        # We need to check if playback is done.
        # Since we use sounddevice, we can just check if we are past the end.
        # But we don't know the total length easily here unless we store it.
        # Let's just rely on user stopping or loop?
        # Or better, check against the longest track.
        pass

    def update_views(self):
        self.waveform_view.setGeometry(self.plot_widget.plotItem.vb.sceneBoundingRect())
        self.waveform_view.linkedViewChanged(self.plot_widget.plotItem.vb, self.waveform_view.XAxis)
        # Sync timeline view X range if needed (already linked via setXLink)
        pass

    def load_model_dialog(self):
        folder = QFileDialog.getExistingDirectory(self, "选择模型文件夹")
        if folder:
            self.load_model(folder)

    def load_default_model(self):
        default_path = config_manager.get_default_model_path()
        if default_path and os.path.exists(default_path):
            try:
                self.processor.load_model(default_path)
                self.model_path = default_path
                if hasattr(self, 'status_label'):
                    self.status_label.setText(f"已加载默认模型: {pathlib.Path(default_path).name}")
            except Exception as e:
                if hasattr(self, 'status_label'):
                    self.status_label.setText(f"加载默认模型失败: {e}")

    def set_default_model_dialog(self):
        folder_path = QFileDialog.getExistingDirectory(self, "选择默认模型文件夹")
        if folder_path:
            config_manager.set_default_model_path(folder_path)
            QMessageBox.information(self, "设置成功", f"默认模型已设置为: {folder_path}")
            # Optionally load it now if no model is loaded
            if self.model_path is None:
                self.load_model(folder_path)

    def load_model(self, folder):
        try:
            self.status_label.setText(f"正在加载模型 {folder}...")
            QApplication.processEvents()
            
            self.processor.load_model(folder)
            self.model_path = folder
            
            self.status_label.setText(f"模型已加载: {pathlib.Path(folder).name}")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载模型失败: {str(e)}")
            self.status_label.setText("模型加载失败。")

    def load_audio_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择音频文件", "", "音频文件 (*.wav *.flac *.mp3)")
        if file_path:
            self.add_track_from_file(file_path)

    def on_files_dropped(self, files):
        for file_path in files:
            self.add_track_from_file(file_path)

    def add_track_from_file(self, file_path):
        if not os.path.exists(file_path):
            return
            
        name = os.path.basename(file_path)
        # Default to vocal
        track = Track(name, file_path, track_type='vocal')
        
        try:
            self.status_label.setText(f"正在加载音轨 {name}...")
            QApplication.processEvents()
            
            track.load(self.processor)
            self.tracks.append(track)
            
            # Update Timeline
            self.timeline_panel.hop_size = self.processor.config['hop_size']
            self.timeline_panel.refresh_tracks(self.tracks)
            self.timeline_panel.select_track(len(self.tracks) - 1)
            
            # Trigger selection logic manually since select_track doesn't emit signal
            self.on_track_selected(len(self.tracks) - 1)
            
            self.status_label.setText(f"已加载音轨: {name}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load track: {e}")
            self.status_label.setText("加载失败")

    def on_track_selected(self, index):
        self.current_track_idx = index
        track = self.current_track
        
        # Sync Timeline Selection (if triggered from elsewhere)
        # self.timeline_panel.select_track(index) # Avoid loop if triggered by timeline
        
        if track:
            self.shift_spin.blockSignals(True)
            self.shift_spin.setValue(track.shift_value)
            self.last_shift_value = track.shift_value
            self.shift_spin.blockSignals(False)
            
        self.update_plot()

    def on_timeline_cursor_moved(self, x_frame):
        # Update playback position
        if self.processor.config:
            hop_size = self.processor.config['hop_size']
            sr = self.processor.config.get('audio_sample_rate', 44100)
            
            time_sec = x_frame * hop_size / sr
            self.current_playback_time = time_sec
            self.playback_start_time = time_sec
            
            self.play_cursor.setValue(x_frame)
            
            # If playing, restart from new position?
            if self.is_playing:
                self.stop_playback(reset=False)
                self.start_playback()

    def convert_track_type(self, track, new_type):
        # track is the Track object passed from signal
        if track.track_type == new_type:
            return
            
        track.track_type = new_type
        # Reload
        try:
            self.status_label.setText(f"正在重新加载音轨 {track.name}...")
            QApplication.processEvents()
            track.load(self.processor)
            self.status_label.setText(f"已重新加载: {track.name}")
            
            self.update_plot()
            
            # Update Timeline
            self.timeline_panel.hop_size = self.processor.config['hop_size']
            self.timeline_panel.refresh_tracks(self.tracks)
            self.timeline_panel.select_track(self.current_track_idx)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to reload track: {e}")

            self.status_label.setText("音频加载失败。")

    def mix_tracks(self):
        max_len = 0
        active_tracks = [t for t in self.tracks if not t.muted]
        solo_tracks = [t for t in self.tracks if t.solo]
        if solo_tracks:
            active_tracks = solo_tracks
        
        if not active_tracks:
            return None

        for track in active_tracks:
            if track.synthesized_audio is not None:
                # Calculate length in samples including offset
                hop_size = self.processor.config['hop_size'] if self.processor.config else 512
                start_sample = track.start_frame * hop_size
                end_sample = start_sample + len(track.synthesized_audio)
                max_len = max(max_len, end_sample)
        
        if max_len == 0:
            return None
            
        mixed_audio = np.zeros(max_len, dtype=np.float32)
        
        for track in active_tracks:
            if track.synthesized_audio is not None:
                audio = track.synthesized_audio
                hop_size = self.processor.config['hop_size'] if self.processor.config else 512
                start_sample = track.start_frame * hop_size
                
                # Ensure we don't go out of bounds (shouldn't happen with max_len logic)
                l = len(audio)
                mixed_audio[start_sample:start_sample+l] += audio * track.volume
                
        return mixed_audio

    def export_audio_dialog(self):
        # Ensure everything is synthesized
        self.synthesize_audio_only()
        
        mixed_audio = self.mix_tracks()
        if mixed_audio is None:
            QMessageBox.warning(self, "警告", "没有可导出的音频。请先合成音频或取消静音。")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(self, "导出音频", "output.wav", "WAV Audio (*.wav)")
        if file_path:
            self.export_audio(file_path, mixed_audio)

    def export_audio(self, file_path, audio):
        try:
            self.status_label.setText(f"正在导出到 {file_path}...")
            QApplication.processEvents()
            
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            sr = self.processor.config['audio_sample_rate'] if self.processor.config else 44100
            wavfile.write(file_path, sr, audio)
            
            self.status_label.setText(f"导出成功: {file_path}")
            QMessageBox.information(self, "成功", f"音频已导出到:\n{file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出失败: {str(e)}")
            self.status_label.setText("导出失败。")

    def open_project_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "打开工程", "", "HifiShifter Project (*.hsp *.json)")
        if file_path:
            self.open_project(file_path)

    def open_project(self, file_path):
        try:
            project_dir = os.path.dirname(os.path.abspath(file_path))
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Clear current
            self.tracks = []
            # self.track_list.clear() # Removed in refactor
            self.current_track_idx = -1
            # self.plot_widget.clear() # This is the editor plot, handled by update_plot
            # self.timeline_widget.clear() # Renamed to timeline_panel
            
            # Load Model
            model_path = data.get('model_path')
            if model_path:
                # Check absolute
                if not os.path.exists(model_path):
                    # Check relative
                    rel_path = os.path.join(project_dir, model_path)
                    if os.path.exists(rel_path):
                        model_path = rel_path
                
                if os.path.exists(model_path):
                    self.load_model(model_path)
                else:
                    QMessageBox.warning(self, "警告", f"找不到模型路径: {model_path}")
            
            # Restore Parameters
            if 'params' in data:
                params = data['params']
                if 'bpm' in params: self.bpm_spin.setValue(params['bpm'])
                if 'beats' in params: self.beats_spin.setValue(params['beats'])
            
            # Load Tracks
            if 'tracks' in data:
                for t_data in data['tracks']:
                    file_p = t_data['file_path']
                    # Resolve path
                    if not os.path.exists(file_p):
                        rel_p = os.path.join(project_dir, file_p)
                        if os.path.exists(rel_p):
                            file_p = rel_p
                    
                    if os.path.exists(file_p):
                        track = Track(t_data['name'], file_p, t_data.get('type', 'vocal'))
                        track.load(self.processor)
                        
                        track.shift_value = t_data.get('shift', 0.0)
                        track.muted = t_data.get('muted', False)
                        track.solo = t_data.get('solo', False)
                        track.volume = t_data.get('volume', 1.0)
                        track.start_frame = t_data.get('start_frame', 0)
                        
                        if 'f0' in t_data and track.f0_edited is not None:
                            saved_f0 = np.array(t_data['f0'])
                            # Handle length mismatch if audio changed slightly or different decoding
                            min_len = min(len(saved_f0), len(track.f0_edited))
                            track.f0_edited[:min_len] = saved_f0[:min_len]
                                
                            # Mark dirty
                            for state in track.segment_states:
                                state['dirty'] = True
                        
                        self.tracks.append(track)
                    else:
                         QMessageBox.warning(self, "警告", f"找不到音频文件: {file_p}")
            
            # Backward compatibility for v1.0
            elif 'audio_path' in data:
                audio_path = data['audio_path']
                if not os.path.exists(audio_path):
                     rel_p = os.path.join(project_dir, audio_path)
                     if os.path.exists(rel_p):
                         audio_path = rel_p
                
                if os.path.exists(audio_path):
                    track = Track(os.path.basename(audio_path), audio_path, 'vocal')
                    track.load(self.processor)
                    if 'f0' in data:
                        saved_f0 = np.array(data['f0'])
                        if len(saved_f0) == len(track.f0_edited):
                            track.f0_edited = saved_f0
                    
                    if 'params' in data and 'shift' in data['params']:
                        track.shift_value = data['params']['shift']
                        
                    self.tracks.append(track)

            # Update Timeline
            if self.processor.config:
                self.timeline_panel.hop_size = self.processor.config['hop_size']
            self.timeline_panel.refresh_tracks(self.tracks)

            self.project_path = file_path
            self.status_label.setText(f"工程已加载: {file_path}")
            self.setWindowTitle(f"HifiShifter - {os.path.basename(file_path)}")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"打开工程失败: {str(e)}")
            import traceback
            traceback.print_exc()

    def save_project(self):
        if self.project_path:
            self._save_project_file(self.project_path)
        else:
            self.save_project_as()

    def save_project_as(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "保存工程", "project.hsp", "HifiShifter Project (*.hsp *.json)")
        if file_path:
            self._save_project_file(file_path)

    def _save_project_file(self, file_path):
        try:
            project_dir = os.path.dirname(os.path.abspath(file_path))
            
            tracks_data = []
            for track in self.tracks:
                # Try to make path relative
                try:
                    rel_path = os.path.relpath(track.file_path, project_dir)
                except ValueError:
                    rel_path = track.file_path # Different drive or cannot be relative

                t_data = {
                    'name': track.name,
                    'file_path': rel_path,
                    'type': track.track_type,
                    'shift': track.shift_value,
                    'muted': track.muted,
                    'solo': track.solo,
                    'volume': track.volume,
                    'start_frame': track.start_frame
                }
                if track.track_type == 'vocal' and track.f0_edited is not None:
                    t_data['f0'] = track.f0_edited.tolist()
                tracks_data.append(t_data)

            # Model path relative
            model_path_save = self.model_path
            if self.model_path:
                try:
                    model_path_save = os.path.relpath(self.model_path, project_dir)
                except ValueError:
                    model_path_save = self.model_path

            data = {
                'version': '2.1',
                'model_path': model_path_save,
                'params': {
                    'bpm': self.bpm_spin.value(),
                    'beats': self.beats_spin.value()
                },
                'tracks': tracks_data
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
            
            self.project_path = file_path
            self.status_label.setText(f"工程已保存: {file_path}")
            self.setWindowTitle(f"HifiShifter - {os.path.basename(file_path)}")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存工程失败: {str(e)}")

    def update_plot(self):
        track = self.current_track
        if not track:
            self.waveform_curve.clear()
            self.f0_orig_curve_item.clear()
            self.f0_curve_item.clear()
            return

        # Waveform
        if track.audio is not None:
            hop_size = self.processor.config['hop_size'] if self.processor.config else 512
            ds_factor = max(1, int(hop_size / 4))
            audio_ds = track.audio[::ds_factor]
            # Add start_frame offset to x
            x_ds = (np.arange(len(audio_ds)) * ds_factor / hop_size) + track.start_frame
            
            # Use waveform_view (Y range -1 to 1)
            # Scale to fit nicely in background
            self.waveform_curve.setData(x_ds, audio_ds * 0.8) 
            self.waveform_curve.setPen(pg.mkPen(color=(255, 255, 255, 100), width=1))
            self.waveform_curve.setBrush(pg.mkBrush(color=(255, 255, 255, 30)))
            self.waveform_curve.setFillLevel(0)
        else:
            self.waveform_curve.clear()

        if track.track_type == 'vocal':
            # Create x axis for F0
            x_f0 = np.arange(len(track.f0_original)) + track.start_frame if track.f0_original is not None else None
            
            if track.f0_original is not None:
                self.f0_orig_curve_item.setData(x_f0, track.f0_original, connect="finite")
            else:
                self.f0_orig_curve_item.clear()

            if track.f0_edited is not None:
                self.f0_curve_item.setData(x_f0, track.f0_edited, connect="finite")
            else:
                self.f0_curve_item.clear()
        else:
            self.f0_orig_curve_item.clear()
            self.f0_curve_item.clear()

    def on_scene_mouse_move(self, pos):
        track = self.current_track
        if not track:
            return
            
        buttons = QApplication.mouseButtons()
        is_left = bool(buttons & Qt.MouseButton.LeftButton)
        is_right = bool(buttons & Qt.MouseButton.RightButton)
        
        if not (is_left or is_right):
            self.last_mouse_pos = None
            return

        mouse_point = self.plot_widget.plotItem.vb.mapSceneToView(pos)
        
        if self.tool_mode == 'move' and is_left and self.move_start_x is not None:
            delta = mouse_point.x() - self.move_start_x
            new_start = int(self.move_start_frame + delta)
            if new_start < 0: new_start = 0
            
            if new_start != track.start_frame:
                track.start_frame = new_start
                self.update_plot()
                self.status_label.setText(f"移动音轨: {new_start} 帧")
                
        elif self.tool_mode == 'draw' and track.track_type == 'vocal' and track.f0_edited is not None:
            self.handle_draw(mouse_point, is_left, is_right)

    def on_scene_mouse_click(self, ev):
        track = self.current_track
        if not track:
            return
        
        # Check if event is already accepted (e.g. by Axis)
        if ev.isAccepted():
            return

        # Check if click is within the ViewBox geometry
        vb = self.plot_widget.plotItem.vb
        pos = ev.scenePos()
        if not vb.sceneBoundingRect().contains(pos):
            return
            
        if ev.button() == Qt.MouseButton.LeftButton or ev.button() == Qt.MouseButton.RightButton:
            ev.accept()
            mouse_point = vb.mapSceneToView(pos)
            
            is_left = (ev.button() == Qt.MouseButton.LeftButton)
            is_right = (ev.button() == Qt.MouseButton.RightButton)
            
            if self.tool_mode == 'move' and is_left:
                self.move_start_x = mouse_point.x()
                self.move_start_frame = track.start_frame
                self.plot_widget.setCursor(Qt.CursorShape.ClosedHandCursor)
            elif self.tool_mode == 'draw' and track.track_type == 'vocal' and track.f0_edited is not None:
                self.last_mouse_pos = None
                self.handle_draw(mouse_point, is_left, is_right)

    def handle_draw(self, point, is_left, is_right):
        track = self.current_track
        if not track or track.track_type != 'vocal' or track.f0_edited is None:
            return

        # Adjust x by start_frame
        x = int(point.x()) - track.start_frame
        y = point.y()
        
        # Start of a new stroke?
        if self.last_mouse_pos is None:
            self.push_undo()
        
        changed = False
        affected_range = (x, x) # Track min/max x affected
        
        f0 = track.f0_edited
        f0_orig = track.f0_original
        
        if 0 <= x < len(f0):
            if self.last_mouse_pos is not None:
                last_x, last_y = self.last_mouse_pos
                # Adjust last_x as well? No, last_mouse_pos stores the *index* in f0, not screen coord?
                # Wait, last_mouse_pos was storing (x, y) where x was int(point.x()).
                # So last_mouse_pos was screen coordinates (frames).
                # If I change x to be relative index, I should store relative index in last_mouse_pos too.
                
                start_x, end_x = sorted((last_x, x))
                start_x = max(0, start_x)
                end_x = min(len(f0) - 1, end_x)
                affected_range = (start_x, end_x)
                
                if start_x < end_x:
                    for i in range(start_x, end_x + 1):
                        if is_left:
                            ratio = (i - last_x) / (x - last_x) if x != last_x else 0
                            interp_y = last_y + ratio * (y - last_y)
                            f0[i] = interp_y
                            changed = True
                        elif is_right:
                            if f0_orig is not None:
                                f0[i] = f0_orig[i]
                                changed = True
                else:
                    if is_left:
                        f0[x] = y
                        changed = True
                    elif is_right and f0_orig is not None:
                        f0[x] = f0_orig[x]
                        changed = True
            else:
                if is_left:
                    f0[x] = y
                    changed = True
                elif is_right and f0_orig is not None:
                    f0[x] = f0_orig[x]
                    changed = True
            
            if changed:
                # Mark affected segments as dirty
                min_x, max_x = affected_range
                for i, (seg_start, seg_end) in enumerate(track.segments):
                    # Check overlap
                    if not (max_x < seg_start or min_x >= seg_end):
                        track.segment_states[i]['dirty'] = True
            
            self.last_mouse_pos = (x, y) # Store relative index
            self.update_plot()
            
            if changed:
                self.is_dirty = True
                self.status_label.setText("音高已修改 (未合成)")
    
    def mouseReleaseEvent(self, ev):
        if self.tool_mode == 'move':
            self.move_start_x = None
            self.plot_widget.setCursor(Qt.CursorShape.OpenHandCursor)
            
        self.last_mouse_pos = None
        super().mouseReleaseEvent(ev)

    def apply_shift(self, semitones):
        track = self.current_track
        if not track or track.track_type != 'vocal' or track.f0_edited is None:
            return
            
        delta = semitones - self.last_shift_value
        track.f0_edited += delta
        track.shift_value = semitones
        self.last_shift_value = semitones
        
        # Mark all segments as dirty on global shift
        for state in track.segment_states:
            state['dirty'] = True
        self.update_plot()

    def play_original(self):
        track = self.current_track
        if track and track.audio is not None:
            sd.stop()
            sd.play(track.audio, track.sr)

    def synthesize_and_play(self):
        self.synthesize_audio_only()
        if self.synthesized_audio is not None:
            self.stop_playback(reset=True)
            self.start_playback()

    def delete_track(self, index):
        if 0 <= index < len(self.tracks):
            reply = QMessageBox.question(self, 'Delete Track', 
                                         f"Are you sure you want to delete track '{self.tracks[index].name}'?",
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, 
                                         QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                del self.tracks[index]
                if self.current_track_idx == index:
                    self.current_track_idx = -1
                elif self.current_track_idx > index:
                    self.current_track_idx -= 1
                
                self.timeline_panel.refresh_tracks(self.tracks)
                self.update_plot()

    def copy_pitch(self, index):
        if 0 <= index < len(self.tracks):
            track = self.tracks[index]
            if track.f0_edited is not None:
                self.pitch_clipboard = track.f0_edited.copy()
                self.status_label.setText(f"Copied pitch from track '{track.name}'")
            else:
                self.status_label.setText("No pitch data to copy")

    def paste_pitch(self, index):
        if 0 <= index < len(self.tracks) and self.pitch_clipboard is not None:
            track = self.tracks[index]
            if track.f0_original is None:
                 QMessageBox.warning(self, "Paste Error", "Target track has no audio/pitch data.")
                 return

            target_len = len(track.f0_original)
            source_len = len(self.pitch_clipboard)
            
            if target_len != source_len:
                reply = QMessageBox.question(self, 'Paste Pitch', 
                                             f"Pitch length mismatch (Source: {source_len}, Target: {target_len}). Paste anyway? (Will be truncated/padded)",
                                             QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, 
                                             QMessageBox.StandardButton.No)
                if reply != QMessageBox.StandardButton.Yes:
                    return
            
            new_f0 = np.zeros(target_len)
            copy_len = min(target_len, source_len)
            new_f0[:copy_len] = self.pitch_clipboard[:copy_len]
            
            if copy_len < target_len:
                 new_f0[copy_len:] = track.f0_original[copy_len:]
            
            track.f0_edited = new_f0
            track.is_edited = True
            
            # Mark all segments as dirty
            for state in track.segment_states:
                state['dirty'] = True
                
            self.update_plot()
            self.status_label.setText(f"Pasted pitch to track '{track.name}'")
        else:
             self.status_label.setText("Clipboard empty or invalid index")

    def stop_audio(self):
        self.stop_playback()
