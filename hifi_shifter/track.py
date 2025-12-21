import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wavfile
import torch
import torchaudio

class Track:
    def __init__(self, name, file_path, track_type='vocal'):
        self.name = name
        self.file_path = file_path
        self.track_type = track_type # 'vocal' or 'bgm'
        
        # Audio Data
        self.audio = None # Original Audio (numpy array)
        self.sr = None
        
        # Vocal Specific Data
        self.mel = None
        self.f0_original = None
        self.f0_edited = None
        self.segments = []
        self.segment_states = [] # List of dicts: {'dirty': bool, 'audio': np.array}
        
        # Playback
        self.synthesized_audio = None # For vocal tracks, this is the result. For BGM, it's just self.audio
        self.volume = 1.0
        self.muted = False
        self.solo = False
        self.shift_value = 0.0
        self.start_frame = 0 # Offset in frames (hop_size)
        
        # Undo/Redo
        self.undo_stack = []
        self.redo_stack = []

    def load(self, processor):
        """
        Load audio data.
        If track_type is 'vocal', use processor to extract features.
        If track_type is 'bgm', just load audio.
        """
        try:
            if self.track_type == 'vocal':
                self.audio, self.sr, self.mel, self.f0_original, self.segments = processor.process_audio(self.file_path)
                self.f0_edited = self.f0_original.copy()

                # Initialize segment states
                self.segment_states = []
                for _ in self.segments:
                    self.segment_states.append({'dirty': True, 'audio': None})

            else:
                # Load BGM
                target_sr = processor.config['audio_sample_rate']

                audio, sr = torchaudio.load(self.file_path)
                if audio.shape[0] > 1:
                    audio = torch.mean(audio, dim=0, keepdim=True)

                if sr != target_sr:
                    resampler = torchaudio.transforms.Resample(sr, target_sr)
                    audio = resampler(audio)
                    sr = target_sr

                self.audio = audio[0].numpy()
                self.sr = sr
                self.synthesized_audio = self.audio

            # Ensure start_frame is initialized correctly
            self.start_frame = int(self.start_frame) if self.start_frame is not None else 0

            # Ensure segments are valid and not None
            self.segments = self.segments if self.segments is not None else []

            # Validate segments to ensure they are non-empty and valid
            self.segments = [(max(0, start), max(start, end)) for start, end in self.segments if start is not None and end is not None]

        except Exception as e:
            raise ValueError(f"Failed to load track: {e}")

    def synthesize_segment(self, processor, segment_idx):
        if self.track_type != 'vocal':
            return

        if not self.segment_states[segment_idx]['dirty']:
            return

        start, end = self.segments[segment_idx]
        f0_segment = self.f0_edited[start:end]
        
        audio_segment = processor.synthesize_segment(self.mel, self.segments[segment_idx], f0_segment)
        self.segment_states[segment_idx]['audio'] = audio_segment
        self.segment_states[segment_idx]['dirty'] = False

    def get_audio_for_playback(self):
        """
        Construct the full audio for playback.
        For vocal tracks, stitch segments together.
        """
        if self.track_type == 'bgm':
            return self.audio
        
        # Stitch segments
        # This might be slow if done every frame, so we should cache it
        # But for now, let's assume we update self.synthesized_audio when segments change
        
        if self.synthesized_audio is None:
             self.synthesized_audio = np.zeros_like(self.audio)
             
        # We need to update synthesized_audio from segments
        # Ideally, we only update the parts that changed
        # But for simplicity, let's reconstruct
        
        # Actually, to avoid gaps, we should probably crossfade or just rely on the segmenting logic
        # The segmenting logic in AudioProcessor returns non-overlapping segments that cover the whole file (including silence)
        # Wait, my segment logic in AudioProcessor skips silence?
        # Let's check AudioProcessor._segment_audio again.
        # It returns segments of speech. Gaps are silence.
        
        # If gaps are silence, we need to fill them with zeros or original audio?
        # Usually vocoder output for silence is noise or silence.
        # If we only synthesize segments, we need to fill the rest with silence.
        
        full_audio = np.zeros_like(self.audio)
        
        for i, (start, end) in enumerate(self.segments):
            seg_audio = self.segment_states[i]['audio']
            if seg_audio is not None:
                # Map frames to samples
                # hop_size is needed here.
                # We need access to hop_size.
                pass
                
        return self.synthesized_audio

    def update_full_audio(self, hop_size):
        if self.track_type == 'bgm':
            return

        if self.synthesized_audio is None:
            self.synthesized_audio = np.zeros_like(self.audio)
            
        for i, (start, end) in enumerate(self.segments):
            seg_audio = self.segment_states[i]['audio']
            if seg_audio is not None:
                s_sample = start * hop_size
                e_sample = s_sample + len(seg_audio)
                
                # Ensure bounds
                if e_sample > len(self.synthesized_audio):
                    e_sample = len(self.synthesized_audio)
                    seg_audio = seg_audio[:e_sample-s_sample]
                
                self.synthesized_audio[s_sample:e_sample] = seg_audio
        
        # Ensure start_frame is always an integer
        self.start_frame = int(self.start_frame) if self.start_frame is not None else 0
