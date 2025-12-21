import os
import json
import pathlib
import numpy as np
import torch
import torchaudio
import yaml

try:
    from training.nsf_HiFigan_task import nsf_HiFigan, dynamic_range_compression_torch
    from utils.config_utils import read_full_config
    from utils.wav2F0 import get_pitch
    from utils.wav2mel import PitchAdjustableMelSpectrogram
except ImportError as e:
    print(f"Error importing modules in AudioProcessor: {e}")

class AudioProcessor:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.config = {}
        self.mel_transform = None
        
    def load_model(self, folder_path):
        """
        Load model and configuration from the specified folder.
        """
        folder_path = pathlib.Path(folder_path)
        
        # Check for config
        config_path = folder_path / 'config.yaml'
        if not config_path.exists():
            config_path = folder_path / 'config.json'
        
        if not config_path.exists():
            raise FileNotFoundError("目录中未找到 config.yaml 或 config.json。")
        
        # Load config
        if config_path.suffix == '.yaml':
            self.config = read_full_config(config_path)
        else:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        
        # Patch config
        self._patch_config()
        
        # Check for checkpoint
        ckpt_path = folder_path / 'model.ckpt'
        if not ckpt_path.exists():
            ckpts = list(folder_path.glob('*.ckpt'))
            if ckpts:
                ckpt_path = ckpts[0]
            else:
                raise FileNotFoundError("目录中未找到 .ckpt 文件。")
        
        # Initialize Model
        self.model = nsf_HiFigan(self.config)
        self.model.build_model()
        
        # Load weights
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)
        
        # Handle nested generator checkpoint
        if 'generator' in state_dict and isinstance(state_dict['generator'], dict) and len(state_dict) == 1:
            self.model.generator.load_state_dict(state_dict['generator'])
        else:
            self.model.load_state_dict(state_dict, strict=False)

        self.model.to(self.device)
        self.model.eval()
        
        # Initialize Mel Transform
        self.mel_transform = PitchAdjustableMelSpectrogram(
            sample_rate=self.config['audio_sample_rate'],
            n_fft=self.config['fft_size'],
            win_length=self.config['win_size'],
            hop_length=self.config['hop_size'],
            f_min=self.config['fmin'],
            f_max=self.config['fmax'],
            n_mels=self.config['audio_num_mel_bins']
        )
        
        return self.config

    def _patch_config(self):
        if 'clip_grad_norm' not in self.config:
            self.config['clip_grad_norm'] = 1.0
        
        key_mapping = {
            'sampling_rate': 'audio_sample_rate',
            'num_mels': 'audio_num_mel_bins',
            'n_fft': 'fft_size'
        }
        for old_key, new_key in key_mapping.items():
            if old_key in self.config and new_key not in self.config:
                self.config[new_key] = self.config[old_key]

        if 'f0_min' not in self.config:
            self.config['f0_min'] = 40
        if 'f0_max' not in self.config:
            self.config['f0_max'] = 1600

        if 'model_args' not in self.config:
            model_arg_keys = [
                'mini_nsf', 'upsample_rates', 'upsample_kernel_sizes', 
                'upsample_initial_channel', 'resblock_kernel_sizes', 
                'resblock_dilation_sizes', 'resblock', 'discriminator_periods'
            ]
            self.config['model_args'] = {}
            for key in model_arg_keys:
                if key in self.config:
                    self.config['model_args'][key] = self.config[key]

    def process_audio(self, file_path):
        """
        Load audio, resample, and extract features (Mel, F0).
        Returns:
            audio_np: numpy array of audio data
            sr: sample rate
            mel: Mel spectrogram tensor
            f0_midi: numpy array of F0 (MIDI)
            segments: list of (start, end) tuples
        """
        if self.model is None:
            raise RuntimeError("请先加载模型以确保采样率正确。")
            
        audio, sr = torchaudio.load(file_path)
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        
        target_sr = self.config['audio_sample_rate']
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            audio = resampler(audio)
            sr = target_sr
        
        # Extract Mel
        mel = dynamic_range_compression_torch(self.mel_transform(audio, key_shift=0))
        
        # Extract F0
        f0_np, uv = get_pitch(
            'parselmouth',
            audio[0].numpy(), 
            hparams=self.config, 
            speed=1, 
            interp_uv=True, 
            length=mel.shape[2]
        )
        
        # Convert F0 to MIDI
        f0_midi = np.zeros_like(f0_np)
        mask = f0_np > 0
        f0_midi[mask] = 69 + 12 * np.log2(f0_np[mask] / 440.0)
        f0_midi[~mask] = np.nan
        
        # Segment audio based on silence
        segments = self._segment_audio(mel)
        
        return audio[0].numpy(), sr, mel, f0_midi, segments

    def _segment_audio(self, mel, threshold_db=-60, min_silence_frames=100):
        """
        Segment audio based on Mel-spectrogram energy.
        """
        # Calculate energy from Mel (approximate)
        mel_np = mel.squeeze().cpu().numpy()
        energy = np.mean(mel_np, axis=0)
        
        # Normalize energy
        energy_db = 20 * np.log10(np.maximum(energy, 1e-5))
        energy_db = energy_db - np.max(energy_db)
        
        is_speech = energy_db > threshold_db
        
        # Dilate to merge close segments (fill short silences)
        from scipy.ndimage import binary_dilation
        # Merge silences shorter than min_silence_frames
        struct = np.ones(min_silence_frames)
        is_speech = binary_dilation(is_speech, structure=struct)
        
        # Find segments
        segments = []
        start = None
        for i, active in enumerate(is_speech):
            if active and start is None:
                start = i
            elif not active and start is not None:
                segments.append((start, i))
                start = None
        if start is not None:
            segments.append((start, len(is_speech)))
            
        # Ensure segments are valid
        if not segments:
            segments = [(0, len(is_speech))]  # Default to one segment if none found
        segments = [(max(0, start), max(start, end)) for start, end in segments]
            
        return segments

    def synthesize_segment(self, mel, segment, f0_midi_segment):
        """
        Synthesize a specific segment with context padding to avoid artifacts.
        """
        if self.model is None:
            raise RuntimeError("模型未加载")
            
        start, end = segment
        pad_frames = 64 # Increased context window size (approx 0.7s)
        
        # Calculate padded range
        p_start = max(0, start - pad_frames)
        p_end = min(mel.shape[2], end + pad_frames)
        
        # Get Mel slice with padding
        mel_slice = mel[:, :, p_start:p_end].to(self.device)
        
        # Prepare F0 with padding
        # f0_midi_segment corresponds to [start:end]
        # We need to pad it to match [p_start:p_end]
        
        pre_pad = start - p_start
        post_pad = p_end - end
        
        # Ensure input segment matches expected length
        expected_len = end - start
        if len(f0_midi_segment) != expected_len:
             if len(f0_midi_segment) < expected_len:
                f0_midi_segment = np.pad(f0_midi_segment, (0, expected_len - len(f0_midi_segment)), constant_values=np.nan)
             else:
                f0_midi_segment = f0_midi_segment[:expected_len]
        
        # Construct full F0 for synthesis
        f0_padded = np.pad(f0_midi_segment, (pre_pad, post_pad), constant_values=np.nan)
        
        # Convert F0
        f0_hz = np.zeros_like(f0_padded)
        mask = ~np.isnan(f0_padded)
        f0_hz[mask] = 440.0 * (2 ** ((f0_padded[mask] - 69) / 12.0))
        f0_hz[~mask] = 0
        
        f0_tensor = torch.from_numpy(f0_hz).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model.Gforward(sample={'mel': mel_slice, 'f0': f0_tensor})['audio']
            
        audio_padded = output[0].cpu().numpy()
        if audio_padded.ndim == 2:
            audio_padded = audio_padded.squeeze(0)
            
        # Trim padding from audio
        hop_size = self.config['hop_size']
        trim_start = pre_pad * hop_size
        trim_end = len(audio_padded) - (post_pad * hop_size)
        
        # Safety check
        if trim_end <= trim_start:
            return audio_padded # Should not happen
            
        return audio_padded[trim_start:trim_end]

    def synthesize(self, mel, f0_midi):
        """
        Synthesize audio using the modified F0.
        Args:
            mel: Mel spectrogram tensor
            f0_midi: numpy array of F0 in MIDI scale
        Returns:
            synthesized_audio: numpy array
        """
        if self.model is None:
            raise RuntimeError("模型未加载")

        mel_tensor = mel.to(self.device)
        
        # Convert MIDI back to Hz
        f0_hz = np.zeros_like(f0_midi)
        mask = ~np.isnan(f0_midi)
        f0_hz[mask] = 440.0 * (2 ** ((f0_midi[mask] - 69) / 12.0))
        f0_hz[~mask] = 0
        
        f0_tensor = torch.from_numpy(f0_hz).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model.Gforward(sample={'mel': mel_tensor, 'f0': f0_tensor})['audio']
        
        synthesized_audio = output[0].cpu().numpy()
        
        if synthesized_audio.ndim == 2 and synthesized_audio.shape[0] == 1:
             synthesized_audio = synthesized_audio.squeeze(0)
             
        return synthesized_audio

        # Ensure config is initialized as a dictionary
        if self.config is None:
            self.config = {'hop_size': 512}  # Default configuration
