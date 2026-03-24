import argparse
import pathlib
import onnx
import onnxsim
import torch
import torch.nn.functional as F
import yaml

from lib.istft import iSTFT
from lib.nets import CascadedNet
from logger.utils import DotDict


class CascadedNetONNX(CascadedNet):
    def __init__(self, n_fft, hop_length, nout=32, nout_lstm=128, fixed_length=True):
        super().__init__(n_fft, hop_length, nout, nout_lstm, True, True, fixed_length)
        self.istft = iSTFT(
            win_len=n_fft,
            win_hop=hop_length,
            fft_len=n_fft,
            window=self.window
        )

    def bounded_mask(self, mask, eps=1e-8):  # [B, 2, N, T]
        mask_real, mask_imag = torch.split(
            mask, [1, 1], dim=1
        )  # [B, 1, N, T]
        mask_abs = torch.sqrt(mask_real ** 2 + mask_imag ** 2).repeat(1, 2, 1, 1)  # [B, 2, N, T]
        mask = torch.tanh(mask_abs) * mask / (mask_abs + eps)
        return mask

    def _forward(self, x):
        x = x[:, :, :self.max_bin]

        bandw = x.size(2) // 2
        l1_in = x[:, :, :bandw]
        h1_in = x[:, :, bandw:]
        l1 = self.stg1_low_band_net(l1_in)
        h1 = self.stg1_high_band_net(h1_in)
        aux1 = torch.cat([l1, h1], dim=2)

        l2_in = torch.cat([l1_in, l1], dim=1)
        h2_in = torch.cat([h1_in, h1], dim=1)
        l2 = self.stg2_low_band_net(l2_in)
        h2 = self.stg2_high_band_net(h2_in)
        aux2 = torch.cat([l2, h2], dim=2)

        f3_in = torch.cat([x, aux1, aux2], dim=1)
        f3 = self.stg3_full_band_net(f3_in)

        mask = self.out(f3)  # [B, 2, N, T]
        mask = self.bounded_mask(mask)

        mask = F.pad(
            input=mask,
            pad=(0, 0, 0, self.output_bin - mask.size()[2]),
            mode='replicate'
        )

        return mask

    def forward(self, x):  # [B, T]
        x_gt = x
        T = x.size(1)
        T1 = T + self.hop_length
        T_pad = self.seg_length * ((T1 - 1) // self.seg_length + 1) - T1
        nl_pad = T_pad // 2 // self.hop_length
        Tl_pad = nl_pad * self.hop_length
        x = F.pad(x, (Tl_pad, T_pad - Tl_pad))
        spec = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            return_complex=False,
            window=self.window,
            pad_mode='constant'
        )  # [B, N, T, 2]
        spec = spec.permute(0, 3, 1, 2)  # [B, 2, N, T]
        mask = self._forward(spec)  # [B, 2, N, T]
        spec_real, spec_imag = torch.split(spec, [1, 1], dim=1)
        mask_real, mask_imag = torch.split(mask, [1, 1], dim=1)
        spec_pred = torch.cat([
            spec_real * mask_real - spec_imag * mask_imag,
            spec_real * mask_imag + spec_imag * mask_real
        ], dim=1)
        spec_pred = spec_pred.permute(0, 2, 3, 1)  # [B, N, T, 2]
        # x_pred = torch.istft(spec_pred, self.n_fft, self.hop_length, window=self.window)
        x_pred = self.istft(spec_pred, length=x.size(1))
        x_pred = x_pred[:, Tl_pad: Tl_pad + T]
        return x_pred, x_gt - x_pred


def load_sep_model(model_path, device='cpu'):
    model_path = pathlib.Path(model_path)
    config_file = model_path.with_name('config.yaml')
    with open(config_file, "r") as config:
        args = yaml.safe_load(config)
    args = DotDict(args)
    assert args.is_mono, 'Only mono mode is supported'
    model = CascadedNetONNX(
        args.n_fft,
        args.hop_length,
        args.n_out,
        args.n_out_lstm,
        True if args.fixed_length is None else args.fixed_length
    )
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('input_path', type=str)
    p.add_argument('output_path', type=str)
    args = p.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_sep_model(args.input_path, device=device)
    waveform = torch.randn(3, 44100, device=device)
    with torch.no_grad():
        torch.onnx.export(
            model,
            waveform,
            args.output_path,
            input_names=['waveform'],
            output_names=['harmonic', 'noise'],
            dynamic_axes={
                'waveform': {0: 'batch_size', 1: 'n_samples'},
                'harmonic': {0: 'batch_size', 1: 'n_samples'},
                'noise': {0: 'batch_size', 1: 'n_samples'},
            },
            opset_version=17,
            dynamo=False
        )
        onnx_model, check = onnxsim.simplify(args.output_path, include_subgraph=True)
        assert check, 'Simplified ONNX model could not be validated'
        onnx.save(onnx_model, args.output_path)
