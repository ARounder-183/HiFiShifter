import pathlib
import argparse
import os
import yaml
import librosa
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm
from lib.nets import CascadedNet

def traverse_dir(
        root_dir,
        extensions: list,
        amount=None,
        str_include=None,
        str_exclude=None,
        is_rel=False,
        is_sort=False,
        is_ext=True
):
    """
    Iterate the files matching the given condition in the given directory and its subdirectories.
    :param root_dir: the root directory
    :param extensions: a list of required file extensions (without ".")
    :param amount: limit the number of files
    :param str_include: require the relative path to include this string
    :param str_exclude: require the relative path not to include this string
    :param is_rel: whether to return the relative path instead of full path
    :param is_sort: whether to sort the final results
    :param is_ext: whether to reserve extensions in the filenames
    """
    root_dir = pathlib.Path(root_dir)
    file_list = []
    cnt = 0
    for file in root_dir.rglob("*"):
        if not any(file.suffix == f".{ext}" for ext in extensions):
            continue
        # path
        pure_path = file.relative_to(root_dir)
        mix_path = pure_path if is_rel else file
        # check string
        if (str_include is not None) and (str_include not in pure_path.as_posix()):
            continue
        if (str_exclude is not None) and (str_exclude in pure_path.as_posix()):
            continue
        # amount
        if (amount is not None) and (cnt == amount):
            if is_sort:
                file_list.sort()
            return file_list

        if not is_ext:
            mix_path = mix_path.with_suffix('')
        file_list.append(mix_path)
        cnt += 1

    if is_sort:
        file_list.sort()
    return file_list
    
    
class DotDict(dict):
    def __getattr__(*args):         
        val = dict.get(*args)         
        return DotDict(val) if type(val) is dict else val   

    __setattr__ = dict.__setitem__    
    __delattr__ = dict.__delitem__


def load_sep_model(model_path, device='cpu'):
    config_file = os.path.join(os.path.split(model_path)[0], 'config.yaml')
    with open(config_file, "r") as config:
        args = yaml.safe_load(config)
    args = DotDict(args)
    model = CascadedNet(
                args.n_fft, 
                args.hop_length, 
                args.n_out, 
                args.n_out_lstm, 
                True, 
                is_mono=args.is_mono,
                fixed_length = True if args.fixed_length is None else args.fixed_length)
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model, args

    
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', '-g', type=int, default=-1)
    p.add_argument('--pretrained_model', '-P', type=str, required=True)
    p.add_argument('--input_dir', '-i', type=str, required=True)
    p.add_argument('--output_dir', '-o', type=str, default="")
    args = p.parse_args()

    print('loading model...', end=' ')
    device = torch.device('cpu')
    if args.gpu >= 0:
        if torch.cuda.is_available():
            device = torch.device('cuda:{}'.format(args.gpu))
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device('mps')
    model, model_args = load_sep_model(args.pretrained_model, device)
    print('done')

    wav_paths = traverse_dir(
        args.input_dir,
        extensions=["wav", "flac"],
        is_rel=True,
        is_sort=True,
        is_ext=True
    )
    for rel_path in tqdm(wav_paths):
        input_path = pathlib.Path(args.input_dir) / rel_path
        output_path = (pathlib.Path(args.output_dir) / rel_path).with_suffix('.wav')
        print('_______________________________')
        print('Input: ' + str(input_path))
        X, sr = librosa.load(
        input_path, sr=model_args.sr, mono=False, dtype=np.float32, res_type='kaiser_fast'
        )
        if X.ndim == 1:
            if model_args.is_mono:
                X = np.asarray([X])
            else:
                X = np.asarray([X, X])
        X = torch.from_numpy(X).float().unsqueeze(0).to(device)
        with torch.no_grad():        
            y = model.predict_fromaudio(X)
            y = y[0].cpu().numpy()
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            sf.write(output_path, y.T, sr)