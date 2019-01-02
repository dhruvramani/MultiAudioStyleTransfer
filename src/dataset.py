import os
import torch
import numpy as np 
from torch.utils.data import Dataset, DataLoader

def load_audio(audio_path):
    signal, fs = librosa.load(audio_path)
    return signal, fs

def collate_fn(data):
    data = list(filter(lambda x: type(x[1]) != int, data))
    audios, captions = zip(*data)
    data = None
    del data
    audios = torch.stack(audios, 0)
    return audios, captions

def inp_transform(inp):
    inp = inp.numpy()
    inp = inp.flatten()
    inp, _ = transform_stft(inp)
    inp = torch.Tensor(inp)
    inp = inp.unsqueeze(0)
    return inp

def test_transform(inp):
    inp = inp.numpy()
    inp = inp.astype(np.float32)
    inp = inp.flatten()
    inp, phase = transform_stft(inp, pad=False)
    inp = torch.Tensor(inp)
    inp = inp.unsqueeze(0)
    inp = inp.unsqueeze(0)
    return inp, phase

def get_style(path='style_lady.wav'):
    path = '../save/style/' + path
    N_FFT = 128
    signal, fs = librosa.load(path)
    del fs
    signal = librosa.stft(signal, n_fft=N_FFT)
    signal, phase = librosa.magphase(signal)
    del phase
    signal = np.log1p(signal)
    signal = signal[ :, 1200:1500]
    signal = torch.from_numpy(signal) # TODO : get style audio
    signal = signal.unsqueeze(0)
    return signal

class CombinedDataset(Dataset):
    def __init__(self, path):
        self.path = path

    def __len__(self):
        return len([name for name in os.listdir(self.path) if os.path.isfile(name)])

    def __getitem__(self, idx):
        return load_audio("{}/file{}.mp3".format(self.path, idx))

class ForegroundDataset(Dataset):
    def __init__(self, combined_ds):
        self.dataloader = DataLoader(combined_ds, batch_size=1, shuffle=True,  collate_fn=collate_fn)
        self.dataloader = iter(self.dataloader)

    def __len__(self):
        return len(self.dataloader)

    def get_foreground(self, audio):
        pass # TODO : FINISH THIS

    def __getitem__(self, idx):
        audio = next(self.dataloader)
        return self.get_foreground(audio)

class BackgroundDataset(Dataset):
    def __init__(self, combined_ds):
        self.dataloader = DataLoader(combined_ds, batch_size=1, shuffle=True,  collate_fn=collate_fn)
        self.dataloader = iter(self.dataloader)

    def __len__(self):
        return len(self.dataloader)

    def get_background(self, audio):
        pass # TODO : FINISH THIS

    def __getitem__(self, idx):
        audio = next(self.dataloader)
        return self.get_background(audio)
