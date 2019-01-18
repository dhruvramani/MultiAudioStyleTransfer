import os
import torch
import librosa
import numpy as np 
from torch.utils.data import Dataset, DataLoader

from feature import *
from new_feature import *

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

def splitAudio(audio, split_size = 300):
    auds = []
    print(audio.shape)
    for i in range(audio.shape[1],split_size):
        a = audio[:,i:i+split_size]
        if(a.shape[1]<split_size):
            a = librosa.util.pad_center(a, split_size)

        print(a)
        auds.append(a)
        del a
    print(auds)
    return torch.stack(auds)

class CombinedDataset(Dataset):
    def __init__(self, path='/home/nevronas/dataset/dualaudio/DSD100/Mixtures/Dev'):
        self.path = path
        self.folder_names = [name for name in os.listdir(self.path)]

    def __len__(self):
        return len(self.folder_names)

    def __getitem__(self, idx):
        return load_audio("{}/{}/mixture.wav".format(self.path, self.folder_names[idx]))

class VocalDataset(Dataset):
    def __init__(self, path='/home/nevronas/dataset/dualaudio/DSD100/Sources/Dev', transform=None):
        self.path = path
        self.folder_names = [name for name in os.listdir(self.path)]
        self.transform = transform

    def __len__(self):
        return len(self.folder_names)

    def __getitem__(self, idx):
        audio, _ = load_audio("{}/{}/vocals.wav".format(self.path, self.folder_names[idx]))
        if(self.transform):
            audio, _ = self.transform(audio)
            audio = torch.Tensor(audio)
            audio = splitAudio(audio, split_size = 300)
            audio = audio.unsqueeze(0)
            
        return audio

class BackgroundDataset(Dataset):
    def __init__(self, path='/home/nevronas/dataset/dualaudio/DSD100/Sources/Dev', n_splits=3):
        self.path = path
        self.n_splits = n_splits
        self.folder_names = [name for name in os.listdir(self.path)]

    def __len__(self):
        return len(self.folder_names)

    def __getitem__(self, idx):
        bass_path = "{}/{}/bass.wav".format(self.path, self.folder_names[idx])
        drums_path = "{}/{}/drums.wav".format(self.path, self.folder_names[idx])
        other_path = "{}/{}/other.wav".format(self.path, self.folder_names[idx])
        paths = [drums_path, bass_path, other_path]
        return (load_audio(paths[i]) for i in range(self.n_splits))


if __name__ == "__main__":
    data = VocalDataset(transform=audioFileToSpectrogram)
    dataloader = DataLoader(data, batch_size=1)
    print(len(data))
    for foo in dataloader:
        print(foo.shape)
        break
