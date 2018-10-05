from torch import nn
import torch
from hparams import hparams
import torch.nn.functional as F
import numpy as np
import librosa

def _normalize(S):
    return (S - hparams.min_level_db)/-hparams.min_level_db

def _build_mel_basis(n_fft, n_mels=80):
    return torch.FloatTensor(librosa.filters.mel(hparams.sample_rate, n_fft, n_mels=n_mels)).transpose(0, 1)

def _amp_to_db(x):
    #return 20 * torch.log10(torch.clamp(x, min=1e-5))
    return 20 * torch.log10(x + 1e-5)


class Spectrogram(nn.Module):
    """Spectrogram transformation.

    Args:
        win_length (int): stft window length
        hop_length (int): stft hop length
        n_fft (int): number of fft basis
        preemp (bool): whether pre-emphasis audio before do stft
    """
    def __init__(self, win_length=1024, hop_length=256, n_fft=2048, preemp=True):
        super(Spectrogram, self).__init__()
        if preemp:
            self.preemp = nn.Conv1d(1, 1, 2, bias=False, padding=1)
            self.preemp.weight.data[0][0][0] = -0.97
            self.preemp.weight.data[0][0][1] = 1.0
            self.preemp.weight.requires_grad = False
        else:
            self.preemp = None

        win = torch.hann_window(win_length)
        self.register_buffer('win', win)
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_fft = n_fft

    def forward(self, x):
        if self.preemp is not None:
            x = x.unsqueeze(1)
            # conv and remove last padding
            x = self.preemp(x)[:, :, :-1]
            x = x.squeeze(1)

        # center=True
        # torch 0.4 doesnt support like librosa
        signal_dim = x.dim()
        extended_shape = [1] * (3 - signal_dim) + list(x.size())
        # pad = int(self.n_fft // 2)
        pad = self.win_length
        x = F.pad(x.view(extended_shape), (pad, pad), 'constant')
        x = x.view(x.shape[-signal_dim:])
        stft = torch.stft(x,
                          self.win_length,
                          self.hop_length,
                          window=self.win,
                          fft_size=self.n_fft)
        real = stft[:, :, :, 0]
        im = stft[:, :, :, 1]
        p = torch.sqrt(torch.pow(real, 2) + torch.pow(im, 2))

        # convert volume to db
        spec = _amp_to_db(p) - hparams.ref_level_db
        return spec, p

class MelSpectrogram(nn.Module):
    """MelSpectrogram transformation.
    
    Args:
        win_length (int): stft window length
        hop_length (int): stft hop length
        n_fft (int): number of fft basis
        n_mels (int): number of mel filters
        preemp (bool): whether pre-emphasis audio before do stft
    """
    def __init__(self, win_length=1024, hop_length=256, n_fft=2048, n_mels=80, preemp=True):
        super(MelSpectrogram, self).__init__()
        if preemp:
            self.preemp = nn.Conv1d(1, 1, 2, bias=False, padding=1)
            self.preemp.weight.data[0][0][0] = -0.97
            self.preemp.weight.data[0][0][1] = 1.0
            self.preemp.weight.requires_grad = False
        else:
            self.preemp = None

        self.register_buffer('mel_basis', _build_mel_basis(n_fft, n_mels))

        win = torch.hann_window(win_length)
        self.register_buffer('win', win)

        self.win_length = win_length
        self.hop_length = hop_length
        self.n_fft = n_fft

    def forward(self, x):
        if self.preemp is not None:
            x = x.unsqueeze(1)
            x = self.preemp(x)
            x = x.squeeze(1)
        stft = torch.stft(x,
                          self.win_length,
                          self.hop_length,
                          fft_size=self.n_fft,
                          window=self.win)
        real = stft[:, :, :, 0]
        im = stft[:, :, :, 1]
        spec = torch.sqrt(torch.pow(real, 2) + torch.pow(im, 2))

        # convert linear spec to mel
        mel = torch.matmul(spec, self.mel_basis)
        # convert to db
        mel = _amp_to_db(mel) - hparams.ref_level_db
        return _normalize(mel)
