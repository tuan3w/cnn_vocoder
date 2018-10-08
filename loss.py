import torch
from torch import nn
from hparams import hparams
import stft

def compute_stft(audio, n_fft=1024, win_length=1024, hop_length=256):
    """
    Computes STFT transformation of given audio

    Args:
        audio (Tensor): B x T, batch of audio

    Returns:
        mag (Tensor): STFT magnitudes
        real (Tensor): Real part of STFT transformation result
        im (Tensor): Imagine part of STFT transformation result
    """
    win = torch.hann_window(win_length).cuda()

    # add some padding because torch 4.0 doesn't
    signal_dim = audio.dim()
    extended_shape = [1] * (3 - signal_dim) + list(audio.size())
    # pad = int(self.n_fft // 2)
    pad = win_length
    audio = F.pad(audio.view(extended_shape), (pad, pad), 'constant')
    audio = audio.view(audio.shape[-signal_dim:])

    stft = torch.stft(audio, win_length, hop_length, fft_size=n_fft, window=win)
    real = stft[:, :, :, 0]
    im = stft[:, :, :, 1]
    power = torch.sqrt(torch.pow(real, 2) + torch.pow(im, 2))
    return power, real, im

def compute_loss(pred, target):
    """
    Computes loss value

    Args:
        pred (Tensor): B x T, predicted wavs
        target (Tensor): B x T, target wavs
    """
    stft_pred, _, _= compute_stft(pred, n_fft=2048, win_length=1024, hop_length=256)
    stft_target, _, _ = compute_stft(target, n_fft=2048, win_length=1024, hop_length=256)
    l1_loss = nn.L1Loss()

    log_stft_pred = torch.log(stft_pred + 1e-8)
    log_stft_target = torch.log(stft_target + 1e-8)
    l1 = l1_loss(log_stft_pred, log_stft_target)
    l2 = l1_loss(log_stft_pred[:, :, :500], log_stft_target[:, :,:500])
    l3 = l1_loss(stft_pred[:,:,:500], stft_target[:,:,:500])
    loss = l1 + l2 + l3
    return loss, l1, l2, l3
