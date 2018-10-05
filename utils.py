#copy from https://github.com/NVIDIA/tacotron2/blob/master/plotting_utils.py
import os
import time, sys, math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt
import numpy as np
import torch



def save_figure_to_numpy(fig):
    # save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data

def plot_spectrogram_to_numpy(spectrogram):
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data

# from https://github.com/NVIDIA/vid2vid/blob/951a52bb38c2aa227533b3731b73f40cbd3843c4/models/networks.py#L17
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and hasattr(m, 'weight'):
        # m.weight.data.normal_(0.0, 0.02)
        torch.nn.init.xavier_uniform_(m.weight)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def load_checkpoint(checkpoint_path, model, optimizer=None, warm_start=False):
    """Loads model from given checkpoint

    Args:
        model (nn.Module): vocoder model
        optimizer (Optimizer): optimizer
    """
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint from '{}'".format(checkpoint_path))
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    lr = ckpt['lr']
    model.load_state_dict(ckpt['model_state'])
    if optimizer != None and not warm_start:
        print('update optimizer')
        optimizer.load_state_dict(ckpt['optim_state'])
    steps = ckpt['steps'] + 1
    if warm_start:
        steps = 0
    return model, optimizer, lr, steps


def save_checkpoint(filename, lr, steps, model, optimizer):
    """Saves model
    Args:
        filename (string): checkpoint path
        lr (float): learning rate
        model (nn.Module): vocoder model
        optimizer (Optimizer): optimizer
    """
    print('Saving checkpoint at step {}'.format(steps))
    torch.save({
        'steps': steps,
        'lr': lr,
        'model_state': model.state_dict(),
        'optim_state': optimizer.state_dict()
    }, filename)

def time_since(started) :
    elapsed = time.time() - started
    m = int(elapsed // 60)
    s = int(elapsed % 60)
    if m >= 60 :
        h = int(m // 60)
        m = m % 60
        return f'{h}h {m}m {s}s'
    else :
        return f'{m}m {s}s'
