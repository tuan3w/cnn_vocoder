import argparse
import os

import matplotlib
import numpy as np
import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

import audio
import librosa
from data import MelDataset
from hparams import hparams, hparams_debug_string
from loss import compute_loss
from model import CNNVocoder
from tensorboardX import SummaryWriter
from tqdm import tqdm
from utils import (load_checkpoint, plot_spectrogram_to_numpy, save_checkpoint,
                   weights_init)

torch.backends.cudnn.benchmark = True
# Set random seed to make training reproducible
np.random.seed(hparams.seed)
torch.manual_seed(hparams.seed)
torch.cuda.manual_seed(hparams.seed)

def add_log(writer, loss, l1, l2, l3, steps):
    writer.add_scalar("loss", loss, steps)
    writer.add_scalar("loss.log_stft", l1, steps)
    writer.add_scalar("loss.log_stft_low_freqs", l2, steps)
    writer.add_scalar("loss.stft_low_freqs", l3, steps)
    # writer.add_scalar("grad_norm", grad_norm, steps)

def add_spec_sample(writer, mel_target, mel_predicted, steps):
    writer.add_image(
        "mel_target",
        plot_spectrogram_to_numpy(mel_target),
        steps)
    writer.add_image(
        "mel_predicted",
        plot_spectrogram_to_numpy(mel_predicted),
        steps)


def prepare_directories(out_dir, log_dir, checkpoint_dir):
    log_dir = os.path.join(out_dir, log_dir)
    checkpoint_dir = os.path.join(out_dir, checkpoint_dir)
    dirs = [out_dir, log_dir, checkpoint_dir]
    for d in dirs:
        print('prepare dir: {}'.format(d))
        if not os.path.isdir(d):
            os.makedirs(d)


def train(args):
    print(hparams_debug_string())

    # prepare logging, checkpoint directories
    prepare_directories(args.out_dir, args.log_dir, args.checkpoint_dir)
    # create model
    model = CNNVocoder(
        n_heads=hparams.n_heads,
        layer_channels=hparams.layer_channels,
        pre_conv_channels=hparams.pre_conv_channels,
        pre_residuals=hparams.pre_residuals,
        up_residuals=hparams.up_residuals,
        post_residuals=hparams.post_residuals
    )
    model.apply(weights_init)
    model = model.cuda()

    # create optimizer
    lr = hparams.lr
    optimizer = optim.Adam(model.parameters(),
                           lr=lr,
                           weight_decay=hparams.weight_decay)

    dataloader = DataLoader(
        MelDataset(hparams.file_list),
        batch_size=hparams.batch_size, shuffle=True,
        num_workers=hparams.n_workers)

    steps = 0
    checkpoint_dir = os.path.join(args.out_dir, args.checkpoint_dir)
    log_dir = os.path.join(args.out_dir, args.log_dir)
    writer = SummaryWriter(log_dir)

    # load model from checkpoint
    if args.checkpoint_path:
        model, optimizer, lr, steps = load_checkpoint(
            args.checkpoint_path, model, optimizer, warm_start=args.warm_start)

    for i in range(hparams.epochs):
        print('Epoch: {}'.format(i))
        for idx, batch in enumerate(dataloader):
            steps += 1
            wav, spec = batch[0].cuda(), batch[1].cuda()
            optimizer.zero_grad()
            pre_predict, predict = model(spec)
            post_loss, l1, l2, l3 = compute_loss(predict, wav)
            loss = post_loss
            print('Step: {:8d}, Loss = {:8.4f}, post_loss = {:8.4f}, pre_loss = {:8.4f}'.format(steps, loss, post_loss, post_loss))
            if torch.isnan(loss).item() != 0:
                print('nan loss, ignore')
                return
            loss.backward()
            # clip grad norm
            grad_norm = clip_grad_norm_(
                model.parameters(), hparams.grad_clip_thresh)
            optimizer.step()

            # log training
            # add_log(writer, loss, p_loss, low_p_loss, phrase_loss, p1, grad_norm, steps)
            add_log(writer, loss, l1, l2, l3, steps)

            if steps > 0 and steps % hparams.checkpoint_interval == 0:
                checkpoint_path = '{}/checkpoint_{}'.format(
                    checkpoint_dir, steps)
                save_checkpoint(checkpoint_path, lr,
                                steps, model, optimizer)

                # saving example
                idx = np.random.randint(wav.shape[0])
                t1 = wav[idx].data.cpu().numpy()
                t2 = predict[idx].data.cpu().numpy()
                audio.save_wav(
                    t2, '{}/generated_{}.wav'.format(checkpoint_dir, steps))
                audio.save_wav(
                    t1, '{}/target_{}.wav'.format(checkpoint_dir, steps))

                # add spec sample
                # spec_pred = audio.melspectrogram(t2)
                # spec_target = audio.melspectrogram(t1)
                # add_spec_sample(writer, spec_target, spec_pred, steps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--out_dir', type=str, default="output",
                        help='directory to save checkpoints')
    parser.add_argument('-l', '--log_dir', type=str, default="logs",
                        help='log directory ${out_directory}/${log_directory}')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='checkpoint model')
    parser.add_argument('-c', '--checkpoint_dir', type=str, default="checkpoints",
                        required=False, help='checkpoint directory ${out_directory}/${checkpoint_dir}')

    parser.add_argument('--warm_start', action='store_true',
                        help='load the model only (warm start)')

    args = parser.parse_args()
    train(args)
