import argparse
from time import time

import numpy as np
import torch

import audio
from hparams import hparams, hparams_debug_string
from model import CNNVocoder
from utils import load_checkpoint


def main(args):
    model = CNNVocoder(
        n_heads=hparams.n_heads,
        layer_channels=hparams.layer_channels,
        pre_conv_channels=hparams.pre_conv_channels,
        pre_residuals=hparams.pre_residuals,
        up_residuals=hparams.up_residuals,
        post_residuals=hparams.post_residuals
    )
    model = model.cuda()

    model, _, _, _ = load_checkpoint(
            args.model_path, model)
    spec = np.load(args.spec_path)
    spec = torch.FloatTensor(spec).unsqueeze(0).cuda()
    t1 = time()
    _, wav = model(spec)
    dt = time() - t1 
    print('Synthesized audio in {}s'.format(dt))
    wav = wav.data.cpu()[0].numpy()
    audio.save_wav(wav, args.out_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True,
                        help='path to model checkpoint')
    parser.add_argument('--spec_path', type=str, default="logs",
                        help='path to spec file')
    parser.add_argument('--out_path', type=str, default=None,
                        help='output wav path')
    args = parser.parse_args()
    main(args)
