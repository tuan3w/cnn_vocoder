import argparse
import os
import glob
from functools import partial
import numpy as np
from tqdm import tqdm
import os
import audio
from random import shuffle
from concurrent.futures import ProcessPoolExecutor
from hparams import hparams

def gen_samples(out_dir, wav_path, n_samples):
    wav = audio.load_wav(wav_path)
    hop_size = hparams.hop_length
    seg_len = hparams.seg_len
    spec_len = hparams.spec_len
    # not sure why we have to minus 1 here ?
    wav_len = wav.shape[0] // hop_size * hop_size -1
    wav = wav[:wav_len]
    spec = audio.spectrogram(wav)
    mel = audio.melspectrogram(wav)
    max_val = spec.shape[1] - 1 - spec_len
    if max_val < 0:
        return []
    idx = np.random.randint(0, max_val, size=(n_samples))
    d = []
    i = 0
    for offset in idx:
        i += 1
        w = wav[offset * hop_size: offset * hop_size + seg_len]
        s = spec[:,offset:offset + spec_len]
        m = mel[:,offset:offset + spec_len]
        wav_name = wav_path.split('/')[-1].split('.')[0]
        file_path = "{0}/{1}_{2:03d}.npz".format(out_dir, wav_name, i)
        np.savez(file_path, wav=w, spec=s, mel=m)
        d.append(file_path)
    return d

def main(args):
    executor = ProcessPoolExecutor(
        max_workers=args.num_workers)
    files = []
    audio_dir = os.path.join(args.data_dir, 'wavs')
    out_dir = args.out_dir
    audio_files = glob.glob(audio_dir + '/*.wav')
    samples_per_audio = args.samples_per_audio
    futures = []
    index = 0
    for wav_path in audio_files:
        futures.append(executor.submit(partial(gen_samples, out_dir, wav_path, samples_per_audio)))
        index += 1
    files = [future.result() for future in tqdm(futures)]
    files =[y for z in files for y in z]
    txt_path = os.path.join(out_dir, "files.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        files = [f + '\n' for f in files]
        # shuffle data
        shuffle(files)
        f.writelines(files)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--out_dir', default='training_data', type=str, help='Output directory to store dataset')
    parser.add_argument('-d', '--data_dir', default="ljspeech", type=str, help='Path to ljspeech dataset')
    parser.add_argument('-s', '--samples_per_audio', type=int, default=400, help='Number of sample per audio')
    parser.add_argument('-n', '--num_workers', type=int, default=4)

    args = parser.parse_args()
    main(args)
