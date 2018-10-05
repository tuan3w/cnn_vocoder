import argparse
import numpy as np
import audio

def main(args):
    wav = audio.load_wav(args.wav_path)
    spec = audio.spectrogram(wav)
    np.save(args.out_path, spec)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--wav_path', type=str, required=True,
                        help='path to audio path')
    parser.add_argument('-o', '--out_path', type=str, required=True,
                        help='output path')
    args = parser.parse_args()
    main(args)
