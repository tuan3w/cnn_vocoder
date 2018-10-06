# CNNVocoder
## A CNN-based vocoder.

This work is inspired from m-cnn model described in [Fast Spectrogram Inversion using Multi-head Convolutional Neural Networks](https://arxiv.org/abs/1808.06719).
The authors show that even a simple upsampling networks is enough to synthesis waveform from spectrogram/mel-spectrogram.

In this repo, I use spectrogram feature for training model because it contains more information than mel-spectrogram feature. However, because the transformation from spectrogram to mel-spectrogram is just a liner projection, so basically, you can train a simple network predict spectrogram from mel-spectrogram. You also can change parameters to be able to train a vocoder from mel-spectrogram feature too.

## Architecture notes

Compare with m-cnn, my proposed network have some differences:
- I use Upsampling + Conv layers instead of TransposedConv layer. This helps to prevent [checkerboard artifacts](https://distill.pub/2016/deconv-checkerboard/).
- The model use a lot of residual blocks pre/after the upsampling module to make network larger/deeper.
- I only used l1 loss between log-scale STFT-magnitude of predicted and target waveform. Evaluation loss on log space is better than on raw STFT-magnitude because it's closer to [human sensation about loudness](http://faculty.tamuc.edu/cbertulani/music/lectures/Lec12/Lec12.pdf). I tried to compute loss on spectrogram feature, but it didn't help much.

## Install requirements

```bash
$ pip install -r requirements.txt
```
## Training vocoder
### 1. Prepare dataset

I use [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) dataset for my experiment. If you don't have it yet, please download dataset and put it somewhere.

After that, you can run command to generate dataset for our experiment:

```bash
$ python preprocessing.py --samples_per_audio 20 \ 
--out_dir ljspeech \
--data_dir path/to/ljspeech/dataset \
--n_workers 4
```

### 2. Train vocoder

```bash
$ python train.py --out_dir ${output_directory}
```
For more training options, please run:
```bash
$ python train.py --help
```

## Generate audio from spectrogram
- Generate spectrogram from audio
```bash
$ python gen_spec.py -i sample.wav -o out.npz
```
- Generate audio from spectrogram

```bash
$ python synthesis.py --model_path path/to/checkpoint \
                      --spec_path out.npz \
                      --out_path out.wav
```

## Acknowledgements
This implementation uses code from [NVIDIA](https://github.com/NVIDIA), [Ryuichi Yamamoto](https://github.com/r9y9), [Keith Ito](https://github.com/keithito) as described in my code.

## License
[MIT](LICENSE)
