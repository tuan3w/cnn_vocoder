import tensorflow as tf
import numpy as np

hparams = tf.contrib.training.HParams(
    #####################################
    # Audio config
    #####################################
    sample_rate=22050,
    silence_threshold=2,
    num_mels=80,
    fmin=125,
    fmax=7600,
    fft_size=1024,
    win_length=1024,
    hop_length=256,
    min_level_db=-100,
    ref_level_db=20,
    rescaling=True,
    rescaling_max=0.999,
    audio_max_value=32767,
    allow_clipping_in_normalization=True,

    #####################################
    # Data config
    #####################################
    seg_len= 81 * 256,
    file_list="training_data/files.txt",
    spec_len= 81,

    #####################################
    # Model parameters
    #####################################
    n_heads = 2,
    pre_residuals = 4,
    up_residuals=0,
    post_residuals = 12,
    pre_conv_channels = [1, 1, 2],
    layer_channels = [1025 * 2, 1024, 512, 256, 128, 64, 32, 16, 8],


    #####################################
    # Training config
    #####################################
    n_workers=2,
    seed=12345,
    batch_size=40,
    lr=1.0 * 1e-3,
    weight_decay=1e-5,
    epochs=50000,
    grad_clip_thresh=5.0,
    checkpoint_interval=1000,
)


def hparams_debug_string():
    values = hparams.values()
    hp = ['  %s: %s' % (name, values[name]) for name in sorted(values)]
    return 'Hyperparameters:\n' + '\n'.join(hp)
