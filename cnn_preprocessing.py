from pathlib import Path
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from utils import read_file, transform_path

DATA = Path('data')
# these folders must be in place
TRAIN_AUDIO_PATH = DATA/'nsynth-train/audio'
VALID_AUDIO_PATH = DATA/'nsynth-valid/audio'
TEST_AUDIO_PATH = DATA/'nsynth-test/audio'

NSYNTH_IMAGES = DATA/'nsynth-images'
TRAIN_IMAGE_PATH = NSYNTH_IMAGES/'train'
VALID_IMAGE_PATH = NSYNTH_IMAGES/'valid'
TEST_IMAGE_PATH = NSYNTH_IMAGES/'test'

# Mel Spectogarm


def log_mel_spec_tfm(fname, src_path, dst_path):
    x, sample_rate = read_file(fname, src_path)

    n_fft = 1024
    hop_length = 256
    n_mels = 40
    fmin = 20
    fmax = sample_rate / 2

    mel_spec_power = librosa.feature.melspectrogram(x, sr=sample_rate, n_fft=n_fft,
                                                    hop_length=hop_length,
                                                    n_mels=n_mels, power=2.0,
                                                    fmin=fmin, fmax=fmax)
    mel_spec_db = librosa.power_to_db(mel_spec_power, ref=np.max)
    dst_fname = dst_path / (fname[:-4] + '.png')
    plt.imsave(dst_fname, mel_spec_db)

# Chroma Images


def chroma_images(fname, src_path, dst_path):
    x, sample_rate = read_file(fname, src_path)

    hop_length = 256
    n_chroma = 12
    fmin = 20

    chroma_spec = librosa.feature.chroma_cqt(y=x, sr=sample_rate,
                                             hop_length=hop_length,
                                             n_chroma=n_chroma, fmin=fmin)
    dst_fname = dst_path / (fname[:-4] + '.png')
    plt.imsave(dst_fname, chroma_spec)


if __name__ == "__main__":
    train_acoustic_fnames = [f.name for f in TRAIN_AUDIO_PATH.iterdir()
                             if 'acoustic' in f.name or 'electronic' in f.name]
    valid_acoustic_fnames = [f.name for f in VALID_AUDIO_PATH.iterdir()
                             if 'acoustic' in f.name or 'electronic' in f.name]
    test_acoustic_fnames = [f.name for f in TEST_AUDIO_PATH.iterdir()
                            if 'acoustic' in f.name or 'electronic' in f.name]
    print('Train Audio Size: {}\nValid Audio Size: {}\nTest Audio Size: {}'.format(
        len(train_acoustic_fnames), len(valid_acoustic_fnames), len(test_acoustic_fnames)))

    transform_path(TRAIN_AUDIO_PATH, TRAIN_IMAGE_PATH, log_mel_spec_tfm,
                   fnames=train_acoustic_fnames, delete=True)
    transform_path(VALID_AUDIO_PATH, VALID_IMAGE_PATH, log_mel_spec_tfm,
                   fnames=valid_acoustic_fnames, delete=True)
    transform_path(TEST_AUDIO_PATH, TEST_IMAGE_PATH, log_mel_spec_tfm,
                   fnames=test_acoustic_fnames, delete=True)
