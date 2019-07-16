from pathlib import Path
import fastai
from fastai.vision import *
import matplotlib as plt
import torch

DATA = Path('data')
NSYNTH_IMAGES = DATA/'nsynth-images'
MODEL_IMAGE_PATH = NSYNTH_IMAGES/'model'
TRAIN_IMAGE_PATH = NSYNTH_IMAGES/'train'
VALID_IMAGE_PATH = NSYNTH_IMAGES/'valid'

if __name__ == "__main__":
    instrument_family_pattern = r'(\w+)_\w+_\d+-\d+-\d+.png$'
    data = (ImageList.from_folder(NSYNTH_IMAGES)
            .split_by_folder()
            .label_from_re(instrument_family_pattern)
            .databunch())
    data.c, data.classes
    xs, ys = data.one_batch()
    xs.shape, ys.shape
    xs.min(), xs.max(), xs.mean(), xs.std()
    data.show_batch(3, figsize=(8, 4), hide_axis=False)

    learn = cnn_learner(data, models.resnet18, metrics=accuracy)
    learn.load('trained_model')
