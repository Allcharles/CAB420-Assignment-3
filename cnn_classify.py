import matplotlib as plt
import torch
import fastai
from fastai.vision import *
from pathlib import Path
from shutil import copyfile
import os
from os import listdir, rename
from os.path import isfile, join

DATA = Path('data')
# these folders must be in place
TRAIN_AUDIO_PATH = DATA/'nsynth-train/audio'
VALID_AUDIO_PATH = DATA/'nsynth-valid/audio'

NSYNTH_IMAGES = DATA/'nsynth-images'
TRAIN_IMAGE_PATH = NSYNTH_IMAGES/'train'
VALID_IMAGE_PATH = NSYNTH_IMAGES/'valid'
MODEL_IMAGE_PATH = DATA/'nsynth-models'
TRAIN_SIZE = 1000


if __name__ == "__main__":
    # Change the size of the dataset

    # folder = "data/nsynth-images/train/mel_spec"
    # output_folder = "data/nsynth-images/train2/mel_spec"
    # train_files = [f for f in listdir(
    #     folder) if isfile(join(folder, f))]
    # num_files = len(train_files)
    # if not os.path.exists(output_folder):
    #     os.makedirs(output_folder)
    # for file_index in range(0, num_files, int(num_files/TRAIN_SIZE)):
    #     copyfile(join(folder, train_files[file_index]), join(
    #         output_folder, train_files[file_index]))

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

    learn = cnn_learner(data, models.resnet34,
                        metrics=accuracy, callback_fns=ShowGraph)
    learn.fit_one_cycle(3, 0.005)
    interp = ClassificationInterpretation.from_learner(learn)
    interp.plot_confusion_matrix(figsize=(10, 10), dpi=60)
    interp.most_confused(min_val=20)
    learn.save('trained_model')
