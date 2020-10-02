from __future__ import print_function
import numpy as np
import pandas as pd
import cv2
import json
import os

from functools import partial
from pathlib import Path
from tqdm import tqdm
from imgaug import augmenters as iaa
from utils import additional_augmenation

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.datasets import cifar10
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers import Input
from keras.models import Model
from keras.optimizers import SGD
from keras.utils import plot_model
from keras.callbacks import Callback
from keras import backend as K

import numpy as np


class PersonDataGenerator(keras.utils.Sequence):
    """Ground truth data generator"""

    def __init__(self, df, batch_size=32, shuffle=True, augmentation=None):
        self.df = df
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
        # image augmentation code
        self.augmentation = augmentation

    def __len__(self):
        return int(np.floor(self.df.shape[0] / self.batch_size))

    def __getitem__(self, index):
        """fetch batched images and targets"""
        batch_slice = slice(index * self.batch_size, (index + 1) * self.batch_size)
        items = self.df.iloc[batch_slice]
        image = np.stack([cv2.imread(item["image_path"]) for _, item in items.iterrows()])
        # image augmentation code
        if self.augmentation is not None:
            image = self.augmentation.flow(image, shuffle=False).next()
        target = {
            "gender_output": items[_gender_cols_].values,
            "image_quality_output": items[_imagequality_cols_].values,
            "age_output": items[_age_cols_].values,
            "weight_output": items[_weight_cols_].values,
            "bag_output": items[_carryingbag_cols_].values,
            "pose_output": items[_bodypose_cols_].values,
            "footwear_output": items[_footwear_cols_].values,
            "emotion_output": items[_emotion_cols_].values,
        }
        return image, target

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        if self.shuffle == True:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

def evaluate_model(model):
    results = model.evaluate_generator(valid_gen, verbose=1)
    accuracies = {}
    losses = {}
    for k, v in zip(model.metrics_names, results):
        if k.endswith('acc'):
            accuracies[k] = round(v * 100, 4)
        else:
            losses[k] = v
    return accuracies, losses

if __name__ == "__main__":
    # load annotations
    df = pd.read_csv("data/ha-tsai/hvc_annotations.csv")
    del df["filename"]  # remove unwanted column
    # df.head()

    # one hot encoding of labels
    one_hot_df = pd.concat([
        df[["image_path"]],
        pd.get_dummies(df.gender, prefix="gender"),
        pd.get_dummies(df.imagequality, prefix="imagequality"),
        pd.get_dummies(df.age, prefix="age"),
        pd.get_dummies(df.weight, prefix="weight"),
        pd.get_dummies(df.carryingbag, prefix="carryingbag"),
        pd.get_dummies(df.footwear, prefix="footwear"),
        pd.get_dummies(df.emotion, prefix="emotion"),
        pd.get_dummies(df.bodypose, prefix="bodypose"),
    ], axis=1)

    # one_hot_df.head().T

    # display(one_hot_df)

    # one_hot_df.shape

    # Label columns per attribute
    _gender_cols_ = [col for col in one_hot_df.columns if col.startswith("gender")]
    _imagequality_cols_ = [col for col in one_hot_df.columns if col.startswith("imagequality")]
    _age_cols_ = [col for col in one_hot_df.columns if col.startswith("age")]
    _weight_cols_ = [col for col in one_hot_df.columns if col.startswith("weight")]
    _carryingbag_cols_ = [col for col in one_hot_df.columns if col.startswith("carryingbag")]
    _footwear_cols_ = [col for col in one_hot_df.columns if col.startswith("footwear")]
    _emotion_cols_ = [col for col in one_hot_df.columns if col.startswith("emotion")]
    _bodypose_cols_ = [col for col in one_hot_df.columns if col.startswith("bodypose")]

    # from sklearn.model_selection import train_test_split
    # train_df, val_df = train_test_split(one_hot_df, test_size=0.10, random_state=42)
    # print(train_df.shape, val_df.shape)
    train_df = one_hot_df[:12150]
    val_df = one_hot_df[12150:13500]
    print(train_df.shape, val_df.shape)

    # create train and validation data generators

    train_gen = PersonDataGenerator(train_df,
                                    batch_size=32,
                                    augmentation=ImageDataGenerator(
                                        # set input mean to 0 over the dataset
                                        featurewise_center=False,
                                        # set each sample mean to 0
                                        samplewise_center=False,
                                        # divide inputs by std of dataset
                                        featurewise_std_normalization=False,
                                        # divide each input by its std
                                        samplewise_std_normalization=False,
                                        # apply ZCA whitening
                                        zca_whitening=False,
                                        # epsilon for ZCA whitening
                                        zca_epsilon=1e-06,
                                        # randomly rotate images in the range (deg 0 to 180)
                                        rotation_range=0,
                                        # randomly shift images horizontally
                                        width_shift_range=0.1,
                                        # randomly shift images vertically
                                        height_shift_range=0.1,
                                        # set range for random shear
                                        shear_range=0.15,
                                        # set range for random zoom
                                        zoom_range=0.,
                                        # set range for random channel shifts
                                        channel_shift_range=0.,
                                        # set mode for filling points outside the input boundaries
                                        fill_mode='nearest',
                                        # value used for fill_mode = "constant"
                                        cval=0.,
                                        # randomly flip images
                                        horizontal_flip=True,
                                        # randomly flip images
                                        vertical_flip=False,
                                        # set rescaling factor (applied before any other transformation)
                                        rescale=None,
                                        # set function that will be applied on each input
                                        preprocessing_function=additional_augmenation,
                                        # image data format, either "channels_first" or "channels_last"
                                        data_format=None,
                                        # fraction of images reserved for validation (strictly between 0 and 1)
                                        validation_split=0.0))

    valid_gen = PersonDataGenerator(val_df, batch_size=64, shuffle=False)

    # get number of output units from data
    # you may get a error incase if you didnt keep the images and annotations in right path as mentioned in readme
    # the reason is we have kept the path of each images in the annotation file.
    images, targets = next(iter(train_gen))
    num_units = {k.split("_output")[0]: v.shape[1] for k, v in targets.items()}

    # This is how my data looks like now
    for j in range(1, 20):
        print("=====================================")
        print("Gender : ", targets["gender_output"][j])
        print("Image Quality : ", targets["image_quality_output"][j])
        print("Age : ", targets["age_output"][j])
        print("Bag : ", targets["bag_output"][j])
        print("Pose : ", targets["pose_output"][j])
        print("Weight : ", targets["weight_output"][j])
        print("Footwear : ", targets["footwear_output"][j])
        print("Emotion : ", targets["emotion_output"][j])
        print("Shape: ", images[j].shape)
        # cv2_imshow(images[j])
        print("=====================================")
    from model import deepmth
    from utils import PlotLearning
    from utils import SaveWeights, CyclicLR
    from backbone.resnet import resnet_v1

    model = deepmth(resnet_v1(), num_units)

    plot = PlotLearning()

    save_weight = SaveWeights()

    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    # Prepare callbacks for model saving and for learning rate adjustment.

    filepath = 'dmth_results/bestvalweights/resnet-epochs:{epoch:03d}-validation.hdf5'

    checkpoint = ModelCheckpoint(filepath=filepath,
                                 verbose=1,
                                 save_best_only=True)

    clr = CyclicLR(base_lr=0.001,
                   max_lr=0.006,
                   step_size=2000.,
                   mode='triangular')

    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   min_lr=0.5e-6)

    callbacks = [checkpoint, lr_reducer, clr, plot, save_weight]

    # Print Model Summary
    model.summary()

    last_executed_epoch = 0
    nb_epoch = 150

    model.load_weights("dmth_results/weights/Assignment5_RESNET__90.hdf5")
    evaluate_model(model)


