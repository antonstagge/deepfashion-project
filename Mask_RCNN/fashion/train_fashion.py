#!/usr/bin/env python
# coding: utf-8

import os
import sys
import tensorflow as tf

# Root directory of the project
ROOT_DIR = os.path.abspath("../")
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

import mrcnn.model as modellib
from mrcnn.model import log

from fashion_config import FashionConfig
from fashion_dataset import FashionDataset

# GPU checks
print("Using GPU: " + str(tf.test.is_gpu_available()))
print("GPU name: " + str(tf.test.gpu_device_name()))

# Hide some tensorflaw warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

""" Load config """ 
config = FashionConfig()
config.display()


""" Prepare dataset """ 
# TODO: change hardcoded dataset path
dataset_train = FashionDataset()
dataset_train.load_fashion(ROOT_DIR + '/datasets/big_deepfashion2', "train")
dataset_train.prepare()

dataset_val = FashionDataset()
dataset_val.load_fashion(ROOT_DIR + '/datasets/big_deepfashion2', "val")
dataset_val.prepare()


""" Create model """ 
# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

# Init weights
init_with = "imagenet"
if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last(), by_name=True)

""" Callbacks """
# These are added to the custom_callbacks in the training of the model
import keras
earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')
history = keras.callbacks.History()

""" Training """
# 
# Train in two stages:
# 1. Only the heads. Here we're freezing all the backbone layers and training only the randomly initialized layers (i.e. the ones that we didn't use pre-trained weights from MS COCO). To train only the head layers, pass `layers='heads'` to the `train()` function.
# 
# 2. Fine-tune all layers. For this simple example it's not necessary, but we're including it to show the process. Simply pass `layers="all` to train all layers.

# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE, 
            epochs=1, 
            layers='heads',
            custom_callbacks=[earlystop, history])


"""
# Fine tune all layers
# Passing layers="all" trains all layers. You can also 
# pass a regular expression to select which layers to
# train by name pattern.
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE / 10,
            epochs=2, 
            layers="all")
"""

# Save weights
# Typically not needed because callbacks save after every epoch
# Uncomment to save manually
model_path = os.path.join(MODEL_DIR, "mask_rcnn_fashion.h5")
model.keras_model.save_weights(model_path)


""" Pickle some additional information """
import pickle

#Store the history dict
pickle_path = os.path.join(MODEL_DIR, "pickled_history")
with open(pickle_path, 'wb') as file_pi:
    pickle.dump(history.history, file_pi)



