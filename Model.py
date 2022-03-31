'''
Model
'''

# Imports
import wandb
from wandb.keras import WandbCallback
import math
from keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

from Library.ModelBlocks import *
from Dataset import *

# Main Functions
# Model Functions
# Part A Functions
# Build Sequential Model Function @ Karthikeyan S CS21M028

# Part B Functions
# Build Pretrained Model Function @ Karthikeyan S CS21M028
# Main Vars

# Common Functions
# Compile Model Function @ Karthikeyan S CS21M028

# Train Model Function @ N Kausik CS21M037
def Model_Train(model, inputs, n_epochs, wandb_data, **params):
    '''
    Train Model
    '''
    # Get Data
    DATASET = inputs["dataset"]
    TRAIN_STEP_SIZE = math.ceil(DATASET["train"].n / DATASET["train"].batch_size)
    VALIDATION_STEP_SIZE = math.ceil(DATASET["val"].n / DATASET["val"].batch_size)

    # Enable Wandb Callback
    WandbCallbackFunc = WandbCallback(
        monitor="val_accuracy", save_model=True, log_evaluation=True, log_weights=True,
        log_best_prefix="best_",
        generator=DATASET["val"], validation_steps=VALIDATION_STEP_SIZE
    )
    # Enable Model Checkpointing
    ModelCheckpointFunc = ModelCheckpoint(
        "Models/best_model.h5",
        monitor="val_accuracy",
        verbose=1,
        save_best_only=True,
        mode="max",
        save_freq="epoch"
    )

    # Train Model
    TRAIN_HISTORY = model.fit(
        DATASET["train"], steps_per_epoch=TRAIN_STEP_SIZE,
        validation_data=DATASET["val"], validation_steps=VALIDATION_STEP_SIZE,
        epochs=n_epochs,
        verbose=1,
        callbacks=[WandbCallbackFunc, ModelCheckpointFunc]
    )

    return model, TRAIN_HISTORY

# Test Model Function @ N Kausik CS21M037
def Model_Test(model, dataset, **params):
    '''
    Test Model
    '''
    # Test Model
    loss_test, eval_test = model.evaluate(dataset, verbose=1)

    return loss_test, eval_test

# Load and Save Model Functions @ N Kausik CS21M037
def Model_LoadModel(path):
    '''
    Load Model
    '''
    return load_model(path)

def Model_SaveModel(model, path):
    '''
    Save Model
    '''
    return model.save(path)