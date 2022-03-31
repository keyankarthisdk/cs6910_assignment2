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
def Model_SequentialBlocks(X_shape, Y_shape, Blocks, **params):
    '''
    Sequential Model
    '''
    # Init Model
    model = Sequential()

    # Add Blocks
    cur_shape = X_shape
    for i in range(len(Blocks)):
        model, cur_shape = Blocks[i](model, cur_shape, block_name="block_" + str(i))

    # Output Layer
    model.add(Flatten(name="flatten"))
    model.add(Dense(params["dense_n_neurons"], activation="relu", name="output_dense"))
    model.add(Dropout(params["dense_dropout_rate"], name="output_dropout"))
    model.add(Dense(Y_shape, activation="softmax", name="output_softmax"))
    return model

# Part B Functions
# Build Pretrained Model Function @ Karthikeyan S CS21M028
# Main Vars

# Common Functions
# Compile Model Function @ Karthikeyan S CS21M028
def Model_Compile(model, loss_fn="categorical_crossentropy", optimizer="adam", metrics=["accuracy"], **params):
    '''
    Compile Model
    '''
    model.compile(loss=loss_fn, optimizer=optimizer, metrics=metrics)
    return model

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