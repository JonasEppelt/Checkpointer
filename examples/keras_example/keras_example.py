
# imports
import numpy as np
import os
from pathlib import Path

# setting the device, kears can train on
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import keras

# import the callback for Keras
from checkpointer.keras_callback import KerasCheckpointerCallback

# Load the data and split it between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")


# Model parameters
num_classes = 10
input_shape = (28, 28, 1)

# create the model
model = keras.Sequential(
    [
        keras.layers.Input(shape=input_shape),
        keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(num_classes, activation="softmax"),
    ]
)

# compile the model
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    metrics=[
        keras.metrics.SparseCategoricalAccuracy(name="acc"),
    ],
)

#defince the callbacks 
callbacks = [
    KerasCheckpointerCallback( # setting up the checkpointer callback
        local_checkpoint_file="/work/jeppelt/checkpointing/checkpointer/checkpoint", # local checkpoint file
        checkpoint_every=1, # checkpointing every epoch
        checkpoint_transfer_mode="shared", # using a shared filesystem to move hte checkpoint to a persisten storage
        checkpoint_transfer_target=Path("/ceph/jeppelt/checkpoint.ckpt") # the target path on the shared filesystem
    ),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=2),
]


# run the training 
# It will now automatically check if a checkpoint exists, load it and continue training from it.
# Test it, by executing this program and interupting it after the first epoch. Then execute it again.
batch_size = 128
epochs = 20

model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.15,
    callbacks=callbacks,
)
score = model.evaluate(x_test, y_test, verbose=0)



