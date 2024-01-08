
# %%
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
from pathlib import Path

import keras
from checkpointer.keras_callback import KerasCheckpointerCallback
from keras.callbacks import BackupAndRestore

# %%
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


# %%
# Model parameters
num_classes = 10
input_shape = (28, 28, 1)

model = keras.Sequential(
    [
        keras.layers.Input(shape=input_shape),
        keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(num_classes, activation="softmax"),
    ]
)


# %%
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    metrics=[
        keras.metrics.SparseCategoricalAccuracy(name="acc"),
    ],
)


# %%
batch_size = 128
epochs = 20
callbacks = [
    KerasCheckpointerCallback(
        local_checkpoint_file="/work/jeppelt/checkpointing/checkpointer/checkpoint", 
        checkpoint_every=1, 
        checkpoint_transfer_mode="shared",
        checkpoint_transfer_target=Path("/ceph/jeppelt/checkpoint.ckpt")
    ),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=2),
]

# %%

model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.15,
    callbacks=callbacks,
)
score = model.evaluate(x_test, y_test, verbose=0)


# %%


# %%



