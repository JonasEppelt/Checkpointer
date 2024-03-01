# Checkpointer for HTC workflows using GPUs

This tool simplifies the creation, storage, detection, and restoration of checkpoints of long-running GPU workflows.

The package can be used as a standalone application that regularly checks for the existence of a new checkpoint or it can interface with your training loops.

## What are checkpoints and why do you need them?

Checkpoints are files that store the current state of a program and allow it to continue from this state.

This allows for interrupting running workflows and continuing them at a different time and place.

Workflows with this ability have various benefits:

* They are resistant to crashes and failures. In cases of technical problems like hardware failures or network issues interrupting the execution, the workflow can be restarted from the last stored checkpoint.

They can execute long-running tasks on sites that have tighter time constraints. If the job runs into a time limit and is evicted, it can be rescheduled and continue from the point it left off.

Time limits of sites can be differently motivated:

* Limiting the usage time of computing resources enables fair scheduling and ensures that workflows once running can block them for a long time.

* The time can also be limited by the availability of renewable power. Since workflows can be energy-intensive and renewable energies are not always produced when needed, it makes sense to increase computing power during times of increased energy supply.

## What is needed to successful checkpoint workflows and restart from them?

Checkpointing a workflow needs five steps:

* Creating a checkpoint that represents the current state of the workflow.

* Transferring the checkpoint to a persistent storage.

* Rescheduling the workflow.

* Recovering the checkpoint from the persistent storage.

* Restoring the workflow's state from the checkpoint.

For each of these steps, multiple solutions exist, and what solution fits best depends on the specific workflow to be checked.

## What is special for ML trainings?

For an arbitrary workflow, the current state can be very complicated to store. Dumping the current state of RAM, CPU, and GPU can cause large checkpoints and introduce dependencies on very specific hardware and software.

In contrast, the state of Machine Learning (ML) training can be defined by very few numbers. In the simplest case, this can be just the current weights of the network. A training workflow could pick up from there and continue. In reality, some more information such as logs of different metrics and losses, the current epoch and batch, the state of the random generator, and others are usually kept. This still makes creating and recovering from checkpoints straightforward. In fact, such functionality is already implemented in common ML libraries like PyTorch and TensorFlow.

## What functionality does this checkpointer provide?

This checkpointer implementation provides a standardised interface to create, transfer and restore checkpoints.

When instantiated, the checkpointer requires at minimum three attributes:

* The name of the checkpoint file as `local_checkpoint_file`,

* A checkpoint function, that takes in an arbitrary Python object and creates a checkpoint file from it.

* A restore function, which returns the Python object created from the checkpoint file.

At every call of the 'step(value)' function, the 'checkpointer' executes the following two steps:

* The `value` is stored as an internal value.

* The checkpoint function is executed for the given value.

* The checkpoint is transferred to persisten storage.

To restore the state, the `restore(default_value)` function can be called. It executes the following three steps:

* It checks if a checkpoint exists.

* If none exists, it returns the default value.

* If a checkpoint exists, it transfers it to the place specified in `local_checkpoint_file`.

* It executes the restore function and returns its result.

## What options for persistent storage are available?

The checkpointer currently can handle three transfer modes to move checkpoints to a persistent location.

* Per default, the mode `None` is used, where the current location in `local_checkpoint_file` is assumed to be persistent, and no transfers occur.

* The `shared` mode assumes a mounted persistent file system and uses `shutil.copy` to move the `local_checkpoint_file` to the location specified in `checkpoint_transfer_target`.

* The `xrootd` mode uses the XRootD protocol to copy the checkpoint to a compatible storage. The storage server is defined by the `xrootd_server_name` attribute. The location on the server is set by `checkpoint_transfer_target`. This mode requires a valid certificate to be installed in the environment.

* The `htcondor` mode assumes that the transfer of checkpoints is handled by the batch system HTCondor. It takes the `checkpoint_transfer_target` from the HTCondor ClassAd `TransferCheckpoint`. Additionally, it changes the exit code to the ClassAd `CheckpointExitCode`. This signals HTCondor to reschedule the jobs and transfer the checkpoint file. For more on checkpointing with HTCondor, read [HTCondor and Self-Checkpointing Applications](https://htcondor.readthedocs.io/en/latest/users-manual/self-checkpointing-applications.html).

The `manual` mode allows for a custom implementation. For this purpose, a `checkpoint_transfer_callback` function needs to be provided. It takes in the `local_checkpoint_file`, the `checkpoint_transfer_target` and `checkpoint_transfer_callback_kwargs`. The same function, with `local_checkpoint_file` and `checkpoint_transfer_target` switched, is used to transfer the checkpoint back from the persistent storage.

## What, if the site signals the workflow to terminate itself?

The checkpointer automatically responds to `SIGTERM` and `SIGINT`. When either of these signals is received, four actions are executed:

* The checkpoint function is executed for the currently stored internal value.

* The checkpoint is transferred to the specified persistent storage.

* The `local_checkpoint_file` is removed if the `checkpoint_transfer_mode` is not set to `None` or `htcondor`.

* The program exits with the `checkpoint_exit_code` (default 85). This signal can be used by a scheduler, that the program exited without finishing but successfully created a checkpoint.

## What if my Python program is not the direct executable?

In some cases, such as running trainings on a batch system, programs are shipped wrapped as an executable that takes care of setting up the environment, copying data, and other things before starting the actual Python program.

While it is possible to use the checkpointer in such cases without additional modifications to the executable, not all of its capabilities can be used. Namely, it must be ensured that the underlying system can properly communicate with the program and vice versa. For that, the `SIGTERM` and `SIGINT` signals need to be trapped and relayed to the Python program so that the checkpointer can react to these signals. Additionally, it must be ensured that the exit code of the executable is the exit code of the Python program. Have a look at the HTCondor example to see how this can be set up.

## Creating and transfering checkpoint files often can slow down my program. Is there a way to do these steps less often?

Setting `checkpoint_every` will cause the `step(value)` function to only update the internal checkpoint and create and transfer the checkpoint only at specified intervals. By default, `checkpoint_every` is set to 1, creating and transferring checkpoints every time `step(value)` is called. Setting it to 10 will trigger the creation and transferring every 10 calls. The reaction to `SIGTERM` and `SIGINT` is unaffected by this.

## I am using Keras or PyTorch Lightning and can not directly access the training loop to call the `step` function. How can I use this checkpointer?

High-level ML libraries like Keras and PyTorch Lightning often provide predefined training routines that cannot easily be accessed by the user. However, callbacks allow modification of these routines.

For both Keras and PyTorch Lightning, these callbacks are provided, which allow the checkpointer to interface with the respective training routines. Additionally, both take advantage of the already defined checkpoint functions, removing the need for you to define them yourself. They also take care to store the state of additional callbacks, optimisers, and loggers used in your training. Check out the examples provided in `examples/keras_example` and `example/lightning_example`.

## The ML package I am using already has something called `ModelCheckpoint`. Why not use this instead?

Checkpoints in the context of ML are often used differently. Often, they are used to find the best-performing model by a given metric, when towards the end of the training, the last model before it is aborted is not necessarily the best. Of course, these checkpoints can also be used to restart the training from a certain point, but they lack the convenient handling of checkpoint transfer and assume one local, persistent filesystem.