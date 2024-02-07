# Checkpointer for HTC workflows using GPUs

This tool simplifies the creation, storing, detection and restoring of checkpoints of long running GPU workflows.
The package can be used as a stand alone application, that regularly checks for the existence of a new checkpoint, or it can interface with your trainings loops.

## Development Installation
```bash
git clone git@gitlab.etp.kit.edu:jeppelt/checkpointer.git 
cd checkpointer 
python3 -m pip install -e .
cd ../
```

## What are checkpoints and why do you need them?
Checkpoints are files, that store the current state of a program and allow the program to continue from this state.
This allows to interrupt running work flows an contiue them at a different time and place.
Workflows with this ability have various benefits:
     - They are resistent to crashes and failures. In cases of technical problems like hardware failures or network issues interrupting the execution, the workflow can be restartet from the last stored checkpoint.
     - They can execute long running tasks on sites, that have thighter time constraints. If the job runs into a time limit and is eveicted, it can be rescheduled and continue from the point it left off. Time limits of sites can be differently motivated:
        - Limiting the usage time of computing resources enables a fair scheduling and that workflows once running, can block them for a long time.
        - The time can also be limited by the availability of renewable power. Since computing can be energy intensive and renewable energies are not always produced when needed, it makes sense to increase computing in times of increased energy supply.

## What is needed to successful checkpoint workflows and restart from them?
Checkpointing a workflow needs five steps:
    - Creating a checkpoint representing the current state of workflow.
    - Transfering the checkpoint to a persistent storage.
    - Rescheduling the workflow.
    - Recovering the checkpoint from the persisten storage.
    - Restoring the workflows state from the checkpoint.
For each of these steps multiple solutions exist and what solution fits best depends on the concrete workflow to be checkpointed.

## What is special for ML trainings?
For an arbitrary workflow, the current state can be very complicated to be stored. Dumping the current state of RAM, CPU, GPU can cause large checkpoints and introduce dependencies on very concrete Hard- and Software.
In contrast, the state of a Machine Learning (ML) training can be defined by very few numbers. In the simplest case this can be just the current weights of the network. A trainings workflow could just pick up from there and continue. In reality some more information like logs of different metrics and losses, the current epoch and batch, the state of the random generator and others are usually kept. This still makes creating and recovering from checkpoints very straight forward. In fact such functionality is already implemented in common ML libraries like pytorch and tensorflow.

## What functionality does this checkpointer provide?
This checkpointer implementation provides a standardized interface to create, transfer and restore checkpoints.
At creation the `checkpointer` requires at minimum 3 things:
    - The name of the checkpoint file as `local_checkpoint_file`,
    - A checkpoint function, that takes in an arbitrary python object and creates a checkpoint file from it.
    - A restor function, that returns the python object created from the checkpoint file.
At every call of the `step(value)` function, the checkpointer the executes the following two steps:
    - The `value` is stored as a internal value.
    - The checkpoint function is executed for the given value.
    - The checkpoint is transfered to a persisten storage.
To restore the state, the `restore(default_value)` function can be called. It executes the following three steps:
    - It checks, if a checkpoint exists.
    - If none exists, it returns the default value.
    - If a checkpoint exists, it transfers it to the place specified in `local_checkpoint_file`.
    - It executes the restore function and returns its result.

## What options for persistent storage are available?
The checkpointer currently is able to handle three transfer modes to move checkpoints to a persistent location.
    - Per default the mode `None` is used, where the current location in `local_checkpoint_file` is assumed to be persistent and no transfers happen.
    - The `shared` mode, assumes a mounted persistent filesystem and uses `shutil.copy` to move the `local_checkpoint_file` to the location specified in `checkpoint_transfer_target`.
    - The `xrootd` mode uses the xrootd protocol to copy the checkpoint to a compatible storage. The storage server is defined by the `xrootd_server_name` attribute. The location on this server is set by `checkpoint_transfer_target`. This mode requires a valid certificate to be installed in the enviroment.
    - The mode `htcondor` assumes, that the transfer of checkpoints is handled by the batch system `htcondor`. It takes the `checkpoint_transfer_target` from the HTcondor ClassAdd `TransferCheckpoint`. Additionally, it changes the exit code to the ClassAdd `CheckpointExitCode`. This signals HTcondor to reschedule the jobs and transfer the checkpoint file. For more on checkpointing with HTcondor, read [HTCondor and Self-Checkpointing Applications](https://htcondor.readthedocs.io/en/latest/users-manual/self-checkpointing-applications.html)
    - The `manual` mode allows for a custom implementation. For that a `checkpoint_transfer_callback` function has to be provided. It takes in the `local_checkpoint_file`, the `checkpoint_transfer_target` and `checkpoint_transfer_callback_kwargs`. The same function, with `local_checkpoint_file` and `checkpoint_transfer_target` switched, is used to transfer the checkpoint back from the persistent storage. 

## What, if the site signals the workflow to terminate itself?
The checkpointer automatically reacts to `SIGTERM` and `SIGINT`. Upon receiving on of these signals, four things are executed:
    - The checkpoint function is executed for the currently stored internal value.
    - The checkpoint is transferred the the specified persisten storage.
    - The `local_checkpoint_file` is removed if the `checkpoint_transfer_mode` is not set to `None` or `htcondor`.
    - The program exits with the `checkpoint_exit_code` (default 85). This signal can be used by a scheduler, that the program exited without finishing, but successfully created a checkpoint.

## What if my python program is not the direct executable?
In some cases, like running trainings on a batch system, programms are shipped wrapped as an executable, that takes care of setting up the enviroment, copying data and other things, before starting the actual python program. While using the checkpointer in such cases without additional modifications to the executable is possible, not all of its capabilities can be used. Namely, it  must be ensured that the underlying system can properly communicate with the program and vice versa. For that the `SIGTERM` and `SIGINT` signals have to be trapped and relayed to the python program, so that the checkpointer can react on these signals. Also it must be ensured, that the exit code of the executable is the exit code of the python program. Have a look at the HTcondor example to see how this can be setup.

## Creating and transfering checkpoint files often can slow down my program. Is there a way to do these steps less often?
Setting `checkpoint_every` will cause the `step(value)` function to only update the internal checkpoint and create and transfer the checkpoint only in the specified intervals. Per default. `checkpoint_every` is set to 1, creating and transfering checkpoints every time `step(value)` is called. Setting it to 10, will trigger the creation and transfering every 10th call. The reaction to `SIGTERM` and `SIGINT` are unaffected by this.

## I am using Keras/Lighning and can not directly access the training loop to call the `step` function. How can I use this checkpointer?
High-level ML libraries like pytorch-lightning and Keras often provide predefined training routines, that can not easily be accessed by the user. However, callbacks allow a modification of these routines. For both Keras and Lightning, this callbacks are provided, that allow the checkpointer ti interface with the respective training routines. Additionally, both take advantage of the already defined checkpoint functions, removing the need for you to define them yourself. Additionally, they also take care to store the state of additional Callbacks, Optimizers and Loggers used in your training. Check out the examples provided in `examples/keras_example` and `example/lightning_example`. 

## The ML package I am using already has somthing called ModelCheckpoint. Why not use this instead?
Checkpoints in the context of Machine Learing are often used differently. Often they are used to find the best performing model by a given metric, when towards the end of the training, to last model before it is aborted, is not necessarily the best. Of course, these checkpoints can also be used to restart the training from a certain point, but they lack the convienent handling of checkpoint transfer and assume one local, persistent file system.