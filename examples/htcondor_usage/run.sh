#!/bin/bash
# setup venv
python3 -m venv venv
source venv/bin/activate

# install checkpointer
tar -xzf checkpointer.tar
cd checkpointer_zipforship
python3 -m pip install -e .
cd ../

xrdcp root://dcachexrootd-kit.gridka.de:1094/pnfs/gridka.de/belle/disk-only/LOCAL/user/jeppelt/keras_checkpoint3.test .

ls

python3 counter.py & # running the script
python_pid=$! # get the pid of the script
trap "kill -15 $python_pid" 15 # kill the script when the job is done
trap "kill -10 $python_pid" 10 # checkpoint the script when the job is done
wait $python_pid # wait for the script to finish
