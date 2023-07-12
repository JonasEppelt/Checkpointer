#!/bin/bash
# install checkpointer
tar -xzf checkpointer.tar
cd checkpointer_zipforship
python3 -m pip install -e .
cd ../

# setting up checkpointing
echo -1 >> pid.txt # file needs to exist before hand TODO: verify this
echo "kill -15 $(cat pid.txt)" >> pipe.sh # creating relay script
trap "bash pipe.sh" 15 # setting up trap
python3 counter.py # running the script