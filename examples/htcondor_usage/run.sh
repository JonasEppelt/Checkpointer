#!/bin/bash
# setup venv
python3 -m venv venv
source venv/bin/activate

# install checkpointer
tar -xzf checkpointer.tar
cd checkpointer_zipforship
python3 -m pip install -e .
cd ../
ls

python3 counter.py & # running the script