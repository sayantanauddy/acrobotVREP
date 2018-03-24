# acrobotVREP
An gym environment for acrobot in VREP, along with a sample script for training the acrobot agent using DDPG (deep deterministic policy gradient). 
Using a similar approach a VREP-based reinforcement learning setup can be created for more complicated control problems, or for other RL algorithms.

![alt text](acrobot.png)

## Folders and scripts
1. acrobotVREP - Python package for acrobotVREP (contains acrobotVrep.py
    1.1 acrobotVREP/env/acrobotVrep.py - Gym class for acrobotVREP
    1.2 acrobotVREP/env/transformations.py - A matrix library by Christoph Gohlke <http://www.lfd.uci.edu/~gohlke/>

2. vrep_scenes - VREP scene file

3. acrobot_train.py - Script for training or testing the acrobot agent


## Dependencies
1. Python 2.7 (not tested on Python 3)
2. OpenAI Gym (https://gym.openai.com/)
3. Pypot (https://poppy-project.github.io/pypot/) - provides an easy way of interacting with VREP
4. Keras (https://keras.io/)
5. Keras-rl (https://github.com/keras-rl/keras-rl)
6. VREP simulator (http://www.coppeliarobotics.com/)


## Setup

1. Set the $HOME location to where the repository can be stored:

    ```bash
    ### Change the path as required
    export HOME=/set/as/needed
    ```
    
2. Clone the repository (or copy it from the disk)

    ```bash
    cd $HOME
    mkdir -p $HOME/computing/repositories/
    cd $HOME/computing/repositories/
    git clone https://github.com/sayantanauddy/acrobotVREP.git
    ```

3. Download VREP

    ```bash
    cd $HOME
    mkdir $HOME/computing/simulators/
    cd $HOME/computing/simulators/
    # Download
    wget http://coppeliarobotics.com/files/V-REP_PRO_EDU_V3_4_0_Linux.tar.gz
    # Extract
    tar -xvf V-REP_PRO_EDU_V3_4_0_Linux.tar.gz
    ```

4. Create the virtual environment

    ```bash
    cd $HOME
    virtualenv --system-site-packages $HOME/matsuoka_virtualenv
    # Activate the virtual environment
    source $HOME/matsuoka_virtualenv/bin/activate
    ```

5. Add the code location to PYTHONPATH

    ```bash
    export PYTHONPATH=$PYTHONPATH:$HOME/computing/repositories/acrobotVREP
    ```

6. Install the dependencies

    ```bash
    # numpy, matplotlib should also be installed
    pip install pypot
    pip install poppy-humanoid
    pip install --upgrade tensorflow
    pip install keras
    pip install gym
    pip install h5py
    ```

7. Start VREP in a terminal

    ```bash
    cd $HOME/computing/simulators/V-REP_PRO_EDU_V3_4_0_Linux
    ./start_vrep.sh
    ```

8. Run acrobot_train.py
    
    ```bash
    python acrobot_train.py
    ```
    

