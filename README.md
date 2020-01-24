# Satmind
A reinforcement learning algorithm controller for a satellite using the Orekit library. The reinforcement learning algorithm 
is based on the Deep Deterministic Policy Gradient (DDPG) algorithm and prioritzed experience replay. The agent is a 
satellite that traverses a spacecraft enviornment. THe spacecraft's thruster is based on an electric proplution system which 
produces a low amount of thrust (< 1 N) with a long mission time (days).


## Purpose
The purpose is use reinforcement learning to solve low-thrust trajectory problems.

## Dependencies
Easiest way to install all the required packages is through Anaconda.

- Python >= 3.5 or later
- Tensorflow = 1.15
- Orekit >=10.0
- matplotlib for displaying results
- openai gym for testing RL algorithm

Use requirements.txt for easy setup using conda

`conda create --name Satmind --file requirements.txt`

## Usage
`python test_rl.py` 

tests to make sure RL algorithm is running correctly and runs in and openAI gym enviornment.

`python Satmind/ddpg-sat.py`  

starts the main RL algorithm in the orekit envornment.

Satmind/orekit-env.py

Starts and configures the orekit enviornment



