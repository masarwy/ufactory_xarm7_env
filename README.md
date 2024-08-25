# ufactory_xarm7_env

## Project Overview

This project consists of two main components:

### 1. Synthetic Data Generation for Manipulator Pose Estimation

The first part of the project focuses on generating synthetic data for the uFactory xArm 7 manipulator, which is simulated in MuJoCo. This process supports multiple views, allowing for the use of a multi-perspective Convolutional Neural Network (CNN). The generated data can be used to train a pose estimation model with higher accuracy by leveraging the different perspectives provided by the multi-view setup.

- **Script to Run**: `scripts/generate_traj_data.py`
- **Example Code to Load the Synthetic Data**: An example of how to load the synthetic data is provided in `tests/test3.py`

### 2. Training a Reinforcement Learning (RL) Agent for Pick-and-Place Tasks

The second part of the project involves training a Reinforcement Learning (RL) agent to perform pick-and-place tasks using the uFactory xArm 7, simulated in MuJoCo. Specifically, the agent is trained to pick lemons from one box and place them in another. The training is done using the `stable-baselines3` library, which provides robust implementations of RL algorithms. The agent learns this task through interactions with the simulated environment, utilizing the policy learned during training.

- **Script to Train the RL Agent**: `scripts/train_agent.py`
- **Script to View the Agent Playing with the Learned Policy**: `scripts/play_agent.py`

### Running the Scripts

1. **Synthetic Data Generation**:
   - To generate the synthetic data, run the following script:
     ```bash
     python scripts/generate_traj_data.py
     ```

2. **Training the RL Agent**:
   - To train the RL agent, use the following command:
     ```bash
     python scripts/train_agent.py
     ```

3. **Playing with the Trained RL Agent**:
   - To see the trained agent in action, use this script:
     ```bash
     python scripts/play_agent.py
     ```
