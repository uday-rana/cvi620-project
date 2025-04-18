# cvi620-project

Self-Driving Car Simulation

Seneca Polytechnic  
Computer Programming and Analysis  
**Course:** CVI620 – Computer Vision  
**Term:** Winter 2025 (2251)

[Watch the demo on YouTube](https://youtu.be/JOZqNO-JRSc)

## Team

| Name             | Role                                                    |
| ---------------- | ------------------------------------------------------- |
| **Sooyeon Kim**  | Data Augmentation, Inference & Testing                  |
| **Sangjune Lee** | Data Preprocessing, Dataset Collection & Testing        |
| **Uday Rana**    | Model Design, Training, Visualization |

## Setup

```sh
# Clone the repository
git clone https://github.com/uday-rana/cvi620-project.git

# Create a Conda environment
conda env create -f environment.yaml

# Activate the environment
conda activate cvi620-project
```

## Usage

1. Collect driving data using [Udacity's Self-Driving Car Simulator](https://github.com/udacity/self-driving-car-sim) in training mode. Save the output into a folder named `data/`.

2. Train the model using the following command:

   ```sh
   python cvi620-project
   ```

   The model will be saved in the current directory as `model.h5`.

3. To test the trained model, launch the simulator in autonomous mode. Then run the following command:

   ```sh
   python scripts/TestSimulation.py
   ```

## Challenges Encountered

- During testing, we encountered an issue where the vehicle did not move at all in autonomous mode. After several attempts debugging the model and dataset, we realized it was due to a package version mismatch. Installing `python-socketio==4.2.1` fixed the issue.

- Our initial dataset was too small for the depth of the neural network. We had to go back and gather more data for the model to be trained properly.

- Our dataset was heavily biased toward a steering angle of 0.00°, so we had to programmatically standardize the distribution by lowering the frequency of values close to zero and dramatically increasing the frequency of values farther away from zero.
