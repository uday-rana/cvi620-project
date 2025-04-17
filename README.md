# cvi620-project

Seneca Polytechnic, Computer Programming and Analysis, Computer Vision, Winter 2025 (2251 CVI620)

Final Project

## Members

- Uday Rana
- Sangjune Lee
- Sooyeon Kim

## Setup

```sh
git clone
conda env create -f environment.yaml
conda activate cvi620-project
```

## Usage

1. Collect data using [Udacity's Self-Driving Car Simulator](https://github.com/udacity/self-driving-car-sim) in training mode and save it to a directory named `data/`.

2. To train a model using your dataset, run `python cvi620-project`. Your model will be saved with the file name `model.h5`.

3. To drive the car using the model, start the simulator again in autonomous mode and run `python scripts/TestSimulation.py` .

## Challenges Encountered

- Our initial dataset was too small for the depth of the neural network. We had to go back and gather more data for the model to be trained properly.

- Our dataset was heavily biased toward a steering angle of 0.00°, so we had to programmatically standardize the distribution by lowering the frequency of values close to zero, and dramatically increasing the frequency of values farther away from zero.

- Our data was collected by driving mostly in the middle of the road, so when driving in autonomous mode, if the car ended up near the edges it had trouble steering. This was fixed by performing horizontal image shifting / panning as part of data augmentation .
