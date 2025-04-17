# cvi620-project

Self-Driving Car Simulation

Seneca Polytechnic  
Computer Programming and Analysis  
**Course:** CVI620 – Computer Vision  
**Term:** Winter 2025 (2251)

[Watch demo on YouTube](https://youtu.be/JOZqNO-JRSc)

## Team

| Name             | Role                                                    |
| ---------------- | ------------------------------------------------------- |
| **Sooyeon Kim**  | Data Augmentation, Inference & Testing                  |
| **Sangjune Lee** | Data Preprocessing, Dataset Collection & Testing        |
| **Uday Rana**    | Model Design, Training, Dataset Batching, Visualization |

Each team member focused on a specific module:

- **Sooyeon Kim** handled the data augmentation strategy and implemented the simulation script for model inference.
- **Sangjune Lee** collected the training data, developed the preprocessing pipeline to clean and standardize the raw driving logs and images, simulation testing/recording.
- **Uday Rana** led the model architecture design, training workflow, and training visualization tools.

## Setup

```sh
# Clone the repository
git clone https://github.com/uday-rana/cvi620-project.git

# Create a Conda environment
conda env create -f environment.yaml

# Activate the environment
conda activate cvi620-project
```

### macOS Setup

If you are using **macOS**, please be aware of the following setup differences:

- The simulator used in this project was downloaded directly from the official [Udacity Self-Driving Car Simulator GitHub repository](https://github.com/udacity/self-driving-car-sim), as the default installer provided may not work reliably on macOS.

- Some Python dependencies included in `requirements.txt` or `environment.yaml` were either incompatible with macOS or not required for local development. In those cases:

  - Incompatible dependencies were either replaced with macOS-friendly versions, or
  - Unnecessary dependencies were commented out or removed to avoid installation issues.

- A separate conda environment was created to isolate these changes. If you are using macOS, it is **recommended** to review and adjust the dependencies before installing with `pip` or `conda`.

This ensures that your environment remains stable and the training pipeline runs without error on macOS systems.

## Usage

1. Collect driving data using [Udacity's Self-Driving Car Simulator](https://github.com/udacity/self-driving-car-sim) in training mode. Save the output into a folder named `data/`.

2. Train the model using the following command:

   ```sh
   python -m cvi620-project
   ```

   This will create a trained model saved as `model.h5`.

3. To test the trained model, launch the simulator in autonomous mode. Then run the following command:

   ```sh
   python scripts/TestSimulation.py
   ```

## Challenges Encountered

- During testing, we encountered an issue where the vehicle did not move at all in autonomous mode.
  After several attempts debugging the model and dataset, we realized it was due to a package version mismatch.
  Installing the following specific dependency fixed the issue:

  ```sh
  pip install python-socketio==4.2.1
  ```

- Our initial dataset was too small for the depth of the neural network. We had to go back and gather more data for the model to be trained properly.

- Our dataset was heavily biased toward a steering angle of 0.00°, so we had to programmatically standardize the distribution by lowering the frequency of values close to zero, and dramatically increasing the frequency of values farther away from zero.

## Final Deliverables

- Modularized codebase with clearly separated functionality:

  - `data_preprocessing.py`: Loads and cleans raw driving log data, preprocesses images (cropping, resizing, normalization).
  - `data_augmentation.py`: Performs data augmentation including flipping, brightness adjustment, shifting, zooming, and rotation to enrich the dataset.
  - `model_training.py`: Defines and trains the convolutional neural network (CNN) using the processed and augmented dataset.
  - `training_visualization.py`: Plots training and validation loss curves to help evaluate model performance.
  - `scripts/TestSimulation.py`: Runs the trained model in inference mode to control the car in the simulator.

- `model.h5`: The final trained model

- README with setup instructions and development challenges
