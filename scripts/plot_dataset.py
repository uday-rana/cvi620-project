import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv

DATASET_PATH = "./data"

df = read_csv(
    f"{DATASET_PATH}/driving_log.csv",
    names=["center", "left", "right", "steering", "throttle", "brake", "speed"],
    usecols=["center", "steering"],
)

data = []

for row in df.itertuples():
    if abs(row.steering) < 0.25 and np.random.rand() < 0.5:
        continue
    if abs(row.steering) < 0.1 and np.random.rand() < 0.3:
        continue

    steering = row.steering
    repeats = 1

    if abs(steering) > 0.4:
        repeats = 100
    elif abs(steering) > 0.3:
        repeats = 16
    elif abs(steering) > 0.15:
        repeats = 8

    for _ in range(repeats):
        data.append(row.steering)

plt.hist(data)
plt.show()
