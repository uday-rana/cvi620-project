import matplotlib.pyplot as plt
from pandas import read_csv

DATASET_PATH = "./data"

df = read_csv(
    f"{DATASET_PATH}/driving_log.csv",
    names=["Center", "Left", "Right", "Steering", "Throttle", "Brake", "Speed"],
    usecols=["Center", "Steering"],
)

plt.hist(df["Steering"])
plt.show()
