import pickle
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

# examined knearest values
k_arr = list(range(1,21))+[22,25,27,30,35,40,45,50,55,60]

# read boxplots data
directory = "box_details_result"
box_arr = []
mean_arr = []
for k in k_arr:
    file_path = os.path.join(directory, f"boxplot_data{k}.pkl")
    with open(file_path, "rb") as f:
        boxplot_data = pickle.load(f)
    box = {
        'whislo': boxplot_data["whiskers"][0][1],  # bottom whisker end
        'q1': boxplot_data["whiskers"][0][0],  # 1st quartile
        'med': boxplot_data["medians"][0],  # median
        'q3': boxplot_data["whiskers"][1][0],  # 3rd quartile
        'whishi': boxplot_data["whiskers"][1][1],  # top whisker end
        'fliers': boxplot_data["fliers"][0],  # fliers
        }
    box_arr.append(box)
    mean_arr.append(boxplot_data["means"][0])  # mean

#boxplot from the data
flierprops = dict(marker='o', markerfacecolor="red", markeredgecolor="black", markersize=4)
medianprops = dict(color="orange")

plt.style.use("fivethirtyeight")
fig, ax = plt.subplots(figsize=(18,6))
ax.bxp(box_arr, positions=k_arr, flierprops=flierprops, medianprops=medianprops)
ax.plot(k_arr, mean_arr, linewidth=0.5, linestyle="--", marker="x", markersize=5)
ax.set_xlabel("k-nearest neighbors")
ax.set_ylabel("Percentage in AD [%]")

plt.tight_layout()
plt.savefig("boxplot.pdf")

#boxplot rounded to integers
for box in box_arr:
    for name in box:
        if name!="fliers":
            box[name] = round(box[name])
        else:
            box[name] = list(np.round(np.array(box[name])))
mean_arr = list(np.round(np.array(mean_arr)))

fig, ax = plt.subplots(figsize=(10,6))
ax.bxp(box_arr, positions=list(range(1,31)), flierprops=flierprops, medianprops=medianprops)
ax.plot(list(range(1,31)), mean_arr, linewidth=0.5, linestyle="--", marker="x", markersize=5)
ax.set_xticklabels(k_arr)
ax.set_xlabel("k-nearest neighbors")
ax.set_ylabel("Percentage in AD [%]")

plt.tight_layout()
plt.savefig("boxplot_rounded_compressed.pdf")

#accuracy plot
accuracy_arr = []
k_arr = list(range(1,21)) + [22,25,27,30,35,40,45,50,55,60]
for i in k_arr:
    accuracy = np.mean(pd.read_csv(f"accuracy/k_accuracy{i}.csv").values)
    accuracy_arr.append(accuracy)

plt.style.use("fivethirtyeight")
plt.figure(figsize=(10,6))
plt.plot(k_arr, accuracy_arr, color="black", linestyle="--", linewidth=1, marker="o", markerfacecolor="deepskyblue")
plt.xlabel("k-nearest neighbors")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.savefig("accuracy_plot.pdf")