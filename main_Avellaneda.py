#import important libraries and the Patient Class file
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics as statistics
from scipy import stats
from patient_Avellaneda import *

DATA_DIR = Path(r"FSHD_MRI_data/FSHD_TA_MRI_data_0020_TP1_TP2")
manifest = pd.read_csv(DATA_DIR / "slice_info.csv")
manifest

row = manifest[manifest["timepoint"] == "TP1"].iloc[3]  # representative mid-muscle slice

water = np.load(DATA_DIR / row["water_file"])
fat = np.load(DATA_DIR / row["fat_file"])
mask = np.load(DATA_DIR / row["mask_file"]).astype(bool)

print("Water image shape:", water.shape)
print("Fat image shape:", fat.shape)
print("Mask shape:", mask.shape)
print("Number of muscle pixels:", mask.sum())

denominator = water + fat
valid_pixels = denominator > 0

fat_fraction = np.divide(
    fat,
    denominator,
    out=np.zeros_like(fat, dtype=float),
    where=valid_pixels
) * 100

# Keep the whole supplied image region for the first analysis.
fat_fraction_full = fat_fraction.copy()

# Then apply the TA mask for the muscle-specific analysis.
fat_fraction_masked = fat_fraction.copy()
fat_fraction_masked[~mask] = np.nan

whole_image_values = fat_fraction_full[valid_pixels]
muscle_values = fat_fraction_masked[~np.isnan(fat_fraction_masked)]

print(f"Whole supplied image region — mean: {np.mean(whole_image_values):.1f}%")
print(f"Whole supplied image region — minimum: {np.min(whole_image_values):.1f}%")
print(f"Whole supplied image region — maximum: {np.max(whole_image_values):.1f}%")
print()
print(f"TA muscle only — mean: {np.mean(muscle_values):.1f}%")
print(f"TA muscle only — minimum: {np.min(muscle_values):.1f}%")
print(f"TA muscle only — maximum: {np.max(muscle_values):.1f}%")

# Find the bounding box of the TA mask
y_pixels, x_pixels = np.where(mask)

y0, y1 = y_pixels.min(), y_pixels.max() + 1
x0, x1 = x_pixels.min(), x_pixels.max() + 1

# Crop the TA-specific fat-fraction map
fat_fraction_TA = fat_fraction_masked[y0:y1, x0:x1]

fig, axes = plt.subplots(1, 4, figsize=(15, 4))

# Water image with TA mask outline
axes[0].imshow(water, cmap="gray")
axes[0].contour(mask, levels=[0.5], colors="red", linewidths=2)
axes[0].set_title("Water image + TA mask")

# Fat image with TA mask outline
axes[1].imshow(fat, cmap="gray")
axes[1].contour(mask, levels=[0.5], colors="red", linewidths=2)
axes[1].set_title("Fat image + TA mask")

# Fat fraction for the whole supplied image region
whole_image = axes[2].imshow(
    fat_fraction_full,
    cmap="plasma",
    vmin=0,
    vmax=100
)
axes[2].contour(mask, levels=[0.5], colors="cyan", linewidths=2)
axes[2].set_title("Fat fraction: whole region")

# Zoomed-in fat fraction for the TA only
ta_image = axes[3].imshow(
    fat_fraction_TA,
    cmap="plasma",
    vmin=0,
    vmax=100
)
axes[3].set_title("TA fat fraction")

for ax in axes:
    ax.axis("off")

# Both fat-fraction plots use the same 0–100% scale
fig.colorbar(
    ta_image,
    ax=axes[2:],
    fraction=0.046,
    pad=0.04,
    label="Fat fraction (%)"
)

plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Whole supplied image region
axes[0].hist(
    whole_image_values,
    bins=30,
    alpha=0.7
)
axes[0].set_xlabel("Fat fraction (%)")
axes[0].set_ylabel("Number of pixels")
axes[0].set_title("Whole supplied image region")

# TA muscle only
axes[1].hist(
    muscle_values,
    bins=30,
    alpha=0.7,
    color="gold"
)
axes[1].set_xlabel("Fat fraction (%)")
axes[1].set_ylabel("Number of pixels")
axes[1].set_title("TA muscle only")

# Use the same fat-fraction range for both plots
axes[0].set_xlim(0, 100)
axes[1].set_xlim(0, 100)

plt.tight_layout()
plt.show()

#####################################

#instantiate patient objects from the csv files
Patient.instantiate_from_csv(Patient, "Module2_FSHD_Dataset.csv")
#reads the dataset when prompted for specific information
df = pd.read_csv("Module2_FSHD_Dataset.csv")

#defines a function to change the title, dataset, and subject order within the bar graph and shows it
def plot_sorted_subject_changes(title, changes):
    sorted_data = sorted(
        zip(changes, [patient.subject for patient in viable_patients])
    )
    
    patient_positions = range(len(sorted_data))
    sorted_changes, subject_ids = zip(*sorted_data)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(patient_positions, sorted_changes, height=0.7, color="orange")
    ax.set_yticks(patient_positions, subject_ids)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Fat fraction change (%)")
    ax.set_ylabel("Patient")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    plt.show()

#creates lists for storing viable patients with fat fraction data for 3 separate time points and
#storing the change in fat fraction over different time frames
viable_patients = []
all_ff_R1_diff = []
all_ff_R2_diff = []
net_ff_diff = []

#adds a patient from all_patients if there are data points for fat fractions and time points
#for years 1 and 2
for patient in Patient.all_patients:
    if patient.get_ff_R1() != 0.0 and patient.get_ff_R2() != 0.0 and patient.get_y1() != 0.0 and patient.get_y2() != 0.0:
        viable_patients.append(patient)
        print(patient)

#finds the maximum shared time from the recording of the baseline to year 1 and year 2
T1 = Patient.find_min_y1(viable_patients)   # earliest Year-1 visit across all subjects
T2 = Patient.find_min_y2(viable_patients)   # earliest Year-2 visit across all subjects

#interpolates the data points for T1 and T2 and calculating the difference between the year 1 and the
#baseline, year 2 and year 1, and year 2 and the baselin
for patient in viable_patients:
    new_ffs = Patient.interpolate(patient, True, T1, T2)
    all_ff_R1_diff.append(new_ffs[0] - patient.get_ff_Rbase())
    all_ff_R2_diff.append(new_ffs[1] - new_ffs[0])
    net_ff_diff.append(new_ffs[1] - patient.get_ff_Rbase())

#calls the method from above, changes the title of the graph and the dataset used for the change
#from year 1 to the baseline, year 2 and year 1, and year 2 and the baseline
plot_sorted_subject_changes("Year-1 Fat Fraction Change by Patient", all_ff_R1_diff)
plot_sorted_subject_changes("Year-2 Fat Fraction Change by Patient", all_ff_R2_diff)
plot_sorted_subject_changes("Total Fat Fraction Change by Patient", net_ff_diff)

#creates d4z4 groups based on the viable patients list to be used in the dotplot
d4z4_groups = sorted({patient.get_d4z4() for patient in viable_patients})
group_positions = {group: index for index, group in enumerate(d4z4_groups)}
group_counts = {group: 0 for group in d4z4_groups}
group_sizes = {
    group: sum(patient.get_d4z4() == group for patient in viable_patients)
    for group in d4z4_groups
}
dot_x = []
#adds a patient into the d4z4 groups depending on the result of the d4z4 getter Patient class method
for patient in viable_patients:
    group = patient.get_d4z4()
    within_group_index = group_counts[group]
    dot_x.append(
        group_positions[group]
        + (within_group_index - (group_sizes[group] - 1) / 2) * 0.08
    )
    group_counts[group] += 1

#settings related to the dotplot made from the d4z4 and year 2 to baseline fat fraction difference
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(dot_x, net_ff_diff, color="darkorange", edgecolor="black", alpha=0.8)
ax.axhline(0, color="black", linewidth=0.8)
group_changes = {
    group: [
        change
        for patient, change in zip(viable_patients, net_ff_diff)
        if patient.get_d4z4() == group
    ]
    for group in d4z4_groups
}
#finds the means of the patients within the groups based on d4z4
group_means = {
    group: statistics.mean(changes)
    for group, changes in group_changes.items()
}
#graphs the data points
ax.scatter(
    [group_positions[group] for group in d4z4_groups],
    [group_means[group] for group in d4z4_groups],
    marker="D",
    color="black",
    s=35,
    label="Group mean",
    zorder=3,
)
#calculating the standard deviations
groups_with_sd = [
    group for group in d4z4_groups if len(group_changes[group]) > 1
]
#adds errorbars based on standard deviations
ax.errorbar(
    [group_positions[group] for group in groups_with_sd],
    [group_means[group] for group in groups_with_sd],
    yerr=[
        statistics.stdev(group_changes[group])
        for group in groups_with_sd
    ],
    fmt="none",
    ecolor="black",
    capsize=5,
    linewidth=1.5,
    label="±1 SD",
    zorder=2,
)
#titles and axises added and plot generated
ax.set_xticks(range(len(d4z4_groups)), d4z4_groups)
ax.set_xlabel("D4Z4 group")
ax.set_ylabel("Total fat fraction change from baseline (%)")
ax.set_title("Fat Fraction Increase by D4Z4 Group")
ax.grid(axis="y", alpha=0.3)
ax.legend()
fig.tight_layout()
plt.show()
