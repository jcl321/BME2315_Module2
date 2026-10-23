import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.interpolate import interp1d

df = pd.read_csv("Module2_FSHD_Dataset.csv")
print(df.shape)
df.head()

T1 = df['lag_months_y1'].min()   # earliest Year-1 visit across all subjects
T2 = df['lag_months_y2'].min()   # earliest Year-2 visit across all subjects
print(f"T1 = {T1:.2f} months")
print(f"T2 = {T2:.2f} months")

row = df.loc[df['subject_id'] == '13-0004'].iloc[0] ## picks example patient to interpolate

# creates arrays of time intervals between datapoints and fat fraction measures
lag_months = np.array([row['lag_months_baseline'], row['lag_months_y1'], row['lag_months_y2']])
ff_values = np.array([row['ff_baseline_R'], row['ff_y1_R'], row['ff_y2_R']])

print("lag_months:", lag_months)
print("ff_values: ", ff_values)

f_linear = interp1d(lag_months, ff_values, kind='linear') # computes linear interpolation between baseline and timepoint
f_quadratic = interp1d(lag_months, ff_values, kind='quadratic') # computes quadratic interpolation between first and second timepoint

print("f_linear(T1)   =", f_linear(T1))
print("f_quadratic(T2) =", f_quadratic(T2))

TEAL = "#1f7a8c"
PINK = "#e0569b"
ORANGE = "#e8821e"

x_curve = np.linspace(lag_months[0], lag_months[-1], 200)
quad_curve = f_quadratic(x_curve)

# plots linear and quadratic interpolations to estimate datapoints
plt.figure(figsize=(7.5, 5))
plt.plot(x_curve, quad_curve, color=ORANGE, lw=2.5, linestyle="--", label="Quadratic (all 3 points)")
plt.plot(lag_months[:2], ff_values[:2], color=PINK, lw=2.5, linestyle="--", label="Linear (baseline -> Year 1)")
plt.scatter(lag_months, ff_values, color=TEAL, s=90, zorder=4, label="Measured visits (known)")
plt.scatter([T1], [f_linear(T1)], color=PINK, marker="D", s=120, zorder=5, label=f"Linear estimate @ T1={T1:.1f}")
plt.scatter([T2], [f_quadratic(T2)], color=ORANGE, marker="D", s=120, zorder=5, label=f"Quadratic estimate @ T2={T2:.1f}")
plt.title("Subject 13-0004 — right tibialis anterior fat fraction")
plt.xlabel("Months since baseline (lag_months)")
plt.ylabel("Fat fraction (%)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# repeats interpolation process for every patient
results = []

for idx, row in df.iterrows():
    subject_id = row['subject_id']

    lag_months = np.array([row['lag_months_baseline'], row['lag_months_y1'], row['lag_months_y2']])
    ff_values = np.array([row['ff_baseline_L'], row['ff_y1_L'], row['ff_y2_L']])
    d4z4 = np.array(row['d4z4'])
    dux4 = np.array(row['dux4_signature_baseline_L'])

    if not (np.isnan(lag_months[:2]).any() or np.isnan(ff_values[:2]).any()):
        est_T1 = float(interp1d(lag_months[:2], ff_values[:2], kind='linear')(T1))
    else:
        est_T1 = np.nan

    if not (np.isnan(lag_months).any() or np.isnan(ff_values).any()):
        est_T2 = float(interp1d(lag_months, ff_values, kind='quadratic')(T2))
    else:
        est_T2 = np.nan

    results.append({'subject_id': subject_id,
                'estimate_T1_linear': est_T1,
                'estimate_T2_quadratic': est_T2, 'd4z4': d4z4, 'dux4_signature': dux4})
    pass


results_df = pd.DataFrame(results)
results_df.head()

results_df.to_csv(
    "interpolated_estimates.csv",
    index=False
)

print(results_df)

percent_change = []

for idx, row in results_df.iterrows():
    if pd.notna(row['estimate_T1_linear']) and pd.notna(row['estimate_T2_quadratic']):
        change = (
            (row['estimate_T2_quadratic'] - row['estimate_T1_linear'])
            / row['estimate_T1_linear']
        ) * 100

        percent_change.append(change)

d4z4_list = []

for idx, row in results_df.iterrows():
    if pd.notna(row['estimate_T1_linear']) and pd.notna(row['estimate_T2_quadratic']):

        d4z4_list.append(int(row['d4z4']))

plt.scatter (d4z4_list, percent_change, color='blue')

dux4_list = []
for idx, row in results_df.iterrows():
    if pd.notna(row['estimate_T1_linear']) and pd.notna(row['estimate_T2_quadratic']) and pd.notna(row['dux4_signature']):
        dux4_list.append(row['dux4_signature'])

percent_change_for_dux4 = []

for idx, row in results_df.iterrows():
    if pd.notna(row['estimate_T1_linear']) and pd.notna(row['estimate_T2_quadratic']) and pd.notna(row['dux4_signature']):
        change = (
            (row['estimate_T2_quadratic'] - row['estimate_T1_linear'])
            / row['estimate_T1_linear']
        ) * 100

        percent_change_for_dux4.append(change)

plt.scatter (d4z4_list, percent_change, color='blue')
plt.xlabel('D4Z4 Repeats')
plt.ylabel('Percent Change in Left Leg Fat Fraction')
plt.title('Scatter Plot of D4Z4 vs. % Change in Fat Fraction')
plt.show()



plt.scatter (dux4_list, percent_change_for_dux4, color = 'blue')
plt.xlabel('Left Leg DUX4 Signature Baseline')
plt.ylabel('Percent Change in Left Leg Fat Fraction')
plt.title('Scatter Plot of DUX4 Signature vs. % Change in Fat Fraction')

model = LinearRegression()
x_reg = np.array(dux4_list).reshape(-1, 1)
y = percent_change_for_dux4

model.fit(x_reg, y)
slope = model.coef_[0]
intercept = model.intercept_
r2 = model.score(x_reg, y)
equation = f"y = {slope:.2f}x + {intercept:.2f}\nR^2 = {r2:.2f}"

plt.text(3, -20, equation, color = "blue", fontsize = 12, verticalalignment = "top")
plt.plot(x_reg, model.predict(x_reg), color = "blue")

plt.show()