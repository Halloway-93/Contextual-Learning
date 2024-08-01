import pandas as pd
import piecewise_regression as pw
import matplotlib.pyplot as plt

# %%
df = pd.read_csv(
    "/Volumes/work/brainets/oueld.h/contextuaLearning/directionCue/rawData.csv"
)
df.head()
# %%
df.columns
# %%
# Getting the region of interest
fixOff = -200
latency = 120
df = df[(df.time >= -200) & (df.time <= 120)]
# %%
# Test on one subject and one condition and one trial

x = df[(df["sub"] == 1) & (df["proba"] == 0.25) & (df["trial"] == 150)].time.values
y = df[(df["sub"] == 1) & (df["proba"] == 0.25) & (df["trial"] == 150)].xp.values
# %%
plt.plot(x, y)
plt.show()
# %%
# piecewise_regression fit
pw_fit = pw.Fit(x, y, n_breakpoints=1)
# pw_fit.summary()
print(pw_fit.summary())
# %%
# Plot the data, fit, breakpoints and confidence intervals
pw_fit.plot_data(color="grey", s=20)
# Pass in standard matplotlib keywords to control any of the plots
pw_fit.plot_fit(color="red", linewidth=4)
pw_fit.plot_breakpoints()
pw_fit.plot_breakpoint_confidence_intervals()
plt.xlabel("x")
plt.ylabel("y")
plt.show()
# %%
# Last 150 trials of the 6th subject
lastTrial = df[(df["sub"] == 6) & (df["proba"] == 1) & (df["trial"] > 150)]
allFit = []
for t in lastTrial.trial.unique():
    x = lastTrial[lastTrial["trial"] == t].time.values
    y = lastTrial[lastTrial["trial"] == t].xp.values
    pw_fit = pw.Fit(x, y, n_breakpoints=3)
    allFit.append(pw_fit)
# %%
for pw_fit in allFit:
    pw_fit.plot_data(color="grey", s=20)
    pw_fit.plot_fit(color="red", linewidth=4)
    pw_fit.plot_breakpoints()
    pw_fit.plot_breakpoint_confidence_intervals()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
