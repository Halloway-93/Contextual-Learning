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
df = df[(df.time >= fixOff+20) & (df.time <= latency)]
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
plt.xlabel("time (ms)")
plt.ylabel("eye x position (deg)")
plt.show()
# %%
# Last 150 trials of the 6th subject
lastTrial = df[(df["sub"] == 6) & (df["proba"] == 1) & (df["trial"] > 150)]
allFit = []
for t in lastTrial.trial.unique():
    x = lastTrial[lastTrial["trial"] == t].time.values
    y = lastTrial[lastTrial["trial"] == t].xp.values
    pw_fit = pw.Fit(x, y, n_breakpoints=2)
    allFit.append(pw_fit)
# %%
for pw_fit in allFit:
    pw_fit.plot_data(color="grey", s=20)
    pw_fit.plot_fit(color="red", linewidth=4)
    pw_fit.plot_breakpoints()
    pw_fit.plot_breakpoint_confidence_intervals()
    plt.xlabel("time (ms)")
    plt.ylabel("eye x position (deg)")
    plt.show()

# %%
ms = pw.ModelSelection(x, y, max_breakpoints=3)
# %%
ms.models
# %%

# Get the key results of the fit
pw_results = pw_fit.get_results()
pw_estimates = pw_results["estimates"]
pw_estimates
# %%

for t in lastTrial.trial.unique():
    x = lastTrial[lastTrial["trial"] == t].time.values
    y = lastTrial[lastTrial["trial"] == t].xp.values
    ms = pw.ModelSelection(x, y, max_breakpoints=3,n_boot=100)

# %%

pw_fit = pw.Fit(x, y, n_breakpoints=2,n_boot=1000)
# %%
print(pw_fit.summary())

# %%

ms = pw.ModelSelection(x, y, max_breakpoints=3,n_boot=1000)
# %%

ms = pw.ModelSelection(x, y, max_breakpoints=5,n_boot=100)
# %%
pw_fit.plot_data(color="grey", s=20)
pw_fit.plot_fit(color="red", linewidth=4)
pw_fit.plot_breakpoints()
pw_fit.plot_breakpoint_confidence_intervals()
plt.xlabel("time (ms)")
plt.ylabel("eye x position (deg)")
plt.show()

