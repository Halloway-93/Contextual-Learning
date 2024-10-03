import pandas as pd
import piecewise_regression as pw
import matplotlib.pyplot as plt
import numpy as np


def detect_saccades(data, mono=True):
    sample_window = 0.001  # 1 kHz eye tracking
    deg = 27.28  # pixel to degree conversion
    tVel = 20  # default velocity threshola in deg/s
    tDist = 5  # minimum distance threshold for saccades in pixels
    trials = data.trial.unique()
    saccades = []
    for iTrial in trials:
        if mono:
            xPos = data[data.trial == iTrial].xp.values
            yPos = data[data.trial == iTrial].yp.values
        else:
            xPos = data[data.trial == iTrial].xpr.values
            yPos = data[data.trial == iTrial].ypr.values
        # Calculate instantaneous eye position and time derivative
        xVel = np.zeros_like(xPos)
        yVel = np.zeros_like(yPos)
        for ii in range(2, len(xPos) - 2):
            xVel[ii] = (xPos[ii + 2] + xPos[ii + 1] - xPos[ii - 1] - xPos[ii - 2]) / (
                6 * sample_window * deg
            )
            yVel[ii] = (yPos[ii + 2] + yPos[ii + 1] - yPos[ii - 1] - yPos[ii - 2]) / (
                6 * sample_window * deg
            )
        euclidVel = np.sqrt(xVel**2 + yVel**2)
        xAcc = np.zeros_like(xPos)
        yAcc = np.zeros_like(yPos)
        for ii in range(2, len(xVel) - 2):
            xAcc[ii] = (xVel[ii + 2] + xVel[ii + 1] - xVel[ii - 1] - xVel[ii - 2]) / (
                6 * sample_window
            )
            yAcc[ii] = (yVel[ii + 2] + yVel[ii + 1] - yVel[ii - 1] - yVel[ii - 2]) / (
                6 * sample_window
            )

        # euclidAcc = np.sqrt(xAcc**2 + yAcc**2)
        candidates = np.where(euclidVel > tVel)[0]
        if len(candidates) > 0:
            diffCandidates = np.diff(candidates)
            breaks = np.concatenate(
                ([0], np.where(diffCandidates > 1)[0] + 1, [len(candidates)])
            )

            for jj in range(len(breaks) - 1):
                saccade = [candidates[breaks[jj]], candidates[breaks[jj + 1] - 1]]
                xDist = xPos[saccade[1]] - xPos[saccade[0]]
                yDist = yPos[saccade[1]] - yPos[saccade[0]]
                euclidDist = np.sqrt(xDist**2 + yDist**2)
                if euclidDist > tDist:
                    peakVelocity = np.max(euclidVel[saccade[0] : saccade[1] + 1])
                    start_time = data[data.trial == iTrial].time.values[saccade[0]]
                    end_time = data[data.trial == iTrial].time.values[saccade[1]]
                    saccades.append(
                        {
                            "trial": iTrial,
                            "start": start_time,
                            "end": end_time,
                            "dur": end_time - start_time,
                            "xDist": xDist,
                            "yDist": yDist,
                            "euclidDist": euclidDist,
                            "peakVelocity": peakVelocity,
                        }
                    )
        # plt.plot(xVel)
        # plt.show()

    saccades_df = pd.DataFrame(saccades)
    return saccades_df


# %%
df = pd.read_csv(
    "~/boubou/contextuaLearning/directionCue/results_voluntaryDirection/rawData.csv"
)
df.head()
# %%
df.columns
# %%
# Getting the region of interest
fixOff = -200
latency = 600
df = df[(df.time >= fixOff) & (df.time <= latency)]
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
for i, pw_fit in enumerate(allFit):
    pw_fit.plot_data(color="grey", s=20)
    pw_fit.plot_fit(color="red", linewidth=4)
    pw_fit.plot_breakpoints()
    pw_fit.plot_breakpoint_confidence_intervals()
    plt.xlabel("time (ms)")
    plt.ylabel("eye x position (deg)")
    plt.title(f"Trial={149+i+1})")
    plt.show()

# %%
# Getting the velocity on each trial btewteen -200 and 600 ms
df.drop(columns="cr.info", inplace=True)
df
# %%
for sub in df["sub"].unique():
    for p in df[(df["sub"] == sub)].proba.unique():
        for t in df[(df["sub"] == sub) & (df["proba"] == p)].trial.unique():
            velo = (
                np.gradient(
                    df[
                        (df["sub"] == sub) & (df["proba"] == p) & (df["trial"] == t)
                    ].xp.values,
                    2,
                )
                * 1000
                / 27.28
            )
            t = df[
                (df["sub"] == sub) & (df["proba"] == p) & (df["trial"] == t)
            ].time.values
            plt.plot(t, velo)
            plt.xlabel("time (ms)")
            plt.ylabel("velocity (deg/ms)")
            plt.show()
# %%
ms = pw.ModelSelection(x, y, max_breakpoints=3)
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
    ms = pw.ModelSelection(x, y, max_breakpoints=7, n_boot=100)

# %%

pw_fit = pw.Fit(x, y, n_breakpoints=2, n_boot=100)
# %%
print(pw_fit.summary())

# %%

ms = pw.ModelSelection(x, y, max_breakpoints=3, n_boot=1000)
# %%

maxBreakPoints = 10
ms = pw.ModelSelection(x, y, max_breakpoints=maxBreakPoints, n_boot=100)

# %%
for pw_fit in ms.models:
    pw_fit.plot_data(color="grey", s=20)
    pw_fit.plot_fit(color="red", linewidth=4)
    pw_fit.plot_breakpoints()
    pw_fit.plot_breakpoint_confidence_intervals()
    plt.xlabel("time (ms)")
    plt.ylabel("eye x position (deg)")
    plt.show()
# %%
# %%

# List of slopes
slopes = [f"alpha{i+1}" for i in range(maxBreakPoints + 1)]
slopes
# %%
for i, pw_fit in enumerate(ms.models):
    pw_results = pw_fit.get_results()
    pw_estimates = pw_results["estimates"]
    for s in slopes:
        if s in pw_estimates.keys():
            print(f"breakpoints={i+1}:", pw_estimates[s]["estimate"])

# %%
