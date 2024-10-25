import numpy as np
from scipy.signal import savgol_filter
import pandas as pd
import matplotlib.pyplot as plt


def detect_saccades(data, mono=True):
    sample_window = 0.001  # 1 kHz eye tracking
    deg = 27.28  # pixel to degree conversion
    tVel = 22  # default velocity threshold in deg/s
    tAcc = 1000  # acceleration threshold in deg/s^2
    trials = data.trial.unique()
    saccades = []

    for iTrial in trials:
        if mono:
            xPos = data[data.trial == iTrial].xp.values
            yPos = data[data.trial == iTrial].yp.values
        else:
            xPos = data[data.trial == iTrial].xpr.values
            yPos = data[data.trial == iTrial].ypr.values

        # Calculate velocity and acceleration
        xVel, yVel = calculate_velocity(xPos, yPos, sample_window, deg)
        xAcc, yAcc = calculate_acceleration(xVel, yVel, sample_window)
        euclidVel = np.sqrt(xVel**2 + yVel**2)
        euclidAcc = np.sqrt(xAcc**2 + yAcc**2)

        # Detect saccade candidates based on velocity and acceleration
        vel_candidates = np.where(euclidVel > tVel)[0]
        acc_candidates = np.where(euclidAcc > tAcc)[0]
        candidates = np.union1d(vel_candidates, acc_candidates)

        # Find saccade start and end points
        saccade_intervals = find_saccade_intervals(candidates, xAcc, yAcc)

        # Process each saccade interval
        for start, end in saccade_intervals:
            peakVelocity = np.max(euclidVel[start : end + 1])
            start_time = data[data.trial == iTrial].time.values[start]
            end_time = data[data.trial == iTrial].time.values[end]
            xDist = xPos[end] - xPos[start]
            yDist = yPos[end] - yPos[start]
            euclidDist = np.sqrt(xDist**2 + yDist**2)

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

    return pd.DataFrame(saccades)


def calculate_velocity(xPos, yPos, sample_window, deg):
    xVel = np.zeros_like(xPos)
    yVel = np.zeros_like(yPos)
    for ii in range(2, len(xPos) - 2):
        xVel[ii] = (xPos[ii + 2] + xPos[ii + 1] - xPos[ii - 1] - xPos[ii - 2]) / (
            6 * sample_window * deg
        )
        yVel[ii] = (yPos[ii + 2] + yPos[ii + 1] - yPos[ii - 1] - yPos[ii - 2]) / (
            6 * sample_window * deg
        )
    return xVel, yVel


def calculate_acceleration(xVel, yVel, sample_window):
    xAcc = np.zeros_like(xVel)
    yAcc = np.zeros_like(yVel)
    for ii in range(2, len(xVel) - 2):
        xAcc[ii] = (xVel[ii + 2] + xVel[ii + 1] - xVel[ii - 1] - xVel[ii - 2]) / (
            6 * sample_window
        )
        yAcc[ii] = (yVel[ii + 2] + yVel[ii + 1] - yVel[ii - 1] - yVel[ii - 2]) / (
            6 * sample_window
        )
    return xAcc, yAcc


def find_saccade_intervals(candidates, xAcc, yAcc):
    intervals = []
    if len(candidates) > 0:
        start = candidates[0]
        for i in range(1, len(candidates)):
            if candidates[i] - candidates[i - 1] > 1:
                end = candidates[i - 1]
                # Check for zero-crossing in acceleration
                if has_zero_crossing(xAcc[start : end + 1]) or has_zero_crossing(
                    yAcc[start : end + 1]
                ):
                    intervals.append((start, end))
                start = candidates[i]
        # Check the last interval
        if has_zero_crossing(xAcc[start : candidates[-1] + 1]) or has_zero_crossing(
            yAcc[start : candidates[-1] + 1]
        ):
            intervals.append((start, candidates[-1]))
    return intervals


def has_zero_crossing(signal):
    return np.any(np.diff(np.sign(signal)) != 0)


# %%
def detect_saccades(data, mono=True):
    sample_window = 0.001  # 1 kHz eye tracking
    deg = 27.28  # pixel to degree conversion
    tVel = 22  # default velocity threshola in deg/s
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

        euclidAcc = np.gradient(euclidVel) / sample_window
        print(euclidAcc)
        candidates = np.where(euclidVel > tVel)[0]
        if len(candidates) > 0:
            diffCandidates = np.diff(candidates)
            breaks = np.concatenate(
                ([0], np.where(diffCandidates > 1)[0] + 1, [len(candidates)])
            )

            for jj in range(len(breaks) - 1):
                saccade = [candidates[breaks[jj]], candidates[breaks[jj + 1] - 1]]
                xDist = xAcc[saccade[1]] - xAcc[saccade[0]]
                yDist = yAcc[saccade[1]] - yAcc[saccade[0]]
                # xDist = xPos[saccade[1]] - xPos[saccade[0]]
                # yDist = yPos[saccade[1]] - yPos[saccade[0]]
                euclidDist = np.sqrt(xDist**2 + yDist**2)
                if euclidDist > 15000:
                    peakVelocity = np.max(euclidVel[saccade[0] : saccade[1] + 1])
                    start_time = data[data.trial == iTrial].time.values[saccade[0]]
                    end_time = data[data.trial == iTrial].time.values[saccade[1]]
                    saccades.append(
                        {
                            "trial": iTrial,
                            "start": start_time,
                            "end": end_time,
                            "dur": end_time - start_time,
                            "xAcc": xDist,
                            "yAcc": yDist,
                            "euclidAcc": euclidDist,
                            "peakVelocity": peakVelocity,
                        }
                    )
        # plt.plot(xVel)
        # plt.show()
    plt.plot(euclidAcc)
    plt.axhline(0)
    plt.show()
    saccades_df = pd.DataFrame(saccades)
    return saccades_df


# %%


def detect_saccades(data, mono=True, window_size=101):
    """
    Detect saccades using adaptive thresholding based on rolling statistics

    Parameters:
    -----------
    data : pandas DataFrame
        Eye tracking data with columns for position and time
    mono : bool
        If True, use monocular data (xp, yp), else use right eye data (xpr, ypr)
    window_size : int
        Size of the rolling window for statistics (should be odd number)

    Returns:
    --------
    pandas DataFrame
        Detected saccades with their properties
    """
    sample_window = 0.001  # 1 kHz eye tracking
    deg = 27.28  # pixel to degree conversion
    trials = data.trial.unique()
    saccades = []

    def calculate_velocity(pos, window_size=5):
        """Calculate velocity using Savitzky-Golay filter"""
        return savgol_filter(pos, window_size, 2, deriv=1) / (sample_window * deg)

    def calculate_acceleration(vel, window_size=5):
        """Calculate acceleration using Savitzky-Golay filter"""
        return savgol_filter(vel, window_size, 2, deriv=1) / sample_window

    def detect_saccade_onset(velocity, acceleration, std_threshold=3.0):
        """
        Detect saccade onset using rolling statistics
        Returns indices where saccades likely begin and end
        """
        # Calculate rolling statistics
        rolling_std = pd.Series(velocity).rolling(window_size, center=True).std()
        rolling_mean = pd.Series(velocity).rolling(window_size, center=True).mean()

        # Find points where velocity exceeds threshold based on local statistics
        threshold = rolling_mean + (std_threshold * rolling_std)
        threshold = threshold.fillna(method="bfill").fillna(method="ffill")

        # Find potential saccade points
        candidates = velocity > threshold

        # Group consecutive True values
        changes = np.diff(candidates.astype(int))
        starts = np.where(changes == 1)[0] + 1
        ends = np.where(changes == -1)[0] + 1

        # Ensure we have paired starts and ends
        if len(starts) == 0 or len(ends) == 0:
            return [], []
        if starts[0] > ends[0]:
            ends = ends[1:]
        if len(starts) > len(ends):
            starts = starts[:-1]

        return starts, ends

    for iTrial in trials:
        # Get position data
        if mono:
            xPos = data[data.trial == iTrial].xp.values
            yPos = data[data.trial == iTrial].yp.values
        else:
            xPos = data[data.trial == iTrial].xpr.values
            yPos = data[data.trial == iTrial].ypr.values

        # Calculate velocities using Savitzky-Golay filter
        xVel = calculate_velocity(xPos)
        yVel = calculate_velocity(yPos)
        euclidVel = np.sqrt(xVel**2 + yVel**2)

        # Calculate accelerations
        xAcc = calculate_acceleration(xVel)
        yAcc = calculate_acceleration(yVel)
        euclidAcc = np.sqrt(xAcc**2 + yAcc**2)

        # Detect saccades using adaptive thresholding
        starts, ends = detect_saccade_onset(euclidVel, euclidAcc)

        # Process detected saccades
        for start, end in zip(starts, ends):
            # Minimum duration check (20ms)
            if (end - start) * sample_window < 0.02:
                continue

            # Calculate saccade properties
            peakVelocity = np.max(euclidVel[start:end])
            mean_acceleration = np.mean(euclidAcc[start:end])

            # Position change during saccade
            x_displacement = xPos[end] - xPos[start]
            y_displacement = yPos[end] - yPos[start]
            amplitude = np.sqrt(x_displacement**2 + y_displacement**2)

            # Only include if amplitude is significant (in pixels)
            if amplitude > 5:  # Adjustable threshold
                start_time = data[data.trial == iTrial].time.values[start]
                end_time = data[data.trial == iTrial].time.values[end]

                saccades.append(
                    {
                        "trial": iTrial,
                        "start": start_time,
                        "end": end_time,
                        "duration": end_time - start_time,
                        "amplitude": amplitude,
                        "peak_velocity": peakVelocity,
                        "mean_acceleration": mean_acceleration,
                        "x_displacement": x_displacement,
                        "y_displacement": y_displacement,
                    }
                )

    return pd.DataFrame(saccades)


# %%
df = pd.read_csv(
    "/Volumes/work/brainets/oueld.h/contextuaLearning/directionCue/results_voluntaryDirection/allRawData.csv"
)
# %%
filtered_df = pd.read_csv(
    "/Volumes/work/brainets/oueld.h/contextuaLearning/directionCue/results_voluntaryDirection/JobLibProcessing.csv"
)
# %%
df.drop(columns=["cr.info"], inplace=True)
df.columns
# %%
cond = df[
    (df["sub"] == 6)
    & (df["proba"] == 75)
    & (df["trial"] == 100)
    & (df["time"] >= -200)
    & (df["time"] <= 600)
]
cond
# %%
condFiltered = filtered_df[
    (filtered_df["sub"] == 6)
    & (filtered_df["proba"] == 75)
    & (filtered_df["trial"] == 100)
]
condFiltered
# %%
saccades = detect_saccades(cond)
saccades
# %%
starts = saccades["start"]
ends = saccades["end"]
plt.plot(cond.time, cond.xp)
for i in range(len(starts)):
    # plot shaded area between srarts[i] and ends [i]
    plt.fill_between(
        [starts.iloc[i], ends.iloc[i]],
        cond.xp.min(),
        cond.xp.max(),
        color="red",
        alpha=0.3,
    )
plt.show()
# %%
plt.plot(condFiltered.time, condFiltered.filtVelo)
plt.plot(condFiltered.time, condFiltered.velo)
for i in range(len(starts)):
    # plot shaded area between srarts[i] and ends [i]
    plt.fill_between(
        [starts.iloc[i], ends.iloc[i]],
        condFiltered.filtVelo.min(),
        condFiltered.filtVelo.max(),
        color="red",
        alpha=0.3,
    )
plt.show()
# %%
plt.plot(condFiltered.time, condFiltered.xp)
plt.plot(condFiltered.time, condFiltered.filtPos)
plt.show()
# %%
df = df.apply(pd.to_numeric, errors="coerce")
# %%
for sub in df["sub"].unique():
    for proba in df[df["sub"] == sub]["proba"].unique():
        cond = df[(df["sub"] == sub) & (df["proba"] == proba)]
        saccades = detect_saccades(cond)
        for t in cond.trial.unique():
            saccTrial = saccades[saccades["trial"] == t]
            starts = saccTrial["start"]
            ends = saccTrial["end"]
            plt.plot(cond[cond.trial == t].time, cond[cond.trial == t].xp, alpha=0.7)
            for i in range(len(starts)):
                # plot shaded area between srarts[i] and ends [i]
                plt.fill_between(
                    [starts.iloc[i], ends.iloc[i]],
                    cond[cond.trial == t].xp.min(),
                    cond[cond.trial == t].xp.max(),
                    color="red",
                    alpha=0.3,
                )
            plt.show()
