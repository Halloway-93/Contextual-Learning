import numpy as np
from scipy.signal import butter, filtfilt
import pandas as pd
from joblib import Parallel, delayed
import time
import os


def detect_saccades_no_plot(
    data, mono=True, velocity_threshold=20, min_duration_ms=10, min_amplitude=3
):
    """
    Detect saccades using Butterworth-filtered velocity and fixed threshold (no plotting version)

    Parameters:
    -----------
    data : pandas DataFrame
        Eye tracking data with columns for position and time
    mono : bool
        If True, use monocular data (xp, yp), else use right eye data (xpr, ypr)
    velocity_threshold : float
        Fixed velocity threshold in degrees/second
    min_duration_ms : float
        Minimum duration in milliseconds for a valid saccade
    min_amplitude : float
        Minimum amplitude in pixels for a valid saccade
    """
    sample_window = 0.001  # 1 kHz eye tracking
    deg = 27.28  # pixel to degree conversion
    trials = data.trial.unique()
    saccades = []

    def butter_lowpass(cutoff, fs, order=2):
        """Design Butterworth lowpass filter"""
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype="low", analog=False)
        return b, a

    def calculate_velocity(pos, fs=1000):
        """Calculate velocity using central difference and Butterworth filter"""
        vel = np.zeros_like(pos)
        vel[1:-1] = (pos[2:] - pos[:-2]) / (2 * sample_window * deg)
        b, a = butter_lowpass(cutoff=50, fs=fs)
        vel_filtered = filtfilt(b, a, vel)
        return vel_filtered

    def calculate_acceleration(vel, fs=1000):
        """Calculate acceleration using Butterworth-filtered derivative"""
        acc = np.zeros_like(vel)
        acc[1:-1] = (vel[2:] - vel[:-2]) / (2 * sample_window)
        b, a = butter_lowpass(cutoff=50, fs=fs)
        acc_filtered = filtfilt(b, a, acc)
        return acc_filtered

    def detect_saccade_onset(velocity):
        """Detect saccade onset using fixed velocity threshold"""
        candidates = velocity > velocity_threshold
        changes = np.diff(candidates.astype(int))
        starts = np.where(changes == 1)[0] + 1
        ends = np.where(changes == -1)[0] + 1

        if len(starts) == 0 or len(ends) == 0:
            return [], []
        if starts[0] > ends[0]:
            ends = ends[1:]
        if len(starts) > len(ends):
            starts = starts[:-1]

        return starts, ends

    for iTrial in trials:
        if mono:
            xPos = data[data.trial == iTrial].xp.values
            yPos = data[data.trial == iTrial].yp.values
        else:
            xPos = data[data.trial == iTrial].xpr.values
            yPos = data[data.trial == iTrial].ypr.values

        xVel = calculate_velocity(xPos)
        yVel = calculate_velocity(yPos)
        euclidVel = np.sqrt(xVel**2 + yVel**2)

        xAcc = calculate_acceleration(xVel)
        yAcc = calculate_acceleration(yVel)
        euclidAcc = np.sqrt(xAcc**2 + yAcc**2)

        starts, ends = detect_saccade_onset(euclidVel)

        for start, end in zip(starts, ends):
            duration_ms = (end - start) * sample_window * 1000
            if duration_ms < min_duration_ms:
                continue

            peakVelocity = np.max(euclidVel[start:end])
            mean_acceleration = np.mean(euclidAcc[start:end])

            x_displacement = xPos[end] - xPos[start]
            y_displacement = yPos[end] - yPos[start]
            amplitude = np.sqrt(x_displacement**2 + y_displacement**2)

            if amplitude < min_amplitude:
                continue

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


def process_subject_probability(df, sub, proba):
    """Process a single subject-probability combination"""
    cond = df[
        (df["sub"] == sub)
        & (df["proba"] == proba)
        & (df["time"] >= -200)
        & (df["time"] <= 600)
    ]

    saccades = detect_saccades_no_plot(
        cond, mono=True, velocity_threshold=20, min_duration_ms=3, min_amplitude=5
    )

    # Add subject and probability information to the saccades DataFrame
    saccades["subject"] = sub
    saccades["probability"] = proba

    return saccades


def parallel_saccade_detection(df, n_jobs=-1):
    """
    Parallel processing of saccade detection across subjects and probabilities

    Parameters:
    -----------
    df : pandas DataFrame
        The complete dataset
    n_jobs : int
        Number of parallel jobs to run. -1 means using all processors

    Returns:
    --------
    pandas DataFrame
        Combined saccade information for all subjects and probabilities
    """
    # Get unique combinations of subjects and probabilities
    combinations = [
        (sub, prob)
        for sub in df["sub"].unique()
        for prob in df[df["sub"] == sub]["proba"].unique()
    ]

    # Run parallel processing
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_subject_probability)(df, sub, prob)
        for sub, prob in combinations
    )

    # Combine all results
    return pd.concat(results, ignore_index=True)


starTime = time.time()
dirDC = "/envau/work/brainets/oueld.h/contextuaLearning/directionCue/results_voluntaryDirection/"
rawDC = "allRawData.csv"

dirCC = "/envau/work/brainets/oueld.h/contextuaLearning/ColorCue/data"
rawCC = "allRawData.csv"

print("Direction Cue Data Set")
allRawDataDC = pd.read_csv(os.path.join(dirDC, rawDC))
print("Getting the Saccades")
saccadeDC = parallel_saccade_detection(allRawDataDC)
saccadeDC.to_csv(os.path.join(dirDC, "saccades.csv"))


print("Color Cue Data Set")
allRawDataCC = pd.read_csv(os.path.join(dirCC, rawCC))
print("Getting the Saccades")
saccadeCC = parallel_saccade_detection(allRawDataCC)
saccadeCC.to_csv(os.path.join(dirCC, "saccades.csv"))

endTime = time.time()
print("computation time:", endTime - starTime)
