import numpy as np
from scipy.signal import butter, sosfiltfilt
import pandas as pd
from joblib import Parallel, delayed
import time
import os


def detect_saccades_no_plot(
    data, mono=True, velocity_threshold=20, min_duration_ms=10, min_acc=1000
):
    """
    Detect saccades using Butterworth-filtered velocity and fixed threshold

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
        sos = butter(order, normal_cutoff, output="sos")
        return sos

    def filter_pos(pos, fs=1000, cutoff=30):

        # First interpolate across NaN values
        valid_indices = ~np.isnan(pos)

        # Get all indices
        all_indices = np.arange(len(pos))

        # Interpolate only if we have some valid data
        if np.any(valid_indices):
            # Use linear interpolation
            interpolated_data = np.interp(
                all_indices, all_indices[valid_indices], pos[valid_indices]
            )

            # Apply butterworth filter
            nyquist = fs * 0.5
            normalized_cutoff = cutoff / nyquist
            sos = butter(2, normalized_cutoff, btype="low", output="sos")

            # Apply filter
            filtered_data = sosfiltfilt(sos, interpolated_data)

            # Put NaN values back in their original positions
            # This is important if you want to exclude these periods from analysis
            final_data = filtered_data.copy()
            final_data[~valid_indices] = np.nan

            return final_data
        else:
            return np.full_like(pos, np.nan)

    def calculate_velocity(pos):
        """
        Calculate velocity using central difference and Butterworth filter

        Parameters:
        -----------
        pos : array
            Position data
        fs : float
            Sampling frequency in Hz
        """
        # Calculate raw velocity using central difference
        vel = np.gradient(pos)

        # Design and apply Butterworth filter
        # Cutoff frequency of 50 Hz is typical for eye movement data
        # sos = butter_lowpass(cutoff=30, fs=fs)
        # vel_filtered = sosfiltfilt(sos, vel)

        return vel / (sample_window * deg)

    def calculate_acceleration(vel, fs=1000):
        """Calculate acceleration using Butterworth-filtered derivative"""
        acc = np.gradient(vel)
        # Apply same Butterworth filter to acceleration
        sos = butter_lowpass(cutoff=30, fs=fs)
        # acc_filtered = sosfiltfilt(sos, acc)
        return acc * fs
        # return acc_filtered

    def detect_saccade_onset(velocity):
        """
        Detect saccade onset using fixed velocity threshold
        Returns indices where saccades likely begin and end
        """
        # Find potential saccade points using fixed threshold
        candidates = velocity > velocity_threshold

        # Group consecutive True values
        changes = np.diff(candidates.astype(int))
        starts = np.where(changes == 1)[0] + 1
        ends = np.where(changes == -1)[0] + 1

        # Ensure we have paired starts and ends
        if len(starts) == 0 or len(ends) == 0:
            print("No saccades detected in this trial")
            return [], []
        if starts[0] > ends[0]:
            ends = ends[1:]
        if len(starts) > len(ends):
            starts = starts[:-1]

        print(f"Found {len(starts)} potential saccades")
        return starts, ends

    for iTrial in trials:
        print(f"\nProcessing trial {iTrial}")

        # Get position data
        if mono:
            xPos = data[data.trial == iTrial].xp.values
            yPos = data[data.trial == iTrial].yp.values
        else:
            xPos = data[data.trial == iTrial].xpr.values
            yPos = data[data.trial == iTrial].ypr.values
        xPosf = filter_pos(xPos)
        yPosf = filter_pos(yPos)
        # Calculate velocities using Butterworth filter
        xVel = calculate_velocity(xPosf)
        yVel = calculate_velocity(yPosf)

        euclidVel = np.where(
            np.isnan(xVel**2 + yVel**2), np.nan, np.sqrt(xVel**2 + yVel**2)
        )
        # Calculate accelerations
        xAcc = calculate_acceleration(xVel)
        yAcc = calculate_acceleration(yVel)
        euclidAcc = np.sqrt(xAcc**2 + yAcc**2)

        # Detect saccades using fixed threshold
        starts, ends = detect_saccade_onset(euclidVel)

        # Process detected saccades
        valid_saccades = 0
        for start, end in zip(starts, ends):
            # Minimum duration check
            duration_ms = (end - start) * sample_window * 1000
            if duration_ms < min_duration_ms:
                continue

            # Calculate saccade properties
            peakVelocity = np.max(euclidVel[start:end])
            acceleration = np.mean(euclidAcc[start:end])

            # Position change during saccade
            x_displacement = xPos[end] - xPos[start]
            y_displacement = yPos[end] - yPos[start]
            amplitude = np.sqrt(x_displacement**2 + y_displacement**2)

            # acceleration=np.sqrt((xAcc[end]-xAcc[start])**2 + (yAcc[end]-yAcc[start])**2)
            # Only include if acceleration is significant
            if acceleration < min_acc:
                continue

            valid_saccades += 1
            start_time = data[data.trial == iTrial].time.values[start]
            end_time = data[data.trial == iTrial].time.values[end]

            saccades.append(
                {
                    "trial": iTrial,
                    "start": start_time,
                    "end": end_time,
                    "duration": end_time - start_time,
                    "acceleration": acceleration,
                    "peak_velocity": peakVelocity,
                    "amplitude": amplitude,
                    "x_displacement": x_displacement,
                    "y_displacement": y_displacement,
                }
            )

    saccades_df = pd.DataFrame(saccades)
    return saccades_df


def process_subject_probability(df, sub, proba):
    """Process a single subject-probability combination"""
    cond = df[
        (df["sub"] == sub)
        & (df["proba"] == proba)
        # & (df["time"] >= -200)
        # & (df["time"] <= 600)
    ]

    saccades = detect_saccades_no_plot(
        cond, mono=True, velocity_threshold=20, min_duration_ms=3, min_acc=200
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
# meso
# dirDC = "/scratch/houeld/contextuaLearning/directionCue/results_voluntaryDirection/"
rawDC = "allRawData.csv"

dirCC = "/envau/work/brainets/oueld.h/contextuaLearning/ColorCue/data"
# meso
# dirCC = "/scratch/houeld/contextuaLearning/ColorCue/data"
rawCC = "allRawData.csv"

print("Direction Cue Data Set")
allRawDataDC = pd.read_csv(os.path.join(dirDC, rawDC))
print("Getting the Saccades")
saccadeDC = parallel_saccade_detection(allRawDataDC)
saccadeDC.to_csv(os.path.join(dirDC, "saccades.csv"), index=False)


print("Color Cue Data Set")
allRawDataCC = pd.read_csv(os.path.join(dirCC, rawCC))
print("Getting the Saccades")
saccadeCC = parallel_saccade_detection(allRawDataCC)
saccadeCC.to_csv(os.path.join(dirCC, "saccades.csv"), index=False)

endTime = time.time()
print("computation time:", endTime - starTime)
