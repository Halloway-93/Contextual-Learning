import pandas as pd
import pwlf
import piecewise_regression as pw
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import filtfilt, butter, sosfiltfilt


# %%
def pwAnalysis(df):
    # Getting the region of interest
    # Test on one subject and one condition and one trial
    # List of slopes
    # starting with 2 breakpoints
    maxBreakPoints = 3
    slopes = [f"alpha{i+1}" for i in range(maxBreakPoints + 1)]
    breakpoints = [f"breakpoint{i+1}" for i in range(maxBreakPoints)]

    allFit = []
    for sub in df["sub"].unique():
        df_sub = df[df["sub"] == sub]
        allFitConditions = []
        for p in df_sub.proba.unique():
            df_sub_p = df_sub[df_sub["proba"] == p]
            # List of slopes for each trial
            allFitTrials = []
            for t in df_sub_p.trial.unique():
                x = df_sub_p[df_sub_p["trial"] == t].time.values
                y = interpolateData(df_sub_p[df_sub_p["trial"] == t].filtPos.values)
                # Fitting the piecewise regression
                pw_fit = pw.Fit(x, y, n_breakpoints=3)
                # Geeting the slopes estimates

                pw_results = pw_fit.get_results()

                pw_estimates = pw_results["estimates"]
                if pw_estimates is None:
                    allFitTrials.append([])
                # storing the alphas for each fitTrial
                else:
                    alphas = []
                    bps = []
                    for s in slopes:
                        if s in pw_estimates.keys():
                            alpha = pw_estimates[s]["estimate"]
                            alphas.append(alpha)
                            print(f"{s}:", alpha)
                    for b in breakpoints:
                        if b in pw_estimates.keys():
                            bp = pw_estimates[b]["estimate"]
                            bps.append(bp)
                            print(f"{b}:", bp)

                    allFitTrials.append((alphas, bps))
            allFitConditions.append(allFitTrials)
        allFit.append(allFitConditions)

    return allFit


def prepare_and_filter_data(eye_position, sampling_freq=1000, cutoff_freq=30):
    """
    Process eye position data with NaN values (from blinks/saccades)
    """
    # First interpolate across NaN values
    valid_indices = ~np.isnan(eye_position)

    # Get all indices
    all_indices = np.arange(len(eye_position))

    # Interpolate only if we have some valid data
    if np.any(valid_indices):
        # Use linear interpolation
        interpolated_data = np.interp(
            all_indices, all_indices[valid_indices], eye_position[valid_indices]
        )

        # Apply butterworth filter
        nyquist = sampling_freq * 0.5
        normalized_cutoff = cutoff_freq / nyquist
        b, a = butter(2, normalized_cutoff, btype="low")

        # Apply filter
        filtered_data = filtfilt(b, a, interpolated_data)

        # Put NaN values back in their original positions
        # This is important if you want to exclude these periods from analysis
        final_data = filtered_data.copy()
        final_data[~valid_indices] = np.nan

        return final_data, interpolated_data
    else:
        return np.full_like(eye_position, np.nan), np.full_like(eye_position, np.nan)


def calculate_velocity(
    position, sampling_freq=1000, velocity_cutoff=20, degToPix=27.28
):
    """
    Calculate velocity from position data, with additional filtering
    """
    # First calculate raw velocity using central difference
    # We do this before filtering to avoid edge effects from the filter
    # velocity = np.zeros_like(position)
    # velocity[1:-1] = (position[2:] - position[:-2]) * (sampling_freq / 2)
    #
    # # Handle edges
    # velocity[0] = (position[1] - position[0]) * sampling_freq
    # velocity[-1] = (position[-1] - position[-2]) * sampling_freq
    velocity = np.gradient(position)
    # Filter velocity separately with lower cutoff
    # nyquist = sampling_freq * 0.5
    # normalized_cutoff = velocity_cutoff / nyquist
    # b, a = signal.butter(2, normalized_cutoff, btype="low")
    #
    # # Filter velocity
    # filtered_velocity = signal.filtfilt(b, a, velocity)

    return velocity * sampling_freq / degToPix


def process_eye_movement(eye_position, sampling_freq=1000, cutoff_freq=20):
    """
    Complete processing pipeline including velocity calculation
    """
    # 1. First handle the NaN values and filter position
    filtered_pos, interpolated_pos = prepare_and_filter_data(
        eye_position, sampling_freq, cutoff_freq  # Position cutoff
    )

    # 2. Calculate velocity from the interpolated position
    # (we use interpolated to avoid NaN issues in velocity calculation)
    velocity = calculate_velocity(
        filtered_pos,
        sampling_freq=sampling_freq,
        velocity_cutoff=20,  # Typically lower cutoff for velocity
    )

    # 3. Put NaN back in velocity where position was NaN
    velocity[np.isnan(eye_position)] = np.nan

    return velocity


def interpolateData(data):

    valid_indices = ~np.isnan(data)

    # Get all indices
    all_indices = np.arange(len(data))

    # Interpolate only if we have some valid data
    if np.any(valid_indices):
        # Use linear interpolation
        interpolated_data = np.interp(
            all_indices, all_indices[valid_indices], data[valid_indices]
        )
        return interpolated_data
    return None


def butter_lowpass(cutoff, fs, order=2):
    """Design Butterworth lowpass filter"""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    sos = butter(order, normal_cutoff, output="sos")
    return sos


def filter_velocity(pos, filPos, fs=1000, degToPix=27.28):
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
    sos = butter_lowpass(cutoff=20, fs=fs)
    vel_filtered = sosfiltfilt(sos, vel)

    vel_filtered[np.isnan(filPos)] = np.nan
    return vel_filtered * fs / degToPix


def calculate_acceleration(vel, fs=1000):
    """Calculate acceleration using Butterworth-filtered derivative"""
    acc = np.gradient(vel)
    # Apply same Butterworth filter to acceleration
    sos = butter_lowpass(cutoff=50, fs=fs)
    acc_filtered = sosfiltfilt(sos, acc)

    return acc_filtered


# %%
csvPath = "/Volumes/work/brainets/oueld.h/contextuaLearning/ColorCue/data/JobLibProcessingCC.csv"
# slopesPath = "/envau/work/brainets/oueld.h/contextuaLearning/directionCue/results_voluntaryDirection/slopes.json"
# allSlopes = pwAnalysis(csvPath)
# Serialize the data to a JSON file
# with open(slopesPath, "w") as file:
#     json.dump(allSlopes, file)
df = pd.read_csv(csvPath)
# %%
df
# %%
example = df[(df["sub"] == 6) & (df["proba"] == 75) & (df["trial"] == 15)]
example = example[(example["time"] >= -180) & (example["time"] <= 500)]

example["interpVelo"] = interpolateData(example.velo.values)
example
# %%
x = example.time.values
y = interpolateData(example.filtVeloFilt.values)
# %%
len(y)
# %%
plt.plot(x, y)
plt.plot(x, example.filtVelo.values, alpha=0.7)
plt.show()
# %%

plt.plot(x, example.xp.values)
plt.plot(x, example.filtPos.values)
plt.show()
# %%
pw_fit = pw.Fit(x, y, n_breakpoints=2)
print(pw_fit.summary())
# %%
pw_results = pw_fit.get_results()
pw_results["estimates"]
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
plt.close()
# %%


my_pwlf = pwlf.PiecewiseLinFit(x, y)
res = my_pwlf.fit(3)
# predict for the determined points
xHat = np.linspace(min(x), max(x), num=10000)
yHat = my_pwlf.predict(xHat)
# %%
plt.figure()
plt.plot(x, y, "o")
plt.plot(xHat, yHat, "-")
plt.show()
# %%
pwAnalysis(example)
# %%

first_nan_after_100_index = example[example.time > 100]["filtPos"].isna().idxmax()
first_nan_after_100_index
# %%
result_df = example.loc[: first_nan_after_100_index - 1]
result_df
# %%
x = result_df.time.values
y = interpolateData(result_df.filtVeloFilt.values)
# %%
len(y)
# %%
plt.plot(x, y)
# plt.plot(x, example.filtVelo.values)
plt.show()
# %%
pw_fit = pw.Fit(x, y, n_breakpoints=2)
print(pw_fit.summary())
# %%
pw_results = pw_fit.get_results()
pw_results["estimates"]
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
plt.close()
# %%
