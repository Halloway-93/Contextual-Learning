import pandas as pd
import piecewise_regression as pw
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import filtfilt, butter
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# %%
def pwAnalysis(csvPath):
    df = pd.read_csv(csvPath)
    # Getting the region of interest
    fixOff = -200
    latency = 120
    df = df[(df.time >= fixOff + 20) & (df.time <= latency)]
    # Test on one subject and one condition and one trial
    # List of slopes
    # starting with 2 breakpoints
    maxBreakPoints = 2
    slopes = [f"alpha{i+1}" for i in range(maxBreakPoints + 1)]

    lastTrial = df[(df["trial"] > 150)]
    allFit = []
    for sub in lastTrial["sub"].unique():
        df_sub = lastTrial[lastTrial["sub"] == sub]
        allFitConditions = []
        for p in df_sub.proba.unique():
            df_sub_p = df_sub[df_sub["proba"] == p]
            # List of slopes for each trial
            allFitTrials = []
            for t in df_sub_p.trial.unique():
                x = df_sub_p[df_sub_p["trial"] == t].time.values
                y = df_sub_p[df_sub_p["trial"] == t].xp.values
                # Fitting the piecewise regression
                pw_fit = pw.Fit(x, y, n_breakpoints=2)
                # Geeting the slopes estimates

                pw_results = pw_fit.get_results()

                pw_estimates = pw_results["estimates"]
                if pw_estimates is None:
                    allFitTrials.append([])
                # storing the alphas for each fitTrial
                else:
                    alphas = []

                    for s in slopes:
                        if s in pw_estimates.keys():
                            alpha = pw_estimates[s]["estimate"]
                            alphas.append(alpha)
                            print(f"{s}:", alpha)
                    allFitTrials.append(alphas)
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


# %%
csvPath = "/Volumes/work/brainets/oueld.h/contextuaLearning/directionCue/results_voluntaryDirection/JobLibProcessing.csv"
# slopesPath = "/envau/work/brainets/oueld.h/contextuaLearning/directionCue/results_voluntaryDirection/slopes.json"
# allSlopes = pwAnalysis(csvPath)
# Serialize the data to a JSON file
# with open(slopesPath, "w") as file:
#     json.dump(allSlopes, file)
df = pd.read_csv(csvPath)
# %%
df
# %%
example = df[(df["sub"] == 1) & (df["proba"] == 0.25) & (df["trial"] == 200)]
example = example[(example["time"] <= 500)]

example["interpVelo"] = interpolateData(example.velo.values)
example["filtVelo2"] = interpolateData(process_eye_movement(example.filtPos.values))
example
# %%
x = example.time.values
y = example.filtVelo2.values
# %%
len(y)
# %%
plt.plot(x, y)
# plt.plot(x, example.filtVelo.values)
plt.show()
# %%
pw_fit = pw.Fit(x, y, n_breakpoints=3)
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
plt.close()
# %%


def piecewise_linear_regression(time, velocity):
    """
    Perform piecewise linear regression on eye movement velocity data.

    Parameters:
    -----------
    time : numpy array
        Time values for the velocity data
    velocity : numpy array
        Velocity data

    Returns:
    --------
    piecewise_model : dict
        Dictionary containing the piecewise linear regression model parameters
    """
    piecewise_model = {"breakpoints": [], "slopes": [], "intercepts": []}

    # Start with a single linear regression
    model = LinearRegression()
    model.fit(time.reshape(-1, 1), velocity)
    piecewise_model["breakpoints"] = []
    piecewise_model["slopes"] = [model.coef_[0]]
    piecewise_model["intercepts"] = [model.intercept_]

    # Iteratively add breakpoints
    prev_mse = mean_squared_error(velocity, model.predict(time.reshape(-1, 1)))
    while len(piecewise_model["breakpoints"]) < 3:
        best_breakpoint = None
        best_mse = prev_mse

        for i in range(1, len(time) - 1):
            # Try inserting a breakpoint at each sample
            left_model = LinearRegression()
            right_model = LinearRegression()

            left_mask = time < time[i]
            right_mask = time >= time[i]

            left_model.fit(time[left_mask].reshape(-1, 1), velocity[left_mask])
            right_model.fit(time[right_mask].reshape(-1, 1), velocity[right_mask])

            left_predict = left_model.predict(time[left_mask].reshape(-1, 1))
            right_predict = right_model.predict(time[right_mask].reshape(-1, 1))
            predict = np.concatenate([left_predict, right_predict])

            new_mse = mean_squared_error(velocity, predict)
            if new_mse < best_mse:
                best_breakpoint = time[i]
                best_mse = new_mse

        if best_breakpoint is not None:
            piecewise_model["breakpoints"].append(best_breakpoint)

            left_model = LinearRegression()
            right_model = LinearRegression()

            left_mask = time < best_breakpoint
            right_mask = time >= best_breakpoint

            left_model.fit(time[left_mask].reshape(-1, 1), velocity[left_mask])
            right_model.fit(time[right_mask].reshape(-1, 1), velocity[right_mask])

            piecewise_model["slopes"].append(left_model.coef_[0])
            piecewise_model["slopes"].append(right_model.coef_[0])
            piecewise_model["intercepts"].append(left_model.intercept_)
            piecewise_model["intercepts"].append(right_model.intercept_)
        else:
            break

    return piecewise_model


def piecewise_linear_regression_with_saccades(time, velocity):
    """
    Perform piecewise linear regression on eye movement velocity data,
    including saccades.

    Parameters:
    -----------
    time : numpy array
        Time values for the velocity data
    velocity : numpy array
        Velocity data

    Returns:
    --------
    piecewise_model : dict
        Dictionary containing the piecewise linear regression model parameters
    """
    piecewise_model = {"breakpoints": [], "slopes": [], "intercepts": []}

    # Start with a single linear regression
    model = LinearRegression()
    model.fit(time.reshape(-1, 1), velocity)
    piecewise_model["breakpoints"] = []
    piecewise_model["slopes"] = [model.coef_[0]]
    piecewise_model["intercepts"] = [model.intercept_]

    # Iteratively add breakpoints
    prev_mse = mean_squared_error(velocity, model.predict(time.reshape(-1, 1)))
    while len(piecewise_model["breakpoints"]) < 3:
        best_breakpoint = None
        best_mse = prev_mse

        for i in range(1, len(time) - 1):
            # Try inserting a breakpoint at each sample
            left_model = LinearRegression()
            right_model = LinearRegression()

            left_mask = time < time[i]
            right_mask = time >= time[i]

            left_model.fit(time[left_mask].reshape(-1, 1), velocity[left_mask])
            right_model.fit(time[right_mask].reshape(-1, 1), velocity[right_mask])

            left_predict = left_model.predict(time[left_mask].reshape(-1, 1))
            right_predict = right_model.predict(time[right_mask].reshape(-1, 1))
            predict = np.concatenate([left_predict, right_predict])

            new_mse = mean_squared_error(velocity, predict)
            if new_mse < best_mse:
                best_breakpoint = time[i]
                best_mse = new_mse

        if best_breakpoint is not None:
            piecewise_model["breakpoints"].append(best_breakpoint)

            left_model = LinearRegression()
            right_model = LinearRegression()

            left_mask = time < best_breakpoint
            right_mask = time >= best_breakpoint

            left_model.fit(time[left_mask].reshape(-1, 1), velocity[left_mask])
            right_model.fit(time[right_mask].reshape(-1, 1), velocity[right_mask])

            piecewise_model["slopes"].append(left_model.coef_[0])
            piecewise_model["slopes"].append(right_model.coef_[0])
            piecewise_model["intercepts"].append(left_model.intercept_)
            piecewise_model["intercepts"].append(right_model.intercept_)
        else:
            break

    return piecewise_model


def plot_piecewise_fit(time, velocity, piecewise_model):
    """
    Plot the piecewise linear regression fit on the original velocity data.

    Parameters:
    -----------
    time : numpy array
        Time values for the velocity data
    velocity : numpy array
        Velocity data
    piecewise_model : dict
        Dictionary containing the piecewise linear regression model parameters
    """
    plt.figure(figsize=(12, 6))
    plt.plot(time, velocity, label="Original Data")

    # Plot the piecewise linear fit
    fitted_velocity = []
    for i in range(len(piecewise_model["breakpoints"]) + 1):
        if i == 0:
            start = 0
            end = piecewise_model["breakpoints"][i]
        elif i == len(piecewise_model["breakpoints"]):
            start = piecewise_model["breakpoints"][-1]
            end = time[-1]
        else:
            start = piecewise_model["breakpoints"][i - 1]
            end = piecewise_model["breakpoints"][i]

        mask = (time >= start) & (time < end)
        segment_time = time[mask]
        segment_velocity = (
            piecewise_model["slopes"][i] * segment_time
            + piecewise_model["intercepts"][i]
        )
        fitted_velocity.extend(segment_velocity)

    plt.plot(time, fitted_velocity, label="Piecewise Linear Fit")
    plt.xlabel("Time")
    plt.ylabel("Velocity")
    plt.title("Piecewise Linear Regression")
    plt.legend()
    plt.show()


# %%
x = example.time.values
y = example.rawVelo.values
# %%
len(x)
# %%
model = piecewise_linear_regression(x, y)
# %%
plot_piecewise_fit(x, y, model)
