import numpy as np
from scipy.signal import butter, sosfiltfilt
import pandas as pd
import matplotlib.pyplot as plt


def detect_saccades(
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
        # sos = butter_lowpass(cutoff=30, fs=fs)
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

        # Debug plot
        plt.figure(figsize=(15, 5))
        plt.plot(velocity, label="Filtered Velocity")
        plt.axhline(
            y=velocity_threshold,
            color="r",
            linestyle="--",
            label=f"Threshold ({velocity_threshold} deg/s)",
        )
        plt.plot(candidates * np.max(velocity), "g-", alpha=0.3, label="Detection")
        plt.scatter(starts, velocity[starts], color="green", label="Starts")
        plt.scatter(ends, velocity[ends], color="red", label="Ends")
        plt.legend()
        plt.title("Saccade Detection Debug Plot")
        plt.xlabel("Samples")
        plt.ylabel("Velocity (deg/s)")
        plt.show()

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

        # Plot velocity profile
        plt.figure(figsize=(15, 5))
        plt.plot(euclidVel)
        plt.axhline(
            y=velocity_threshold,
            color="r",
            linestyle="--",
            label=f"Threshold ({velocity_threshold} deg/s)",
        )
        plt.title(f"Velocity Profile - Trial {iTrial}")
        plt.ylabel("Velocity (deg/s)")
        plt.xlabel("Samples")
        plt.legend()
        plt.show()

        # Detect saccades using fixed threshold
        starts, ends = detect_saccade_onset(euclidVel)

        # Process detected saccades
        valid_saccades = 0
        for start, end in zip(starts, ends):
            # Minimum duration check
            duration_ms = (end - start) * sample_window * 1000
            if duration_ms < min_duration_ms:
                print(f"Rejecting saccade: duration {duration_ms:.1f}ms too short")
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
                print(
                    f"Rejecting saccade: acceleration {amplitude:.1f} acceleration too low"
                )
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

        print(f"Found {valid_saccades} valid saccades in trial {iTrial}")

    saccades_df = pd.DataFrame(saccades)
    print(f"\nTotal saccades detected across all trials: {len(saccades_df)}")
    return saccades_df


# %%

filtered_df = pd.read_csv(
    "/Volumes/work/brainets/oueld.h/contextuaLearning/directionCue/results_voluntaryDirection/allRawData.csv"
)
# %%
messages = pd.read_csv(
    "/Volumes/work/brainets/oueld.h/contextuaLearning/directionCue/results_voluntaryDirection/allMessages.csv"
)
# %%
messages
# %%
condFiltered = filtered_df[
    (filtered_df["sub"] == 3)
    # & (filtered_df["proba"] == 0.25)
    & (filtered_df["trial"] == 200)
]
condFiltered
# %%
saccades = detect_saccades(
    condFiltered, mono=True, velocity_threshold=20, min_duration_ms=10, min_acc=1000
)
# %%
saccades["acceleration"]
# %%
starts = saccades["start"]
ends = saccades["end"]
plt.plot(condFiltered.time, condFiltered.xp)

for i in range(len(starts)):
    # plot shaded area between srarts[i] and ends [i]
    plt.fill_between(
        [starts.iloc[i], ends.iloc[i]],
        condFiltered.xp.min(),
        condFiltered.xp.max(),
        color="red",
        alpha=0.3,
    )
plt.show()
# %%
plt.plot(condFiltered.time, condFiltered.filtVelo, label="Velo")
plt.plot(condFiltered.time, condFiltered.filtVeloFilt, label="filtered Velo")
plt.plot(condFiltered.time, condFiltered.velo, alpha=0.5, label="Raw Velo")
for i in range(len(starts)):
    # plot shaded area between srarts[i] and ends [i]
    plt.fill_between(
        [starts.iloc[i], ends.iloc[i]],
        condFiltered.filtVelo.min(),
        condFiltered.filtVelo.max(),
        color="red",
        alpha=0.3,
    )
plt.legend()
plt.show()
# %%
plt.plot(condFiltered.time, condFiltered.xp)
plt.plot(condFiltered.time, condFiltered.filtPos)
plt.show()
# %%
for proba in filtered_df[filtered_df["sub"] == 3]["proba"].unique():
    cond = filtered_df[
        (filtered_df["sub"] == 3)
        & (filtered_df["proba"] == proba)
        # & (filtered_df["time"] >= -200)
        # & (filtered_df["time"] <= 600)
    ]
    saccades = detect_saccades(
        cond, mono=True, velocity_threshold=20, min_duration_ms=5, min_acc=1000
    )
    for t in cond.trial.unique():
        saccTrial = saccades[saccades["trial"] == t]
        starts = saccTrial["start"]
        ends = saccTrial["end"]
        plt.plot(
            cond[cond.trial == t].time,
            cond[cond.trial == t].xp,
            alpha=0.7,
        )
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
