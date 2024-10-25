import numpy as np
from scipy.signal import butter, filtfilt
import pandas as pd
import matplotlib.pyplot as plt


# %%


def detect_saccades(
    data, mono=True, velocity_threshold=20, min_duration_ms=10, min_amplitude=3
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
        b, a = butter(order, normal_cutoff, btype="low", analog=False)
        return b, a

    def calculate_velocity(pos, fs=1000):
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
        vel = np.zeros_like(pos)
        vel[1:-1] = (pos[2:] - pos[:-2]) / (2 * sample_window * deg)

        # Design and apply Butterworth filter
        # Cutoff frequency of 50 Hz is typical for eye movement data
        b, a = butter_lowpass(cutoff=50, fs=fs)
        vel_filtered = filtfilt(b, a, vel)

        return vel_filtered

    def calculate_acceleration(vel, fs=1000):
        """Calculate acceleration using Butterworth-filtered derivative"""
        acc = np.zeros_like(vel)
        acc[1:-1] = (vel[2:] - vel[:-2]) / (2 * sample_window)

        # Apply same Butterworth filter to acceleration
        b, a = butter_lowpass(cutoff=50, fs=fs)
        acc_filtered = filtfilt(b, a, acc)

        return acc_filtered

    def detect_saccade_onset(velocity, acceleration):
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

        # Calculate velocities using Butterworth filter
        xVel = calculate_velocity(xPos)
        yVel = calculate_velocity(yPos)
        euclidVel = np.sqrt(xVel**2 + yVel**2)

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
        starts, ends = detect_saccade_onset(euclidVel, euclidAcc)

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
            mean_acceleration = np.mean(euclidAcc[start:end])

            # Position change during saccade
            x_displacement = xPos[end] - xPos[start]
            y_displacement = yPos[end] - yPos[start]
            amplitude = np.sqrt(x_displacement**2 + y_displacement**2)

            # Only include if amplitude is significant
            if amplitude < min_amplitude:
                print(f"Rejecting saccade: amplitude {amplitude:.1f} pixels too small")
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
                    "amplitude": amplitude,
                    "peak_velocity": peakVelocity,
                    "mean_acceleration": mean_acceleration,
                    "x_displacement": x_displacement,
                    "y_displacement": y_displacement,
                }
            )

        print(f"Found {valid_saccades} valid saccades in trial {iTrial}")

    saccades_df = pd.DataFrame(saccades)
    print(f"\nTotal saccades detected across all trials: {len(saccades_df)}")
    return saccades_df


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
    & (df["proba"] == 0.75)
    & (df["trial"] == 100)
    & (df["time"] >= -200)
    & (df["time"] <= 600)
]
cond
# %%
condFiltered = filtered_df[
    (filtered_df["sub"] == 6)
    & (filtered_df["proba"] == 0.75)
    & (filtered_df["trial"] == 100)
]
condFiltered
# %%
saccades = detect_saccades(
    cond, mono=True, velocity_threshold=20, min_duration_ms=5, min_amplitude=5
)
# %%
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
plt.plot(condFiltered.time, condFiltered.velo, alpha=0.5)
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
        cond = df[
            (df["sub"] == sub)
            & (df["proba"] == proba)
            & (df["time"] >= -200)
            & (df["time"] <= 600)
        ]
        saccades = detect_saccades(
            cond, mono=True, velocity_threshold=20, min_duration_ms=3, min_amplitude=5
        )
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
