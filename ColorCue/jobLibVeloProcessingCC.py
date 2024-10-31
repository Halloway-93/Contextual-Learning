import numpy as np
import io
import os
import re
from datetime import datetime
import pandas as pd
from scipy.signal import butter, filtfilt
from joblib import Parallel, delayed


# Example velocity threshold for ASP detection
def detect_asp_onset(velocity, threshold=2.0):  # deg/s
    """
    Detect ASP onset using a conservative velocity threshold
    Returns the index of ASP onset
    """
    # Look for sustained velocity above threshold
    sustained_samples = 10  # Number of samples to confirm it's not noise
    above_threshold = np.where(velocity > threshold)[0]

    for i in range(len(above_threshold) - sustained_samples):
        if np.all(velocity[above_threshold[i : i + sustained_samples]] > threshold):
            return above_threshold[i]

    return None


def detect_saccades(
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
        b, a = butter_lowpass(cutoff=30, fs=fs)
        vel_filtered = filtfilt(b, a, vel)
        return vel_filtered

    def calculate_acceleration(vel, fs=1000):
        """Calculate acceleration using Butterworth-filtered derivative"""
        acc = np.zeros_like(vel)
        acc[1:-1] = (vel[2:] - vel[:-2]) / (2 * sample_window)
        b, a = butter_lowpass(cutoff=30, fs=fs)
        # acc_filtered = filtfilt(b, a, acc)
        return acc

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


def process_events(rows, blocks, colnames):
    # If no data, create empty dataframe w/ all cols and types
    if len(rows) == 0:
        rows = ["", ""]
        blocks = []
    # Parse data, dropping useless first column
    if len(rows) == 1:
        list(rows).append("")
    # first col is event type, which we drop later
    colnames = ["type"] + colnames
    get_coltypes(colnames)
    df = pd.read_csv(
        io.StringIO("\n".join(rows)),
        delimiter=r"\s+",
        header=None,
        names=colnames,
        na_values=".",
        index_col=False,
    )
    df = df.iloc[:, 1:]  # drop the first column
    # Move eye column to end & make factor, append block numbers to beginning of data frame
    if "eye" in colnames:
        df = df.iloc[:, [1] + list(range(2, df.shape[1])) + [0]]
        df["eye"] = pd.Categorical(df["eye"], categories=["L", "R"], ordered=False)
    df.insert(loc=0, column="trial", value=blocks)
    return df


def process_saccades(saccades, blocks, info):
    sacc_df = process_events(saccades, blocks, get_sacc_header(info))
    # Set amplitudes for any saccades missing start/end coords to NAs because they're wonky
    ampl_cols = [col for col in sacc_df.columns if re.search(r"ampl\d*$", col)]
    partial = sacc_df["sxp"].isna() | sacc_df["exp"].isna()
    if any(partial):
        sacc_df.loc[partial, ampl_cols] = pd.NA
    return sacc_df


def process_fixations(fixations, blocks, info):
    return process_events(fixations, blocks, get_fix_header(info))


def process_blinks(blinks, blocks):
    return process_events(blinks, blocks, ["eye", "stime", "etime", "dur"])


def process_messages(msgs, blocks):
    # Process messages from tracker
    msg_mat = [msg.split(" ", 1) for msg in msgs]
    msg_mat = [[msg[0][4:], msg[1].rstrip()] for msg in msg_mat]
    msg_df = pd.DataFrame(msg_mat, columns=np.array(["time", "text"]))
    msg_df["time"] = pd.to_numeric(msg_df["time"])

    # Append trial numbers to beginning of data frame
    msg_df.insert(0, "trial", blocks)

    return msg_df


def process_input(input_data, blocks):
    return process_events(input_data, blocks, ["time", "value"])


def process_buttons(button, blocks):
    return process_events(button, blocks, ["time", "button", "state"])


def from_header(header, field):
    pattern = r"\*\* {}\s*: (.*)".format(re.escape(field))
    matches = [re.findall(pattern, line) for line in header]
    matches = [match for match in matches if match]
    return matches[0][0]


def get_resolution(nonsample):
    res = [None, None]
    for pattern in ["DISPLAY_COORDS", "GAZE_COORDS", "RESOLUTION"]:
        display_xy = [s for s in nonsample if pattern in s]
        if len(display_xy) == 0:
            continue
        display_xy = re.sub(f".* {pattern}\\D+(.*)", "\\1", display_xy[0])
        try:
            display_xy = [int(float(s)) for s in display_xy.split()]
        except ValueError:
            continue
        res = [display_xy[2] - display_xy[0] + 1, display_xy[3] - display_xy[1] + 1]
        break
    return res


def get_mount(mount_str):
    # Older EyeLink 1000s may be missing "R" in table mount names, we add one if needed
    if re.search("TABLE$", mount_str):
        mount_str = mount_str + "R"

    mounts = {
        "MTABLER": "Desktop / Monocular / Head Stabilized",
        "BTABLER": "Desktop / Binocular / Head Stabilized",
        "RTABLER": "Desktop / Monocular / Remote",
        "RBTABLER": "Desktop / Binocular / Remote",
        "AMTABLER": "Arm Mount / Monocular / Head Stabilized",
        "ARTABLER": "Arm Mount / Monocular / Remote",
        "TOWER": "Tower Mount / Monocular / Head Stabilized",
        "BTOWER": "Tower Mount / Binocular / Head Stabilized",
        "MPRIM": "Primate Mount / Monocular / Head Stabilized",
        "BPRIM": "Primate Mount / Binocular / Head Stabilized",
        "MLRR": "Long-Range Mount / Monocular / Head Stabilized",
        "BLRR": "Long-Range Mount / Binocular / Head Stabilized",
    }

    return mounts[mount_str] if mount_str in mounts else None


def get_raw_header(info):
    eyev = ["xp", "yp", "ps"]

    if not info["mono"]:
        eyev = [f"{e}{s}" for s in ["l", "r"] for e in eyev]

    if info["velocity"]:
        if info["mono"]:
            eyev += ["xv", "yv"]
        else:
            eyev += [f"{e}{s}" for s in ["vl", "vr"] for e in ["x", "y"]]

    if info["resolution"]:
        eyev += ["xr", "yr"]

    if info["input"]:
        eyev += ["input"]

    if info["buttons"]:
        eyev += ["buttons"]

    if info["tracking"]:
        eyev += ["cr.info"]

    if info["htarg"]:
        eyev += ["tx", "ty", "td", "remote.info"]

    return ["time"] + eyev


def get_event_header(info, xy_cols):
    base = ["eye", "stime", "etime", "dur"]

    if info["event.dtype"] == "HREF":
        xy_cols = [f"href.{xy}" for xy in xy_cols] + xy_cols

    if info["resolution"]:
        xy_cols += ["xr", "yr"]

    return base + xy_cols


def get_sacc_header(info):
    return get_event_header(info, ["sxp", "syp", "exp", "eyp", "ampl", "pv"])


def get_fix_header(info):
    return get_event_header(info, ["axp", "ayp", "aps"])


def get_model(header):
    version_str = from_header(header, "VERSION")
    version_str2 = [x for x in header if re.search("\\*\\* EYELINK II", x)]
    if version_str is None:
        model = "Unknown"
        ver_num = "Unknown"
    elif version_str != "EYELINK II 1":
        model = "EyeLink I"
        ver_num = re.search(r"(\d+.\d+)", version_str).group(1)
    else:
        ver_num = re.search(r"v(\d+.\d+)", version_str2[0]).group(1)
        model = (
            "EyeLink II"
            if float(ver_num) < 2.4
            else (
                "EyeLink 1000"
                if float(ver_num) < 5
                else (
                    "EyeLink 1000 Plus"
                    if float(ver_num) < 6
                    else "EyeLink Portable Duo"
                )
            )
        )
    return [model, ver_num]


def get_coltypes(colnames, float_time=True):
    chr_cols = ["type", "eye", "cr.info", "remote.info"]
    int_cols = ["button", "state", "value"]
    time_cols = ["time", "stime", "etime", "dur"]
    if not float_time:
        int_cols += time_cols

    coltypes = [
        "str" if col in chr_cols else "int64" if col in int_cols else "float64"
        for col in colnames
    ]

    return coltypes


def get_htarg_regex(binocular):
    htarg_errs = "MANCFTBLRTBLRTBLR" if binocular else "MANCFTBLRTBLR"
    htarg_errs = list(htarg_errs)
    htarg_regex = "(" + "|".join(htarg_errs + ["\\."]) + ")"

    return htarg_regex


def is_float(string):
    return bool(re.search("\\.", string))


def get_info(nonsample, firstcol):
    header = [f for f in nonsample if f.startswith("**")]
    info = {}
    hh = from_header(header, "DATE")
    # Get date/time of recording from file
    datetime.strptime(hh, "%a %b %d %H:%M:%S %Y")
    # Get tracker model/version info
    version_info = get_model(header)
    info["model"] = version_info[0]
    info["version"] = version_info[1]

    # Get tracker mount info
    elclcfg = [line for line in nonsample if "ELCLCFG" in line]
    if len(elclcfg) > 0:
        info["mount"] = get_mount(re.findall(r"ELCLCFG\s+(.*)", elclcfg[0])[0])

    # Get display size from file
    screen_res = get_resolution(nonsample)
    info["screen.x"] = screen_res[0]
    info["screen.y"] = screen_res[1]

    # Get pupil size data type (area or diameter)
    pupil_config = [line for i, line in enumerate(nonsample) if firstcol[i] == "PUPIL"]
    if len(pupil_config) > 0:
        info["pupil.dtype"] = pupil_config[-1].split()[1]

    # Find the samples and events config lines in the non-sample input, get data types
    events_config = [
        line for i, line in enumerate(nonsample) if firstcol[i] == "EVENTS"
    ]
    samples_config = [
        line for i, line in enumerate(nonsample) if firstcol[i] == "SAMPLES"
    ]

    # Find the samples and events config lines in the non-sample input, get data types
    events_config = [
        line for i, line in enumerate(nonsample) if firstcol[i] == "EVENTS"
    ]
    samples_config = [
        line for i, line in enumerate(nonsample) if firstcol[i] == "SAMPLES"
    ]
    if len(events_config) > 0:
        info["event.dtype"] = events_config[-1].split()[1]
    if len(samples_config) > 0:
        info["sample.dtype"] = samples_config[-1].split()[1]

    # Get last config line in file (preferring sample config) and extract remaining info
    config = events_config + samples_config[-1:]
    config = config[-1] if len(config) > 0 else ""
    if config:
        info["sample.rate"] = (
            float(re.findall(r"RATE\s+([0-9]+\.[0-9]+)", config)[0])
            if "RATE" in config
            else None
        )
        info["tracking"] = "\tTRACKING" in config
        info["cr"] = "\tCR" in config
        info["filter.level"] = (
            int(re.findall(r"FILTER\s+([0-9]+)", config)[0])
            if "FILTER" in config
            else None
        )
        info["velocity"] = "\tVEL" in config
        info["resolution"] = "\tRES" in config
        info["htarg"] = "\tHTARG" in config
        info["input"] = "\tINPUT" in config
        info["buttons"] = "\tBUTTONS" in config
        info["left"] = "\tLEFT" in config
        info["right"] = "\tRIGHT" in config
        info["mono"] = not (info["right"] & info["left"])

    return info


def get_raw(raw, blocks, info):
    if len(raw) == 0:
        # If no sample data in file, create empty raw DataFrame w/ all applicable columns
        raw = ["", ""]
        blocks = pd.Series([], dtype=int)
        colnames = get_raw_header(info)
        get_coltypes(colnames, float_time=False)
    else:
        # Determine if timestamps stored as floats (edf2asc option -ftime, useful for 2000 Hz)
        float_time = is_float(re.split(r"\s+", raw[0])[0])
        # Generate column names and types based in info in header
        colnames = get_raw_header(info)
        get_coltypes(colnames, float_time)
        # Discard any rows with too many or too few columns (usually rows where eye is missing)
        row_length = [len(re.split(r"\t", r)) for r in raw]
        med_length = np.median(row_length)
        raw = [r for r, l in zip(raw, row_length) if l == med_length]  # noqa: E741
        blocks = blocks[row_length == med_length]
    # Process raw sample data using pandas
    if len(raw) == 1:
        raw.append("")

    raw_df = pd.read_csv(
        io.StringIO("".join(raw)),
        sep="\t",
        header=None,
        names=colnames,
        na_values=np.nan,
        low_memory=False,
    )

    if info["tracking"] and not info["cr"]:
        # Drop CR column when not actually used
        raw_df = raw_df.drop(columns=["cr.info"])
    # Append block numbers to beginning of DataFrame
    raw_df.insert(0, "trial", blocks)
    # Replace missing pupil data (zeros) with NaNs
    if "X1" not in raw_df.columns:
        if info["mono"]:
            raw_df.loc[raw_df["ps"] == 0, "ps"] = np.nan
        else:
            raw_df.loc[raw_df["psl"] == 0, "psl"] = np.nan
            raw_df.loc[raw_df["psr"] == 0, "psr"] = np.nan
    return raw_df


def read_asc(fname, samples=True, events=True, parse_all=False):
    with open(fname, "r", encoding="ISO-8859-1", errors="ignore") as f:
        inp = f.readlines()

    # Convert to ASCII
    inp = [line.encode("ascii", "ignore").decode() for line in inp]

    # Get strings prior to first tab for each line for faster string matching
    inp_first = [re.split(r"\s", s)[0] for s in inp]

    starts = [i for i, x in enumerate(inp_first) if x == "START"]
    if not starts:
        raise ValueError("No samples or events found in .asc file.")

    # Read metadata from file before processing
    is_raw = [bool(re.match("^[0-9]", line)) for line in inp_first]

    info = get_info(
        [line for line, raw in zip(inp, is_raw) if not raw],
        [first for first, raw in zip(inp_first, is_raw) if not raw],
    )

    # Do some extra processing/sanitizing if there's HTARG info in the file
    # if info["htarg"]:
    #     inp, info = handle_htarg(inp, info, is_raw)  # noqa: F821

    # Find blocks and mark lines between block ENDs and next block STARTs
    dividers = starts + [len(inp)]
    block = np.cumsum([x == "START" for x in inp_first])
    block = block.astype(float)

    for i in range(1, len(dividers)):
        start = dividers[i - 1]
        end = dividers[i]
        endline = [j for j, x in enumerate(inp_first[start:end]) if x == "END"]
        if endline and endline[-1] < end - start:
            block[endline[0] + start : end] += 0.5

    # Unless parsing all input, drop any lines not within a block
    block[: dividers[0] + 1] += 0.5
    if not parse_all:
        in_block = np.floor(block) == block
        inp = [line for line, block_match in zip(inp, in_block) if block_match]
        inp_first = [
            first for first, block_match in zip(inp_first, in_block) if block_match
        ]
        is_raw = [raw for raw, block_match in zip(is_raw, in_block) if block_match]
        block = block[in_block]

    block = np.array(block)

    # Initialize dictionary of data output and process different data types
    out = {}
    if samples:
        out["raw"] = get_raw(
            [line for line, raw in zip(inp, is_raw) if raw], block[is_raw], info
        )
    if events:
        is_sacc = np.array(inp_first) == "ESACC"
        out["sacc"] = process_saccades(
            np.array(inp)[is_sacc], np.array(block)[is_sacc], info
        )

        is_fix = np.array(inp_first) == "EFIX"
        out["fix"] = process_fixations(
            np.array(inp)[is_fix], np.array(block)[is_fix], info
        )

        is_blink = np.array(inp_first) == "EBLINK"
        out["blinks"] = process_blinks(
            np.array(inp)[is_blink], np.array(block)[is_blink]
        )

        is_msg = np.array(inp_first) == "MSG"
        out["msg"] = process_messages(np.array(inp)[is_msg], np.array(block)[is_msg])

        is_input = np.array(inp_first) == "INPUT"
        out["input"] = process_input(np.array(inp)[is_input], np.array(block)[is_input])

        is_button = np.array(inp_first) == "BUTTON"
        out["button"] = process_buttons(
            np.array(inp)[is_button], np.array(block)[is_button]
        )

    # needed for parsing, but otherwise redundant with CR
    info["tracking"] = None

    out["info"] = info

    return out


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

        return final_data
    else:
        return np.full_like(eye_position, np.nan)


def calculate_velocity(position, sampling_freq=1000, degToPix=27.28):
    """
    Calculate velocity from position data, with additional filtering
    """
    velocity = np.gradient(position)

    return velocity * sampling_freq / degToPix


def filter_velocity(velocity, sampling_freq=1000, velocity_cutoff=20):
    """
    Calculate velocity from position data, with additional filtering
    """
    valid_indices = ~np.isnan(velocity)

    # Get all indices
    all_indices = np.arange(len(velocity))

    # Interpolate only if we have some valid data
    if np.any(valid_indices):
        # Use linear interpolation
        interpolated_data = np.interp(
            all_indices, all_indices[valid_indices], velocity[valid_indices]
        )
        # Filter velocity separately with lower cutoff
        nyquist = sampling_freq * 0.5
        normalized_cutoff = velocity_cutoff / nyquist
        b, a = butter(2, normalized_cutoff, btype="low")
        #
        # # Filter velocity
        filtered_velocity = filtfilt(b, a, interpolated_data)

        # Put NaN values back in their original positions
        # This is important if you want to exclude these periods from analysis
        final_data = filtered_velocity.copy()
        final_data[~valid_indices] = np.nan
        return final_data
    else:
        np.full_like(velocity, np.nan)


def process_eye_movement(eye_position, sampling_freq=1000, cutoff_freq=30):
    """
    Complete processing pipeline including velocity calculation
    """
    # 1. First handle the NaN values and filter position
    filtered_pos = prepare_and_filter_data(
        eye_position, sampling_freq, cutoff_freq  # Position cutoff
    )

    # 2. Calculate velocity from the filtered position as n.gradient handle the NaN.
    # (we use interpolated to avoid NaN issues in velocity calculation)
    velocity = calculate_velocity(
        filtered_pos, sampling_freq=sampling_freq, degToPix=27.28
    )
    filtered_velocity = filter_velocity(
        velocity, sampling_freq=1000, velocity_cutoff=20
    )

    return pd.DataFrame(
        dict(
            {
                "filtPos": filtered_pos,
                "filtVelo": velocity,
                "filtVeloFilt": filtered_velocity,
            }
        )
    )


def process_single_condition(sub, p, df_condition, fixOff=-200, endOftrial=600):
    """
    Process data for a single subject and probability condition
    """
    condition_data = df_condition.copy()

    # Detect saccades for this condition
    sacc = detect_saccades(
        condition_data,
        mono=True,
        velocity_threshold=20,
        min_duration_ms=3,
        min_amplitude=5,
    )

    # Process saccades if any were detected
    if not sacc.empty:
        for t in sacc.trial.unique():
            start = sacc[sacc.trial == t]["start"].values
            end = sacc[sacc.trial == t]["end"].values
            for i in range(len(start)):
                condition_data.loc[
                    (condition_data.trial == t)
                    & (condition_data.time >= start[i])
                    & (condition_data.time <= end[i]),
                    "xp",
                ] = np.nan

    # Process each trial in parallel
    def process_trial(t):
        trial = condition_data[condition_data.trial == t]
        filtered_trial = process_eye_movement(trial.xp)
        filtered_trial["time"] = trial["time"].values

        if not sacc.empty and t in sacc.trial.unique():
            start = sacc[sacc.trial == t]["start"].values
            end = sacc[sacc.trial == t]["end"].values
            for i in range(len(start)):
                filtered_trial.loc[
                    (filtered_trial.time >= start[i] - 25)
                    & (filtered_trial.time <= end[i] + 25),
                    ["filtPos", "filtVelo", "filtVeloFilt"],
                ] = np.nan

        filtered_trial.drop(columns=["time"], inplace=True)

        # Calculate velocity for the trial
        trial_velo = np.gradient(trial["xp"].values, 1) * 1000 / 27.28

        return {"filtered_trial": filtered_trial, "trial": t, "velocity": trial_velo}

    # Process all trials in parallel
    trial_results = Parallel(n_jobs=-1)(
        delayed(process_trial)(t) for t in condition_data.trial.unique()
    )

    return sub, p, trial_results


def processAllRawData(path, fileName, newFileName, fixOff=-200, endOftrial=600):
    """
    Parallelized version of the raw data processing function
    """
    print("Reading data from ", os.path.join(path, fileName))
    df = pd.read_csv(os.path.join(path, fileName))

    # Getting the region of interest
    df = df[(df.time >= fixOff) & (df.time <= endOftrial)]
    if "cr.info" in df.columns:
        df.drop(columns=["cr.info"], inplace=True)

    # Prepare conditions for parallel processing
    conditions = [
        (sub, p, df[(df["sub"] == sub) & (df["proba"] == p)])
        for sub in df["sub"].unique()
        for p in df[df["sub"] == sub].proba.unique()
    ]

    # Process all conditions in parallel
    results = Parallel(n_jobs=-1)(
        delayed(process_single_condition)(sub, p, condition_df, fixOff, endOftrial)
        for sub, p, condition_df in conditions
    )

    # Combine results back into the main dataframe
    df["filtPos"] = np.nan
    df["filtVelo"] = np.nan
    df["filtVeloFilt"] = np.nan
    df["velo"] = np.nan

    for sub, p, trial_results in results:
        for result in trial_results:
            trial = result["trial"]
            mask = (df["sub"] == sub) & (df["proba"] == p) & (df["trial"] == trial)

            # Update filtered data
            filtered_trial = result["filtered_trial"]
            df.loc[mask, "filtPos"] = filtered_trial["filtPos"].values
            df.loc[mask, "filtVelo"] = filtered_trial["filtVelo"].values
            df.loc[mask, "filtVeloFilt"] = np.array(filtered_trial["filtVeloFilt"].values)

            # Update velocity
            df.loc[mask, "velo"] = result["velocity"]

    # Save the processed data
    df.to_csv(os.path.join(path, newFileName), index=False)

    return df



# %%

# %%
# Path in niolon
path = "/envau/work/brainets/oueld.h/contextuaLearning/ColorCue/data/"
fileName = "allRawData.csv"
newFileName = "JobLibProcessingCC.csv"
processAllRawData(path, fileName, newFileName)

path = "/envau/work/brainets/oueld.h/contextuaLearning/ColorCue/imposedColorData"
fileName = "allRawData.csv"
newFileName = "JobLibProcessingCC.csv"
processAllRawData(path, fileName, newFileName)

# Attention Task
path = "/envau/work/brainets/oueld.h/attentionalTask/data/"
fileName = "allRawData.csv"
newFileName = "JobLibProcessingCC.csv"
processAllRawData(path, fileName, newFileName)
