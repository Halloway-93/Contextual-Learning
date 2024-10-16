import io
import matplotlib.animation as animation
import os
import re
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import researchpy as rp
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats, signal
from scipy.stats import kruskal
from statsmodels.formula.api import ols
from statsmodels.stats.diagnostic import het_white

# %%


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
    coltypes = get_coltypes(colnames)
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
    msg_df = pd.DataFrame(msg_mat, columns=["time", "text"])
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
    return matches[0][0] if matches else None


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

    # Get date/time of recording from file
    datetime.strptime(from_header(header, "DATE"), "%a %b %d %H:%M:%S %Y")
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


def process_raw(raw, blocks, info):
    if len(raw) == 0:
        # If no sample data in file, create empty raw DataFrame w/ all applicable columns
        raw = ["", ""]
        blocks = pd.Series([], dtype=int)
        colnames = get_raw_header(info)
        coltypes = get_coltypes(colnames, float_time=False)
    else:
        # Determine if timestamps stored as floats (edf2asc option -ftime, useful for 2000 Hz)
        float_time = is_float(re.split(r"\s+", raw[0])[0])
        # Generate column names and types based in info in header
        colnames = get_raw_header(info)
        coltypes = get_coltypes(colnames, float_time)
        # Discard any rows with too many or too few columns (usually rows where eye is missing)
        row_length = [len(re.split(r"\t", r)) for r in raw]
        med_length = np.median(row_length)
        raw = [r for r, l in zip(raw, row_length) if l == med_length]
        blocks = blocks[row_length == med_length]
        # Verify that generated columns match up with actual maximum row length
        length_diff = med_length - len(colnames)
        # if length_diff > 0:
        #    warnings.warn("Unknown columns in raw data. Assuming first one is time, please check the others")
        #    colnames = ["time"] + [f"X{i+1}" for i in range(med_length-1)]
        #    coltypes = "i" + "?"*(med_length-1)
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


# %%
def read_asc(fname, samples=True, events=True, parse_all=False):
    with open(fname, "r", encoding="ISO-8859-1", errors="ignore") as f:
        inp = f.readlines()

    # Convert to ASCII
    inp = [line.encode("ascii", "ignore").decode() for line in inp]

    # Get strings prior to first tab for each line for faster string matching
    inp_first = [re.split(r"\s", s)[0] for s in inp]

    # # Get the Trial info for each trial:
    # bias = [
    #     s.split()[4] for s in inp if len(s.split()) > 4 and s.split()[2] == "Trialinfo:"
    # ]
    # direct = [
    #     s.split()[5] for s in inp if len(s.split()) > 4 and s.split()[2] == "Trialinfo:"
    # ]
    # Check if any actual data recorded in file
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
    #     inp, info = handle_htarg(inp, info, is_raw)

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
        out["raw"] = process_raw(
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


def filter_asp_data(eye_position, sampling_freq=1000, cutoff_freq=30):
    # For ASP, we typically want a slightly higher cutoff
    # to preserve the subtle movements
    # Use a lower order filter to minimize ringing
    order = 2

    nyquist = sampling_freq * 0.5
    normalized_cutoff = cutoff_freq / nyquist

    # Design filter
    b, a = signal.butter(order, normalized_cutoff, btype="low")

    # Use filtfilt for zero-phase filtering
    filtered_pos = signal.filtfilt(b, a, eye_position)

    # Calculate velocity (using central difference)
    velocity = np.zeros_like(filtered_pos)
    velocity[1:-1] = (filtered_pos[2:] - filtered_pos[:-2]) * (sampling_freq / 2)

    # Filter velocity separately with lower cutoff
    vel_cutoff = 20  # Hz
    normalized_vel_cutoff = vel_cutoff / nyquist
    b_vel, a_vel = signal.butter(order, normalized_vel_cutoff, btype="low")
    filtered_vel = signal.filtfilt(b_vel, a_vel, velocity)

    return pd.DataFrame(
        dict({"filtered_pos": filtered_pos, "filtered_vel": filtered_vel})
    )


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


# %%
def preprocess_data_file(filename):
    """
    Preprocessing the blinks and the saccades from the asc file.
    Returning a dataframe for that containes the raw data.
    """
    # Read data from file
    data = read_asc(filename)

    # Extract relevant data from the DataFrame
    df = data["raw"]
    mono = data["info"]["mono"]

    # Convert columns to numeric
    numeric_columns = ["trial", "time", "input"]
    if not mono:
        numeric_columns.extend(["xpl", "ypl", "psl", "xpr", "ypr", "psr"])
    else:
        numeric_columns.extend(["xp", "yp", "ps"])

    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors="coerce")

    # Drop rows where trial is equal to 1
    df = df[df["trial"] != 1]

    # Reset index after dropping rows and modifying the 'trial' column
    # df = df.reset_index(drop=True)

    # Extract messages from eyelink
    MSG = data["msg"]
    tON = MSG.loc[MSG.text == "FixOn", ["trial", "time"]]
    t0 = MSG.loc[MSG.text == "FixOff", ["trial", "time"]]
    Zero = MSG.loc[MSG.text == "TargetOn", ["trial", "time"]]

    # Reset time based on 'Zero' time
    for t in Zero.trial.unique():
        df.loc[df["trial"] == t, "time"] = (
            df.loc[df["trial"] == t, "time"] - Zero.loc[Zero.trial == t, "time"].values
        )
    tON.loc[:, "time"] = tON.time.values - Zero.time.values
    t0.loc[:, "time"] = t0.time.values - Zero.time.values

    # Extract the blinks
    blinks = data["blinks"]
    blinks = blinks[blinks["trial"] != 1]

    # Reset blinks time
    for t in blinks["trial"].unique():
        blinks.loc[blinks.trial == t, ["stime", "etime"]] = (
            blinks.loc[blinks.trial == t, ["stime", "etime"]].values
            - Zero.loc[Zero.trial == t, "time"].values
        )
    # Preocessing the blinks.
    for t in blinks["trial"].unique():
        start = blinks.loc[(blinks.trial == t) & (blinks.eye == "R"), "stime"]
        end = blinks.loc[(blinks.trial == t) & (blinks.eye == "R"), "etime"]

        for i in range(len(start)):
            if not mono:
                df.loc[
                    (df.trial == t)
                    & (df.time >= start.iloc[i] - 50)
                    & (df.time <= end.iloc[i] + 50),
                    "xpr",
                ] = np.nan
            else:
                df.loc[
                    (df.trial == t)
                    & (df.time >= start.iloc[i] - 50)
                    & (df.time <= end.iloc[i] + 50),
                    "xp",
                ] = np.nan

    sacc = detect_saccades(df, mono)
    for t in sacc.trial.unique():
        start = sacc.loc[(sacc.trial == t), "start"]
        end = sacc.loc[(sacc.trial == t), "end"]

        for i in range(len(start)):
            if not mono:
                df.loc[
                    (df.trial == t)
                    & (df.time >= start.iloc[i] - 20)
                    & (df.time <= end.iloc[i] + 20),
                    "xpr",
                ] = np.nan
            else:
                df.loc[
                    (df.trial == t)
                    & (df.time >= start.iloc[i] - 20)
                    & (df.time <= end.iloc[i] + 20),
                    "xp",
                ] = np.nan

    # Decrement the values in the 'trial' column by 1
    df.loc[:, "trial"] = df["trial"] - 1

    return df


def process_data(
    df, frequencyRate=1000, degToPix=27.28, fOFF=80, latency=120, mono=True
):
    """
    Process the data without  filtering

    Returns the value of velocity and postion offset on the window chosen between fOFF and latency
    """
    # Extract position and velocity data
    selected_values = df[(df.time >= fOFF) & (df.time <= latency)]

    pos = (
        selected_values[["trial", "xp"]] if mono else selected_values[["trial", "xpr"]]
    )

    # stdPos = np.std(pos, axis=1) / 27.28
    # # Reshaping veloSteadyState
    # veloSteadyState = np.array(veloSteadyState[: trial_dim * time_dim]).reshape(
    #     trial_dim, time_dim
    # )
    if mono:
        velo = (
            np.array(
                np.gradient(
                    np.array([pos[pos["trial"] == t].xp for t in pos.trial.unique()]),
                    axis=1,
                )
            )
            * frequencyRate
            / degToPix
        )

    # velo[(velo > 20) | (velo < -20)] = np.nan
    else:
        velo = (
            (
                np.gradient(
                    np.array([pos[pos["trial"] == t].xpr for t in pos.trial.unique()]),
                    axis=1,
                )
            )
            * frequencyRate
            / degToPix
        )

    print(velo.shape)
    if mono:
        posOffSet = (
            np.array(
                [
                    pos[pos["trial"] == t].xp.values[-1]
                    - pos[pos["trial"] == t].xp.values[0]
                    for t in pos.trial.unique()
                ]
            )
            / degToPix
        )
    else:
        posOffSet = (
            np.array(
                [
                    pos[pos["trial"] == t].xpr.values[-1]
                    - pos[pos["trial"] == t].xpr.values[0]
                    for t in pos.trial.unique()
                ]
            )
            / degToPix
        )
    meanVelo = np.nanmean(velo, axis=1)
    # stdVelo = np.std(velo, axis=1)
    # meanVSS = np.nanmean(veloSteadyState, axis=1)
    # TS = trialSacc
    # SaccD = saccDir
    # SACC = Sacc

    return pd.DataFrame({"posOffSet": posOffSet, "meanVelo": meanVelo})


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
        b, a = signal.butter(2, normalized_cutoff, btype="low")

        # Apply filter
        filtered_data = signal.filtfilt(b, a, interpolated_data)

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
    nyquist = sampling_freq * 0.5
    normalized_cutoff = velocity_cutoff / nyquist
    b, a = signal.butter(2, normalized_cutoff, btype="low")

    # Filter velocity
    filtered_velocity = signal.filtfilt(b, a, velocity)

    return filtered_velocity * sampling_freq / degToPix


def process_eye_movement(eye_position, sampling_freq=1000, cutoff_freq=30):
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
        interpolated_pos,
        sampling_freq=sampling_freq,
        velocity_cutoff=20,  # Typically lower cutoff for velocity
    )

    # 3. Put NaN back in velocity where position was NaN
    velocity[np.isnan(eye_position)] = np.nan

    return pd.DataFrame(dict({"filtPos": filtered_pos, "filtVelo": velocity}))


def analyze_smooth_pursuit(position, velocity, target_velocity=11.0):
    """
    Analyze smooth pursuit performance
    """
    # Get valid samples (non-NaN)
    valid_samples = ~np.isnan(position)

    # Calculate gain during valid samples
    pursuit_gain = velocity[valid_samples] / target_velocity

    # Basic metrics
    mean_gain = np.mean(pursuit_gain)
    gain_std = np.std(pursuit_gain)

    return mean_gain, gain_std, pursuit_gain


def process_filtered_data(df, mono=True, degToPix=27.28, fOFF=80, latency=120):
    """
    Process the filtered data.
    Returns the position offset and the velocity on the desired window[fOFF,latency].
    """
    data = df[["trial", "time", "xp"]]
    data = data.apply(pd.to_numeric, errors="coerce")
    if mono:
        filtered_data = process_eye_movement(data.xp)
    else:
        filtered_data = process_eye_movement(data.xpr)
    data["filtered_pos"] = filtered_data["filtPos"].values
    data["filtered_velo"] = filtered_data["filtVelo"].values
    # Extract position and velocity data
    selected_values = data[(data.time >= fOFF) & (data.time <= latency)]

    pos = selected_values[["trial", "filtered_pos"]]
    posOffSet = (
        np.array(
            [
                pos[pos["trial"] == t]["filtered_pos"].values[-1]
                - pos[pos["trial"] == t]["filtered_pos"].values[0]
                for t in pos.trial.unique()
            ]
        )
        / degToPix
    )
    meanVelo = np.array(
        [
            np.nanmean(data[data["trial"] == t]["filtered_velo"])
            for t in data.trial.unique()
        ]
    )

    return pd.DataFrame({"posOffSet": posOffSet, "meanVelo": meanVelo})


# %%
def process_all_asc_files(data_dir):
    """
    Go across the data_dir and combine the processed data(Position offset and ASEM) with events tsv file.
    This gives us the information about the chosen cue in each trial and its target direction.
    """
    allDFs = []
    allEvents = []

    for root, _, files in sorted(os.walk(data_dir)):
        for filename in sorted(files):
            if filename.endswith(".asc"):
                filepath = os.path.join(root, filename)
                print(f"Read data from {filepath}")
                df = preprocess_data_file(filepath)
                data = process_data(df)
                # Extract proba from filename
                proba = int(re.search(r"dir(\d+)", filename).group(1))
                data["proba"] = proba

                allDFs.append(data)
                print(len(data))

            if filename.endswith(".tsv"):
                filepath = os.path.join(root, filename)
                print(f"Read data from {filepath}")
                events = pd.read_csv(filepath, sep="\t")
                # Extract proba from filename
                # proba = int(re.search(r"dir(\d+)", filename).group(1))
                # events['proba'] = proba
                # print(len(events))
                allEvents.append(events)

    bigDF = pd.concat(allDFs, axis=0, ignore_index=True)
    # print(len(bigDF))
    bigEvents = pd.concat(allEvents, axis=0, ignore_index=True)
    # print(len(bigEvents))
    # Merge DataFrames based on 'proba'
    merged_data = pd.concat([bigEvents, bigDF], axis=1)
    # print(len(merged_data))

    merged_data.to_csv(os.path.join(data_dir, "results.csv"), index=False)
    return merged_data


# %%
def process_all_filtered_files(data_dir):
    """
    Go across the data_dir and combine the processed data(Position offset and ASEM) with events tsv file.
    This gives us the information about the chosen cue in each trial and its target direction.
    """
    allDFs = []
    allEvents = []

    for root, _, files in sorted(os.walk(data_dir)):
        for filename in sorted(files):
            if filename.endswith(".asc"):
                filepath = os.path.join(root, filename)
                print(f"Read data from {filepath}")
                df = preprocess_data_file(filepath)
                data = process_filtered_data(df)
                # Extract proba from filename
                proba = int(re.search(r"dir(\d+)", filename).group(1))
                data["proba"] = proba

                allDFs.append(data)
                print(len(data))

            if filename.endswith(".tsv"):
                filepath = os.path.join(root, filename)
                print(f"Read data from {filepath}")
                events = pd.read_csv(filepath, sep="\t")
                # Extract proba from filename
                # proba = int(re.search(r"dir(\d+)", filename).group(1))
                # events['proba'] = proba
                # print(len(events))
                allEvents.append(events)

    bigDF = pd.concat(allDFs, axis=0, ignore_index=True)
    # print(len(bigDF))
    bigEvents = pd.concat(allEvents, axis=0, ignore_index=True)
    # print(len(bigEvents))
    # Merge DataFrames based on 'proba'
    merged_data = pd.concat([bigEvents, bigDF], axis=1)
    # print(len(merged_data))

    merged_data.to_csv(os.path.join(data_dir, "filtered_results.csv"), index=False)
    return merged_data


# %%
path = "/Volumes/work/brainets/oueld.h/contextuaLearning/ColorCue/data"

# %%
filename = (
    "/Users/mango/contextuaLearning/ColorCue/data/sub-06/sub-06_col50-dir50_eyeData.asc"
)
# %%
data = read_asc(filename)
df = data["raw"]
df.head()
# %%
firsTrial = df[df.trial == 2]
firsTrial = firsTrial.apply(pd.to_numeric, errors="coerce")
plt.plot(firsTrial.time, firsTrial.xp)
plt.plot(firsTrial.time, firsTrial.yp)
plt.show()
# %%

# Sample DataFrame
# Create a figure and axis
fig, ax = plt.subplots()

# Initialize the line object
(line,) = ax.plot([], [], "bo-")

# Set the limits of the plot
# Set the limits of the plot
ax.set_xlim(firsTrial["xp"].min() - 1, firsTrial["xp"].max() + 1)
ax.set_ylim(firsTrial["yp"].min() - 1, firsTrial["yp"].max() + 1)


# Initialization function: plot the background of each frame
def init():
    line.set_data([], [])
    return (line,)


# Animation function: update the line with new data
def animate(i):
    line.set_data(firsTrial["xp"].values[:i], firsTrial["yp"].values[:i])
    return (line,)


# Create the animation
ani = animation.FuncAnimation(
    fig, animate, init_func=init, frames=len(firsTrial.time), interval=5, blit=True
)

# Show the plot
plt.show()

# %%
df = preprocess_data_file(filename)
# df = df[df["trial"] != 1]
df.trial.unique()
# %%
df
# %%
data = df[["trial", "time", "xp"]]
data = data.apply(pd.to_numeric, errors="coerce")
data
# %%
filtered_data = process_eye_movement(data.xp)
filtered_data
# %%
data["filtered_pos"] = filtered_data["filtPos"].values
data["filtered_velo"] = filtered_data["filtVelo"].values
data
# %%
data = data[(data.time > -200) & (data.time < 600)]
data
# %%
for t in data.trial.unique():
    plt.plot(data[data.trial == t].time, data[data.trial == t].filtered_velo)
    plt.xlabel("Time in ms")
    plt.ylabel("Filtered Velocity")
    plt.title(f"Butteworth Filter on Velocity: Trial {t}")
    plt.show()
# %%
for t in data.trial.unique():
    plt.plot(data[data.trial == t].time, data[data.trial == t].filtered_pos)
    plt.xlabel("Time in ms")
    plt.ylabel("Filtered Velocity")
    plt.title(f"Butteworth Filter on Velocity: Trial {t}")
    plt.show()
# %%

# %%
df = process_all_asc_files(path)
# %%
process_all_filtered_files(path)
# %%

# %%
df.columns
# %%

df.meanVelo.isna().sum()

# %%
df.posOffSet.isna().sum()
# %%
# %%
r"""°°°
# Start Running the code from Here
°°°"""

df = pd.read_csv(path + "/results.csv")
# [print(df[df["sub_number"] == i]["meanVelo"].isna().sum()) for i in range(1, 13)]
# df.dropna(inplace=True)
df["color"] = df["trial_color_chosen"].apply(lambda x: "green" if x == 0 else "red")


df = df.dropna(subset=["meanVelo"])
# Assuming your DataFrame is named 'df' and the column you want to rename is 'old_column'
# df.rename(columns={'old_column': 'new_column'}, inplace=True)
df.head()

# %%

df.meanVelo.isna().sum()
# %%

# df = df[df["sub_number"] != 9]

# %%

colors = ["green", "red"]
# Set style to whitegrid

# # Set font size for labels
# sns.set(
#     rc={
#         "axes.labelsize": 25,
#
#         "axes.titlesize": 20,
#     }
# )
#
# sns.set_style("whitegrid")

# %%

sns.lmplot(
    x="proba",
    y="meanVelo",
    data=df,
    hue="color",
    scatter_kws={"alpha": 0.2},
    palette=colors,
)
plt.show()
# %%
sns.lmplot(
    x="proba",
    y="posOffSet",
    data=df,
    hue="color",
    scatter_kws={"alpha": 0.2},
    palette=colors,
)
plt.show()
# %%

l = (
    df.groupby(["sub_number", "trial_color_chosen", "proba"])[["meanVelo", "posOffSet"]]
    .mean()
    .reset_index()
)


l


l["color"] = l["trial_color_chosen"].apply(lambda x: "green" if x == 0 else "red")
l

# %%

bp = sns.barplot(x="proba", y="meanVelo", hue="color", data=l, palette=colors)
bp.legend(fontsize="larger")
plt.xlabel("P(Right|Red)", fontsize=30)
plt.ylabel("Anticipatory Velocity", fontsize=30)
plt.savefig("clccbp.png")
plt.show()
# %%

bp = sns.boxplot(x="proba", y="posOffSet", hue="color", data=l, palette=colors)
bp.legend(fontsize="larger")
plt.xlabel("P(Right|Red)", fontsize=30)
plt.ylabel("Pos OffSet", fontsize=30)
plt.savefig("clccbp.png")
plt.show()
# %%
lm = sns.lmplot(
    x="proba", y="meanVelo", hue="trial_color_chosen", data=l, palette=colors, height=8
)
# Adjust font size for axis labels
lm.set_axis_labels("P(Right|Red)", "Anticipatory Velocity")
# lm.ax.legend(fontsize='large')
plt.savefig("clcclp.png")
plt.show()

# %%
df.columns
# %%

# Create the box plot with transparent fill and black borders, and without legend
bp = sns.boxplot(
    x="proba",
    y="meanVelo",
    hue="color",
    data=l,
    palette=colors,
    boxprops=dict(facecolor="none", edgecolor="black"),
    legend=False,
)

# Add scatter plot on top
sns.stripplot(
    x="proba",
    y="meanVelo",
    hue="color",
    data=l,
    dodge=True,
    palette=colors,
    jitter=True,
    size=8,
    alpha=0.7,
)
# Set labels for both top and bottom x-axes
plt.xlabel("P(Right|Red)", fontsize=30)
plt.ylabel("Anticipatory Velocity", fontsize=30)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
# Overlay regplot on top of the boxplot and stripplot

plt.twiny().set_xlabel("P(Right|Green)", fontsize=30)
# Set the tick positions for both top and bottom x-axes
tick_positions = [0.2, 0.5, 0.8]
tick_labels = [25, 50, 75]

# Set the ticks and labels for both top and bottom x-axes
plt.xticks(tick_positions, tick_labels, fontsize=20)
plt.xticks(fontsize=30)
# Invert the top x-axis
plt.gca().invert_xaxis()

# Manually add stars indicating statistical significance
# Adjust the coordinates based on your plot
plt.text(0.6, 0.6, "**", fontsize=30, ha="center", va="center", color="red")
plt.text(
    0.6, 0.65, "_______________", fontsize=30, ha="center", va="center", color="red"
)
# plt.text(0.6, 0.6, 'p < 0.001', fontsize=15, ha='center', va='center', color='red')

plt.text(0.75, 0.75, "***", fontsize=30, ha="center", va="center", color="green")
plt.text(
    0.75, 0.8, "_______________", fontsize=30, ha="center", va="center", color="green"
)

# Right side

plt.text(0.25, -1, "**", fontsize=30, ha="center", va="center", color="red")
plt.text(
    0.25, -0.95, "_______________", fontsize=30, ha="center", va="center", color="red"
)
# plt.text(0.6, 0.6, 'p < 0.001', fontsize=15, ha='center', va='center', color='red')

plt.text(0.45, -1, "***", fontsize=30, ha="center", va="center", color="green")
plt.text(
    0.45, -1, "_______________", fontsize=30, ha="center", va="center", color="green"
)

# plt.text(0.333, 0.6, 'p < 0.001', fontsize=15, ha='center', va='center', color='green')
# Adjust legend
bp.legend(fontsize=25)


# Save the plot
plt.savefig("clccbp.png")

# Show the plot
plt.show()


lm = sns.lmplot(
    x="trial_color_chosen",
    y="meanVelo",
    hue="proba",
    data=l,
    height=8,
    palette="viridis",
)
# Adjust font size for axis labels
lm.set_axis_labels("Color Chosen", "Anticipatory Velocity", fontsize=20)

# %%

bp = sns.barplot(
    x="color",
    y="meanVelo",
    hue="proba",
    data=l,
    palette="viridis",
)

bp.legend(fontsize=25)
plt.xlabel("Color Chosen", fontsize=30)
plt.ylabel("Anticipatory Velocity", fontsize=30)
plt.savefig("antihueproba.png")
plt.show()
# %%
df[(df.sub_number == 8)].trial_color_chosen

# %%
dd = (
    df.groupby(["sub_number", "color", "proba"])[["meanVelo", "posOffSet"]]
    .mean()
    .reset_index()
)
# %%
model = sm.OLS.from_formula("meanVelo ~ C(proba) ", data=dd[dd.color == "red"])
result = model.fit()

print(result.summary())

# %%

model = sm.OLS.from_formula("meanVelo ~ C(proba) ", data=dd[dd.color == "green"])
result = model.fit()

print(result.summary())

# %%

model = sm.OLS.from_formula("meanVelo ~ C(color) ", data=dd[dd.proba == 25])
result = model.fit()

print(result.summary())

# %%

model = sm.OLS.from_formula("meanVelo ~ C(color) ", data=dd[dd.proba == 75])
result = model.fit()

print(result.summary())

# %%

model = sm.OLS.from_formula("meanVelo ~ C(color) ", data=dd[dd.proba == 50])
result = model.fit()

print(result.summary())

# %%

model = ols("meanVelo ~ C(proba) ", data=dd[dd.color == "green"]).fit()
anova_table = sm.stats.anova_lm(model, typ=3)

print(anova_table)

# %%

rp.summary_cont(df.groupby(["sub_number", "color", "proba"])["meanVelo"])

# %%

model = ols("meanVelo ~ C(proba)*C(color) ", data=dd).fit()
anova_table = sm.stats.anova_lm(model, typ=3)

print(anova_table)
# %%

model = smf.mixedlm(
    "posOffSet ~ C(color)*C(proba)",
    data=dd,
    groups=dd["sub_number"],
).fit()
model.summary()
# %%
model = smf.mixedlm(
    "meanVelo ~ C(color)*C(proba)",
    data=dd,
    groups=dd["sub_number"],
).fit()
model.summary()
# %%
model = smf.mixedlm(
    "meanVelo ~ C(color)*C(proba)",
    data=dd,
    groups=dd["sub_number"],
).fit()
model.summary()

# %%
# Plottign the model

sns.barplot(data=dd, x="proba", y="meanVelo", hue="color", palette=["green", "red"])
plt.show()
# %%

summary = rp.summary_cont(df.groupby(["sub_number", "color", "proba"])["meanVelo"])


# %%

summary.reset_index(inplace=True)
print(summary)
# %%


sns.boxplot(data=summary, x="proba", y="Mean", hue="color", palette=["green", "red"])
plt.show()


# %%

# Get unique sub_numbers
unique_sub_numbers = summary["sub_number"].unique()

# Set up the FacetGrid
facet_grid = sns.FacetGrid(
    data=summary,
    col="sub_number",
    col_wrap=4,
    sharex=True,
    sharey=True,
    height=3,
    aspect=1.5,
)

# Create pointplots for each sub_number
facet_grid.map_dataframe(
    sns.pointplot,
    x="proba",
    y="Mean",
    hue="color",
    markers=["o", "s", "d"],
    palette=["green", "red"],
)

# Add legends
facet_grid.add_legend()

# Set titles for each subplot
for ax, sub_number in zip(facet_grid.axes.flat, unique_sub_numbers):
    ax.set_title(f"Subject {sub_number}")

# Adjust spacing between subplots
facet_grid.fig.subplots_adjust(
    wspace=0.2, hspace=0.2
)  # Adjust wspace and hspace as needed

# Show the plot
plt.savefig("allSubjectanti.png")
plt.show()

# %%
grid = sns.FacetGrid(df, col="sub_number", hue="proba", col_wrap=4, height=3)


grid.map(plt.scatter, "trial_number", "meanVelo")
plt.show()
# %%

# Create a KDE plot of residuals
sns.displot(model.resid, kind="kde", fill=True, lw=1)

# Overlay normal distribution on the same plot
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 1000)
p = stats.norm.pdf(x, np.mean(model.resid), np.std(model.resid))
plt.plot(x, p, "k", linewidth=1)

# Set title and labels
plt.title("KDE Plot of Model Residuals (Red) and Normal Distribution (Black)")
plt.xlabel("Residuals")
plt.show()
# %%

fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(111)

sm.qqplot(model.resid, dist=stats.norm, line="s", ax=ax)

ax.set_title("Q-Q Plot")

plt.show()
# %%
labels = ["Statistic", "p-value"]

norm_res = stats.normaltest(model.resid)

for key, val in dict(zip(labels, norm_res)).items():
    print(key, val)


# %%
fig = plt.figure(figsize=(16, 9))

ax = sns.boxplot(x=model.model.groups, y=model.resid)

ax.set_title("Distribution of Residuals for Anticipatory Velocity by Subject")
ax.set_ylabel("Residuals")
ax.set_xlabel("Subject")
plt.show()
# %%

het_white_res = het_white(model.resid, model.model.exog)

labels = ["LM Statistic", "LM-Test p-value", "F-Statistic", "F-Test p-value"]

for key, val in dict(zip(labels, het_white_res)).items():
    print(key, val)

# %%

# t test to comprare proba 25/red and proba75/green
stats.ttest_ind(
    dd[(l.proba == 25) & (dd.color == "red")].meanVelo,
    dd[(l.proba == 75) & (dd.color == "green")].meanVelo,
)

# %%

# %%

# t test to comprare proba 25/red and proba75/green
stats.ttest_ind(
    dd[(l.proba == 75) & (dd.color == "red")].meanVelo,
    dd[(l.proba == 25) & (dd.color == "green")].meanVelo,
)

# %%

# t test to comprare proba 25/red and proba75/green
stats.ttest_ind(
    dd[(dd.proba == 75) & (dd.color == "red")].meanVelo,
    dd[(dd.proba == 75) & (dd.color == "green")].meanVelo,
)

# %%

# %%

# t test to comprare proba 25/red and proba75/green
stats.ttest_ind(
    dd[(dd.proba == 25) & (dd.color == "red")].meanVelo,
    dd[(dd.proba == 25) & (dd.color == "green")].meanVelo,
)

# %%


stats.ttest_ind(
    dd[(dd.proba == 50) & (dd.color == "red")].meanVelo,
    dd[(dd.proba == 50) & (dd.color == "green")].meanVelo,
)

# %%

stats.ttest_ind(
    df[(df.proba == 50) & (df.color == "green")].meanVelo,
    df[(df.proba == 75) & (df.color == "green")].meanVelo,
)

# %%

# Example assuming 'proba' and 'color' are categorical variables in your DataFrame
colors = df["color"].unique()

for color in colors:
    # Filter data for the current color
    color_data = df[df["color"] == color]

    # Group data by 'proba' and get meanVelo for each group
    grouped_data = [group["meanVelo"] for proba, group in color_data.groupby("proba")]
    print(grouped_data)

    # Perform Kruskal-Wallis test
    statistic, p_value = kruskal(*grouped_data)

    # Print results for each color
    print(f"Color: {color}")
    print(f"Kruskal-Wallis Statistic: {statistic}")
    print(f"P-value: {p_value}")

    # Check if the result is statistically significant
    if p_value < 0.01:
        print(
            "The probabilities within this color have significantly different distributions."
        )
    else:
        print(
            "There is not enough evidence to suggest significant differences between probabilities within this color."
        )
    print("\n")

# %%
sns.histplot(
    data=dd[dd.color == "red"],
    x="meanVelo",
    hue="proba",
    kde=True,
)
plt.show()
# %%
r"""
# Analysis of subject who did Vanessa's task
"""
# %%

df_prime = df[(df.sub_number > 12)]

# |%%--%%| <9e6bJW7zSd|aAEDXXm0yJ>

# %%
l_prime = (
    df_prime.groupby(["sub_number", "trial_color_chosen", "proba"])
    .meanVelo.mean()
    .reset_index()
)
l_prime

# |%%--%%| <aAEDXXm0yJ|QXsRE0iCWU>
# %%

bp = sns.boxplot(
    x="proba", y="meanVelo", hue="trial_color_chosen", data=l_prime, palette=colors
)
bp.legend(fontsize="larger")
plt.xlabel("P(Dir=Right|Color=Red)", fontsize=30)
plt.ylabel("Anticipatory Velocity", fontsize=30)

# |%%--%%| <QXsRE0iCWU|5EX22XRGSK>

# %%
lm = sns.lmplot(
    x="proba",
    y="meanVelo",
    hue="trial_color_chosen",
    data=l_prime,
    palette=colors,
    height=8,
)
# Adjust font size for axis labels
lm.set_axis_labels("P(R|Red)", "Anticipatory Velocity")

# |%%--%%| <5EX22XRGSK|qxewuIGTnt>
# %%

# Participants balanced their choices
print(df.trial_color_chosen.value_counts())
# |%%--%%| <qxewuIGTnt|ZwGowoTUlq>
# %%


from collections import Counter


def compute_probability_distribution_tplus1_given_t(
    df, subject_col, condition_col, choice_col
):
    # df is your DataFrame
    # subject_col is the column name for the subjects
    # condition_col is the column name for the conditions
    # choice_col is the column name for the choices

    # Create a dictionary to store probability distributions for each subject and condition group
    probability_distributions = {}

    # Iterate over unique subject-condition pairs
    for (subject, condition), group_df in df.groupby([subject_col, condition_col]):
        choices = group_df[choice_col].tolist()

        # Count occurrences of each pair (C_t, C_{t+1})
        transition_counts = Counter(zip(choices[:-1], choices[1:]))

        # Compute total counts for each choice at time t
        total_counts_t = Counter(choices[:-1])

        # Calculate the conditional probabilities
        probability_distribution = {}
        for (choice_t, choice_tplus1), count in transition_counts.items():
            probability_distribution[(choice_tplus1, choice_t)] = (
                count / total_counts_t[choice_t]
            )

        # Store the probability distribution in the dictionary
        probability_distributions[(subject, condition)] = probability_distribution

    return probability_distributions


# %%
# |%%--%%| <ZwGowoTUlq|ggwrOHSS1C>

probability_distributions_by_group = compute_probability_distribution_tplus1_given_t(
    df, "sub_number", "proba", "trial_color_chosen"
)
probability_distributions_by_group

# |%%--%%| <ggwrOHSS1C|xhpGhcUYO3>


# %%

# Example usage:
# with columns "subject", "condition", and "choice"
for i in df["sub_number"].unique():
    for p in df["proba"].unique():
        print(f"Probability Distribution for subject {i} and condition {p}:")
        for key, probability in probability_distributions_by_group[(i, p)].items():
            print(f"P(C_{key[0]} | C_{key[1]}) = {probability:.2f}")

# %%
# |%%--%%| <xhpGhcUYO3|go3EAD1SFB>

# Get unique subjects and probabilities
unique_subjects = df["sub_number"].unique()
unique_probabilities = df["proba"].unique()

# Iterate over subjects
for subject in unique_subjects:
    # Set up subplots for the current subject
    fig, axes = plt.subplots(nrows=1, ncols=len(unique_probabilities), figsize=(15, 5))

    # Iterate over probabilities
    for j, probability in enumerate(unique_probabilities):
        # Get probability distribution for the current subject and probability
        probability_distribution = probability_distributions_by_group.get(
            (subject, probability), {}
        )

        # Get unique pairs and corresponding probabilities
        # The pai is C_t+1 and C_t
        unique_pairs = sorted(set(pair for pair in probability_distribution.keys()))
        probabilities = [probability_distribution.get(pair, 0) for pair in unique_pairs]

        # Set bar width and offsets
        bar_width = 0.35
        bar_offsets = np.arange(len(unique_pairs))

        # Plot the bar chart
        axes[j].bar(
            bar_offsets, probabilities, bar_width, label=f"Probability {probability}"
        )
        axes[j].set_xticks(bar_offsets)

        axes[j].set_xticklabels(
            [f"({pair[0]}, {pair[1]})" for pair in unique_pairs], size=20
        )
        axes[j].set_title("$\mathbb{P}(Right|Red)$=" + f"{probability}", fontsize=30)
        axes[j].set_xlabel("Pairs $(C_{t+1}, C_t)$")
        # axes[j].set_ytickslabels(size=20)
    # Set common labels and legend for the entire figure
    # fig.text(0.5, 0.04, 'Pairs (C_t, C_{t+1})', ha='center', va='center')
    # fig.text(0.06, 0.5, 'Probability', ha='center', va='center', rotation='vertical')
    fig.suptitle(
        "$\mathbb{P}(Choice_{t+1}|Choice_t)$ for each condition:"
        + f"Subject {subject}",
        size=35,
    )
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# |%%--%%| <go3EAD1SFB|LIDfUD5MWo>
# %%


# Computing the mean over all subjects
def compute_mean_probability_distribution_tplus1_given_t(dictionary):
    # Create a dictionary to store the mean probability distribution for each condition
    mean_probability_distribution = {}

    # Iterate over unique conditions
    for (subject, condition), distribution in dictionary.items():
        if condition not in mean_probability_distribution:
            mean_probability_distribution[condition] = Counter()

        mean_probability_distribution[condition].update(distribution)

    # Calculate the mean probability distribution over all subjects for each condition
    for condition, distribution in mean_probability_distribution.items():
        total_subjects = len(dictionary) // len(mean_probability_distribution)
        mean_probability_distribution[condition] = {
            key: count / total_subjects for key, count in distribution.items()
        }

    return mean_probability_distribution


# %%
# |%%--%%| <LIDfUD5MWo|lKAgzJCVuI>


# Assuming you already have the probability_distributions_by_group_tplus1_given_t dictionary
probability_distributions_by_group_tplus1_given_t = (
    compute_probability_distribution_tplus1_given_t(
        df, "sub_number", "proba", "trial_color_chosen"
    )
)
mean_probability_distribution_tplus1_given_t = (
    compute_mean_probability_distribution_tplus1_given_t(
        probability_distributions_by_group_tplus1_given_t
    )
)


# |%%--%%| <lKAgzJCVuI|5oKxlGF93G>
# %%


# Extract unique pairs (C_t, C_{t+1}) from the first condition (assuming all conditions have the same pairs)
unique_pairs_t_tplus1 = list(mean_probability_distribution_tplus1_given_t.values())[
    0
].keys()

# Prepare data for plotting
num_conditions = len(mean_probability_distribution_tplus1_given_t)
num_pairs = len(unique_pairs_t_tplus1)

# Create subplots
fig, axes = plt.subplots(1, num_conditions, figsize=(15, 5), sharey=True)

bar_width = 0.2
bar_offsets = np.arange(num_pairs)

# Plotting
for idx, (condition, mean_distribution) in enumerate(
    mean_probability_distribution_tplus1_given_t.items()
):
    probabilities = [mean_distribution[pair] for pair in unique_pairs_t_tplus1]

    axes[idx].bar(bar_offsets, probabilities, bar_width, label=f"Condition {condition}")
    axes[idx].set_xticks(bar_offsets)
    axes[idx].set_xticklabels(unique_pairs_t_tplus1)
    # axes[idx].set_xlabel('Pairs (C_t, C_{t+1})')
    axes[idx].set_title(f"Probability:  {condition}")

# Set common labels and legend
fig.text(0.5, 0.04, "Pairs (C_{t+1},C_t,)", ha="center", va="center")
fig.text(0.06, 0.5, "Probability", ha="center", va="center", rotation="vertical")
fig.suptitle("Mean Probability Distribution for Each Condition and Pair (C_{t+1},C_t)")
# plt.legend()

plt.show()


# %%
# |%%--%%| <5oKxlGF93G|4SSNV0ixKb>
"""°°°
Computing P(C_{t+2} | C_{t+1}, C_t)
°°°"""
# |%%--%%| <4SSNV0ixKb|7AfY2O0mA0>
# %%


def compute_probability_distribution_tplus2_given_tplus1_and_t(
    df, subject_col, condition_col, choice_col
):
    # df is your DataFrame
    # subject_col is the column name for the subjects
    # condition_col is the column name for the conditions
    # choice_col is the column name for the choices

    # Create a dictionary to store probability distributions for each subject and condition group
    probability_distributions_tplus2_given_tplus1_and_t = {}

    # Iterate over unique subject-condition pairs
    for (subject, condition), group_df in df.groupby([subject_col, condition_col]):
        choices = group_df[choice_col].tolist()

        # Count occurrences of each triplet (C_t, C_{t+1}, C_{t+2})
        transition_counts_t_tplus1_tplus2 = Counter(
            zip(choices[:-2], choices[1:-1], choices[2:])
        )

        # Compute total counts for each pair (C_{t+1}, C_t)
        total_counts_tplus1_t = Counter(zip(choices[:-1], choices[1:]))

        # Calculate the conditional probabilities for P(C_{t+2} | C_{t+1} & C_t)
        probability_distribution_tplus2_given_tplus1_and_t = {}
        for (
            choice_t,
            choice_tplus1,
            choice_tplus2,
        ), count in transition_counts_t_tplus1_tplus2.items():
            probability_distribution_tplus2_given_tplus1_and_t[
                (choice_tplus2, choice_tplus1, choice_t)
            ] = (count / total_counts_tplus1_t[choice_t, choice_tplus1])

        # Store the probability distribution in the dictionary
        probability_distributions_tplus2_given_tplus1_and_t[(subject, condition)] = (
            probability_distribution_tplus2_given_tplus1_and_t
        )

    return probability_distributions_tplus2_given_tplus1_and_t


# |%%--%%| <7AfY2O0mA0|JIDuF3osMK>
# %%


probability_distributions_by_group = (
    compute_probability_distribution_tplus2_given_tplus1_and_t(
        df, "sub_number", "proba", "trial_color_chosen"
    )
)
probability_distributions_by_group
# |%%--%%| <JIDuF3osMK|ZfiNVKwozc>

# %%
for i in df["sub_number"].unique():
    for p in df["proba"].unique():
        print(f"Probability Distribution for subject {i} and condition {p}:")
        for key, probability in probability_distributions_by_group[(i, p)].items():
            print(f"P(C_{key[0]} | C_{key[1]},C_{key[2]}) = {probability:.2f}")

# |%%--%%| <ZfiNVKwozc|B6f78cRdCl>

# %%

unique_triplets = list(probability_distributions_by_group.values())[0].keys()
unique_triplets
# |%%--%%| <B6f78cRdCl|4uikRnrdl5>

# %%
probability_distributions_by_group.items()

# |%%--%%| <4uikRnrdl5|37XUO5s80z>
# %%

# Extract unique triplets from the first condition (assuming all conditions have the same triplets)

# Prepare data for plotting
num_conditions = len(probability_distributions_by_group)
num_triplets = len(unique_triplets)

# Create subplots
fig, axes = plt.subplots(1, num_conditions, figsize=(15, 5), sharey=True)

bar_width = 0.2
bar_offsets = np.arange(num_triplets)


# Plotting
for idx, (condition, mean_distribution) in enumerate(
    probability_distributions_by_group.items()
):
    subject = condition[0]
    condition_value = condition[1]

    print(f"Subject: {subject}, Condition: {condition_value}")
    print(f"Keys in mean_distribution: {mean_distribution.keys()}")

    probabilities = [mean_distribution[triplet] for triplet in unique_triplets]

    axes[idx].bar(
        bar_offsets,
        probabilities,
        bar_width,
        label=f"Subject {subject}, Condition {condition_value}",
    )
    axes[idx].set_xticks(bar_offsets)
    axes[idx].set_xticklabels(unique_triplets, fontsize=8)
    axes[idx].set_title(f"Subject {subject}, Condition {condition_value}")

# Set common labels
fig.text(
    0.5,
    0.04,
    "Triplets (C_{t+2}, C_{t+1}, C_t))",
    ha="center",
    va="center",
    fontsize=20,
)
fig.text(0.06, 0.5, "Probability", ha="center", va="center", rotation="vertical")
fig.suptitle("Mean Probability Distribution for P(C_t+2 | C_t+1, C_t)", fontsize=30)

plt.show()


# |%%--%%| <37XUO5s80z|jfgTiohOAU>

# %%


def compute_mean_probability_distribution(dictionary):
    # Create a dictionary to store the mean probability distribution for each condition
    mean_probability_distribution = {}

    # Iterate over unique conditions
    for condition in set(key[1] for key in dictionary.keys()):
        mean_distribution = Counter()
        num_participants = 0

        # Aggregate distributions for the same condition
        for subject, cond in dictionary.keys():
            if cond == condition:
                distribution = dictionary[(subject, cond)]
                mean_distribution.update(distribution)
                num_participants += 1

        # Calculate the mean by dividing each count by the number of participants
        mean_distribution = {
            key: count / num_participants for key, count in mean_distribution.items()
        }

        # Store the mean distribution for the condition
        mean_probability_distribution[condition] = mean_distribution

    return mean_probability_distribution


# |%%--%%| <jfgTiohOAU|ki3x5jpfAR>

# %%
meanOverSubjects = compute_mean_probability_distribution(
    probability_distributions_by_group
)
meanOverSubjects

# %%
# |%%--%%| <ki3x5jpfAR|4ol2GdpoqJ>


for p in df["proba"].unique():
    print(f"Probability Distribution for proba {p}:")
    for key, probability in meanOverSubjects[(p)].items():
        print(f"P(C_{key[0]} | C_{key[1]},C_{key[2]}) = {probability:.2f}")


# |%%--%%| <4ol2GdpoqJ|RAx7oDpABa>

# %%

# Extract unique triplets from the first condition (assuming all conditions have the same triplets)
unique_triplets = list(meanOverSubjects.values())[0].keys()

# Prepare data for plotting
num_conditions = len(meanOverSubjects)
num_triplets = len(unique_triplets)

# Create subplots
fig, axes = plt.subplots(1, num_conditions, figsize=(15, 5), sharey=True)

bar_width = 0.2
bar_offsets = np.arange(num_triplets)

# Plotting
for idx, (condition, mean_distribution) in enumerate(meanOverSubjects.items()):
    probabilities = [mean_distribution[triplet] for triplet in unique_triplets]

    axes[idx].bar(bar_offsets, probabilities, bar_width, label=f"Condition {condition}")
    axes[idx].set_xticks(bar_offsets)
    axes[idx].set_xticklabels(unique_triplets, fontsize=8)
    # axes[idx].set_xlabel('Triplets (C_t, C_{t+1}, C_{t+2})')
    axes[idx].set_title(f"Condition {condition}")

# Set common labels and legend
fig.text(
    0.5,
    0.04,
    "Triplets (C_{t+2}, C_{t+1}, C_t))",
    ha="center",
    va="center",
    fontsize=20,
)
fig.text(0.06, 0.5, "Probability", ha="center", va="center", rotation="vertical")
fig.suptitle("Mean Probability Distribution for P(C_t+2 | C_t+1, C_t)", fontsize=30)
plt.legend()

plt.show()
# |%%--%%| <RAx7oDpABa|YvO4edwPiv>

# %%


unique_sub_numbers = df["sub_number"].unique()
# Custom color palette for 'color' categories
custom_palette = {"green": "green", "red": "red"}
for sub_number_value in unique_sub_numbers:
    subset_df = df[df["sub_number"]] == sub_number_value

    # Set up subplots for each proba
    fig, axes = plt.subplots(
        nrows=1, ncols=len(subset_df["proba"].unique()), figsize=(15, 5), sharey=True
    )

    # Plot each subplot
    for i, proba_value in enumerate(subset_df["proba"].unique()):
        proba_subset_df = subset_df[subset_df["proba"] == proba_value]
        ax = axes[i]

        # Group by both "proba" and "color" and compute rolling mean with a window of 20
        rolling_mean = (
            proba_subset_df.groupby(["proba", "color"])["meanVelo"]
            .rolling(window=10, min_periods=1)
            .mean()
            .reset_index(level=[0, 1], drop=True)
        )

        # Plot the rolling mean with color as a hue
        sns.lineplot(
            x="trial_number",
            y=rolling_mean,
            hue="color",
            data=proba_subset_df,
            ax=ax,
            palette=custom_palette,
            markers=True,
        )

        ax.set_title(f"sub_number = {sub_number_value}, proba = {proba_value}")
        ax.set_xlabel("Trial Number")
        ax.set_ylabel("Mean Velocity (Rolling Average)")
        ax.legend(title="Color", loc="upper right")

    # Adjust layout for subplots for each subject
    plt.tight_layout()
    plt.show()

# |%%--%%| <YvO4edwPiv|wY6MEnWD9g>
# %%

rolling_mean

# |%%--%%| <wY6MEnWD9g|bB5rAoyuwj>
# %%

# Score of percistency

# |%%--%%| <bB5rAoyuwj|btcBZVF8lR>

probability_distributions_by_group_tplus1_given_t.keys()

# |%%--%%| <btcBZVF8lR|U1RCI8clxw>
# %%


probability_distributions_by_group_tplus1_given_t
# |%%--%%| <U1RCI8clxw|jGu7UdQ3iH>
# %%

tplus1GivenT = pd.DataFrame(probability_distributions_by_group_tplus1_given_t)
# |%%--%%| <jGu7UdQ3iH|eaOt3gMsmF>
# %%

tplus1GivenT

# |%%--%%| <eaOt3gMsmF|htGZUflqhx>
# %%


# |%%--%%| <htGZUflqhx|m3jza5TNn8>


# Convert dictionary to DataFrame
# %%
tplus1GivenT = pd.DataFrame.from_dict(
    {
        (k1, k2): v2
        for k1, d in probability_distributions_by_group_tplus1_given_t.items()
        for k2, v2 in d.items()
    },
    orient="index",
)

# Reset index and rename columns
tplus1GivenT = tplus1GivenT.reset_index().rename(
    columns={"level_0": "Group", "level_1": "Distribution", 0: "Probability"}
)

print(tplus1GivenT)

# |%%--%%| <m3jza5TNn8|ypzJK0eb37>
# %%

tplus1GivenT.columns

# |%%--%%| <ypzJK0eb37|kAaihO8ylP>
# %%

# Dictionary to store the sums for each main key
sums_by_main_key = {}

# Iterate through the main keys
for main_key, sub_dict in probability_distributions_by_group_tplus1_given_t.items():
    # Initialize the sum for the current main key
    current_sum = 0
    # Iterate through the sub-dictionary
    for sub_key, probability in sub_dict.items():
        # Extract the sub-key components
        x, y = sub_key
        # Perform the arithmetic operations and update the sum
        if x == 1 and y == 1:
            current_sum += probability
        elif x == 0 and y == 0:
            current_sum += probability
        elif x == 0 and y == 1:
            current_sum -= probability
        elif x == 1 and y == 0:
            current_sum -= probability
    # Store the sum for the current main key
    sums_by_main_key[main_key] = current_sum

# Print the sums for each main key
# %%
for main_key, sum_value in sums_by_main_key.items():
    print(f"Main Key: {main_key}, Sum: {sum_value}")

# |%%--%%| <kAaihO8ylP|FIuoAdIe71>
# %%

sums_by_main_key

# |%%--%%| <FIuoAdIe71|rgRPlJFvLL>
# %%


# Initialize lists to store data
subjects = []
probabilities = []
persistence_scores = []

# Iterate through the dictionary items
for key, value in sums_by_main_key.items():
    # Extract subject and probability from the key
    subject, probability = key

    # Append data to lists
    subjects.append(subject)
    probabilities.append(probability)
    persistence_scores.append(value)

# Create DataFrame
percistenceScore = pd.DataFrame(
    {
        "Subject": subjects,
        "Probability": probabilities,
        "Persistence Score": persistence_scores,
    }
)

# Display DataFrame
print(percistenceScore)

# |%%--%%| <rgRPlJFvLL|3CJbtd4FeR>
# %%

percistenceScore.groupby("Subject")["Persistence Score"].mean()

# |%%--%%| <3CJbtd4FeR|iCpM9sHN6j>
# %%

learning = df.groupby(["sub_number", "color", "proba"]).meanVelo.mean().reset_index()
learning
# |%%--%%| <iCpM9sHN6j|in2fBOKfLB>
# %%


# Group by 'sub_number' and 'color'
grouped = learning.groupby(["sub_number", "color"])

# Calculate the mean velocity for probability 75 and 25, respectively
mean_velo_75 = grouped.apply(lambda x: x[x["proba"] == 75]["meanVelo"].mean())
mean_velo_25 = grouped.apply(lambda x: x[x["proba"] == 25]["meanVelo"].mean())

# Calculate the difference
difference = np.abs(mean_velo_75 - mean_velo_25)

# Display the result
print(difference)

# |%%--%%| <in2fBOKfLB|XVJ4Z1UtaB>
# %%

grouped

# %%
# |%%--%%| <XVJ4Z1UtaB|bLdHdui9o8>

difference_green = difference.xs("green", level="color")
difference_red = difference.xs("red", level="color")
percistence = (
    percistenceScore.groupby("Subject")["Persistence Score"].mean().reset_index()
)
percistence

# |%%--%%| <bLdHdui9o8|mJbZBBTDrX>
# %%

percistence["learningScore"] = np.mean([difference_green, difference_red], axis=0)

# |%%--%%| <mJbZBBTDrX|sQipqrPL2s>
# %%

percistence["learningGreen"] = difference_green.values
percistence["learningRed"] = difference_red.values
percistence
# |%%--%%| <sQipqrPL2s|Yf0nwdODe4>
# %%

sns.scatterplot(
    data=percistence, x="Persistence Score", y="learningGreen", hue="Subject"
)
plt.show()
# |%%--%%| <Yf0nwdODe4|gXeKBR5VSb>

# %%
sns.scatterplot(data=percistence, x="Persistence Score", y="learningRed", hue="Subject")
plt.show()
# %%
# |%%--%%| <gXeKBR5VSb|zSwvDk70Ay>

sns.scatterplot(
    data=percistence, x="Persistence Score", y="learningScore", hue="Subject"
)
plt.show()
# |%%--%%| <zSwvDk70Ay|mOf9eDNwlV>
# %%

# Plotting
plt.figure(figsize=(10, 6))

# Scatter plot for Learning Green
plt.scatter(
    percistence["Persistence Score"],
    percistence["learningGreen"],
    color="green",
    label="Learning Green",
)
plt.show()
# Scatter plot for Learning Red
# %%
plt.scatter(
    percistence["Persistence Score"],
    percistence["learningRed"],
    color="red",
    label="Learning Red",
)

# Adding labels and title
plt.xlabel("Persistence Score")
plt.ylabel("Learning Score")
plt.title("Persistence Score vs Learning Score")
plt.legend()

# Show plot
plt.show()
# |%%--%%| <mOf9eDNwlV|CVwlJ9uY8v>


# %%
# Plotting
plt.figure(figsize=(10, 6))

# Scatter plot for Learning Green
plt.scatter(
    percistence["Persistence Score"],
    percistence["learningGreen"],
    color="green",
    label="Learning Green",
)

# Scatter plot for Learning Red
plt.scatter(
    percistence["Persistence Score"],
    percistence["learningRed"],
    color="red",
    label="Learning Red",
)

# Adding fitting lines using seaborn's lmplot
sns.regplot(
    x="Persistence Score",
    y="learningGreen",
    data=percistence,
    scatter=False,
    color="green",
)
sns.regplot(
    x="Persistence Score", y="learningRed", data=percistence, scatter=False, color="red"
)

# Adding labels and title
plt.xlabel("Persistence Score")
plt.ylabel("Learning Score")
plt.title("Persistence Score vs Learning Score")
plt.legend()
plt.savefig("Persistence_Score_vs_Learning_Score.png")
# Show plot
plt.show()


# |%%--%%| <CVwlJ9uY8v|bjMFu3kNZl>
# %%
plt.scatter(data=percistence, x="Persistence Score", y="learningScore")
sns.regplot(
    x="Persistence Score",
    y="learningScore",
    data=percistence,
    scatter=False,
    color="black",
)
# Adding labels and title
plt.xlabel("Persistence Score", fontsize=30)
plt.ylabel("Learning Score", fontsize=30)
plt.title("Persistence Score vs Learning Score", fontsize=40)
plt.legend()
plt.savefig("Persistence_Score_vs_Learning_Score.png")
# Show plot
plt.show()

# |%%--%%| <bjMFu3kNZl|tnRAsPqMvM>
# %%


import statsmodels.api as sm

# Define the independent variables (Xs) and dependent variables (Ys)
X = percistence[["Persistence Score"]]
Y_green = percistence["learningGreen"]
Y_red = percistence["learningRed"]
Y = percistence["learningScore"]
# Add a constant to the independent variables for the intercept term
X = sm.add_constant(X)

# Fit the multiple linear regression models
model_green = sm.OLS(Y_green, X).fit()
model_red = sm.OLS(Y_red, X).fit()
model = sm.OLS(Y, X).fit()
# Print the summary of the regression results
print("Regression Results for Learning Green:")
print(model_green.summary())

print("\nRegression Results for Learning Red:")
print(model_red.summary())

# |%%--%%| <tnRAsPqMvM|9b0P2daokY>

# %%
print("\nRegression Results for Learning Score:")
print(model.summary())

# |%%--%%| <9b0P2daokY|KjTNRknrah>
# %%

df_green = df[df.color == "green"]
df_red = df[df.color == "red"]

# |%%--%%| <KjTNRknrah|NfyGTpWFos>
# %%

# For df_green
df_green_last_40_trials = df_green.groupby(["sub_number", "proba"]).tail(40)
df_green_last_40_trials
# |%%--%%| <MeklYuccDP|iLt0dZ28I6>

# Create a dictionary to map each subject to a specific color
# %%
subject_colors = {
    sub: sns.color_palette(
        "husl", n_colors=len(df_green_last_40_trials["sub_number"].unique())
    )[i]
    for i, sub in enumerate(sorted(df_green_last_40_trials["sub_number"].unique()))
}

# Plot mean velocity for each combination of 'sub_number' and 'proba', manually assigning colors
ax = sns.catplot(
    x="proba",
    y="meanVelo",
    hue="sub_number",
    kind="point",
    data=df_green,
    palette=subject_colors,
    legend=False,
)
plt.xlabel("Probability")
plt.ylabel("Mean Velocity")
plt.title("Mean Velocity (Green) across the last 40 Trials \n for Each Sub and Proba")

# Get the current axes
ax = plt.gca()

# Manually create legend with subject labels and corresponding colors
handles = [
    plt.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor=subject_colors[sub],
        markersize=10,
        label=f"Sub {sub}",
    )
    for sub in sorted(df_green["sub_number"].unique())
]
ax.legend(handles=handles, title="Subject", loc="upper right", fontsize="small")

plt.show()


# %%


# Create a dictionary to map each subject to a specific color
subject_colors = {
    sub: sns.color_palette("husl", n_colors=len(df_red["sub_number"].unique()))[i]
    for i, sub in enumerate(sorted(df_red["sub_number"].unique()))
}

# Plot mean velocity for each combination of 'sub_number' and 'proba', manually assigning colors
ax = sns.catplot(
    x="proba",
    y="meanVelo",
    hue="sub_number",
    kind="point",
    data=df_red,
    palette=subject_colors,
    legend=False,
)
plt.xlabel("Probability")
plt.ylabel("Mean Velocity")
plt.title("Mean Velocity across 240 Trials for Each Sub and Proba")

# Get the current axes
ax = plt.gca()

# Manually create legend with subject labels and corresponding colors
handles = [
    plt.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor=subject_colors[sub],
        markersize=10,
        label=f"Sub {sub}",
    )
    for sub in sorted(df_red["sub_number"].unique())
]
ax.legend(handles=handles, title="Subject", loc="upper right", fontsize="small")

plt.show()

# |%%--%%| <ibJic2WQyh|QdkOZdFx9N>
# %%

df_green_last_40_trials.proba.unique()

# |%%--%%| <QdkOZdFx9N|iu5cRdvDKF>
# %%

l_green = (
    df_green_last_40_trials.groupby(["sub_number", "proba"])
    .meanVelo.mean()
    .reset_index()
)
l_green

# |%%--%%| <iu5cRdvDKF|xDbH5HWsVR>
# %%

df_red_last_40_trials = df_red.groupby(["sub_number", "proba"]).tail(40)
df_red_last_40_trials

# |%%--%%| <xDbH5HWsVR|uvEDDrZ7B8>
# %%

df_red_last_40_trials.proba.unique()

# |%%--%%| <uvEDDrZ7B8|qeUJHOs7qz>

l_red = (
    df_red_last_40_trials.groupby(["sub_number", "proba"]).meanVelo.mean().reset_index()
)
l_red


# %%
# |%%--%%| <qeUJHOs7qz|u41mblaBsi>

# Plot the last 40 trials for each color across the 3 probabilities:
# Concatenate the two DataFrames and create a new column 'group' to distinguish between them
df_green_last_40_trials["color"] = "Green"
df_red_last_40_trials["color"] = "Red"
las40Trials = pd.concat([df_green_last_40_trials, df_red_last_40_trials])

# Plot the boxplot
sns.boxplot(
    x="proba",
    y="meanVelo",
    hue="color",
    data=las40Trials,
    palette={"Green": "green", "Red": "red"},
)
plt.show()
# |%%--%%| <u41mblaBsi|CHoiD1szdN>


# %%
df.columns


df.trial_RT_colochoice
RT = (
    df.groupby(["sub_number", "proba"])
    .trial_RT_colochoice.mean()
    .reset_index()["trial_RT_colochoice"]
)


# %%
plt.hist(RT, color="lightblue", edgecolor="black")
plt.vlines(RT.mean(), 0, 10, color="red", linestyle="--", label="Mean RT")
# plt.vlines(0.6, 0, 10, color='black', label='Mean RT', linewidth=2,label="Vanessa's Exp")
plt.legend()
plt.xlabel("RT", fontsize=40)
plt.title("RT Distribution", fontsize=40)
plt.savefig("RT_Distribution.png")
plt.show()

# %%
df.trial_color_chosen == df.trial_color_UP


df["arrowChosen"] = df.trial_color_chosen == df.trial_color_UP


df.arrowChosen = ["UP" if x == True else "DOWN" for x in df.arrowChosen]


df.arrowChosen

# %%

df[(df.sub_number == 16) & (df.proba == 75) & (df.arrowChosen == "UP")]
