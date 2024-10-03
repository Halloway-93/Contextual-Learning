import io
import os
import re
from datetime import datetime
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import researchpy as rp
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.diagnostic import het_white
from statsmodels.stats.anova import AnovaRM


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


def process_raw_data(data):

    df = data["raw"]
    mono = data["info"]["mono"]
    MSG = data["msg"]
    Zero = MSG.loc[MSG.text == "TargetOnSet", ["trial", "time"]]
    Sacc = data["sacc"]

    for t in Zero.trial:
        Sacc.loc[Sacc.trial == t, ["stime", "etime"]] = (
            Sacc.loc[Sacc.trial == t, ["stime", "etime"]].values
            - Zero.loc[Zero.trial == t, "time"].values
        )

    for i in range(len(Zero)):
        df.loc[df["trial"] == i + 1, "time"] = (
            df.loc[df["trial"] == i + 1, "time"] - Zero.time.values[i]
        )
    numeric_columns = ["trial", "time"]
    if not mono:
        numeric_columns.extend(["xpl", "ypl", "psl", "xpr", "ypr", "psr"])
    else:
        numeric_columns.extend(["xp", "yp", "ps"])

    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors="coerce")

    return df


def process_all_raw_data(data_dir):
    allRawData = []
    # allEvents = []

    for root, _, files in sorted(os.walk(data_dir)):
        for filename in sorted(files):
            if filename.endswith(".asc"):
                filepath = os.path.join(root, filename)
                print(f"Read data from {filepath}")
                data = read_asc(filepath)
                df = process_raw_data(data)
                allRawData.append(df)

            if filename.endswith(".csv"):
                filepath = os.path.join(root, filename)
                print(f"Read data from {filepath}")
                events = pd.read_csv(filepath)
                proba = events["proba"].unique().values[-1]  # print(len(events))
                # allEvents.append(events)
            # Adding the proba to the raw data
            df.proba = proba

    bigDF = pd.concat(allRawData, axis=0, ignore_index=True)
    bigDF.to_csv("rawData.csv", index=False)


def process_data_file(f):
    # Read data from file
    data = read_asc(f)
    sampling_rate = data["info"]["sample.rate"]
    deg = 27.28  # pixel to degree conversion
    # Extract relevant data from the DataFrame
    df = process_raw_data(data)
    mono = data["info"]["mono"]

    # # Convert columns to numeric
    # numeric_columns = ["trial", "time"]
    # if not mono:
    #     numeric_columns.extend(["xpl", "ypl", "psl", "xpr", "ypr", "psr"])
    # else:
    #     numeric_columns.extend(["xp", "yp", "ps"])
    #
    # df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors="coerce")

    # Drop rows where trial is equal to 1
    # df = df[df["trial"] != 1]

    # Decrement the values in the 'trial' column by 1
    # df.loc[:, "trial"] = df["trial"] - 1

    # Reset index after dropping rows and modifying the 'trial' column
    # df = df.reset_index(drop=True)

    # Extract messages from eyelink
    MSG = data["msg"]
    tON = MSG.loc[MSG.text == "FixOn", ["trial", "time"]]
    t0 = MSG.loc[MSG.text == "FixOff", ["trial", "time"]]
    Zero = MSG.loc[MSG.text == "TargetOnSet", ["trial", "time"]]
    tOFF = MSG.loc[MSG.text == "TargetOffSet", ["trial", "time"]]
    # Reset time based on 'Zero' time
    for i in range(len(Zero)):
        df.loc[df["trial"] == i + 1, "time"] = (
            df.loc[df["trial"] == i + 1, "time"] - Zero.time.values[i]
        )

    # target off set
    TOFF = []
    if len(tOFF) < len(Zero):
        TOFF = np.array([600] * len(Zero))
    else:
        TOFF = tOFF.time.values - Zero.time.values
    SON = tON.time.values - Zero.time.values
    SOFF = t0.time.values - Zero.time.values
    # ZEROS = Zero.time.values

    # Extract saccades data
    # Sacc = data["sacc"]

    # # Reset saccade times
    # for t in Zero.trial:
    #     Sacc.loc[Sacc.trial == t, ["stime", "etime"]] = (
    #         Sacc.loc[Sacc.trial == t, ["stime", "etime"]].values
    #         - Zero.loc[Zero.trial == t, "time"].values
    #     )

    # Sacc = drop_bad_trials(Sacc, badTrials)

    # Extract trials with saccades within the time window [0, 80ms]
    # Sacc[(Sacc.stime >= -200) & (Sacc.etime < 80) & (Sacc.eye == "R")]["trial"].values
    #
    # np.sign(
    #     (
    #         Sacc[(Sacc.stime >= -200) & (Sacc.etime < 80) & (Sacc.eye == "R")].exp
    #         - Sacc[(Sacc.stime >= -200) & (Sacc.etime < 80) & (Sacc.eye == "R")].sxp
    #     ).values
    # )
    #
    # for t in Sacc.trial.unique():
    #     start = Sacc.loc[(Sacc.trial == t) & (Sacc.eye == "R"), "stime"]
    #     end = Sacc.loc[(Sacc.trial == t) & (Sacc.eye == "R"), "etime"]
    #
    #     for i in range(len(start)):
    #         if not mono:
    #             df.loc[
    #                 (df.trial == t)
    #                 & (df.time >= start.iloc[i] - 20)
    #                 & (df.time <= end.iloc[i] + 20),
    #                 "xpr",
    #             ] = np.nan
    #         else:
    #             df.loc[
    #                 (df.trial == t)
    #                 & (df.time >= start.iloc[i] - 20)
    #                 & (df.time <= end.iloc[i] + 20),
    #                 "xp",
    #             ] = np.nan

    # Extract first bia
    # first_bias = np.where(bias == 1)[0][0]

    # Extract position and velocity data
    selected_values = (
        df.xpr[(df.time >= 80) & (df.time <= 120)]
        if not mono
        else df.xp[(df.time >= 80) & (df.time <= 120)]
    )
    posSteadyState = df.xpr[(df.time >= 300)] if not mono else df.xp[(df.time >= 300)]
    veloSteadyState = np.gradient(posSteadyState.values)  # Rescale position
    pos_before = (
        df.xpr[(df.time >= -40) & (df.time <= 0)]
        if not mono
        else df.xp[(df.time >= -40) & (df.time <= 0)]
    )

    time_dim = 41
    trial_dim = len(selected_values) // time_dim

    pos = np.array(selected_values[: time_dim * trial_dim]).reshape(trial_dim, time_dim)
    stdPos = np.std(pos, axis=1)

    pos_before_reshaped = np.array(pos_before[: time_dim * trial_dim]).reshape(
        trial_dim, time_dim
    )
    pos_before_mean = np.nanmean(pos_before_reshaped, axis=1)
    # Reshaping veloSteadyState
    veloSteadyState = np.array(veloSteadyState[: trial_dim * time_dim]).reshape(
        trial_dim, time_dim
    )
    velo = np.gradient(pos, axis=1) * sampling_rate / deg  # deg/s
    # velo[(velo > 20) | (velo < -20)] = np.nan

    for i, pp in enumerate(pos_before_mean):
        if pd.notna(pp):
            pos[i] = pos[i] - pp

    meanPos = np.nanmean(pos, axis=1)
    meanVelo = np.nanmean(velo, axis=1)
    stdVelo = np.std(velo, axis=1)
    meanVSS = (
        np.nanmean(veloSteadyState, axis=1) * sampling_rate / deg
    )  # TS = trialSacc

    return pd.DataFrame(
        {
            "SON": SON,
            "SOFF": SOFF,
            "TOFF": TOFF,
            "meanPos": meanPos,
            "stdPos": stdPos,
            "meanVelo": meanVelo,
            "stdVelo": stdVelo,
            "meanVSS": meanVSS,
            # "TS": TS,
            # "SaccD": SaccD,
            # "SACC": SACC
        }
    )


def process_all_asc_files(data_dir):
    allDFs = []
    allEvents = []

    for root, _, files in sorted(os.walk(data_dir)):
        for filename in sorted(files):
            if filename.endswith(".asc"):
                filepath = os.path.join(root, filename)
                print(f"Read data from {filepath}")
                # cell                data = process_data_file(filepath)
                # Extract proba from filename

                allDFs.append(data)
                print(len(data))

            if filename.endswith(".csv"):
                filepath = os.path.join(root, filename)
                print(f"Read data from {filepath}")
                events = pd.read_csv(filepath)
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

    return merged_data


# %%
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
dirPath = "/Volumes/work/brainets/oueld.h/contextuaLearning/directionCue/results_voluntaryDirection/"
dirFig = "/Volumes/work/brainets/oueld.h/contextuaLearning/directionCue/results_voluntaryDirection/figures/"
# %%
filePath = "~/boubou/contextuaLearning/directionCue/results_voluntaryDirection/sub-002/session-04/sub-002_ses-04_proba-100.asc"

# %%
eventsPath = "~/boubou/contextuaLearning/directionCue/results_voluntaryDirection/sub-002/session-04/sub-002_ses-04_proba-100.csv"
# %%
data = read_asc(filePath)
# %%
df = data["raw"]
df.head()

# %%
mono = data["info"]["mono"]
events = pd.read_csv(eventsPath)
events
# %%
data["info"]
# %%
MSG = data["msg"]
tON = MSG.loc[MSG.text == "FixOn", ["trial", "time"]]
t0 = MSG.loc[MSG.text == "FixOff", ["trial", "time"]]
Zero = MSG.loc[MSG.text == "TargetOnSet", ["trial", "time"]]
Zero
# %%
# Extract saccades data
Sacc = data["sacc"]

# Reset saccade times
for t in Zero.trial:
    Sacc.loc[Sacc.trial == t, ["stime", "etime"]] = (
        Sacc.loc[Sacc.trial == t, ["stime", "etime"]].values
        - Zero.loc[Zero.trial == t, "time"].values
    )


# %%
Sacc[Sacc.trial == 36]
# %%
print(Sacc)
# Keeping only the saccades inside the window we're interested in: [-200, 600] ms
Sacc = Sacc[(Sacc.stime >= -200) & (Sacc.etime <= 150)]
# %%
sns.histplot(Sacc.dur.values)
plt.show()
# %%
badTrial = int(Sacc[Sacc.dur == Sacc.dur.max()].trial.values[0])
badTrial
# %%
Sacc[Sacc.trial == badTrial]
# %%
# for t in Sacc.trial.unique():
#     start = Sacc.loc[(Sacc.trial == t) & (Sacc.eye == "R"), "stime"]
#     end = Sacc.loc[(Sacc.trial == t) & (Sacc.eye == "R"), "etime"]
#
#     for i in range(len(start)):
#         if not mono:
#             df.loc[
#                 (df.trial == t)
#                 & (df.time >= start.iloc[i] - 5)
#                 & (df.time <= end.iloc[i] + 10),
#                 "xpr",
#             ] = np.nan
#         else:
#             df.loc[
#                 (df.trial == t)
#                 & (df.time >= start.iloc[i] - 5)
#                 & (df.time <= end.iloc[i] + 10),
#                 "xp",
#             ] = np.nan

# %%
tOFF = MSG.loc[MSG.text == "TargetOffSet", ["trial", "time"]]
tOFF
# %%
tOFF.time = tOFF.time.values - Zero.time.values
tOFF
# %%
gap = Zero.time.values - t0.time.values
# gap
# %%
for i in range(len(Zero)):
    df.loc[df["trial"] == i + 1, "time"] = (
        df.loc[df["trial"] == i + 1, "time"] - Zero.time.values[i]
    )
# %%
# Convert columns to numeric
numeric_columns = ["trial", "time"]
if not mono:
    numeric_columns.extend(["xpl", "ypl", "psl", "xpr", "ypr", "psr"])
else:
    numeric_columns.extend(["xp", "yp", "ps"])

df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors="coerce")

# %%
trial = 220
print(-gap[trial - 1])
plt.subplot(1, 2, 1)
velT1 = np.diff(df[(df.trial == 99) & (df.time > 0) & (df.time < 120)].xp)
# plt.plot(np.convolve(velT1*1000/27.28, np.ones(10)/10, mode='valid'))
# plt.plot(velT1)
velT1 = np.diff(
    df[(df.trial == trial) & (df.time > -gap[trial - 1]) & (df.time < 120)].xp
)
plt.plot(np.convolve(velT1 * 1000 / 27.28, np.ones(60) / 60, mode="valid"))
# plt.plot(velT1*1000/27.28)
plt.subplot(1, 2, 2)
plt.plot(
    df[(df.trial == trial) & (df.time > -gap[trial - 1]) & (df.time < 120)].time,
    df[(df.trial == trial) & (df.time > -gap[trial - 1]) & (df.time < 120)].xp,
)
plt.show()
## %%
trial = badTrial
trial
# %%
# Testin detect_saccades
sac = detect_saccades(df, mono)
sac
# %%
# getting only the saccasde in the window -200 to 150
sac = sac[(sac.start >= -200) & (sac.end <= 150)]
# %%
# Extract position and velocity data
t = df[
    (df.trial == trial)
    & (df.time > -gap[trial - 1])
    & (df.time < tOFF.time.iloc[trial - 1])
].time.values
velT1 = np.gradient(
    df[
        (df.trial == trial)
        & (df.time > -gap[trial - 1])
        & (df.time <= tOFF.time.iloc[trial - 1])
    ].xp
)

# Perform the convolution on the velocity data

convolved_velT1 = np.convolve(velT1 * 1000 / 27.28, np.ones(20) / 20, mode="valid")
# %%
# Generate a new time axis with the same number of points as the convolved data
new_t = np.linspace(t.min(), t.max(), len(convolved_velT1))
# %%
plt.figure(figsize=(20, 12))
plt.subplot(2, 1, 1)
plt.xlabel("Time (ms)", fontsize=20)
plt.ylabel("Velocity (deg/s)", fontsize=20)
plt.plot(new_t, convolved_velT1)
# plt.plot(velT1*1000/27.28)
plt.subplot(2, 1, 2)
plt.plot(
    df[
        (df.trial == trial)
        & (df.time > -gap[trial - 1])
        & (df.time <= tOFF.time.iloc[trial - 1])
    ].time,
    df[
        (df.trial == trial)
        & (df.time > -gap[trial - 1])
        & (df.time <= tOFF.time.iloc[trial - 1])
    ].xp,
)

plt.xlabel("Time (ms)", fontsize=20)
plt.ylabel("Position (deg)", fontsize=20)
plt.suptitle(f"Trial{trial}:Top Velocity, Bottom Position", fontsize=40)
plt.show()
# %%


trials = [100]  # Replace with the actual trial numbers you want to plot

plt.figure(figsize=(20, 12))

# Subplot for velocities
plt.subplot(2, 1, 1)
plt.xlabel("Time (ms)", fontsize=20)
plt.ylabel("Velocity (deg/s)", fontsize=20)

# Subplot for positions
plt.subplot(2, 1, 2)
plt.xlabel("Time (ms)", fontsize=20)
plt.ylabel("Position (deg)", fontsize=20)

for trial in trials:
    # Filter the dataframe based on the trial and time conditions
    filtered_df = df[
        (df.trial == trial)
        & (df.time > -gap[trial - 1])
        & (df.time <= tOFF.time.iloc[trial - 1])
    ]

    # Extract the time and velocity data
    t = filtered_df.time.values
    velT1 = np.diff(
        filtered_df.xp
    )  # Assuming velT1 is calculated as the difference of xp

    # Perform the convolution on the velocity data
    convolved_velT1 = np.convolve(velT1 * 1000 / 27.44, np.ones(60) / 60, mode="valid")

    # Generate a new time axis with the same number of points as the convolved data
    new_t = np.linspace(t.min(), t.max(), len(convolved_velT1))

    # Plot the convolved velocity data
    plt.subplot(2, 1, 1)
    plt.plot(new_t, convolved_velT1, label=f"Trial {trial}")

    # Plot the original position data
    plt.subplot(2, 1, 2)
    plt.plot(filtered_df.time, filtered_df.xp, label=f"Trial {trial}")

# Add legend to the plots
plt.subplot(2, 1, 1)
plt.legend(fontsize=15)

plt.subplot(2, 1, 2)
plt.legend(fontsize=15)

# Add a title to the figure
plt.suptitle("Multiple Trials: Top Velocity, Bottom Position", fontsize=40)

plt.tight_layout()
plt.show()

# %%
# Focusing on the Anticipatory part: -200 t0 120 ms
t = df[(df.trial == trial) & (df.time > -gap[trial - 1]) & (df.time < 120)].time.values
velT1 = np.diff(
    df[(df.trial == trial) & (df.time > -gap[trial - 1]) & (df.time <= 120)].xp
)
# Perform the convolution on the velocity data
convolved_velT1 = np.convolve(velT1 * 1000 / 27.28, np.ones(20) / 20, mode="valid")
# %%
# Generate a new time axis with the same number of points as the convolved data
new_t = np.linspace(t.min(), t.max(), len(convolved_velT1))
# %%
plt.figure(figsize=(20, 12))
plt.subplot(2, 1, 1)
plt.xlabel("Time (ms)", fontsize=20)
plt.ylabel("Velocity (deg/s)", fontsize=20)
plt.plot(new_t, convolved_velT1)
# plt.plot(velT1*1000/27.28)
plt.subplot(2, 1, 2)
plt.plot(
    df[(df.trial == trial) & (df.time > -gap[trial - 1]) & (df.time <= 120)].time,
    df[(df.trial == trial) & (df.time > -gap[trial - 1]) & (df.time <= 120)].xp,
)
plt.xlabel("Time (ms)", fontsize=20)
plt.ylabel("Position (deg)", fontsize=20)
plt.suptitle(f"Trial{trial}:Anticipatory Interval", fontsize=40)
plt.show()
df.head()

# %%
np.mean(convolved_velT1)
# %%
# Separating the trials based on the chosen arrowChosen
events
# %%
np.mean(events.RT.values)
sns.histplot(events.RT)
plt.show()
# %%
chosenUpTrials = events[events.chosen_arrow == "up"].index.values + 1
chosenDownTrials = events[events.chosen_arrow == "down"].index.values + 1
# %%
trials = chosenUpTrials[chosenUpTrials > 1]

plt.figure(figsize=(20, 12))

# Subplot for velocities
plt.subplot(2, 1, 1)
plt.xlabel("Time (ms)", fontsize=20)
plt.ylabel("Velocity (deg/s)", fontsize=20)

# Subplot for positions
plt.subplot(2, 1, 2)
plt.xlabel("Time (ms)", fontsize=20)
plt.ylabel("Position (deg)", fontsize=20)
allVelUp = []
for trial in trials:
    # Filter the dataframe based on the trial and time conditions
    filtered_df = df[
        (df.trial == trial) & (df.time > -gap[trial - 1]) & (df.time <= 120)
    ]

    # Extract the time and velocity data
    t = filtered_df.time.values
    velT1 = (
        np.gradient(filtered_df.xp) * 1000 / 27.44
    )  # Assuming velT1 is calculated as the difference of xp
    # Perform the convolution on the velocity data
    convolved_velT1 = np.convolve(velT1, np.ones(20) / 20, mode="valid")

    allVelUp.append(np.mean(velT1))
    # Generate a new time axis with the same number of points as the convolved data
    new_t = np.linspace(t.min(), t.max(), len(convolved_velT1))

    # Plot the convolved velocity data
    plt.subplot(2, 1, 1)
    plt.plot(new_t, convolved_velT1, label=f"Trial {trial}")

    # Plot the original position data
    plt.subplot(2, 1, 2)
    plt.plot(filtered_df.time, filtered_df.xp, label=f"Trial {trial}")

# Add legend to the plots
plt.subplot(2, 1, 1)
plt.legend(fontsize=15)

plt.subplot(2, 1, 2)
plt.legend(fontsize=15)

# Add a title to the figure
plt.suptitle(
    r"Chosen arrow is Up ($\mathbb{P}(Right|UP)$"
    + f"={events.proba.unique()[0]}) :Velocity(Top) & Position(Bottom)",
    fontsize=30,
)

plt.tight_layout()
plt.show()
# %%
sns.histplot(allVelUp)
plt.show()
# %%
sns.boxplot(allVelUp)
plt.show()

# %%
np.nanmean(allVelUp)
# %%

# Arrow chosen is pointing up
trials = chosenUpTrials[chosenUpTrials > 50]
allVelUp = []
# Plotting the trials Separately
for trial in trials:
    # Create a new figure for each trial
    plt.figure(figsize=(20, 12))

    # Filter the dataframe based on the trial and time conditions
    filtered_df = df[
        (df.trial == trial) & (df.time > -gap[trial - 1]) & (df.time <= 120)
    ]

    # Extract the time and position data
    t = filtered_df.time.values
    xp = filtered_df.xp.values

    # Compute velocity using np.gradient
    velT1 = np.gradient(xp) * 1000 / 27.28  # Convert to deg/s

    # Perform the convolution on the velocity data with a filter length of 40 data points
    filter_length = 20
    convolved_velT1 = np.convolve(
        velT1, np.ones(filter_length) / filter_length, mode="valid"
    )
    allVelUp.append(np.mean(velT1))
    # Generate a new time axis with the same number of points as the convolved data
    new_t = np.linspace(t.min(), t.max(), len(convolved_velT1))
    # Calculate the time shift introduced by the convolution
    time_shift = filter_length // 2
    # Plot the convolved velocity data
    plt.subplot(2, 1, 1)
    plt.plot(new_t, convolved_velT1, label=f"Trial {trial}")
    plt.xlabel("Time (ms)", fontsize=20)
    plt.ylabel("Velocity (deg/s)", fontsize=20)
    plt.title(f"Trial {trial}: Velocity", fontsize=25)
    plt.legend(fontsize=15)

    # Plot the original position data
    plt.subplot(2, 1, 2)
    plt.plot(filtered_df.time, filtered_df.xp, label=f"Trial {trial}")
    plt.xlabel("Time (ms)", fontsize=20)
    plt.ylabel("Position (deg)", fontsize=20)
    plt.title(f"Trial {trial}: Position", fontsize=25)
    plt.legend(fontsize=15)

    # Add saccade intervals
    saccades = sac[sac.trial == trial]
    for _, saccade in saccades.iterrows():
        stime = saccade["start"]
        etime = saccade["end"]

        # Adjust the saccade intervals for the convolved velocity plot
        adjusted_stime = stime + time_shift
        adjusted_etime = etime + time_shift

        # Add shaded region for saccade intervals on the velocity plot
        plt.subplot(2, 1, 1)
        # plt.axvspan(adjusted_stime, adjusted_etime, color="red", alpha=0.3)

        plt.axvspan(stime, etime, color="red", alpha=0.3)
        # Add shaded region for saccade intervals on the position plot
        plt.subplot(2, 1, 2)
        plt.axvspan(stime, etime, color="red", alpha=0.3)
    # Add a title to the figure
    plt.suptitle(
        f"Trial {trial}: Chosen arrow-(UP) \n P(Right|UP)= {events.proba.values[0]} ",
        fontsize=40,
    )

    plt.tight_layout()
    plt.show()
# %%
# Arrow chosed is down
trials = chosenDownTrials[chosenDownTrials > 50]
allVelDown = []
# Plotting the trials Separately
for trial in trials:
    # Create a new figure for each trial
    plt.figure(figsize=(20, 12))

    # Filter the dataframe based on the trial and time conditions
    filtered_df = df[
        (df.trial == trial) & (df.time > -gap[trial - 1]) & (df.time <= 120)
    ]

    # Extract the time and position data
    t = filtered_df.time.values
    xp = filtered_df.xp.values

    # Compute velocity using np.gradient
    velT1 = np.gradient(xp) * 1000 / 27.28  # Convert to deg/s

    # Perform the convolution on the velocity data with a filter length of 40 data points
    filter_length = 20
    convolved_velT1 = np.convolve(
        velT1, np.ones(filter_length) / filter_length, mode="valid"
    )
    allVelDown.append(np.mean(convolved_velT1))
    # Generate a new time axis with the same number of points as the convolved data
    new_t = np.linspace(t.min(), t.max(), len(convolved_velT1))

    # Plot the convolved velocity data
    plt.subplot(2, 1, 1)
    plt.plot(new_t, convolved_velT1, label=f"Trial {trial}")
    plt.xlabel("Time (ms)", fontsize=20)
    plt.ylabel("Velocity (deg/s)", fontsize=20)
    plt.title(f"Trial {trial}: Velocity", fontsize=25)
    plt.legend(fontsize=15)

    # Plot the original position data
    plt.subplot(2, 1, 2)
    plt.plot(filtered_df.time, filtered_df.xp, label=f"Trial {trial}")
    plt.xlabel("Time (ms)", fontsize=20)
    plt.ylabel("Position (deg)", fontsize=20)
    plt.title(f"Trial {trial}: Position", fontsize=25)
    plt.legend(fontsize=15)

    # Add saccade intervals
    saccades = sac[sac.trial == trial]
    for _, saccade in saccades.iterrows():
        stime = saccade["start"]
        etime = saccade["end"]

        # Adjust the saccade intervals for the convolved velocity plot
        adjusted_stime = stime + time_shift
        adjusted_etime = etime + time_shift

        # Add shaded region for saccade intervals on the velocity plot
        plt.subplot(2, 1, 1)
        # plt.axvspan(adjusted_stime, adjusted_etime, color="red", alpha=0.3)

        plt.axvspan(stime, etime, color="red", alpha=0.3)
        # Add shaded region for saccade intervals on the position plot
        plt.subplot(2, 1, 2)
        plt.axvspan(stime, etime, color="red", alpha=0.3)
    # Add a title to the figure
    plt.suptitle(
        f"Trial {trial}: Chosen arrow-(DOWN) \n P(Right|UP)= {events.proba.values[0]} ",
        fontsize=40,
    )

    plt.tight_layout()
    plt.show()
# %%
sns.histplot(allVelDown)
plt.show()
# %%
plt.figure(figsize=(12, 10))
sns.histplot(allVelDown, label="Down")
sns.histplot(allVelUp, label="Up")
plt.title(r"$\mathbb{P}(Right|Up)=" + f"{events.proba.unique()[-1]}$", fontsize=30)
plt.xlabel("Anticipatory Velocity", fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend()
plt.show()

# %%
# Looking at the position offest on the x-axis during the Anticipatory window ([-200, 120] ms)
# Arrow chosed is down
trials = chosenDownTrials[chosenDownTrials > 0]
allPosDown = []
# Plotting the trials Separately
for trial in trials:

    # Filter the dataframe based on the trial and time conditions
    filtered_df = df[
        (df.trial == trial) & (df.time > -gap[trial - 1]) & (df.time <= 120)
    ]

    # Extract the time and position data
    t = filtered_df.time.values
    xp = filtered_df.xp.values
    allPosDown.append(xp[-1] - xp[0])

# %%
trials = chosenUpTrials[chosenUpTrials > 0]
allPosUp = []
# Plotting the trials Separately
for trial in trials:

    # Filter the dataframe based on the trial and time conditions
    filtered_df = df[
        (df.trial == trial) & (df.time > -gap[trial - 1]) & (df.time <= 120)
    ]

    # Extract the time and position data
    t = filtered_df.time.values
    xp = filtered_df.xp.values
    allPosUp.append(xp[-1] - xp[0])


# %%
plt.figure(figsize=(12, 10))
sns.histplot(allPosDown, label="Down")
sns.histplot(allPosUp, label="Up")
plt.title(r"$\mathbb{P}(Right|Up)=" + f"{events.proba.unique()[-1]}$", fontsize=30)
plt.xlabel("Position Offset", fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=14)
plt.show()


# %%
plt.subplot(1, 2, 1)

sns.histplot(allPosDown, label="Down")
sns.histplot(allPosUp, label="Up")
plt.title(r"$\mathbb{P}(Right|Up)=" + f"{events.proba.unique()[-1]}$", fontsize=30)
plt.xlabel("Position Offset", fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)

plt.subplot(1, 2, 2)
sns.histplot(allVelDown, label="Down")
sns.histplot(allVelUp, label="Up")
plt.title(r"$\mathbb{P}(Right|Up)=" + f"{events.proba.unique()[-1]}$", fontsize=30)
plt.xlabel("Anticipatory Velocity", fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)
plt.show()
plt.savefig("positionOffsetVsAntiVelo.png")

# %%
# Reading the full data.
data = pd.read_csv(dirPath + "rawData.csv")

# %%
data.head()
# %%
data.drop(columns=["cr.info"], inplace=True)
# %%
# Getting the time window of ineterst.
filtered_data = data[(data["time"] > -200) & (data["time"] <= 120)]
filtered_data
# %%
filtered_data.loc[:, "time"] = pd.to_numeric(filtered_data["time"], errors="coerce")
# %%
filtered_data
# %%
allEvents = pd.read_csv(dirPath + "allEvents.csv")
allEvents
# %%
allEvents.loc[(allEvents["sub"] == 1), "training"] = "no"
allEvents
# %%
unique_sub = allEvents["sub"].unique()
# %%
filtered_sacc = detect_saccades(filtered_data)
filtered_sacc
# %%
# Getting the Position Offset for each Participant and each proba
df_results = []
for sub in unique_sub:
    unique_prob = allEvents[allEvents["sub"] == sub]["proba"].unique()
    for p in unique_prob:
        trials = filtered_data[
            (filtered_data["sub"] == sub) & (filtered_data["proba"] == p)
        ]["trial"].unique()
        # getting rid off the training trials.
        if len(trials) > 240:
            trials = trials[trials > len(trials) - 240]
        for trial in trials:
            # Filter the dataframe based on the trial and time conditions
            xPos = filtered_data[
                (filtered_data["trial"] == trial)
                & (filtered_data["sub"] == sub)
                & (filtered_data["proba"] == p)
            ]
            # print(xPos[xPos["time"] >= 80].values)
            meanVelo = np.nanmean(
                np.gradient(xPos[xPos["time"] >= 80]["xp"].values) * 1000 / 27.28
            )
            # print(meanVelo)
            posOffset = (
                xPos.xp.values[-1] - xPos.xp.values[0]
            )  # Append the result to the DataFrame
            df_results.append(
                {
                    "trial": trial,
                    "sub": sub,
                    "proba": p,
                    "posOffSet": posOffset,
                    "meanVelo": meanVelo,
                }
            )
# %%
df_results = pd.DataFrame(df_results)
df_results
# %%
df_results["sub"].unique()
# %%
# Adding the colum of chosen arrow to the df_results.
for sub in unique_sub:
    for unique_prob in df_results[df_results["sub"] == sub]["proba"].unique():
        # Get the trials for the current subject and probability
        trials = df_results[
            (df_results["sub"] == sub) & (df_results["proba"] == unique_prob)
        ]["trial"].values

        # Filter allEvents to get the chosen_arrow values for the current trials
        eventsOfInterest = allEvents[
            (allEvents["sub"] == sub)
            & (allEvents["proba"] == unique_prob)
            & (allEvents["trial"].isin(trials))
        ]

        # Create a dictionary to map trials to chosen_arrow values
        trial_arrow_map = dict(
            zip(eventsOfInterest["trial"], eventsOfInterest["chosen_arrow"])
        )

        # Update the df_results DataFrame with the chosen_arrow values
        df_results.loc[
            (df_results["sub"] == sub) & (df_results["proba"] == unique_prob), "arrow"
        ] = df_results.loc[
            (df_results["sub"] == sub) & (df_results["proba"] == unique_prob), "trial"
        ].map(
            trial_arrow_map
        )

        # Create a dictionary to map trials to target direction values
        trial_td_map = dict(
            zip(eventsOfInterest["trial"], eventsOfInterest["target_direction"])
        )

        # Update the df_results DataFrame with the chosen_arrow values
        df_results.loc[
            (df_results["sub"] == sub) & (df_results["proba"] == unique_prob), "TD"
        ] = df_results.loc[
            (df_results["sub"] == sub) & (df_results["proba"] == unique_prob), "trial"
        ].map(
            trial_td_map
        )
df_results
# %%
# getting all the saccaades for all participatns and all conditions (proba) in the Anticipatory window.
# allSacc = detect_saccades(filtered_data)
# %%
# allSacc
# %%
# Saving the dataframe to a csv file.
df_results.to_csv(dirPath + "results.csv", index=False)
# %%
# getting the results of the anticipatory position offset and mean velocity on [80, 120] ms
df = pd.read_csv(dirPath + "results.csv")
# Create a DataFrame
df["proba"] = pd.to_numeric(df["proba"], errors="coerce")
df["posOffSet"] = pd.to_numeric(df["posOffSet"], errors="coerce")
df["meanVelo"] = pd.to_numeric(df["meanVelo"], errors="coerce")
# Print the resulting DataFrame
print(df)
# %%
balance = df.groupby(["arrow", "sub", "proba"])["trial"].count().reset_index()
balance
# %%
for sub in balance["sub"].unique():
    sns.barplot(x="proba", y="trial", hue="arrow", data=balance[balance["sub"] == sub])
    plt.title(f"Subject {sub}")
    plt.show()
# %%
# getting previous TD for each trial for each subject and each proba
for sub in df["sub"].unique():
    for p in df[df["sub"] == sub]["proba"].unique():
        df.loc[(df["sub"] == sub) & (df["proba"] == p), "TD_prev"] = df.loc[
            (df["sub"] == sub) & (df["proba"] == p), "TD"
        ].shift(1)
        df.loc[(df["sub"] == sub) & (df["proba"] == p), "arrow_prev"] = df.loc[
            (df["sub"] == sub) & (df["proba"] == p), "arrow"
        ].shift(1)
# %%
df["interaction"] = list(zip(df["TD_prev"], df["arrow_prev"]))
# Clean the data
# df.dropna(subset=["posOffSet", "meanVelo"], inplace=True)
df.dropna(inplace=True)
df = df[(df["posOffSet"] >= -150) & (df["posOffSet"] <= 150)]
df = df[(df["meanVelo"] >= -8) & (df["meanVelo"] <= 8)]
# %%
df
# %%
df[(df.proba == 0) & (df.TD_prev == -1) & (df.arrow == "up")]
# %%
# Need to shift it for each proba and each sub
# df['TD_prev']=df['TD'].shift(1)
# df
# %%
df_prime = df[["trial", "proba", "arrow", "TD_prev", "posOffSet", "meanVelo"]]
learningCurve = (
    df_prime.groupby(["proba", "arrow", "TD_prev"])
    .mean()[["posOffSet", "meanVelo"]]
    .reset_index()
)


learningCurve
# %%
df_prime.groupby(["proba", "arrow", "TD_prev"]).count()[["posOffSet", "meanVelo"]]

# %%
sns.barplot(
    x="proba",
    y="posOffSet",
    hue="TD_prev",
    data=learningCurve[learningCurve.arrow == "up"],
)
plt.title("Position Offset: Arrow UP")
plt.xlabel("P(Right|UP)")
plt.show()
# %%

# %%
sns.barplot(
    x="proba",
    y="posOffSet",
    data=learningCurve[learningCurve.arrow == "up"],
)
plt.title("Position Offset: Arrow UP")
plt.xlabel("P(Right|UP)")
plt.show()
# %%


sns.barplot(
    x="proba",
    y="meanVelo",
    hue="TD_prev",
    data=learningCurve[learningCurve.arrow == "up"],
)
plt.title("Anticipatory Velocity: Arrow UP")
plt.xlabel("P(Right|UP)")
plt.show()
# %%
sns.barplot(
    x="proba",
    y="posOffSet",
    hue="TD_prev",
    data=learningCurve[learningCurve.arrow == "down"],
)
plt.title("Position Offset: Arrow DOWN")
plt.xlabel("P(Left|DOWN)")
plt.show()
# %%
sns.barplot(
    x="proba",
    y="posOffSet",
    data=learningCurve[learningCurve.arrow == "down"],
)
plt.title("Position Offset: Arrow DOWN")
plt.xlabel("P(Left|DOWN)")
plt.show()
# %%
sns.barplot(
    x="proba",
    y="meanVelo",
    hue="TD_prev",
    data=learningCurve[learningCurve.arrow == "down"],
)
plt.title("meanVelo: Arrow DOWN")
plt.xlabel("P(Left|DOWN)")
plt.show()
# %%
learningCurve = (
    df_prime.groupby(["trial", "proba", "arrow", "TD_prev"])
    .mean()[["posOffSet", "meanVelo"]]
    .reset_index()
)
learningCurve
# %%
for p in learningCurve.proba.unique():
    # y = np.convolve(
    #     learningCurve[
    #         (learningCurve.arrow == "up")
    #         & (learningCurve.proba == p)
    #         & (learningCurve.TD_prev == 1)
    #     ].posOffSet,
    #     np.ones(5) / 5,
    #     mode="valid",
    # )
    # plt.plot(y, label=f"Up: {p}, TD_prev=1")
    # y = np.convolve(
    #     learningCurve[
    #         (learningCurve.arrow == "up")
    #         & (learningCurve.proba == p)
    #         & (learningCurve.TD_prev == -1)
    #     ].posOffSet,
    #     np.ones(5) / 5,
    #     mode="valid",
    # )
    sns.scatterplot(
        x="trial",
        y="posOffSet",
        hue="TD_prev",
        palette="viridis",
        data=learningCurve[(learningCurve.arrow == "up") & (learningCurve.proba == p)],
    )
    # plt.plot(y, label=f"Up: {p}, TD_prev=-1")
    plt.legend()
    plt.title(f"Learning Curve: Arrow UP P(Right|UP)={p}")
    plt.show()
# %%
# %%
df_prime = df[["trial", "proba", "arrow", "interaction", "posOffSet", "meanVelo"]]
df_prime
# %%
# %%
learningCurve = (
    df_prime.groupby(["trial", "proba", "arrow", "interaction"])
    .mean()[["posOffSet", "meanVelo"]]
    .reset_index()
)
learningCurve
# %%
for p in learningCurve.proba.unique():
    # y = np.convolve(
    #     learningCurve[
    #         (learningCurve.arrow == "up")
    #         & (learningCurve.proba == p)
    #         & (learningCurve.TD_prev == 1)
    #     ].posOffSet,
    #     np.ones(5) / 5,
    #     mode="valid",
    # )
    # plt.plot(y, label=f"Up: {p}, TD_prev=1")
    # y = np.convolve(
    #     learningCurve[
    #         (learningCurve.arrow == "up")
    #         & (learningCurve.proba == p)
    #         & (learningCurve.TD_prev == -1)
    #     ].posOffSet,
    #     np.ones(5) / 5,
    #     mode="valid",
    # )
    sns.scatterplot(
        x="trial",
        y="posOffSet",
        hue="interaction",
        palette="viridis",
        data=learningCurve[(learningCurve.arrow == "up") & (learningCurve.proba == p)],
    )
    # plt.plot(y, label=f"Up: {p}, TD_prev=-1")
    plt.legend()
    plt.title(f"Learning Curve: Arrow UP P(Right|UP)={p}")
    plt.show()

# %%
learningCurveInteraction = (
    df_prime.groupby(["proba", "arrow", "interaction"])
    .mean()[["posOffSet", "meanVelo"]]
    .reset_index()
)


learningCurveInteraction

# %%
sns.barplot(
    x="proba",
    y="posOffSet",
    hue="interaction",
    data=learningCurveInteraction[learningCurveInteraction.arrow == "up"],
)
plt.title("Position Offset: Arrow UP")
plt.xlabel("P(Right|UP)")
plt.show()
# %%
sns.barplot(
    x="proba",
    y="meanVelo",
    hue="interaction",
    data=learningCurveInteraction[learningCurveInteraction.arrow == "up"],
)
plt.title("Position Offset: Arrow UP")
plt.xlabel("P(Right|UP)")
plt.show()
# %%
sns.barplot(
    x="proba",
    y="posOffSet",
    hue="interaction",
    data=learningCurveInteraction[learningCurveInteraction.arrow == "down"],
)
plt.title("Position Offset: Arrow DOWN")
plt.xlabel("P(Left|DOWN)")
plt.show()
# %%
sns.barplot(
    x="proba",
    y="meanVelo",
    hue="interaction",
    data=learningCurveInteraction[learningCurveInteraction.arrow == "down"],
)
plt.title("Position Offset: Arrow DOWN")
plt.xlabel("P(Left|DOWN)")
plt.show()
# %%
#
for p in learningCurveInteraction.proba.unique():
    # y = np.convolve(
    #     learningCurve[
    #         (learningCurve.arrow == "up")
    #         & (learningCurve.proba == p)
    #         & (learningCurve.TD_prev == 1)
    #     ].meanVelo,
    #     np.ones(20) / 20,
    #     mode="valid",
    # )
    sns.scatterplot(
        x="trial",
        y="posOffSet",
        hue="interaction",
        palette="viridis",
        data=learningCurveInteraction[
            (learningCurveInteraction.arrow == "up")
            & (learningCurveInteraction.proba == p)
        ],
    )
    # plt.plot(y, label=f"Up: {p}, TD_prev=1")
    # y = np.convolve(
    #     learningCurve[
    #         (learningCurve.arrow == "up")
    #         & (learningCurve.proba == p)
    #         & (learningCurve.TD_prev == -1)
    #     ].meanVelo,
    #     np.ones(20) / 20,
    #     mode="valid",
    # )
    # sns.lineplot(
    #     x="trial",
    #     y="meanVelo",
    #     hue="TD_prev",
    #     palette="viridis",
    #     data=learningCurve[
    #         (learningCurve.arrow == "up")
    #         & (learningCurve.proba == p)
    #         & (learningCurve.TD_prev == -1)
    #     ],
    # )
    # plt.plot(y, label=f"Up: {p}, TD_prev=-1")
    plt.legend()
    plt.title(f"Learning Curve: Arrow UP,\n Average over participants P(Right|UP)={p}")
    plt.show()
# %%
# Plotting for each subject and each of their proba: the position offset across trials by taking into account thhe previous target
for sub in df["sub"].unique():
    for p in df[df["sub"] == sub]["proba"].unique():
        sns.scatterplot(
            x="trial",
            y="posOffSet",
            hue="TD_prev",
            data=df[(df["sub"] == sub) & (df["proba"] == p) & (df["arrow"] == "up")],
        )

        plt.title(f"Subject {sub}: Arrow UP, P(Right|UP)={p}")
        plt.savefig(dirFig + f"sub{sub}ArrowUp{p}.png")
        plt.show()
# %%
# Plot the data
sns.lmplot(
    x="proba",
    y="posOffSet",
    hue="arrow",
    data=df,
    scatter_kws={"alpha": 0.2},
)

# Show the plot
plt.show()
# %%

bp = sns.boxplot(x="proba", y="posOffSet", hue="arrow", data=df)
bp.legend(fontsize="larger")
plt.xlabel("P(Right|Up)", fontsize=30)
plt.ylabel("Position Offset", fontsize=30)

plt.show()

# %%
# Merging probablity 1 and 0 together.
# Replace proba 0 with 1 and invert the sign of the corresponding posOffset
df.loc[df["proba"] == 0, "posOffSet"] = -df.loc[df["proba"] == 0, "posOffSet"]
df.loc[df["proba"] == 0, "meanVelo"] = -df.loc[df["proba"] == 0, "meanVelo"]
df.loc[df["proba"] == 0, "proba"] = 1

# %%
# Groupping the data
df_melted = (
    df.groupby(["sub", "proba", "arrow"])[["posOffSet", "meanVelo"]]
    .mean()
    .reset_index()
)
df_melted
# %%
bp = sns.boxplot(x="proba", y="posOffSet", hue="arrow", data=df_melted)
bp.legend(fontsize="larger")
plt.xlabel("P(Right|Up)", fontsize=30)
plt.ylabel("Position Offset", fontsize=30)

plt.savefig(dirFig + "cldcbp.png")
plt.show()
# %%
bp = sns.boxplot(x="proba", y="meanVelo", hue="arrow", data=df_melted)
bp.legend(fontsize="larger")
plt.xlabel("P(Right|Up)", fontsize=30)
plt.ylabel("Anticipatory Velocity", fontsize=30)

plt.savefig(dirFig + "cldcbpvelo.png")
plt.show()
# %%
lm = sns.lmplot(x="proba", y="posOffSet", hue="arrow", data=df_melted, height=8)
# Adjust font size for axis labels
lm.set_axis_labels("P(Right|Up)", "Position OffSet")
# lm.ax.legend(fontsize='large')
plt.savefig(dirFig + "cldclp.png")
plt.show()
# %%

lm = sns.lmplot(x="proba", y="meanVelo", hue="arrow", data=df_melted, height=8)
# Adjust font size for axis labels
lm.set_axis_labels("P(Right|Up)", "Anticipatory Velocity")
# lm.ax.legend(fontsize='large')
plt.savefig(dirFig + "cldclpvelo.png")
plt.show()
# %%

# Create the box plot with transparent fill and black borders, and without legend
bp = sns.boxplot(
    x="proba",
    y="posOffSet",
    hue="arrow",
    data=df_melted,
    boxprops=dict(facecolor="none", edgecolor="black"),
    legend=False,
)

# Add scatter plot on top
sns.stripplot(
    x="proba",
    y="posOffSet",
    hue="arrow",
    data=df_melted,
    dodge=True,
    jitter=True,
    size=8,
    alpha=0.7,
)
# Set labels for both top and bottom x-axes
plt.xlabel("P(Right|UP)", fontsize=30)
plt.ylabel("posOffSet", fontsize=30)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
# Overlay regplot on top of the boxplot and stripplot

plt.twiny().set_xlabel("P(Right|DOWN)", fontsize=30)
# Set the tick positions for both top and bottom x-axes
tick_positions = [0.15, 0.4, 0.65, 0.85]
tick_labels = [0, 0.25, 0.50, 0.75]

# Set the ticks and labels for both top and bottom x-axes
plt.xticks(tick_positions, tick_labels, fontsize=20)
plt.xticks(fontsize=30)
# Invert the top x-axis
plt.gca().invert_xaxis()

# # Manually add stars indicating statistical significance
# # Adjust the coordinates based on your plot
# plt.text(0.6, 0.6, "**", fontsize=30, ha="center", va="center", color="red")
# plt.text(
#     0.6, 0.65, "_______________", fontsize=30, ha="center", va="center", color="red"
# )
# # plt.text(0.6, 0.6, 'p < 0.001', fontsize=15, ha='center', va='center', color='red')
#
# plt.text(0.75, 0.75, "***", fontsize=30, ha="center", va="center", color="green")
# plt.text(
#     0.75, 0.8, "_______________", fontsize=30, ha="center", va="center", color="green"
# )
#
# # Right side
#
# plt.text(0.25, -1, "**", fontsize=30, ha="center", va="center", color="red")
# plt.text(
#     0.25, -0.95, "_______________", fontsize=30, ha="center", va="center", color="red"
# )
# # plt.text(0.6, 0.6, 'p < 0.001', fontsize=15, ha='center', va='center', color='red')
#
# plt.text(0.45, -1, "***", fontsize=30, ha="center", va="center", color="green")
# plt.text(
#     0.45, -1, "_______________", fontsize=30, ha="center", va="center", color="green"
# )

# plt.text(0.333, 0.6, 'p < 0.001', fontsize=15, ha='center', va='center', color='green')
# Adjust legend
bp.legend(fontsize=25)


# Save the plot
plt.savefig(dirFig + "cldcbpsc.png")

# Show the plot
plt.show()

# # %%
#
# Create the box plot with transparent fill and black borders, and without legend
bp = sns.boxplot(
    x="proba",
    y="meanVelo",
    hue="arrow",
    data=df_melted,
    boxprops=dict(facecolor="none", edgecolor="black"),
    legend=False,
)

# Add scatter plot on top
sns.stripplot(
    x="proba",
    y="meanVelo",
    hue="arrow",
    data=df_melted,
    dodge=True,
    jitter=True,
    size=8,
    alpha=0.7,
)
# Set labels for both top and bottom x-axes
plt.xlabel("P(Right|UP)", fontsize=30)
plt.ylabel("meanVelo", fontsize=30)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
# Overlay regplot on top of the boxplot and stripplot

plt.twiny().set_xlabel("P(Right|DOWN)", fontsize=30)
# Set the tick positions for both top and bottom x-axes
tick_positions = [0.15, 0.4, 0.65, 0.85]
tick_labels = [0, 0.25, 0.50, 0.75]

# Set the ticks and labels for both top and bottom x-axes
plt.xticks(tick_positions, tick_labels, fontsize=20)
plt.xticks(fontsize=30)
# Invert the top x-axis
plt.gca().invert_xaxis()

# # Manually add stars indicating statistical significance
# # Adjust the coordinates based on your plot
# plt.text(0.6, 0.6, "**", fontsize=30, ha="center", va="center", color="red")
# plt.text(
#     0.6, 0.65, "_______________", fontsize=30, ha="center", va="center", color="red"
# )
# # plt.text(0.6, 0.6, 'p < 0.001', fontsize=15, ha='center', va='center', color='red')
#
# plt.text(0.75, 0.75, "***", fontsize=30, ha="center", va="center", color="green")
# plt.text(
#     0.75, 0.8, "_______________", fontsize=30, ha="center", va="center", color="green"
# )
#
# # Right side
#
# plt.text(0.25, -1, "**", fontsize=30, ha="center", va="center", color="red")
# plt.text(
#     0.25, -0.95, "_______________", fontsize=30, ha="center", va="center", color="red"
# )
# # plt.text(0.6, 0.6, 'p < 0.001', fontsize=15, ha='center', va='center', color='red')
#
# plt.text(0.45, -1, "***", fontsize=30, ha="center", va="center", color="green")
# plt.text(
#     0.45, -1, "_______________", fontsize=30, ha="center", va="center", color="green"
# )

# plt.text(0.333, 0.6, 'p < 0.001', fontsize=15, ha='center', va='center', color='green')
# Adjust legend
bp.legend(fontsize=25)


# Save the plot
plt.savefig(dirFig + "cldcbpscvelo.png")

# Show the plot
plt.show()

# # %%
# lm = sns.lmplot(
#     x="arrow",
#     y="posOffSet",
#     hue="proba",
#     data=df_melted,
#     height=8,
#     palette="viridis",
# )
# # Adjust font size for axis labels
# lm.set_axis_labels("Arrow Chosen", "Position Offset", fontsize=20)
#
# %%
# Create a figure and axis
fig, ax = plt.subplots()

# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()

# Show the plot
bp = sns.boxplot(
    x="arrow",
    y="posOffSet",
    hue="proba",
    data=df_melted,
    palette="viridis",
)

bp.legend(fontsize=25)
plt.xlabel("Arrow Chosen", fontsize=30)
plt.ylabel("Position Offset", fontsize=30)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.title(
    "Position Offset vs Arrow Chosen \n for different $\mathbb{P}(Right|UP)$",
    fontsize=30,
)
plt.savefig(dirFig + "antihueproba.png")
plt.show()

# %%

# %%
# Create a figure and axis
fig = plt.figure()

# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()

# Show the plot
bp = sns.boxplot(
    x="arrow",
    y="meanVelo",
    hue="proba",
    data=df_melted,
    palette="viridis",
)

bp.legend(fontsize=25)
plt.xlabel("Arrow Chosen", fontsize=30)
plt.ylabel("Anticipatory Velocity", fontsize=30)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.title(
    "Position Offset vs Arrow Chosen \n for different $\mathbb{P}(Right|UP)$= $\mathbb{P}(Left|DOWN)$",
    fontsize=30,
)
plt.savefig(dirFig + "antihueprobavelo.png")
plt.show()

# %%
# %%

model = sm.OLS.from_formula(
    "posOffSet ~ C(proba) ",
    data=df_melted[(df_melted["arrow"] == "up") & (df_melted["proba"] != 1)],
)
result = model.fit()

print(result.summary())

# %%

model = sm.OLS.from_formula(
    "posOffSet ~ C(proba) ",
    data=df_melted[(df_melted["arrow"] == "down") & (df_melted["proba"] != 1)],
)
result = model.fit()

print(result.summary())

# %%
model = sm.OLS.from_formula(
    "posOffSet ~ C(proba) ", data=df_melted[df_melted["arrow"] == "down"]
)
result = model.fit()

print(result.summary())
# %%

# %%
model = sm.OLS.from_formula(
    "posOffSet ~ C(proba) ", data=df_melted[df_melted["arrow"] == "up"]
)
result = model.fit()

print(result.summary())
# %%

# %%
model = sm.OLS.from_formula(
    "meanVelo ~ C(proba) ", data=df_melted[df_melted["arrow"] == "down"]
)
result = model.fit()

print(result.summary())
# %%

# %%
model = sm.OLS.from_formula(
    "meanVelo ~ C(proba) ", data=df_melted[df_melted["arrow"] == "up"]
)
result = model.fit()

print(result.summary())
# %%
model = sm.OLS.from_formula(
    "posOffSet ~ C(arrow) ", data=df_melted[df_melted.proba == 0.25]
)
result = model.fit()

print(result.summary())

# %%

model = sm.OLS.from_formula(
    "posOffSet ~ C(arrow) ", data=df_melted[df_melted.proba == 0.75]
)
result = model.fit()

print(result.summary())

# %%

model = sm.OLS.from_formula(
    "posOffSet ~ C(arrow) ", data=df_melted[df_melted.proba == 0.5]
)
result = model.fit()

print(result.summary())


# %%
rp.summary_cont(df.groupby(["sub", "proba", "arrow"])["posOffSet"])
# %%
model = ols("meanVelo ~ C(proba)*C(arrow) ", data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=3)

print(anova_table)

# %%

# Perform repeated measures ANOVA
aovrm = AnovaRM(
    df_melted[df_melted["sub"] != 1], "posOffSet", "sub", within=["proba", "arrow"]
)
print(aovrm.fit())
# %%
model = smf.mixedlm(
    "posOffSet ~C(proba)*C(arrow)",
    data=df,
    groups=df["sub"],
).fit()
model.summary()

# %%

model = smf.mixedlm(
    "meanVelo ~C(proba)*C(arrow)",
    data=df,
    groups=df["sub"],
    re_formula="C(proba)",
).fit(method=["lbfgs"])
model.summary()

# %%
summary = rp.summary_cont(df.groupby(["sub_number", "color", "proba"])["meanVelo"])


# |%%--%%| <0egQ5Pt63g|pWlRpGw6rk>

summary.reset_index(inplace=True)

# |%%--%%| <pWlRpGw6rk|De2pkM9jay>


sns.boxplot(data=summary, x="proba", y="Mean", hue="color", palette=["green", "red"])


# |%%--%%| <De2pkM9jay|OoqkKYL40A>


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

# |%%--%%| <OoqkKYL40A|n3x6xbZn3K>

grid = sns.FacetGrid(df, col="sub_number", hue="proba", col_wrap=4, height=3)

# |%%--%%| <n3x6xbZn3K|P5OlKynfTc>

grid.map(plt.scatter, "trial_number", "meanVelo")

# |%%--%%| <P5OlKynfTc|O5W1mDptee>

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

# |%%--%%| <O5W1mDptee|AxGXf1axFL>

fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(111)

sm.qqplot(model.resid, dist=stats.norm, line="s", ax=ax)

ax.set_title("Q-Q Plot")

# |%%--%%| <AxGXf1axFL|Jpl7HunDxN>

labels = ["Statistic", "p-value"]

norm_res = stats.shapiro(model.resid)

for key, val in dict(zip(labels, norm_res)).items():
    print(key, val)

# |%%--%%| <Jpl7HunDxN|zQIeHV1DXu>

fig = plt.figure(figsize=(16, 9))

ax = sns.boxplot(x=model.model.groups, y=model.resid)

ax.set_title("Distribution of Residuals for Anticipatory Velocity by Subject")
ax.set_ylabel("Residuals")
ax.set_xlabel("Subject")

# |%%--%%| <zQIeHV1DXu|txSp7GI86s>

het_white_res = het_white(model.resid, model.model.exog)

labels = ["LM Statistic", "LM-Test p-value", "F-Statistic", "F-Test p-value"]

for key, val in dict(zip(labels, het_white_res)).items():
    print(key, val)

# |%%--%%| <txSp7GI86s|0IbD46YHQY>

# t test to comprare proba 25/red and proba75/green
stats.ttest_ind(
    df[(df.proba == 25) & (df.color == "red")].meanVelo,
    df[(df.proba == 75) & (df.color == "green")].meanVelo,
)

# |%%--%%| <0IbD46YHQY|bs8JxWTHJZ>


# t test to comprare proba 25/red and proba75/green
stats.ttest_ind(
    df[(df.proba == 75) & (df.color == "red")].meanVelo,
    df[(df.proba == 25) & (df.color == "green")].meanVelo,
)

# |%%--%%| <bs8JxWTHJZ|9chAimtOga>


stats.ttest_ind(
    df[(df.proba == 50) & (df.color == "red")].meanVelo,
    df[(df.proba == 50) & (df.color == "green")].meanVelo,
)

# |%%--%%| <9chAimtOga|ORF419I7zO>


stats.ttest_ind(
    df[(df.proba == 50) & (df.color == "green")].meanVelo,
    df[(df.proba == 75) & (df.color == "green")].meanVelo,
)

# |%%--%%| <ORF419I7zO|7Pnq20YKvY>

# Example assuming 'proba' and 'color' are categorical variables in your DataFrame
colors = df["color"].unique()

for color in colors:
    # Filter data for the current color
    color_data = df[df["color"] == color]

    # Group data by 'proba' and get meanVelo for each group
    grouped_data = [group["meanVelo"] for proba, group in color_data.groupby("proba")]

    # Perform Kruskal-Wallis test
    statistic, p_value = stats.kruskal(*grouped_data)

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

# |%%--%%| <7Pnq20YKvY|fsmlwWdIC0>
r"""°°°
# Analysis of subject who did Vanessa's task
°°°"""
# |%%--%%| <fsmlwWdIC0|9e6bJW7zSd>

df_prime = df[(df.sub_number > 12)]

# |%%--%%| <9e6bJW7zSd|aAEDXXm0yJ>

l_prime = (
    df_prime.groupby(["sub_number", "trial_color_chosen", "proba"])
    .meanVelo.mean()
    .reset_index()
)
l_prime

# |%%--%%| <aAEDXXm0yJ|QXsRE0iCWU>

bp = sns.boxplot(
    x="proba", y="meanVelo", hue="trial_color_chosen", data=l_prime, palette=colors
)
bp.legend(fontsize="larger")
plt.xlabel("P(Dir=Right|Color=Red)", fontsize=30)
plt.ylabel("Anticipatory Velocity", fontsize=30)

# |%%--%%| <QXsRE0iCWU|5EX22XRGSK>

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

# Participants balanced their choices
print(df.trial_color_chosen.value_counts())
# |%%--%%| <qxewuIGTnt|ZwGowoTUlq>


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


# |%%--%%| <ZwGowoTUlq|ggwrOHSS1C>

probability_distributions_by_group = compute_probability_distribution_tplus1_given_t(
    df, "sub_number", "proba", "trial_color_chosen"
)
probability_distributions_by_group

# |%%--%%| <ggwrOHSS1C|xhpGhcUYO3>


# Example usage:
# with columns "subject", "condition", and "choice"
for i in df["sub_number"].unique():
    for p in df["proba"].unique():
        print(f"Probability Distribution for subject {i} and condition {p}:")
        for key, probability in probability_distributions_by_group[(i, p)].items():
            print(f"P(C_{key[0]} | C_{key[1]}) = {probability:.2f}")

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
#
# # |%%--%%| <go3EAD1SFB|LIDfUD5MWo>
#
#
# # Computing the mean over all subjects
# def compute_mean_probability_distribution_tplus1_given_t(dictionary):
#     # Create a dictionary to store the mean probability distribution for each condition
#     mean_probability_distribution = {}
#
#     # Iterate over unique conditions
#     for (subject, condition), distribution in dictionary.items():
#         if condition not in mean_probability_distribution:
#             mean_probability_distribution[condition] = Counter()
#
#         mean_probability_distribution[condition].update(distribution)
#
#     # Calculate the mean probability distribution over all subjects for each condition
#     for condition, distribution in mean_probability_distribution.items():
#         total_subjects = len(dictionary) // len(mean_probability_distribution)
#         mean_probability_distribution[condition] = {
#             key: count / total_subjects for key, count in distribution.items()
#         }
#
#     return mean_probability_distribution
#
#
# # |%%--%%| <LIDfUD5MWo|lKAgzJCVuI>
#
#
# # Assuming you already have the probability_distributions_by_group_tplus1_given_t dictionary
# probability_distributions_by_group_tplus1_given_t = (
#     compute_probability_distribution_tplus1_given_t(
#         df, "sub_number", "proba", "trial_color_chosen"
#     )
# )
# mean_probability_distribution_tplus1_given_t = (
#     compute_mean_probability_distribution_tplus1_given_t(
#         probability_distributions_by_group_tplus1_given_t
#     )
# )
#
#
# # |%%--%%| <lKAgzJCVuI|5oKxlGF93G>
#
#
# # Extract unique pairs (C_t, C_{t+1}) from the first condition (assuming all conditions have the same pairs)
# unique_pairs_t_tplus1 = list(mean_probability_distribution_tplus1_given_t.values())[
#     0
# ].keys()
#
# # Prepare data for plotting
# num_conditions = len(mean_probability_distribution_tplus1_given_t)
# num_pairs = len(unique_pairs_t_tplus1)
#
# # Create subplots
# fig, axes = plt.subplots(1, num_conditions, figsize=(15, 5), sharey=True)
#
# bar_width = 0.2
# bar_offsets = np.arange(num_pairs)
#
# # Plotting
# for idx, (condition, mean_distribution) in enumerate(
#     mean_probability_distribution_tplus1_given_t.items()
# ):
#     probabilities = [mean_distribution[pair] for pair in unique_pairs_t_tplus1]
#
#     axes[idx].bar(bar_offsets, probabilities, bar_width, label=f"Condition {condition}")
#     axes[idx].set_xticks(bar_offsets)
#     axes[idx].set_xticklabels(unique_pairs_t_tplus1)
#     # axes[idx].set_xlabel('Pairs (C_t, C_{t+1})')
#     axes[idx].set_title(f"Probability:  {condition}")
#
# # Set common labels and legend
# fig.text(0.5, 0.04, "Pairs (C_{t+1},C_t,)", ha="center", va="center")
# fig.text(0.06, 0.5, "Probability", ha="center", va="center", rotation="vertical")
# fig.suptitle("Mean Probability Distribution for Each Condition and Pair (C_{t+1},C_t)")
# # plt.legend()
#
# plt.show()
#
#
# # |%%--%%| <5oKxlGF93G|4SSNV0ixKb>
# """°°°
# Computing P(C_{t+2} | C_{t+1}, C_t)
# °°°"""
# # |%%--%%| <4SSNV0ixKb|7AfY2O0mA0>
#
#
# def compute_probability_distribution_tplus2_given_tplus1_and_t(
#     df, subject_col, condition_col, choice_col
# ):
#     # df is your DataFrame
#     # subject_col is the column name for the subjects
#     # condition_col is the column name for the conditions
#     # choice_col is the column name for the choices
#
#     # Create a dictionary to store probability distributions for each subject and condition group
#     probability_distributions_tplus2_given_tplus1_and_t = {}
#
#     # Iterate over unique subject-condition pairs
#     for (subject, condition), group_df in df.groupby([subject_col, condition_col]):
#         choices = group_df[choice_col].tolist()
#
#         # Count occurrences of each triplet (C_t, C_{t+1}, C_{t+2})
#         transition_counts_t_tplus1_tplus2 = Counter(
#             zip(choices[:-2], choices[1:-1], choices[2:])
#         )
#
#         # Compute total counts for each pair (C_{t+1}, C_t)
#         total_counts_tplus1_t = Counter(zip(choices[:-1], choices[1:]))
#
#         # Calculate the conditional probabilities for P(C_{t+2} | C_{t+1} & C_t)
#         probability_distribution_tplus2_given_tplus1_and_t = {}
#         for (
#             choice_t,
#             choice_tplus1,
#             choice_tplus2,
#         ), count in transition_counts_t_tplus1_tplus2.items():
#             probability_distribution_tplus2_given_tplus1_and_t[
#                 (choice_tplus2, choice_tplus1, choice_t)
#             ] = (count / total_counts_tplus1_t[choice_t, choice_tplus1])
#
#         # Store the probability distribution in the dictionary
#         probability_distributions_tplus2_given_tplus1_and_t[(subject, condition)] = (
#             probability_distribution_tplus2_given_tplus1_and_t
#         )
#
#     return probability_distributions_tplus2_given_tplus1_and_t
#
#
# # |%%--%%| <7AfY2O0mA0|JIDuF3osMK>
#
#
# probability_distributions_by_group = (
#     compute_probability_distribution_tplus2_given_tplus1_and_t(
#         df, "sub_number", "proba", "trial_color_chosen"
#     )
# )
# probability_distributions_by_group
# # |%%--%%| <JIDuF3osMK|ZfiNVKwozc>
#
# for i in df["sub_number"].unique():
#     for p in df["proba"].unique():
#         print(f"Probability Distribution for subject {i} and condition {p}:")
#         for key, probability in probability_distributions_by_group[(i, p)].items():
#             print(f"P(C_{key[0]} | C_{key[1]},C_{key[2]}) = {probability:.2f}")
#
# # |%%--%%| <ZfiNVKwozc|B6f78cRdCl>
#
#
# unique_triplets = list(probability_distributions_by_group.values())[0].keys()
# unique_triplets
# # |%%--%%| <B6f78cRdCl|4uikRnrdl5>
#
# probability_distributions_by_group.items()
#
# # |%%--%%| <4uikRnrdl5|37XUO5s80z>
#
# # Extract unique triplets from the first condition (assuming all conditions have the same triplets)
#
# # Prepare data for plotting
# num_conditions = len(probability_distributions_by_group)
# num_triplets = len(unique_triplets)
#
# # Create subplots
# fig, axes = plt.subplots(1, num_conditions, figsize=(15, 5), sharey=True)
#
# bar_width = 0.2
# bar_offsets = np.arange(num_triplets)
#
#
# # Plotting
# for idx, (condition, mean_distribution) in enumerate(
#     probability_distributions_by_group.items()
# ):
#     subject = condition[0]
#     condition_value = condition[1]
#
#     print(f"Subject: {subject}, Condition: {condition_value}")
#     print(f"Keys in mean_distribution: {mean_distribution.keys()}")
#
#     probabilities = [mean_distribution[triplet] for triplet in unique_triplets]
#
#     axes[idx].bar(
#         bar_offsets,
#         probabilities,
#         bar_width,
#         label=f"Subject {subject}, Condition {condition_value}",
#     )
#     axes[idx].set_xticks(bar_offsets)
#     axes[idx].set_xticklabels(unique_triplets, fontsize=8)
#     axes[idx].set_title(f"Subject {subject}, Condition {condition_value}")
#
# # Set common labels
# fig.text(
#     0.5,
#     0.04,
#     "Triplets (C_{t+2}, C_{t+1}, C_t))",
#     ha="center",
#     va="center",
#     fontsize=20,
# )
# fig.text(0.06, 0.5, "Probability", ha="center", va="center", rotation="vertical")
# fig.suptitle("Mean Probability Distribution for P(C_t+2 | C_t+1, C_t)", fontsize=30)
#
# plt.show()
#
#
# # |%%--%%| <37XUO5s80z|jfgTiohOAU>
#
#
# def compute_mean_probability_distribution(dictionary):
#     # Create a dictionary to store the mean probability distribution for each condition
#     mean_probability_distribution = {}
#
#     # Iterate over unique conditions
#     for condition in set(key[1] for key in dictionary.keys()):
#         mean_distribution = Counter()
#         num_participants = 0
#
#         # Aggregate distributions for the same condition
#         for subject, cond in dictionary.keys():
#             if cond == condition:
#                 distribution = dictionary[(subject, cond)]
#                 mean_distribution.update(distribution)
#                 num_participants += 1
#
#         # Calculate the mean by dividing each count by the number of participants
#         mean_distribution = {
#             key: count / num_participants for key, count in mean_distribution.items()
#         }
#
#         # Store the mean distribution for the condition
#         mean_probability_distribution[condition] = mean_distribution
#
#     return mean_probability_distribution
#
#
# # |%%--%%| <jfgTiohOAU|ki3x5jpfAR>
#
# meanOverSubjects = compute_mean_probability_distribution(
#     probability_distributions_by_group
# )
# meanOverSubjects
#
# # |%%--%%| <ki3x5jpfAR|4ol2GdpoqJ>
#
#
# for p in df["proba"].unique():
#     print(f"Probability Distribution for proba {p}:")
#     for key, probability in meanOverSubjects[(p)].items():
#         print(f"P(C_{key[0]} | C_{key[1]},C_{key[2]}) = {probability:.2f}")
#
#
# # |%%--%%| <4ol2GdpoqJ|RAx7oDpABa>
#
#
# # Extract unique triplets from the first condition (assuming all conditions have the same triplets)
# unique_triplets = list(meanOverSubjects.values())[0].keys()
#
# # Prepare data for plotting
# num_conditions = len(meanOverSubjects)
# num_triplets = len(unique_triplets)
#
# # Create subplots
# fig, axes = plt.subplots(1, num_conditions, figsize=(15, 5), sharey=True)
#
# bar_width = 0.2
# bar_offsets = np.arange(num_triplets)
#
# # Plotting
# for idx, (condition, mean_distribution) in enumerate(meanOverSubjects.items()):
#     probabilities = [mean_distribution[triplet] for triplet in unique_triplets]
#
#     axes[idx].bar(bar_offsets, probabilities, bar_width, label=f"Condition {condition}")
#     axes[idx].set_xticks(bar_offsets)
#     axes[idx].set_xticklabels(unique_triplets, fontsize=8)
#     # axes[idx].set_xlabel('Triplets (C_t, C_{t+1}, C_{t+2})')
#     axes[idx].set_title(f"Condition {condition}")
#
# # Set common labels and legend
# fig.text(
#     0.5,
#     0.04,
#     "Triplets (C_{t+2}, C_{t+1}, C_t))",
#     ha="center",
#     va="center",
#     fontsize=20,
# )
# fig.text(0.06, 0.5, "Probability", ha="center", va="center", rotation="vertical")
# fig.suptitle("Mean Probability Distribution for P(C_t+2 | C_t+1, C_t)", fontsize=30)
# plt.legend()
#
# plt.show()
# |%%--%%| <RAx7oDpABa|YvO4edwPiv>


# unique_sub_numbers = df['sub_number'].unique()
# # Custom color palette for 'color' categories
# custom_palette = {'green': 'green', 'red': 'red'}
# for sub_number_value in unique_sub_numbers:
#     subset_df = df[df['sub_number'] == sub_number_value
#
#     # Set up subplots for each proba
#     fig, axes = plt.subplots(nrows=1, ncols=len(subset_df['proba'].unique()), figsize=(15, 5), sharey=True)
#
#     # Plot each subplot
#     for i, proba_value in enumerate(subset_df['proba'].unique()):
#         proba_subset_df = subset_df[subset_df['proba'] == proba_value]
#         ax = axes[i]
#
#         # Group by both "proba" and "color" and compute rolling mean with a window of 20
#         rolling_mean = proba_subset_df.groupby(["proba", "color"])["meanVelo"].rolling(window=10, min_periods=1).mean().reset_index(level=[0, 1], drop=True)
#
#         # Plot the rolling mean with color as a hue
#         sns.lineplot(x="trial_number", y=rolling_mean, hue="color", data=proba_subset_df, ax=ax, palette=custom_palette, markers=True)
#
#         ax.set_title(f'sub_number = {sub_number_value}, proba = {proba_value}')
#         ax.set_xlabel('Trial Number')
#         ax.set_ylabel('Mean Velocity (Rolling Average)')
#         ax.legend(title='Color', loc='upper right')
#
#     # Adjust layout for subplots for each subject
#     plt.tight_layout()
#     plt.show()
#
# #|%%--%%| <YvO4edwPiv|wY6MEnWD9g>
#
# rolling_mean
#
# #|%%--%%| <wY6MEnWD9g|bB5rAoyuwj>
#
# #Score of percistency
#
# #|%%--%%| <bB5rAoyuwj|btcBZVF8lR>
#
# probability_distributions_by_group_tplus1_given_t.keys ()
#
# #|%%--%%| <btcBZVF8lR|U1RCI8clxw>
#
#
# probability_distributions_by_group_tplus1_given_t
# #|%%--%%| <U1RCI8clxw|jGu7UdQ3iH>
#
# tplus1GivenT=pd.DataFrame(probability_distributions_by_group_tplus1_given_t)
# #|%%--%%| <jGu7UdQ3iH|eaOt3gMsmF>
#
# tplus1GivenT
#
# #|%%--%%| <eaOt3gMsmF|htGZUflqhx>
#
#
#
# #|%%--%%| <htGZUflqhx|m3jza5TNn8>
#
#
#
#
#
# # Convert dictionary to DataFrame
# tplus1GivenT = pd.DataFrame.from_dict({(k1, k2): v2 for k1, d in probability_distributions_by_group_tplus1_given_t.items() for k2, v2 in d.items()}, orient='index')
#
# # Reset index and rename columns
# tplus1GivenT = tplus1GivenT.reset_index().rename(columns={'level_0': 'Group', 'level_1': 'Distribution', 0: 'Probability'})
#
# print(tplus1GivenT)
#
# #|%%--%%| <m3jza5TNn8|ypzJK0eb37>
#
# tplus1GivenT.columns
#
# #|%%--%%| <ypzJK0eb37|kAaihO8ylP>
#
# # Dictionary to store the sums for each main key
# sums_by_main_key = {}
#
# # Iterate through the main keys
# for main_key, sub_dict in probability_distributions_by_group_tplus1_given_t.items():
#     # Initialize the sum for the current main key
#     current_sum = 0
#     # Iterate through the sub-dictionary
#     for sub_key, probability in sub_dict.items():
#         # Extract the sub-key components
#         x, y = sub_key
#         # Perform the arithmetic operations and update the sum
#         if x == 1 and y == 1:
#             current_sum += probability
#         elif x == 0 and y == 0:
#             current_sum += probability
#         elif x == 0 and y == 1:
#             current_sum -= probability
#         elif x == 1 and y == 0:
#             current_sum -= probability
#     # Store the sum for the current main key
#     sums_by_main_key[main_key] = current_sum
#
# # Print the sums for each main key
# for main_key, sum_value in sums_by_main_key.items():
#     print(f"Main Key: {main_key}, Sum: {sum_value}")
#
# #|%%--%%| <kAaihO8ylP|FIuoAdIe71>
#
# sums_by_main_key
#
# #|%%--%%| <FIuoAdIe71|rgRPlJFvLL>
#
#
#
# # Initialize lists to store data
# subjects = []
# probabilities = []
# persistence_scores = []
#
# # Iterate through the dictionary items
# for key, value in sums_by_main_key.items():
#     # Extract subject and probability from the key
#     subject, probability = key
#
#     # Append data to lists
#     subjects.append(subject)
#     probabilities.append(probability)
#     persistence_scores.append(value)
#
# # Create DataFrame
# percistenceScore = pd.DataFrame({
#     'Subject': subjects,
#     'Probability': probabilities,
#     'Persistence Score': persistence_scores
# })
#
# # Display DataFrame
# print(percistenceScore)
#
# #|%%--%%| <rgRPlJFvLL|3CJbtd4FeR>
#
# percistenceScore.groupby('Subject')['Persistence Score'].mean()
#
# #|%%--%%| <3CJbtd4FeR|iCpM9sHN6j>
#
# learning=df.groupby(['sub_number','color','proba']).meanVelo.mean().reset_index()
# learning
# #|%%--%%| <iCpM9sHN6j|in2fBOKfLB>
#
#
# # Group by 'sub_number' and 'color'
# grouped = learning.groupby(['sub_number', 'color'])
#
# # Calculate the mean velocity for probability 75 and 25, respectively
# mean_velo_75 = grouped.apply(lambda x: x[x['proba'] == 75]['meanVelo'].mean())
# mean_velo_25 = grouped.apply(lambda x: x[x['proba'] == 25]['meanVelo'].mean())
#
# # Calculate the difference
# difference = np.abs(mean_velo_75 - mean_velo_25)
#
# # Display the result
# print(difference)
#
# #|%%--%%| <in2fBOKfLB|XVJ4Z1UtaB>
#
# grouped
#
# #|%%--%%| <XVJ4Z1UtaB|bLdHdui9o8>
#
# difference_green=difference.xs('green', level='color')
# difference_red=difference.xs('red', level='color')
# percistence=percistenceScore.groupby('Subject')['Persistence Score'].mean().reset_index()
# percistence
#
# #|%%--%%| <bLdHdui9o8|mJbZBBTDrX>
#
# percistence['learningScore']=np.mean([difference_green,difference_red], axis=0)
#
# #|%%--%%| <mJbZBBTDrX|sQipqrPL2s>
#
# percistence["learningGreen"]=difference_green.values
# percistence["learningRed"]=difference_red.values
# percistence
# #|%%--%%| <sQipqrPL2s|Yf0nwdODe4>
#
# sns.scatterplot(data=percistence, x="Persistence Score", y="learningGreen", hue="Subject")
#
# #|%%--%%| <Yf0nwdODe4|gXeKBR5VSb>
#
# sns.scatterplot(data=percistence, x="Persistence Score", y="learningRed", hue="Subject")
#
# #|%%--%%| <gXeKBR5VSb|zSwvDk70Ay>
#
# sns.scatterplot(data=percistence, x="Persistence Score", y="learningScore", hue="Subject")
#
# #|%%--%%| <zSwvDk70Ay|mOf9eDNwlV>
#
# # Plotting
# plt.figure(figsize=(10, 6))
#
# # Scatter plot for Learning Green
# plt.scatter(percistence['Persistence Score'], percistence['learningGreen'], color='green', label='Learning Green')
#
# # Scatter plot for Learning Red
# plt.scatter(percistence['Persistence Score'], percistence['learningRed'], color='red', label='Learning Red')
#
# # Adding labels and title
# plt.xlabel('Persistence Score')
# plt.ylabel('Learning Score')
# plt.title('Persistence Score vs Learning Score')
# plt.legend()
#
# # Show plot
# plt.show()
#
# #|%%--%%| <mOf9eDNwlV|CVwlJ9uY8v>
#
#
#
# # Plotting
# plt.figure(figsize=(10, 6))
#
# # Scatter plot for Learning Green
# plt.scatter(percistence['Persistence Score'], percistence['learningGreen'], color='green', label='Learning Green')
#
# # Scatter plot for Learning Red
# plt.scatter(percistence['Persistence Score'], percistence['learningRed'], color='red', label='Learning Red')
#
# # Adding fitting lines using seaborn's lmplot
# sns.regplot(x='Persistence Score', y='learningGreen', data=percistence, scatter=False, color='green')
# sns.regplot(x='Persistence Score', y='learningRed', data=percistence, scatter=False, color='red')
#
# # Adding labels and title
# plt.xlabel('Persistence Score')
# plt.ylabel('Learning Score')
# plt.title('Persistence Score vs Learning Score')
# plt.legend()
# plt.savefig('Persistence_Score_vs_Learning_Score.png')
# # Show plot
# plt.show()
#
#
# #|%%--%%| <CVwlJ9uY8v|bjMFu3kNZl>
# plt.scatter(data=percistence, x="Persistence Score", y="learningScore")
# sns.regplot(x='Persistence Score', y='learningScore', data=percistence, scatter=False, color='black')
# # Adding labels and title
# plt.xlabel('Persistence Score',fontsize=30)
# plt.ylabel('Learning Score',fontsize=30)
# plt.title('Persistence Score vs Learning Score',fontsize=40)
# plt.legend()
# plt.savefig('Persistence_Score_vs_Learning_Score.png')
# # Show plot
# plt.show()
#
# #|%%--%%| <bjMFu3kNZl|tnRAsPqMvM>
#
#
# import statsmodels.api as sm
#
# # Define the independent variables (Xs) and dependent variables (Ys)
# X = percistence[['Persistence Score']]
# Y_green = percistence['learningGreen']
# Y_red = percistence['learningRed']
# Y=percistence['learningScore']
# # Add a constant to the independent variables for the intercept term
# X = sm.add_constant(X)
#
# # Fit the multiple linear regression models
# model_green = sm.OLS(Y_green, X).fit()
# model_red = sm.OLS(Y_red, X).fit()
# model=sm.OLS(Y, X).fit()
# # Print the summary of the regression results
# print("Regression Results for Learning Green:")
# print(model_green.summary())
#
# print("\nRegression Results for Learning Red:")
# print(model_red.summary())
#
# #|%%--%%| <tnRAsPqMvM|9b0P2daokY>
#
# print("\nRegression Results for Learning Score:")
# print(model.summary())
#
# #|%%--%%| <9b0P2daokY|KjTNRknrah>
#
# df_green = df[df.color == 'green']
# df_red = df[df.color == 'red']
#
# #|%%--%%| <KjTNRknrah|NfyGTpWFos>
#
# df_green
#
# #|%%--%%| <NfyGTpWFos|z0mPLT9NxN>
#
#
#
# #|%%--%%| <z0mPLT9NxN|Jt0QcNYDDr>
#
# df_green
#
# #|%%--%%| <Jt0QcNYDDr|pqxOhOH9Q7>
#
# df_red
#
#
# #|%%--%%| <pqxOhOH9Q7|MeklYuccDP>
#
# # For df_green
# df_green_last_40_trials = df_green.groupby(['sub_number','proba']).tail(40)
# df_green_last_40_trials
# #|%%--%%| <MeklYuccDP|iLt0dZ28I6>
#
# # Create a dictionary to map each subject to a specific color
# subject_colors = {sub: sns.color_palette('husl', n_colors=len(df_green['sub_number'].unique()))[i]
#                   for i, sub in enumerate(sorted(df_green['sub_number'].unique()))}
#
# # Plot mean velocity for each combination of 'sub_number' and 'proba', manually assigning colors
# ax = sns.catplot(x='proba', y='meanVelo', hue='sub_number', kind='point', data=df_green, palette=subject_colors, legend=False)
# plt.xlabel('Probability')
# plt.ylabel('Mean Velocity')
# plt.title('Mean Velocity across 240 Trials for Each Sub and Proba')
#
# # Get the current axes
# ax = plt.gca()
#
# # Manually create legend with subject labels and corresponding colors
# handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=subject_colors[sub], markersize=10, label=f'Sub {sub}') for sub in sorted(df_green['sub_number'].unique())]
# ax.legend(handles=handles, title='Subject', loc='upper right',fontsize='small')
#
# plt.show()
#
#
# #|%%--%%| <iLt0dZ28I6|ibJic2WQyh>
#
#
# # Create a dictionary to map each subject to a specific color
# subject_colors = {sub: sns.color_palette('husl', n_colors=len(df_red['sub_number'].unique()))[i]
#                   for i, sub in enumerate(sorted(df_red['sub_number'].unique()))}
#
# # Plot mean velocity for each combination of 'sub_number' and 'proba', manually assigning colors
# ax = sns.catplot(x='proba', y='meanVelo', hue='sub_number', kind='point', data=df_red, palette=subject_colors, legend=False)
# plt.xlabel('Probability')
# plt.ylabel('Mean Velocity')
# plt.title('Mean Velocity across 240 Trials for Each Sub and Proba')
#
# # Get the current axes
# ax = plt.gca()
#
# # Manually create legend with subject labels and corresponding colors
# handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=subject_colors[sub], markersize=10, label=f'Sub {sub}') for sub in sorted(df_red['sub_number'].unique())]
# ax.legend(handles=handles, title='Subject', loc='upper right',fontsize='small')
#
# plt.show()
#
# #|%%--%%| <ibJic2WQyh|QdkOZdFx9N>
#
# df_green_last_40_trials.proba.unique()
#
# #|%%--%%| <QdkOZdFx9N|iu5cRdvDKF>
#
# l_green=df_green_last_40_trials.groupby(["sub_number", "proba"]).meanVelo.mean().reset_index()
# l_green
#
# #|%%--%%| <iu5cRdvDKF|xDbH5HWsVR>
#
# df_red_last_40_trials = df_red.groupby(['sub_number','proba']).tail(40)
# df_red_last_40_trials
#
# #|%%--%%| <xDbH5HWsVR|uvEDDrZ7B8>
#
# df_red_last_40_trials.proba.unique()
#
# #|%%--%%| <uvEDDrZ7B8|qeUJHOs7qz>
#
# l_red=df_red_last_40_trials.groupby(["sub_number", "proba"]).meanVelo.mean().reset_index()
# l_red
#
#
#
# #|%%--%%| <qeUJHOs7qz|u41mblaBsi>
#
# # Plot the last 40 trials for each color across the 3 probabilities:
# # Concatenate the two DataFrames and create a new column 'group' to distinguish between them
# df_green_last_40_trials['color'] = 'Green'
# df_red_last_40_trials['color'] = 'Red'
# las40Trials = pd.concat([df_green_last_40_trials, df_red_last_40_trials])
#
# # Plot the boxplot
# sns.boxplot(x="proba", y="meanVelo", hue="color", data=las40Trials, palette={'Green': 'green', 'Red': 'red'})
# plt.show()
# #|%%--%%| <u41mblaBsi|CHoiD1szdN>
#
#
# df.columns
#
# #|%%--%%| <CHoiD1szdN|FMJjdHBMPo>
#
# df.trial_RT_colochoice
# RT=df.groupby(['sub_number','proba']).trial_RT_colochoice.mean().reset_index()['trial_RT_colochoice']
#
# #|%%--%%| <FMJjdHBMPo|T0kMpwlTeB>
#
# plt.hist(RT, color='lightblue', edgecolor='black')
# plt.vlines(RT.mean(), 0, 10, color='red', linestyle='--', label='Mean RT')
# plt.vlines(0.6, 0, 10, color='black', label='Mean RT', linewidth=2,label="Vanessa's Exp")
# plt.legend()
# plt.xlabel('RT',fontsize=40)
# plt.title('RT Distribution',fontsize=40)
# plt.savefig('RT_Distribution.png')
#
# #|%%--%%| <T0kMpwlTeB|1ldsJf9Jcw>
#
# df.trial_color_chosen==df.trial_color_UP
#
# #|%%--%%| <1ldsJf9Jcw|xNFlcjxvCT>
#
# df['arrowChosen']=df.trial_color_chosen==df.trial_color_UP
#
# #|%%--%%| <xNFlcjxvCT|iwjRvfwHn4>
#
# df.arrowChosen=['UP' if x==True else 'DOWN' for x in df.arrowChosen]
#
# #|%%--%%| <iwjRvfwHn4|NjDRDCyRHM>
#
# df.arrowChosen
#
# df[(df.sub_number==2)&(df.proba==75)]
#
# #|%%--%%| <NjDRDCyRHM|rs8Wn7TSfU>
#
# df[(df.sub_number==8) & (df.proba==75)]
#
# #|%%--%%| <rs8Wn7TSfU|7XipOxjoBb>
#
#
# df = df.dropna(subset=['meanVelo'])
#
# #|%%--%%| <7XipOxjoBb|hkoROFsHKx>
#
# df['meanVelo'].isna().sum()
# #|%%--%%| <hkoROFsHKx|W3R9L8xWIT>
#
# df.trial_direction
#
#
# #|%%--%%| <W3R9L8xWIT|HhGq8jrCUB>
#
# df[(df.sub_number==16)&(df.proba==75)& (df.arrowChosen=='UP')]
#
