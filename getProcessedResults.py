import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import os


def process_single_condition(sub, proba, pos, degToPix=27.28):
    """
    Process data for a single subject and probability condition.
    """
    trials = pos[(pos["sub"] == sub) & (pos["proba"] == proba)]["trial"].unique()

    # Handle training trials
    numOfTrials = len(trials)
    print("Number of Trials:", numOfTrials)
    if numOfTrials > 240 and numOfTrials < 350:
        pos = pos[pos["trial"] > (numOfTrials - 240)]

    # Taking just the non-training trials
    trials = pos[(pos["sub"] == sub) & (pos["proba"] == proba)].trial.unique()
    # Calculate position offset
    pos_offsets = []
    mean_velocities = []

    for t in trials:
        trial_data = pos[
            (pos["trial"] == t) & (pos["sub"] == sub) & (pos["proba"] == proba)
        ]

        # Calculate position offset
        pos_offset = (
            trial_data["filtPos"].values[-1] - trial_data["filtPos"].values[0]
        ) / degToPix
        pos_offsets.append(pos_offset)

        # Calculate mean velocity
        mean_velo = np.nanmean(trial_data["filtVelo"])
        mean_velocities.append(mean_velo)

    # Create DataFrame for this condition
    condition_df = pd.DataFrame(
        {
            "sub": [sub for i in range(len(mean_velocities))],
            "proba": [proba for i in range(len(mean_velocities))],
            "trial": trials,
            "posOffSet": pos_offsets,
            "meanVelo": mean_velocities,
        }
    )
    # print(condition_df)
    return condition_df


def process_filtered_data_parallel(
    df,
    events,
    fOFF=-50,
    latency=50,
    mono=True,
    degToPix=27.28,
    n_jobs=-1,
    output_file=None,
):
    """
    Parallel processing version of process_filtered_data.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame with filtered data
    events : pandas.DataFrame
        Events data to merge with processed data
    fOFF : int
        Start of the time window
    latency : int
        End of the time window
    mono : bool
        Monotonicity flag
    degToPix : float
        Degrees to pixels conversion factor
    n_jobs : int
        Number of parallel jobs (-1 for all cores)
    output_file : str, optional
        Path to save the CSV output file

    Returns:
    --------
    pandas.DataFrame
        Processed and merged data
    """

    # Extract position and velocity data for the specified time window
    selected_values = df[(df.time >= fOFF) & (df.time <= latency)]
    pos = selected_values[["sub", "proba", "trial", "filtPos", "filtVelo"]]

    # Get unique combinations of subject and probability
    conditions = [
        (sub, proba)
        for sub in pos["sub"].unique()
        for proba in pos[pos["sub"] == sub]["proba"].unique()
    ]

    # Process conditions in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_single_condition)(sub, proba, pos, degToPix)
        for sub, proba in conditions
    )

    # Combine all results
    allData = pd.concat(results, axis=0, ignore_index=True)
    # Ensure data types are consistent
    allData["sub"] = allData["sub"].astype(int)
    allData["proba"] = allData["proba"].astype(float)
    allData["trial"] = allData["trial"].astype(int)

    events["sub"] = events["sub"].astype(int)
    events["proba"] = events["proba"].astype(float)
    events["trial"] = events["trial"].astype(int)

    finalData = allData.merge(events, on=["sub", "proba", "trial"])

    # Save to CSV if output file is specified
    if output_file is not None:
        finalData.to_csv(output_file, index=False)

    return finalData


# Example usage:
# Load your data
dirPath1 = "/envau/work/brainets/oueld.h/contextuaLearning/ColorCue/data/"
filteredRawData1 = "JobLibProcessingCC.csv"
allEventsFile1 = "allEvents.csv"

dirPath2 = "/envau/work/brainets/oueld.h/contextuaLearning/directionCue/results_voluntaryDirection"
filteredRawData2 = "JobLibProcessing.csv"
allEventsFile2 = "allEvents.csv"


dirPath3 = "/envau/work/brainets/oueld.h/contextuaLearning/ColorCue/imposedColorData"
filteredRawData3 = "JobLibProcessingCC.csv"
allEventsFile3 = "allEvents.csv"

dirPath4 = "/envau/work/brainets/oueld.h/contextuaLearning/directionCue/results_imposeDirection"
filteredRawData4 = "JobLibProcessing.csv"
allEventsFile4 = "allEvents.csv"

dirPath5 = "/envau/work/brainets/oueld.h/attentionalTask/data"
filteredRawData5 = "JobLibProcessingCC.csv"
allEventsFile5 = "allEvents.csv"

paths = [dirPath1, dirPath2, dirPath3, dirPath4, dirPath5]
filteredRawDatas = [
    filteredRawData1,
    filteredRawData2,
    filteredRawData3,
    filteredRawData4,
    filteredRawData5,
]
allEventsFiles = [
    allEventsFile1,
    allEventsFile2,
    allEventsFile4,
    allEventsFile4,
    allEventsFile5,
]
windows = [(-50, 50), (80, 120), (-100, 100), (-200, 120)]
for w in windows:
    for p, f, e in zip(paths, filteredRawDatas, allEventsFiles):
        df = pd.read_csv(os.path.join(p, f))
        events = pd.read_csv(os.path.join(p, e))
        output_file = os.path.join(p, f"processedResultsWindow({w[0]},{w[1]}).csv")
        # Process data and save to CSV
        results = process_filtered_data_parallel(
            df=df, events=events, fOFF=w[0], latency=w[1], output_file=output_file
        )
