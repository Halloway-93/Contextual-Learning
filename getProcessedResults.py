import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import os


def process_single_condition(sub, proba, pos, degToPix=27.28):
    """
    Process data for a single subject and probability condition.
    """
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
            "posOffSet": pos_offsets,
            "sub": sub,
            "proba": proba,
            "trial": [i + 1 for i in range(len(pos_offsets))],
            "meanVelo": mean_velocities,
        }
    )

    # Handle training trials
    numOfTrials = len(condition_df)
    if numOfTrials > 240:
        condition_df = condition_df[condition_df["trial"] > numOfTrials - 240]

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
    allData.drop(colums=['sub'],inplace=True)
    # Merge with events data
    finalData = pd.concat(events, allData, axis=1)

    # Save to CSV if output file is specified
    if output_file is not None:
        finalData.to_csv(output_file, index=False)

    return finalData


# Example usage:
# Load your data
dirPath1 = "/envau/work/brainets/oueld.h/contextuaLearning/directionCue/results_voluntaryDirection"
filteredRawData1 = "JobLibProcessing.csv"
allEventsFile1 = "allEvents.csv"

dirPath2 = "/envau/work/brainets/oueld.h/contextuaLearning/ColorCue/data/"
filteredRawData2 = "JobLibProcessingCC.csv"
allEventsFile2 = "allEvents.csv"
paths = [dirPath1, dirPath2]
filteredRawDatas = [filteredRawData1, filteredRawData2]
allEventsFiles = [allEventsFile1, allEventsFile2]
for p, f, e in zip(paths, filteredRawDatas, allEventsFiles):
    df = pd.read_csv(os.path.join(p, f))
    events = pd.read_csv(os.path.join(p, e))
    output_file = os.path.join(p, "processedResults.csv")
    # Process data and save to CSV
    results = process_filtered_data_parallel(
        df=df, events=events, fOFF=80, latency=120, output_file="processedResults.csv"
    )