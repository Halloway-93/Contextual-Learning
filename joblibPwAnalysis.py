import piecewise_regression as pw
import numpy as np
import json
import pandas as pd
import os


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


def fit_trial(df_sub_p, trial, slopes, breakpoints):
    x = df_sub_p[df_sub_p["trial"] == trial].time.values
    y = interpolateData(df_sub_p[df_sub_p["trial"] == trial].filtPos.values)
    # Fitting the piecewise regression
    pw_fit = pw.Fit(x, y, n_breakpoints=3)
    # Getting the slopes estimates
    pw_results = pw_fit.get_results()
    pw_estimates = pw_results["estimates"]
    if pw_estimates is None:
        return ([], [])
    # Storing the alphas for each fitTrial
    alphas = []
    bps = []
    for s in slopes:
        if s in pw_estimates.keys():
            alpha = pw_estimates[s]["estimate"]
            alphas.append(alpha)
    for b in breakpoints:
        if b in pw_estimates.keys():
            bp = pw_estimates[b]["estimate"]
            bps.append(bp)
    return (alphas, bps)


def fit_condition(df_sub, proba, slopes, breakpoints):
    df_sub_p = df_sub[df_sub["proba"] == proba]
    trials = df_sub_p.trial.unique()
    allFitTrials = {
        trial: fit_trial(df_sub_p, trial, slopes, breakpoints) for trial in trials
    }
    return allFitTrials


def fit_subject(df, sub, slopes, breakpoints):
    df_sub = df[df["sub"] == sub]
    probas = df_sub.proba.unique()
    allFitConditions = {
        proba: fit_condition(df_sub, proba, slopes, breakpoints) for proba in probas
    }
    return allFitConditions


def pwAnalysis(df):
    maxBreakPoints = 3
    slopes = [f"alpha{i+1}" for i in range(maxBreakPoints + 1)]
    breakpoints = [f"breakpoint{i+1}" for i in range(maxBreakPoints)]
    subjects = df["sub"].unique()
    allFit = {sub: fit_subject(df, sub, slopes, breakpoints) for sub in subjects}
    return allFit


def save_fit_data(fit_data, filename):
    with open(filename, "w") as f:
        json.dump(fit_data, f)


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
for p, f, e in zip(paths, filteredRawDatas):
    df = pd.read_csv(os.path.join(p, f))
    output_file = os.path.join(p, "fitData.csv")
    fit_data = pwAnalysis(df)
    save_fit_data(fit_data,output_file)
    # Process data and save to CSV
# Example usage
# df = pd.read_csv('your_data.csv')
# Save fit_data to a file or database as needed
