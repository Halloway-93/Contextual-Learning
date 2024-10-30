from joblib import Parallel, delayed
import piecewise_regression as pw
import numpy as np


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
    allFitTrials = {trial: fit_trial(df_sub_p, trial, slopes, breakpoints) for trial in trials}
    return allFitTrials

def fit_subject(df, sub, slopes, breakpoints):
    df_sub = df[df["sub"] == sub]
    probas = df_sub.proba.unique()
    allFitConditions = {proba: fit_condition(df_sub, proba, slopes, breakpoints) for proba in probas}
    return allFitConditions

def pwAnalysis(df):
    maxBreakPoints = 3
    slopes = [f"alpha{i+1}" for i in range(maxBreakPoints + 1)]
    breakpoints = [f"breakpoint{i+1}" for i in range(maxBreakPoints)]
    subjects = df["sub"].unique()
    allFit = {sub: fit_subject(df, sub, slopes, breakpoints) for sub in subjects}
    return allFit

# Example usage
# df = pd.read_csv('your_data.csv')
# fit_data = pwAnalysis(df)
# Save fit_data to a file or database as needed
