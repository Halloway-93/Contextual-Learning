
import pandas as pd
import piecewise_regression as pw
import matplotlib.pyplot as plt

# %%
df = pd.read_csv(
    "/envau/work/brainets/oueld.h/contextuaLearning/directionCue/rawData.csv"
)
# %%
# Getting the region of interest
fixOff = -200
latency = 120
df = df[(df.time >= fixOff + 20) & (df.time <= latency)]
# %%
# Test on one subject and one condition and one trial
# List of slopes
# starting with 2 breakpoints
maxBreakPoints=2
slopes = [f"alpha{i+1}" for i in range(maxBreakPoints+1)]
# %%

lastTrial = df[(df["trial"] > 150)]
allFit = []
for sub in lastTrial.subject.unique():
    df_sub = lastTrial[lastTrial["subject"] == sub]
    allFitConditions= []
    for p in df_sub.proba.unique():
        df_sub_p = df_sub[df_sub["proba"] == p]
        # List of slopes for each trial
        allFitTrials = []
        for t in df_sub_p.trial.unique():
            x = df_sub_p[df_sub_p["trial"] == t].time.values
            y = df_sub_p[df_sub_p["trial"] == t].xp.values
            pw_fit = pw.Fit(x, y, n_breakpoints=2)

            pw_results = pw_fit.get_results()
            pw_estimates = pw_results["estimates"]
            #storing the alphas for each fitTrial
            alphas=[]
            for s in slopes:
                if s in pw_estimates.keys():
                    alpha=pw_estimates[s]['estimate']
                    alphas.append(alpha)
                    print(f"{s}:",alpha)
            allFitTrials.append(alphas)
        allFitConditions.append(allFitTrials)
    allFit.append(allFitConditions)

