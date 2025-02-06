"""
Script to analyze the data from the voluntaryDirection task.
Saccades and blinks has been removed from the data. on the window -200ms(Fixation Offset) to 600ms the end of the trials
"""

import os
from scipy import stats
from scipy.stats import spearmanr
import statsmodels.formula.api as smf
import pingouin as pg
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

path = "/Volumes/work/brainets/oueld.h/contextuaLearning/directionCue/results_imposeDirection/"
jobLibData = "JobLibProcessing.csv"

# %%
# jlData = pd.read_csv(os.path.join(path, jobLibData))
# %%
# exampleJL = jlData[
#     (jlData["sub"] == 9)
#     & (jlData["proba"] == 0.75)
#     & (jlData["trial"] == 130)
#     #   & (jlData["time"] <= 100)
# ]
# # %%
# # Plotting one example
# # Plotting one example
# for t in exampleJL.trial.unique():
#     plt.plot(
#         exampleJL[exampleJL["trial"] == t].time,
#         exampleJL[exampleJL["trial"] == t].filtVelo,
#         alpha=0.5,
#     )
#     plt.plot(
#         exampleJL[exampleJL["trial"] == t].time,
#         exampleJL[exampleJL["trial"] == t].velo,
#         alpha=0.5,
#     )
#     plt.xlabel("Time in ms", fontsize=20)
#     plt.ylabel("Filteup Velocity in deg/s", fontsize=20)
#     plt.title(f"Filteup Velocity of trial {t} ", fontsize=30)
#     #plt.show()
# # %%
# for t in exampleJL.trial.unique():
#     plt.plot(
#         exampleJL[exampleJL["trial"] == t].time,
#         exampleJL[exampleJL["trial"] == t].xp,
#         alpha=0.5,
#     )
#     plt.plot(
#         exampleJL[exampleJL["trial"] == t].time,
#         exampleJL[exampleJL["trial"] == t].filtPos,
#         alpha=0.5,
#     )
#     plt.xlabel("Time in ms", fontsize=20)
#     plt.ylabel("Eye Position", fontsize=20)
#     plt.title(f"Filteup Velocity of trial {t} ", fontsize=30)
#     #plt.show()


# %%
pathFig = "/Users/mango/Contextual-Learning/directionCue/figures/imposeDirection"
allEventsFile = "/Volumes/work/brainets/oueld.h/contextuaLearning/directionCue/results_imposeDirection/allEvents.csv"
allEvents = pd.read_csv(allEventsFile)
df = pd.read_csv(
    "/Volumes/work/brainets/oueld.h/contextuaLearning/directionCue/results_imposeDirection/processedResultsWindow(80,120).csv"
)

# %%

df = df[(df["sub"] != 10) & (df["sub"] != 11)]
if "arrow" not in df.columns:
    df["arrow"] = df["chosen_arrow"].values
print(df)
# %%
# getting previous TD for each trial for each subject and each proba
for sub in df["sub"].unique():
    for p in df[df["sub"] == sub]["proba"].unique():
        df.loc[(df["sub"] == sub) & (df["proba"] == p), "TD_prev"] = df.loc[
            (df["sub"] == sub) & (df["proba"] == p), "target_direction"
        ].shift(1)
        df.loc[(df["sub"] == sub) & (df["proba"] == p), "arrow_prev"] = df.loc[
            (df["sub"] == sub) & (df["proba"] == p), "arrow"
        ].shift(1)
# %%
df[df["TD_prev"].isna()]
# %%
# df = df[~((df["sub"] == 6) & (df["proba"] == 0.5))]
# df = df[~((df["sub"] == 6) & (df["proba"] == 0.5))]
df = df[~(df["TD_prev"].isna())]
df[df["TD_prev"].isna()]
# %%
df["TD_prev"] = df["TD_prev"].apply(lambda x: "left" if x == -1 else "right")
df["interaction"] = list(zip(df["TD_prev"], df["arrow_prev"]))
# %%
badTrials = df[(df["meanVelo"] <= -8) | (df["meanVelo"] >= 8)]
badTrials
# %%
df = df[(df["meanVelo"] <= 8) & (df["meanVelo"] >= -8)]
# %%
df[df["meanVelo"] == df["meanVelo"].max()]
# %%
sns.histplot(data=df, x="meanVelo")
plt.show()
# %%
balance = df.groupby(["arrow", "sub", "proba"])["trial"].count().reset_index()
balance
# %%
for sub in balance["sub"].unique():
    sns.barplot(x="proba", y="trial", hue="arrow", data=balance[balance["sub"] == sub])
    plt.title(f"Subject {sub}")
    # plt.show()
# %%
dd = (
    df.groupby(["sub", "arrow", "proba"])[["meanVelo", "posOffSet"]]
    .mean()
    .reset_index()
)
dd
# %%
np.abs(dd.meanVelo.values).max()
# %%

meanVelo = dd[dd.arrow == "up"]["meanVelo"]
proba = dd[dd.arrow == "up"]["proba"]

# Spearman's rank correlation
correlation, p_value = spearmanr(meanVelo, proba)
print(f"Spearman's correlation (up): {correlation}, p-value: {p_value}")
# %%
meanVelo = dd[dd.arrow == "down"]["meanVelo"]
proba = dd[dd.arrow == "down"]["proba"]

# Spearman's rank correlation
correlation, p_value = spearmanr(meanVelo, proba)
print(f"Spearman's correlation (down): {correlation}, p-value: {p_value}")

# %%
meanVelo = dd[dd.proba == 0.75]["meanVelo"]
arrow = dd[dd.proba == 0.75]["arrow"]

# Spearman's rank correlation
correlation, p_value = spearmanr(meanVelo, arrow)
print(f"Spearman's correlation(Proba 0.75): {correlation}, p-value: {p_value}")


# %%

meanVelo = dd[dd.proba == 0.25]["meanVelo"]
arrow = dd[dd.proba == 0.25]["arrow"]

# Spearman's rank correlation
correlation, p_value = spearmanr(meanVelo, arrow)
print(f"Spearman's correlation(Proba 0.25): {correlation}, p-value: {p_value}")


# %%

meanVelo = dd[dd.proba == 0.5]["meanVelo"]
arrow = dd[dd.proba == 0.5]["arrow"]

# Spearman's rank correlation
correlation, p_value = spearmanr(meanVelo, arrow)
print(f"Spearman's correlation(Proba 0.5): {correlation}, p-value: {p_value}")


# %%
# cehcking the normality of the data
print(pg.normality(dd[dd.proba == 0.5]["meanVelo"]))
# %%
stat, p = stats.kstest(
    dd["meanVelo"], "norm", args=(dd["meanVelo"].mean(), dd["meanVelo"].std(ddof=1))
)
print(f"Statistic: {stat}, p-value: {p}")
# %%
x = dd["meanVelo"]
ax = pg.qqplot(x, dist="norm")
# plt.show()
# %%
sns.histplot(data=df, x="meanVelo", alpha=0.5)
# plt.show()
# %%
sns.histplot(data=df, x="meanVelo", hue="arrow", bins=20, alpha=0.5)
# plt.show()
# %%
sns.histplot(data=df, x="meanVelo", hue="proba", bins=20, palette="viridis", alpha=0.5)
# plt.show()
# %%


# Set up the FacetGrid
facet_grid = sns.FacetGrid(data=df, col="proba", col_wrap=3, height=8, aspect=1.5)

facet_grid.add_legend()
# Create pointplots for each sub
facet_grid.map_dataframe(
    sns.histplot,
    x="meanVelo",
    hue="arrow",
    hue_order=["down", "up"],
    alpha=0.3,
)
# Set titles for each subplot
for ax, p in zip(facet_grid.axes.flat, np.sort(df.proba.unique())):
    ax.set_title(f"ASEM: P(Right|up)=P(Left|down)={p}")
    ax.legend(["up", "down"])
# Adjust spacing between subplots
facet_grid.fig.subplots_adjust(
    wspace=0.2, hspace=0.2
)  # Adjust wspace and hspace as needed

# Show the plot
# plt.show()

# %%
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.pointplot(
    data=df,
    x="proba",
    y="meanVelo",
    capsize=0.1,
    errorbar="ci",
    hue="arrow",
    hue_order=["down", "up"],
)
_ = plt.title("ASEM Across Probabilities", fontsize=30)
plt.legend(fontsize=20)
plt.xlabel("P(Right|UP)=P(Left|DOWN)", fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel("ASEM (deg/s)", fontsize=30)
plt.ylim(-1.25, 1.25)
plt.savefig(pathFig + "/asemAcrossProbappFullProba.svg")
# plt.show()
# %%
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.pointplot(
    data=df[df.arrow == "up"],
    x="proba",
    y="meanVelo",
    capsize=0.1,
    errorbar="ci",
    hue="sub",
    alpha=0.7,
    palette="tab20",
)
_ = plt.title("ASEM Per Subject: Arrow UP", fontsize=30)
plt.legend(fontsize=20)
plt.xlabel("P(Right|UP)", fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel("ASEM (deg/s)", fontsize=30)
plt.savefig(pathFig + "/individualsUPFullProba.svg")
# plt.show()
# %%
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.pointplot(
    data=df[df.arrow == "down"],
    x="proba",
    y="meanVelo",
    capsize=0.1,
    errorbar="ci",
    hue="sub",
    alpha=0.7,
    palette="tab20",
)
_ = plt.title("ASEM Per Subject: Arrow DOWN", fontsize=30)
plt.legend(fontsize=20)
plt.xlabel("P(Left|DOWN)", fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel("ASEM (deg/s)", fontsize=30)
plt.savefig(pathFig + "/individualsDOWNFullProba.svg")
# plt.show()
# %%

model = smf.mixedlm(
    "meanVelo~C( arrow )",
    data=df[df.proba == 1.0],
    # re_formula="~arrow",
    groups=df[df.proba == 1.0]["sub"],
).fit()
model.summary()
# %%
model = smf.mixedlm(
    "meanVelo~C( arrow )",
    data=df[df.proba == 0.0],
    re_formula="~arrow",
    groups=df[df.proba == 0.0]["sub"],
).fit()
model.summary()

# %%
model = smf.mixedlm(
    "meanVelo~C( arrow )",
    data=df[df.proba == 0.75],
    re_formula="~arrow",
    groups=df[df.proba == 0.75]["sub"],
).fit()
model.summary()

# %%
model = smf.mixedlm(
    "meanVelo~C( arrow )",
    data=df[df.proba == 0.25],
    # re_formula="~arrow",
    groups=df[df.proba == 0.25]["sub"],
).fit()
model.summary()
# %%
model = smf.mixedlm(
    "meanVelo~C( arrow,Treatment('up') )",
    data=df[df.proba == 0.5],
    # re_formula="~arrow",
    groups=df[df.proba == 0.5]["sub"],
).fit()
model.summary()


# %%
model = smf.mixedlm(
    "meanVelo~ C(proba,Treatment(0.5))",
    data=df[df.arrow == "up"],
    re_formula="~proba",
    groups=df[df.arrow == "up"]["sub"],
).fit()
model.summary()

# %%
model = smf.mixedlm(
    "meanVelo~ C(proba,Treatment(0.5))",
    data=df[df.arrow == "down"],
    re_formula="~proba",
    groups=df[df.arrow == "down"]["sub"],
).fit()
model.summary()


# %%
downarrowsPalette = ["#0000FF", "#A2D9FF"]
uparrowsPalette = ["#FFA500", "#FFD699"]
# %%
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.barplot(
    x="proba",
    y="meanVelo",
    hue="arrow",
    hue_order=["down", "up"],
    data=df,
    errorbar="ci",
    palette=[downarrowsPalette[1], uparrowsPalette[1]],
)
plt.title("Anticipatory Velocity across  probabilites", fontsize=30)
plt.xlabel("P(Right|up)=P(Left|down)", fontsize=30)
plt.ylabel("ASEM (deg/s)", fontsize=30)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
ylim = (-1.5, 1.5)
plt.legend(fontsize=20)
plt.savefig(pathFig + "/meanVeloarrowsFullProba.svg", transparent=True)
plt.show()
# %%
df_prime = df[
    [
        "sub",
        "trial",
        "proba",
        "arrow",
        "target_direction",
        "TD_prev",
        "posOffSet",
        "meanVelo",
    ]
]
df_prime
# %%
df["TD_prev"].unique()
# %%
df["arrow_prev"].unique()
# %%
learningCurve = (
    df_prime.groupby(["sub", "proba", "arrow", "TD_prev"])
    .mean()[["posOffSet", "meanVelo"]]
    .reset_index()
)


learningCurve
# %%
df_prime.groupby(["proba", "arrow", "TD_prev"]).count()[["meanVelo"]]

# %%
df_prime.groupby(["sub", "proba", "arrow", "TD_prev"]).count()[["meanVelo"]]

# %%
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.barplot(
    x="proba",
    y="posOffSet",
    data=df[df.arrow == "up"],
    color=uparrowsPalette[0],
)
plt.title("Position Offset: arrow up", fontsize=30)
plt.xlabel("P(Right|up)")
plt.ylabel("Position Offset", fontsize=20)
plt.savefig(pathFig + "/posOffSetupFullProba.svg")
# plt.show()
# %%
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.barplot(
    x="proba",
    y="posOffSet",
    hue="TD_prev",
    palette=uparrowsPalette,
    data=df[df.arrow == "up"],
)
plt.legend(fontsize=20)
plt.title("Position Offset: arrow up Given Previous Target Direction", fontsize=30)
plt.xlabel("P(Right|up)", fontsize=20)
plt.ylabel("Position Offset", fontsize=20)
plt.savefig(pathFig + "/posOffSetupTDFullProba.svg")
# plt.show()
# %%
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.barplot(
    x="proba",
    y="meanVelo",
    data=df[df.arrow == "up"],
    errorbar="se",
    color=uparrowsPalette[1],
)
plt.title("Anticipatory Smooth Eye Movement: Arrow Up", fontsize=30)
plt.xlabel("P(Right|UP)", fontsize=25)
plt.ylabel("ASEM (deg/s)", fontsize=25)
# plt.ylim(-1.5, 1.5)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.savefig(pathFig + "/meanVeloupFullProba.svg")
# plt.show()
# %%
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.barplot(
    x="proba",
    y="meanVelo",
    hue="TD_prev",
    palette=uparrowsPalette,
    data=df[df.arrow == "up"],
    errorbar="ci",
)
plt.legend(fontsize=20)
plt.title("Anticipatory Velocity Given Previous TD: arrow UP ", fontsize=30)
plt.xlabel("P(Right|UP)", fontsize=25)
plt.ylabel("ASEM (deg/s)", fontsize=25)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.savefig(os.path.join(pathFig, "meanVeloupTDFullProba.svg"))
# plt.show()
# %%
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.barplot(
    x="proba",
    y="posOffSet",
    data=df[df.arrow == "down"],
)
plt.title("Position Offset: arrow down", fontsize=30)
plt.xlabel("P(Left|down)", fontsize=30)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.ylabel("Position Offset", fontsize=30)
plt.savefig(pathFig + "/posOffSetdownFullProba.svg")
# plt.show()
# %%
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.barplot(
    x="proba",
    y="posOffSet",
    hue="TD_prev",
    palette=downarrowsPalette,
    data=df[df.arrow == "down"],
)
plt.legend(fontsize=20)
plt.title("Position Offset:arrow down \n  ", fontsize=30)
plt.xlabel("P(Left|down)", fontsize=20)
plt.ylabel("Position Offset", fontsize=20)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.savefig(pathFig + "/posOffSetdownTDFullProba.svg")
# plt.show()
# %%
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.barplot(
    x="proba", y="meanVelo", data=df[df.arrow == "down"], color=downarrowsPalette[1]
)
plt.title("Anticipatory Smooth Eye Movement: Arrow Down", fontsize=30)
plt.xlabel("P(Left|down)", fontsize=30)
plt.ylabel("ASEM (deg/s)", fontsize=30)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.savefig(pathFig + "/meanVelodownFullProba.svg")
# plt.show()
# %%
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.barplot(
    x="proba",
    y="meanVelo",
    hue="TD_prev",
    hue_order=["right", "left"],
    palette=downarrowsPalette,
    data=df[df.arrow == "down"],
)
plt.legend(fontsize=20)
plt.title("Anticipatory Velocity Given Previous TD: Arrow DOWN ", fontsize=30)
plt.xlabel("P(Left|Down)", fontsize=25)
plt.ylabel("ASEM (deg/s)", fontsize=25)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.savefig(pathFig + "/meanVelodownTDFullProba.svg")
# plt.show()
# Adding the interacton between  previous arrow and previous TD.
# %%
df["interaction"] = list(zip(df["TD_prev"], df["arrow_prev"]))
df_prime = df[
    [
        "sub",
        "trial",
        "proba",
        "arrow",
        "interaction",
        "posOffSet",
        "meanVelo",
    ]
]
df_prime
# %%
learningCurveInteraction = (
    df_prime.groupby(["sub", "proba", "interaction", "arrow"])
    .mean()[["posOffSet", "meanVelo"]]
    .reset_index()
)

# %%
learningCurveInteraction
# %%
df_prime.groupby(["sub", "proba", "interaction", "arrow"]).count()[["meanVelo"]]

# %%
df_prime.groupby(["proba", "interaction", "arrow"]).count()[["posOffSet", "meanVelo"]]

# %%
learningCurveInteraction
# %%
df["interaction"].unique()
# %%
colorsPalettes = ["#0000FF", "#FFA500", "#A2D9FF", "#FFD699"]
hue_order = [("right", "down"), ("right", "up"), ("left", "down"), ("left", "up")]
# %%
# Create a figure and axis
fig = plt.figure()

# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()

sns.barplot(
    x="proba",
    y="posOffSet",
    palette=colorsPalettes,
    hue="interaction",
    hue_order=hue_order,
    data=df[df.arrow == "up"],
)
plt.legend(fontsize=20)
plt.title(
    "Position Offset:arrow up\n Interaction of Previous Target Direction & arrow Chosen ",
    fontsize=30,
)
plt.xlabel("P(Right|up)", fontsize=20)
plt.ylabel("Position Offset", fontsize=20)
plt.savefig(pathFig + "/posOffSetUpupInteractionFullProba.svg")
# plt.show()
# %%
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.barplot(
    x="proba",
    y="meanVelo",
    palette=colorsPalettes,
    hue="interaction",
    hue_order=hue_order,
    data=df[df.arrow == "up"],
)
plt.title(
    "ASEM: Arrow Up\n Interaction of Previous Target Direction & arrow Chosen",
    fontsize=30,
)
plt.xlabel("P(Right|Up)", fontsize=30)
plt.ylabel("ASEM (deg/s)", fontsize=30)
plt.ylim(-1.5, 1.5)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.legend(fontsize=20)
plt.savefig(pathFig + "/meanVeloupInteractionFullProba.svg")
# plt.show()
# %%
fig = plt.figure()

# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.barplot(
    x="proba",
    y="posOffSet",
    palette="viridis",
    hue="interaction",
    data=df[df.arrow == "down"],
)
plt.title(
    "Position Offset: arrow down\n Interaction of Previous Target Direction & arrow Chosen",
    fontsize=30,
)
plt.xlabel("P(Left|down)", fontsize=20)
plt.ylabel("Position Offset", fontsize=20)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.legend(fontsize=20)
plt.savefig(pathFig + "/posOffSetdownInteractionFullProba.svg")
# plt.show()
# %%
fig = plt.figure()

# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.barplot(
    x="proba",
    y="meanVelo",
    palette=colorsPalettes,
    hue="interaction",
    hue_order=hue_order,
    data=df[df.arrow == "down"],
)
plt.title(
    "ASEM:Arrow Down\n Interaction of Previous Target Direction & arrow Chosen",
    fontsize=30,
)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.legend(fontsize=20)
plt.ylabel("ASEM (deg/s)", fontsize=30)
plt.ylim(-1.5, 1.5)
plt.xlabel("P(Left|Down)", fontsize=30)
plt.savefig(pathFig + "/meanVeloDownInteractionFullProba.svg")
# plt.show()
# %%
df
# %%
dd = df.groupby(["sub", "proba", "arrow", "TD_prev"])["meanVelo"].mean().reset_index()
dd
# %%
model = smf.mixedlm(
    "meanVelo~  C(arrow)*C(TD_prev)",
    data=df[df.proba == 0.25],
    # re_formula="~arrow",
    groups=df[df.proba == 0.25]["sub"],
).fit()
model.summary()
# %%
model = smf.mixedlm(
    "meanVelo~  C(arrow)*C(TD_prev)",
    data=df[df.proba == 0.75],
    # re_formula="~arrow",
    groups=df[df.proba == 0.75]["sub"],
).fit()
model.summary()
# %%
model = smf.mixedlm(
    "meanVelo~  C(arrow)*C(TD_prev)",
    data=df[df.proba == 0.5],
    # re_formula="~arrow",
    groups=df[df.proba == 0.5]["sub"],
).fit()
model.summary()
# %%


# Doing the same analysis without the determinstic condition
# %%
df = df[~((df["proba"] == 0) | (df["proba"] == 1))]
# df = df[~((df["sub"] == 6) & (df["proba"] == 0.5))]
# %%

balance = df.groupby(["arrow", "sub", "proba"])["trial"].count().reset_index()
balance
# %%
for sub in balance["sub"].unique():
    sns.barplot(x="proba", y="trial", hue="arrow", data=balance[balance["sub"] == sub])
    plt.title(f"Subject {sub}")
    # plt.show()
# %%
dd = (
    df.groupby(["sub", "arrow", "proba"])[["meanVelo", "posOffSet"]]
    .mean()
    .reset_index()
)
dd
# %%
np.abs(dd.meanVelo.values).max()
# %%

meanVelo = dd[dd.arrow == "up"]["meanVelo"]
proba = dd[dd.arrow == "up"]["proba"]

# Spearman's rank correlation
correlation, p_value = spearmanr(meanVelo, proba)
print(f"Spearman's correlation (up): {correlation}, p-value: {p_value}")
# %%
meanVelo = dd[dd.arrow == "down"]["meanVelo"]
proba = dd[dd.arrow == "down"]["proba"]

# Spearman's rank correlation
correlation, p_value = spearmanr(meanVelo, proba)
print(f"Spearman's correlation (down): {correlation}, p-value: {p_value}")

# %%
meanVelo = dd[dd.proba == 0.75]["meanVelo"]
arrow = dd[dd.proba == 0.75]["arrow"]

# Spearman's rank correlation
correlation, p_value = spearmanr(meanVelo, arrow)
print(f"Spearman's correlation(Proba 0.75): {correlation}, p-value: {p_value}")


# %%

meanVelo = dd[dd.proba == 0.25]["meanVelo"]
arrow = dd[dd.proba == 0.25]["arrow"]

# Spearman's rank correlation
correlation, p_value = spearmanr(meanVelo, arrow)
print(f"Spearman's correlation(Proba 0.25): {correlation}, p-value: {p_value}")


# %%

meanVelo = dd[dd.proba == 0.5]["meanVelo"]
arrow = dd[dd.proba == 0.5]["arrow"]

# Spearman's rank correlation
correlation, p_value = spearmanr(meanVelo, arrow)
print(f"Spearman's correlation(Proba 0.5): {correlation}, p-value: {p_value}")


# %%
# cehcking the normality of the data
print(pg.normality(dd[dd.proba == 0.5]["meanVelo"]))
# %%
stat, p = stats.kstest(
    dd["meanVelo"], "norm", args=(dd["meanVelo"].mean(), dd["meanVelo"].std(ddof=1))
)
print(f"Statistic: {stat}, p-value: {p}")
# %%
x = dd["meanVelo"]
ax = pg.qqplot(x, dist="norm")
plt.show()
# %%
sns.histplot(data=df, x="meanVelo", alpha=0.5)
plt.show()
# %%
sns.histplot(data=df, x="meanVelo", hue="arrow", bins=20, alpha=0.5)
# plt.show()
# %%
sns.histplot(data=df, x="meanVelo", hue="proba", bins=20, palette="viridis", alpha=0.5)
# plt.show()
# %%


# Set up the FacetGrid
facet_grid = sns.FacetGrid(data=df, col="proba", col_wrap=3, height=8, aspect=1.5)

# Create pointplots for each sub
facet_grid.map_dataframe(
    sns.histplot,
    x="meanVelo",
    hue="arrow",
    hue_order=["down", "up"],
    alpha=0.3,
    # palette=[downarrowsPalette[0], uparrowsPalette[0]],
)

# Add legends
facet_grid.add_legend()

# Set titles for each subplot
for ax, p in zip(facet_grid.axes.flat, np.sort(df.proba.unique())):
    ax.set_title(f"ASEM: P(Right|up)=P(Left|down)={p}")
    ax.legend(["up", "down"])
# Adjust spacing between subplots
facet_grid.fig.subplots_adjust(
    wspace=0.2, hspace=0.2
)  # Adjust wspace and hspace as needed

# Show the plot
# plt.show()

# %%
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.pointplot(
    data=df,
    x="proba",
    y="meanVelo",
    capsize=0.1,
    errorbar="ci",
    hue="arrow",
    hue_order=["down", "up"],
)
_ = plt.title("ASEM across porbabilities", fontsize="30")
plt.legend(fontsize=20)
plt.xlabel("P(Right|UP)", fontsize=30)
plt.ylabel("ASEM (deg/s)", fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylim(-0.5, 0.5)
plt.savefig(pathFig + "/asemAcrossProbapp.svg")
# plt.show()
# %%
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.pointplot(
    data=df[df.arrow == "up"],
    x="proba",
    y="meanVelo",
    capsize=0.1,
    errorbar="ci",
    hue="sub",
    alpha=0.7,
    palette="tab20",
)
_ = plt.title("ASEM Per Subject: Arrow UP", fontsize=30)
plt.legend(fontsize=20)
plt.xlabel("P(Right|UP)", fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylim(-1.5, 1.5)
plt.ylabel("ASEM (deg/s)", fontsize=30)
plt.savefig(pathFig + "/individualsUP.svg")
# plt.show()
# %%
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.pointplot(
    data=df[df.arrow == "down"],
    x="proba",
    y="meanVelo",
    capsize=0.1,
    errorbar="ci",
    hue="sub",
    alpha=0.7,
    palette="tab20",
)
_ = plt.title("ASEM Per Subject: Arrow DOWN", fontsize=30)
plt.legend(fontsize=20)
plt.xlabel("P(Left|DOWN)", fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel("ASEM (deg/s)", fontsize=30)
plt.ylim(-1.75, 1.75)
plt.savefig(pathFig + "/individualsDOWN.svg")
# plt.show()
# %%

model = smf.mixedlm(
    "meanVelo~C( arrow )",
    data=df[df.proba == 0.75],
    re_formula="~arrow",
    groups=df[df.proba == 0.75]["sub"],
).fit()
model.summary()

# %%
model = smf.mixedlm(
    "meanVelo~C( arrow )",
    data=df[df.proba == 0.25],
    # re_formula="~arrow",
    groups=df[df.proba == 0.25]["sub"],
).fit()
model.summary()
# %%
model = smf.mixedlm(
    "meanVelo~C( arrow )",
    data=df[df.proba == 0.5],
    # re_formula="~arrow",
    groups=df[df.proba == 0.5]["sub"],
).fit()
model.summary()


# %%
model = smf.mixedlm(
    "meanVelo~ C(proba, Treatment(0.5))",
    data=df[df.arrow == "up"],
    # re_formula="~proba",
    groups=df[df.arrow == "up"]["sub"],
).fit()
model.summary()

# %%
model = smf.mixedlm(
    "meanVelo~ C(proba, Treatment(0.5))",
    data=df[df.arrow == "down"],
    # re_formula="~proba",
    groups=df[df.arrow == "down"]["sub"],
).fit()
model.summary()


# %%
downarrowsPalette = ["#0000FF", "#A2D9FF"]
uparrowsPalette = ["#FFA500", "#FFD699"]
# %%
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.barplot(
    x="proba",
    y="meanVelo",
    hue="arrow",
    hue_order=dd["arrow"].unique(),
    data=df,
    errorbar="ci",
    palette=[downarrowsPalette[1], uparrowsPalette[1]],
)
plt.title("ASEM Across 3 Different Probabilities", fontsize=30)
plt.xlabel("P(Right|UP)=P(Left|DOWN)", fontsize=30)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.ylabel("ASEM (deg/s)", fontsize=30)
plt.ylim(-0.75, 0.75)
plt.legend(fontsize=20)
plt.savefig(pathFig + "/meanVeloarrows.svg", transparent=True)
plt.show()
# %%

df_prime = df[
    [
        "sub",
        "trial",
        "proba",
        "arrow",
        "target_direction",
        "TD_prev",
        "posOffSet",
        "meanVelo",
    ]
]
df_prime
# %%
learningCurve = (
    df_prime.groupby(["sub", "proba", "arrow", "TD_prev"])
    .mean()[["posOffSet", "meanVelo"]]
    .reset_index()
)


learningCurve
# %%
df_prime.groupby(["proba", "arrow", "TD_prev"]).count()[["posOffSet", "meanVelo"]]

# %%
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.barplot(
    x="proba",
    y="posOffSet",
    data=df[df.arrow == "up"],
    color=uparrowsPalette[0],
)
plt.title("Position Offset: arrow up", fontsize=30)
plt.xlabel("P(Right|up)")
plt.ylabel("Position Offset", fontsize=20)
plt.savefig(pathFig + "/posOffSetup.svg")
# plt.show()
# %%
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.barplot(
    x="proba",
    y="posOffSet",
    hue="TD_prev",
    palette=uparrowsPalette,
    data=df[df.arrow == "up"],
)
plt.legend(fontsize=20)
plt.title("Position Offset: arrow up Given Previous Target Direction", fontsize=30)
plt.xlabel("P(Right|up)", fontsize=20)
plt.ylabel("Position Offset", fontsize=20)
plt.savefig(pathFig + "/posOffSetupTD.svg")
# plt.show()
# %%
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.barplot(
    x="proba",
    y="meanVelo",
    data=df[df.arrow == "up"],
    errorbar="ci",
    color=uparrowsPalette[1],
)
plt.title("Anticipatory Smooth Eye Movement: Arrow Up", fontsize=30)
plt.xlabel("P(Right|UP)", fontsize=25)
plt.ylabel("ASEM (deg/s)", fontsize=25)
plt.ylim(-0.75, 0.75)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.savefig(pathFig + "/meanVeloup.svg")
# plt.show()
# %%
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.barplot(
    x="proba",
    y="meanVelo",
    hue="TD_prev",
    palette=uparrowsPalette,
    data=df[df.arrow == "up"],
    errorbar="ci",
)
plt.legend(fontsize=20)
plt.title("Anticipatory Velocity Given Previous TD: arrow UP ", fontsize=30)
plt.xlabel("P(Right|UP)", fontsize=25)
plt.ylabel("ASEM (deg/s)", fontsize=25)
plt.ylim(-0.75, 0.75)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.savefig(os.path.join(pathFig, "meanVeloupTD.svg"))
# plt.show()
# %%
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.barplot(
    x="proba",
    y="posOffSet",
    data=df[df.arrow == "down"],
)
plt.title("Position Offset: arrow down", fontsize=30)
plt.xlabel("P(Left|down)", fontsize=30)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.ylabel("Position Offset", fontsize=30)
plt.savefig(pathFig + "/posOffSetdown.svg")
# plt.show()
# %%
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.barplot(
    x="proba",
    y="posOffSet",
    hue="TD_prev",
    palette=downarrowsPalette,
    data=df[df.arrow == "down"],
)
plt.legend(fontsize=20)
plt.title("Position Offset:arrow down \n  ", fontsize=30)
plt.xlabel("P(Left|down)", fontsize=20)
plt.ylabel("Position Offset", fontsize=20)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.savefig(pathFig + "/posOffSetdownTD.svg")
# plt.show()
# %%
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.barplot(
    x="proba", y="meanVelo", data=df[df.arrow == "down"], color=downarrowsPalette[1]
)
plt.title("Anticipatory Smooth Eye Movement: Arrow Down", fontsize=30)
plt.xlabel("P(Left|down)", fontsize=30)
plt.ylabel("ASEM (deg/s)", fontsize=30)
plt.ylim(-0.75, 0.75)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.savefig(pathFig + "/meanVelodown.svg")
# plt.show()
# %%
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.barplot(
    x="proba",
    y="meanVelo",
    hue="TD_prev",
    hue_order=df["TD_prev"].unique(),
    palette=downarrowsPalette,
    data=df[df.arrow == "down"],
)
plt.legend(fontsize=20)
plt.title("Anticipatory Velocity Given Previous TD: Arrow DOWN ", fontsize=30)
plt.xlabel("P(Left|Down)", fontsize=25)
plt.ylabel("ASEM (deg/s)", fontsize=25)
plt.ylim(-0.75, 0.75)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.savefig(pathFig + "/meanVelodownTD.svg")
# plt.show()
# Adding the interacrion between  previous arrow and previous TD.
# %%
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.barplot(
    x="proba",
    y="meanVelo",
    hue="arrow",
    hue_order=["down", "up"],
    palette=[downarrowsPalette[0], uparrowsPalette[0]],
    data=df[df.TD_prev == "right"],
)
plt.legend(fontsize=20)
plt.title("Anticipatory Velocity Given Previous TD(Right) ", fontsize=30)
plt.xlabel("P(Left|Down)", fontsize=25)
plt.ylabel("ASEM (deg/s)", fontsize=25)
plt.ylim(-0.75, 0.75)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.savefig(pathFig + "/meanVeloTDRight.svg")
# plt.show()
# Adding the interacrion between  previous arrow and previous TD.
# %%
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.barplot(
    x="proba",
    y="meanVelo",
    hue="arrow",
    hue_order=["down", "up"],
    palette=[downarrowsPalette[1], uparrowsPalette[1]],
    data=df[df.TD_prev == "left"],
)
plt.legend(fontsize=20)
plt.title("Anticipatory Velocity Given Previous TD(Left) ", fontsize=30)
plt.xlabel("P(Left|Down)", fontsize=25)
plt.ylabel("ASEM (deg/s)", fontsize=25)
plt.ylim(-0.75, 0.75)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.savefig(pathFig + "/meanVeloTDLeft.svg")
# plt.show()
# Adding the interacrion between  previous arrow and previous TD.
# %%
df_prime = df[
    [
        "sub",
        "trial",
        "proba",
        "arrow",
        "interaction",
        "posOffSet",
        "meanVelo",
    ]
]
df_prime
# %%

learningCurveInteraction = (
    df_prime.groupby(["sub", "proba", "interaction", "arrow"])
    .mean()[["posOffSet", "meanVelo"]]
    .reset_index()
)

# %%
df.columns
# %%
df_prime.groupby(["sub", "proba", "interaction", "arrow"]).count()[["meanVelo"]]

# %%
learningCurveInteraction
# %%
df["interaction"].unique()
# %%
colorsPalettes = ["#0000FF", "#FFA500", "#A2D9FF", "#FFD699"]
hue_order = [("right", "down"), ("right", "up"), ("left", "down"), ("left", "up")]
# %%
# Create a figure and axis
fig = plt.figure()

# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()

sns.barplot(
    x="proba",
    y="posOffSet",
    palette=colorsPalettes,
    hue="interaction",
    hue_order=hue_order,
    data=df[df.arrow == "up"],
)
plt.legend(fontsize=20)
plt.title(
    "Position Offset:arrow up\n Interaction of Previous Target Direction & arrow Chosen ",
    fontsize=30,
)
plt.xlabel("P(Right|up)", fontsize=20)
plt.ylabel("Position Offset", fontsize=20)
plt.savefig(pathFig + "/posOffSetUpupInteraction.svg")
# plt.show()
# %%
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.barplot(
    x="proba",
    y="meanVelo",
    palette=colorsPalettes,
    hue="interaction",
    hue_order=hue_order,
    data=df[df.arrow == "up"],
)
plt.title(
    "Anticipatory Smooth Eye Movement: Arrow Up\n Interaction of Previous Target Direction & arrow Chosen",
    fontsize=30,
)
plt.xlabel("P(Right|Up)", fontsize=30)
plt.ylabel("ASEM (deg/s)", fontsize=30)
plt.ylim(-1.25, 1.25)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.legend(fontsize=20)
plt.savefig(pathFig + "/meanVeloUpInteraction.svg")
# plt.show()
# %%
fig = plt.figure()

# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.barplot(
    x="proba",
    y="posOffSet",
    palette="viridis",
    hue="interaction",
    data=df[df.arrow == "down"],
)
plt.title(
    "Position Offset: arrow down\n Interaction of Previous Target Direction & arrow Chosen",
    fontsize=30,
)
plt.xlabel("P(Left|down)", fontsize=20)
plt.ylabel("Position Offset", fontsize=20)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.legend(fontsize=20)
plt.savefig(pathFig + "/posOffSetdownInteraction.svg")
# plt.show()
# %%
fig = plt.figure()

# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.barplot(
    x="proba",
    y="meanVelo",
    palette=colorsPalettes,
    hue="interaction",
    hue_order=hue_order,
    data=df[df.arrow == "down"],
)
plt.title(
    "ASEM:Arrow Down\n Interaction of Previous Target Direction & arrow Chosen",
    fontsize=30,
)
plt.ylim(-1.25, 1.25)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.legend(fontsize=20)
plt.xlabel("P(Left|Down)", fontsize=30)
plt.ylabel("ASEM (deg/s)", fontsize=30)
plt.savefig(pathFig + "/meanVeloDownInteraction.svg")
# plt.show()
# %%
df
# %%
dd = df.groupby(["sub", "proba", "arrow", "TD_prev"])["meanVelo"].mean().reset_index()
dd
# %%
model = smf.mixedlm(
    "meanVelo~  C(arrow,Treatment('up'))*C(TD_prev)",
    data=df[df.proba == 0.25],
    # re_formula="~arrow",
    groups=df[df.proba == 0.25]["sub"],
).fit()
model.summary()
# %%
model = smf.mixedlm(
    "meanVelo~  C(arrow,Treatment('up'))*C(TD_prev)",
    data=df[df.proba == 0.75],
    re_formula="~arrow",
    groups=df[df.proba == 0.75]["sub"],
).fit()
model.summary()
# %%
model = smf.mixedlm(
    "meanVelo~  C(arrow)*C(TD_prev)",
    data=df[df.proba == 0.5],
    # re_formula="~arrow",
    groups=df[df.proba == 0.5]["sub"],
).fit()
model.summary()
# %%
# %%
# sampling Bias:
df["sampling"] = list(zip(df["arrow"], df["arrow_prev"]))
# %%
# Group by and count
# Group by and count
grouped = (
    df.groupby(["sub", "proba", "sampling"])["meanVelo"]
    .count()
    .reset_index(name="count")
)

grouped["sampling"] = grouped["sampling"].astype(str)
# Calculate percentages using transform to keep the original DataFrame structure
grouped["percentage"] = grouped.groupby(["sub", "proba"])["count"].transform(
    lambda x: x / x.sum() * 100
)
grouped
# %%
grouped["sampling"].count()
# %%
sns.histplot(data=grouped, x="sampling")
plt.show()
# %%
for s in grouped["sub"].unique():
    # Set up the FacetGrid
    facet_grid = sns.FacetGrid(
        data=grouped[grouped["sub"] == s], col="proba", col_wrap=3, height=8, aspect=1.5
    )

    facet_grid.add_legend()
    # Create pointplots for each sub
    facet_grid.map_dataframe(
        sns.barplot,
        x="sampling",
        y="percentage",
    )
    # Set titles for each subplot
    for ax, p in zip(facet_grid.axes.flat, np.sort(df.proba.unique())):
        ax.set_title(f"Sampling Bias p={p} : P(C(t+1)|C(t))")
    # Adjust spacing between subplots
    facet_grid.fig.subplots_adjust(
        wspace=0.2, hspace=0.2
    )  # Adjust wspace and hspace as needed

    # Show the plot
    plt.show()
