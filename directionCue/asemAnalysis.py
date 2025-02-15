"""
Script to analyze the data from the voluntaryDirection task.
Saccades and blinks has been removed from the data. on the window -200ms(Fixation Offset) to 600ms the end of the trials
"""

import os
import statsmodels.api as sm
from scipy import stats
from scipy.stats import spearmanr
import statsmodels.formula.api as smf
import pingouin as pg
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

path = "/Volumes/work/brainets/oueld.h/contextuaLearning/directionCue/results_voluntaryDirection"
jobLibData = "JobLibProcessing.csv"

# %%
jlData = pd.read_csv(os.path.join(path, jobLibData))
# jlData = pd.read_csv("/Users/mango/PhD/output.csv")
# %%
jlData.columns
# %%
exampleJL = jlData[
    (jlData["sub"] == 1)
    & (jlData["proba"] == 0.25)
    & (jlData["trial"] == 193)
    #   & (jlData["time"] <= 100)
].copy()
# %%
# exampleJL.time = exampleJL.time.values / 1000
# %%
# emTypes = exampleJL["EYE_MOVEMENT_TYPE"].unique()
# emTypes
# %%
assert isinstance(exampleJL, pd.DataFrame)
# # Plotting one example
for t in exampleJL.trial.unique():
    sns.lineplot(
        data=exampleJL,
        x="time",
        # y="speed_8",
        y="filtVelo",
        alpha=0.5,
    )
    sns.scatterplot(
        data=exampleJL,
        x="time",
        # y="speed_16",
        y="filtVeloFilt",
        alpha=0.5,
    )
    plt.xlabel("Time in ms", fontsize=20)
    plt.ylabel("Filteup Velocity in deg/s", fontsize=20)
    plt.title(f"Filteup Velocity of trial {t} ", fontsize=30)
    plt.show()
# # %%
for t in exampleJL.trial.unique():
    assert isinstance(exampleJL[exampleJL.trial == t], pd.DataFrame)
    sns.scatterplot(
        data=exampleJL[exampleJL.trial == t],
        x="time",
        # y="x",
        y="filtPos",
        alpha=0.5,
        # hue="EYE_MOVEMENT_TYPE",
    )

    plt.xlabel("Time in ms", fontsize=20)
    plt.ylabel("Eye position in pix", fontsize=20)
    plt.title(f"Eye position of trial {t} ", fontsize=30)
    plt.show()
# %%
for t in exampleJL.trial.unique():
    sns.scatterplot(
        data=exampleJL,
        x="time",
        y="xp",
        alpha=0.5,
    )
    # plt.plot(
    #     exampleJL[exampleJL["trial"] == t].time,
    #     exampleJL[exampleJL["trial"] == t].filtPos,
    #     alpha=0.5,
    # )
    plt.xlabel("Time in ms", fontsize=20)
    plt.ylabel("Eye Position", fontsize=20)
    plt.title(f"Filteup Velocity of trial {t} ", fontsize=30)
    plt.show()


# %%
pathFig = "Contextual-Learning/directionCue/figures/voluntaryDirection/"
allEventsFile = "/Volumes/work/brainets/oueld.h/contextuaLearning/directionCue/results_voluntaryDirection/allEvents.csv"
allEvents = pd.read_csv(allEventsFile)
df = pd.read_csv(
    "/Volumes/work/brainets/oueld.h/contextuaLearning/directionCue/results_voluntaryDirection/processedResultsWindow(80,120).csv"
)

# %%
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
df = df[df["sub"] != 1]
df[df["TD_prev"].isna()]
# %%
df["TD_prev"] = df["TD_prev"].apply(lambda x: "left" if x == -1 else "right")
df["interaction"] = list(zip(df["TD_prev"], df["arrow_prev"]))
# %%
badTrials = df[(df["meanVelo"] <= -8) | (df["meanVelo"] >= 8)]
print(badTrials)
# %%
df = df[(df["meanVelo"] <= 8) & (df["meanVelo"] >= -8)]
# %%
df[df["meanVelo"] == df["meanVelo"].max()]
# %%
sns.histplot(data=df, x="meanVelo")
plt.show()
# %%
balance = df.groupby(["arrow", "sub", "proba"])["trial"].count().reset_index()
print(balance)
# %%
for sub in balance["sub"].unique():
    sns.barplot(x="proba", y="trial", hue="arrow", data=balance[balance["sub"] == sub])
    plt.title(f"Subject {sub}")
    plt.show()
# %%
dd = (
    df.groupby(["sub", "arrow", "proba", "TD_prev"])[["meanVelo", "posOffSet"]]
    .mean()
    .reset_index()
)
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
sns.histplot(data=df, x="meanVelo", hue="arrow", bins=20, alpha=0.5)
plt.show()
# %%
sns.histplot(data=df, x="meanVelo", hue="proba", bins=20, palette="viridis", alpha=0.5)
plt.show()
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
facet_grid.figure.subplots_adjust(
    wspace=0.2, hspace=0.2
)  # Adjust wspace and hspace as needed

# Show the plot
plt.show()

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
plt.savefig(pathFig + "/asemAcrossProbappFullProba.svg")
plt.show()
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
plt.show()
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
plt.show()

model = smf.mixedlm(
    "meanVelo~C( arrow )",
    data=df[df.proba == 1.0],
    re_formula="~arrow",
    groups=df[df.proba == 1.0]["sub"],
).fit()
model.summary()
# %%
model = smf.mixedlm(
    "meanVelo~C( arrow )",
    data=df[df.proba == 0.0],
    re_formula="~arrow",
    groups=df[df.proba == 0.0]["sub"],
).fit(method=["lbfgs"])
model.summary()

# %%
model = smf.mixedlm(
    "meanVelo~C( arrow )",
    data=df[df.proba == 0.75],
    re_formula="~arrow",
    groups=df[df.proba == 0.75]["sub"],
).fit(method=["lbfgs"])
model.summary()

# %%
model = smf.mixedlm(
    "meanVelo~C( arrow )",
    data=df[df.proba == 0.25],
    re_formula="~arrow",
    groups=df[df.proba == 0.25]["sub"],
).fit()
model.summary()
# a %%
model = smf.mixedlm(
    "meanVelo~C( arrow,Treatment('up') )",
    data=df[df.proba == 0.5],
    re_formula="~arrow",
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
plt.title("ASEM Across 5 Probabilities", fontsize=30)
plt.xlabel("P(Right|UP)=P(Left|DOWN)", fontsize=30)
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


print(learningCurve)
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
plt.show()
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
plt.show()
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
plt.ylim(-1.5, 1.5)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.savefig(pathFig + "/meanVeloupFullProba.svg")
plt.show()
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
plt.show()
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
plt.show()
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
plt.show()
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
plt.show()
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
plt.show()
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
# %%
learningCurveInteraction = (
    df_prime.groupby(["sub", "proba", "interaction", "arrow"])
    .mean()[["posOffSet", "meanVelo"]]
    .reset_index()
)

# %%
# %%
df_prime.groupby(["sub", "proba", "interaction", "arrow"]).count()[["meanVelo"]]

# %%
df_prime.groupby(["proba", "interaction", "arrow"]).count()[["posOffSet", "meanVelo"]]

# %%
print(learningCurveInteraction)
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
plt.show()
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
plt.show()
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
plt.show()
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
plt.show()
# %%
dd = df.groupby(["sub", "proba", "arrow", "TD_prev"])["meanVelo"].mean().reset_index()
# %%
model = smf.mixedlm(
    "meanVelo~  C(arrow)*C(TD_prev)",
    data=df[df.proba == 0.25],
    re_formula="~arrow",
    groups=df[df.proba == 0.25]["sub"],
).fit(method="lbfgs")
model.summary()
# %%
model = smf.mixedlm(
    "meanVelo~  C(arrow)*C(TD_prev)",
    data=df[df.proba == 0.75],
    re_formula="~arrow",
    groups=df[df.proba == 0.75]["sub"],
).fit(method="lbfgs")
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

# Define transition counts for previous state = down
down_transitions = (
    df[df["arrow_prev"] == "down"]
    .groupby(["sub", "proba", "arrow"])["meanVelo"]
    .count()
    .reset_index(name="count")
)
down_transitions["total"] = down_transitions.groupby(["sub", "proba"])[
    "count"
].transform("sum")
down_transitions["conditional_prob"] = (
    down_transitions["count"] / down_transitions["total"]
)
down_transitions = down_transitions.rename(columns={"arrow": "current_state"})
down_transitions["previous_state"] = "down"

# Define transition counts for previous state = up
up_transitions = (
    df[df["arrow_prev"] == "up"]
    .groupby(["sub", "proba", "arrow"])["meanVelo"]
    .count()
    .reset_index(name="count")
)
up_transitions["total"] = up_transitions.groupby(["sub", "proba"])["count"].transform(
    "sum"
)
up_transitions["conditional_prob"] = up_transitions["count"] / up_transitions["total"]
up_transitions = up_transitions.rename(columns={"arrow": "current_state"})
up_transitions["previous_state"] = "up"
# %%
# Combine results
conditional_probabilities = pd.concat([down_transitions, up_transitions])
print(conditional_probabilities)
# %%
conditional_probabilities["transition_state"] = list(
    zip(
        conditional_probabilities["current_state"],
        conditional_probabilities["previous_state"],
    )
)

conditional_probabilities["transition_state"] = conditional_probabilities[
    "transition_state"
].astype(str)
print(conditional_probabilities)
# %%
for s in conditional_probabilities["sub"].unique():
    # Set up the FacetGrid
    facet_grid = sns.FacetGrid(
        data=conditional_probabilities[conditional_probabilities["sub"] == s],
        col="proba",
        col_wrap=3,
        height=8,
        aspect=1.5,
    )

    # Create barplots for each sub
    facet_grid.map_dataframe(
        sns.barplot,
        x="transition_state",
        y="conditional_prob",
    )

    # Adjust the layout to prevent title overlap
    plt.subplots_adjust(top=0.85)  # Increases space above subplots

    # Add a main title for the entire figure
    facet_grid.figure.suptitle(f"Subject {s}", fontsize=16, y=0.98)

    # Set titles for each subplot
    for ax, p in zip(
        facet_grid.axes.flat, np.sort(conditional_probabilities.proba.unique())
    ):
        ax.set_title(f"Sampling Bias p={p} : P(C(t+1)|C(t))")
        ax.set_xlabel("Transition State")
        ax.set_ylabel("Conditional Probability")
        # ax.tick_params(axis='x', rotation=45)

    # Adjust spacing between subplots
    facet_grid.figure.subplots_adjust(
        wspace=0.2, hspace=0.3  # Slightly increased to provide more vertical space
    )

    # Show the plot
    plt.show()
# %%
# Define transition counts for previous state = down
down_transitions = (
    df[df["arrow_prev"] == "down"]
    .groupby(["sub", "proba", "arrow"])["meanVelo"]
    .count()
    .reset_index(name="count")
)
down_transitions["total"] = down_transitions.groupby(["sub", "proba"])[
    "count"
].transform("sum")
down_transitions["conditional_prob"] = (
    down_transitions["count"] / down_transitions["total"]
)
down_transitions = down_transitions.rename(columns={"arrow": "current_state"})
down_transitions["previous_state"] = "down"

# Define transition counts for previous state = up
up_transitions = (
    df[df["arrow_prev"] == "up"]
    .groupby(["sub", "proba", "arrow"])["meanVelo"]
    .count()
    .reset_index(name="count")
)
up_transitions["total"] = up_transitions.groupby(["sub", "proba"])["count"].transform(
    "sum"
)
up_transitions["conditional_prob"] = up_transitions["count"] / up_transitions["total"]
up_transitions = up_transitions.rename(columns={"arrow": "current_state"})
up_transitions["previous_state"] = "up"
# %%
# Combine results
conditional_probabilities = pd.concat([down_transitions, up_transitions])
# %%
conditional_probabilities["transition_state"] = list(
    zip(
        conditional_probabilities["current_state"],
        conditional_probabilities["previous_state"],
    )
)

conditional_probabilities["transition_state"] = conditional_probabilities[
    "transition_state"
].astype(str)
conditional_probabilities["transition_state"].unique()


# %%
def classify_subject_behavior(conditional_probabilities):
    # Create a function to categorize behavior for a single probability condition
    def categorize_single_proba(group):
        # Transition probabilities for this probability condition
        down_to_down = group[group["transition_state"] == "('down', 'down')"][
            "conditional_prob"
        ].values[0]
        up_to_up = group[group["transition_state"] == "('up', 'up')"][
            "conditional_prob"
        ].values[0]
        down_to_up = group[group["transition_state"] == "('up', 'down')"][
            "conditional_prob"
        ].values[0]
        up_to_down = group[group["transition_state"] == "('down', 'up')"][
            "conditional_prob"
        ].values[0]

        # Persistent: high probability of staying in the same state
        if down_to_down > 0.6 and up_to_up > 0.6:
            return "Persistent"

        # Alternating: high probability of switching states
        if down_to_up > 0.6 and up_to_down > 0.6:
            return "Alternating"

        return "Random"

    # Classify behavior for each subject and probability
    subject_proba_behavior = (
        conditional_probabilities.groupby(["sub", "proba"])
        .apply(lambda x: categorize_single_proba(x))
        .reset_index(name="behavior")
    )
    print(subject_proba_behavior)

    # Count behaviors for each subject across probabilities
    behavior_counts = (
        subject_proba_behavior.groupby(["sub", "behavior"]).size().unstack(fill_value=0)
    )

    # Classify subject based on behavior consistency across at least two probabilities
    def final_classification(row):
        if row["Persistent"] >= 2:
            return "Persistent"
        elif row["Alternating"] >= 2:
            return "Alternating"
        else:
            return "Random"

    subject_classification = behavior_counts.apply(
        final_classification, axis=1
    ).reset_index()
    subject_classification.columns = ["sub", "behavior_class"]

    # Visualize classification
    plt.figure(figsize=(10, 6))
    behavior_counts = subject_classification["behavior_class"].value_counts()
    plt.pie(behavior_counts, labels=behavior_counts.index, autopct="%1.1f%%")
    plt.title("Subject Behavior Classification\n(Consistent Across Probabilities)")
    plt.show()

    # Print detailed results
    print(subject_classification)

    # Additional detailed view
    detailed_behavior = subject_proba_behavior.pivot_table(
        index="sub", columns="proba", values="behavior", aggfunc="first"
    )
    print("\nDetailed Behavior Across Probabilities:")
    print(detailed_behavior)

    return subject_classification


subject_classification = classify_subject_behavior(conditional_probabilities)
# %%
# Perform classification
# Optional: Create a more detailed summary
summary = subject_classification.groupby("behavior_class").size()
print("\nBehavior Classification Summary:")
print(summary)
# %%

# Doing the analysis without the the deterministic blocks

df = df[~((df["proba"] == 0) | (df["proba"] == 1))]
# df = df[~((df["sub"] == 6) & (df["proba"] == 0.5))]
# %%

balance = df.groupby(["arrow", "sub", "proba"])["trial"].count().reset_index()
# %%
for sub in balance["sub"].unique():
    sns.barplot(x="proba", y="trial", hue="arrow", data=balance[balance["sub"] == sub])
    plt.title(f"Subject {sub}")
    plt.show()
# %%
dd = (
    df.groupby(["sub", "arrow", "proba"])[["meanVelo", "posOffSet"]]
    .mean()
    .reset_index()
)
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
plt.show()
# %%
sns.histplot(data=df, x="meanVelo", hue="proba", bins=20, palette="viridis", alpha=0.5)
plt.show()
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
facet_grid.figure.subplots_adjust(
    wspace=0.2, hspace=0.2
)  # Adjust wspace and hspace as needed

# Show the plot
plt.show()

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
_ = plt.title("ASEM Across 3 Different Probabilities", fontsize="30")
plt.legend(fontsize=20)
plt.xlabel("P(Right|UP)", fontsize=30)
plt.ylabel("ASEM (deg/s)", fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylim(-0.5, 0.5)
plt.savefig(pathFig + "/asemAcrossProbapp.svg")
plt.show()
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
plt.show()
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
plt.show()
# %%

model = smf.mixedlm(
    "meanVelo~C( arrow )",
    data=df[df.proba == 0.75],
    re_formula="~arrow",
    groups=df[df.proba == 0.75]["sub"],
).fit(method=["lbfgs"])
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
plt.savefig(pathFig + "/meanVeloarrows.svg",transparent=True)
plt.show()

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
# %%
learningCurve = (
    df_prime.groupby(["sub", "proba", "arrow", "TD_prev"])
    .mean()[["posOffSet", "meanVelo"]]
    .reset_index()
)


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
plt.show()
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
plt.show()
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
plt.show()
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
plt.show()
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
plt.show()
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
plt.show()
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
plt.show()
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
plt.ylim(-1, 1)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.savefig(pathFig + "/meanVelodownTD.svg")
plt.show()
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
plt.show()
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
plt.show()
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
plt.show()
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
plt.show()
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
plt.show()
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
plt.show()
# %%
# %%
dd = df.groupby(["sub", "proba", "arrow", "TD_prev"])["meanVelo"].mean().reset_index()
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


# Sampling Bias Analysis:

df["sampling"] = list(zip(df["arrow"], df["arrow_prev"]))
# %%
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
    facet_grid.figure.subplots_adjust(
        wspace=0.2, hspace=0.2
    )  # Adjust wspace and hspace as needed

    # Show the plot
    plt.show()

# %%


# Define transition counts for previous state = down
down_transitions = (
    df[df["arrow_prev"] == "down"]
    .groupby(["sub", "proba", "arrow"])["meanVelo"]
    .count()
    .reset_index(name="count")
)
down_transitions["total"] = down_transitions.groupby(["sub", "proba"])[
    "count"
].transform("sum")
down_transitions["conditional_prob"] = (
    down_transitions["count"] / down_transitions["total"]
)
down_transitions = down_transitions.rename(columns={"arrow": "current_state"})
down_transitions["previous_state"] = "down"

# Define transition counts for previous state = up
up_transitions = (
    df[df["arrow_prev"] == "up"]
    .groupby(["sub", "proba", "arrow"])["meanVelo"]
    .count()
    .reset_index(name="count")
)
up_transitions["total"] = up_transitions.groupby(["sub", "proba"])["count"].transform(
    "sum"
)
up_transitions["conditional_prob"] = up_transitions["count"] / up_transitions["total"]
up_transitions = up_transitions.rename(columns={"arrow": "current_state"})
up_transitions["previous_state"] = "up"
# %%
# Combine results
conditional_probabilities = pd.concat([down_transitions, up_transitions])
# %%
conditional_probabilities["transition_state"] = list(
    zip(
        conditional_probabilities["current_state"],
        conditional_probabilities["previous_state"],
    )
)

conditional_probabilities["transition_state"] = conditional_probabilities[
    "transition_state"
].astype(str)


# %%
for s in conditional_probabilities["sub"].unique():
    # Set up the FacetGrid
    facet_grid = sns.FacetGrid(
        data=conditional_probabilities[conditional_probabilities["sub"] == s],
        col="proba",
        col_wrap=3,
        height=8,
        aspect=1.5,
    )

    # Create barplots for each sub
    facet_grid.map_dataframe(
        sns.barplot,
        x="transition_state",
        y="conditional_prob",
    )

    # Adjust the layout to prevent title overlap
    plt.subplots_adjust(top=0.85)  # Increases space above subplots

    # Add a main title for the entire figure
    facet_grid.figure.suptitle(f"Subject {s}", fontsize=16, y=0.98)

    # Set titles for each subplot
    for ax, p in zip(
        facet_grid.axes.flat, np.sort(conditional_probabilities.proba.unique())
    ):
        ax.set_title(f"Sampling Bias p={p} : P(C(t+1)|C(t))")
        ax.set_xlabel("Transition State")
        ax.set_ylabel("Conditional Probability")
        # ax.tick_params(axis='x', rotation=45)

    # Adjust spacing between subplots
    facet_grid.figure.subplots_adjust(
        wspace=0.2, hspace=0.3  # Slightly increased to provide more vertical space
    )

    # Show the plot
    plt.show()


# %%
def classify_subject_behavior(conditional_probabilities):
    # Create a function to categorize behavior for a single probability condition
    def categorize_single_proba(group):
        # Transition probabilities for this probability condition
        if (
            len(
                group[group["transition_state"] == "('down', 'down')"][
                    "conditional_prob"
                ]
            )
            > 0
        ):
            down_to_down = group[group["transition_state"] == "('down', 'down')"][
                "conditional_prob"
            ].values[0]
        else:
            down_to_down = 0

        if (
            len(group[group["transition_state"] == "('up', 'up')"]["conditional_prob"])
            > 0
        ):
            up_to_up = group[group["transition_state"] == "('up', 'up')"][
                "conditional_prob"
            ].values[0]
        else:
            up_to_up = 0

        if (
            len(
                group[group["transition_state"] == "('up', 'down')"]["conditional_prob"]
            )
            > 0
        ):
            down_to_up = group[group["transition_state"] == "('up', 'down')"][
                "conditional_prob"
            ].values[0]
        else:
            down_to_up = 0

        if len(
            group[group["transition_state"] == "('down', 'up')"]["conditional_prob"]
        ):

            up_to_down = group[group["transition_state"] == "('down', 'up')"][
                "conditional_prob"
            ].values[0]
        else:
            up_to_down = 0

        # Persistent: high probability of staying in the same state
        if down_to_down > 0.6 and up_to_up > 0.6:
            return "Persistent"

        # Alternating: high probability of switching states
        if down_to_up > 0.6 and up_to_down > 0.6:
            return "Alternating"

        return "Random"

    # Classify behavior for each subject and probability
    subject_proba_behavior = (
        conditional_probabilities.groupby(["sub", "proba"])
        .apply(lambda x: categorize_single_proba(x))
        .reset_index(name="behavior")
    )
    print(subject_proba_behavior)

    # Count behaviors for each subject across probabilities
    behavior_counts = (
        subject_proba_behavior.groupby(["sub", "behavior"]).size().unstack(fill_value=0)
    )

    # Classify subject based on behavior consistency across at least two probabilities
    def final_classification(row):
        if row["Persistent"] >= 2:
            return "Persistent"
        elif row["Alternating"] >= 2:
            return "Alternating"
        else:
            return "Random"

    subject_classification = behavior_counts.apply(
        final_classification, axis=1
    ).reset_index()
    subject_classification.columns = ["sub", "behavior_class"]

    # Visualize classification
    plt.figure(figsize=(10, 6))
    behavior_counts = subject_classification["behavior_class"].value_counts()
    plt.pie(behavior_counts, labels=behavior_counts.index, autopct="%1.1f%%")
    plt.title("Subject Behavior Classification\n(Consistent Across Probabilities)")
    plt.show()

    # Print detailed results
    print(subject_classification)

    # Additional detailed view
    detailed_behavior = subject_proba_behavior.pivot_table(
        index="sub", columns="proba", values="behavior", aggfunc="first"
    )
    print("\nDetailed Behavior Across Probabilities:")
    print(detailed_behavior)

    return subject_classification


subject_classification = classify_subject_behavior(conditional_probabilities)
# %%
# Perform classification
# Optional: Create a more detailed summary
summary = subject_classification.groupby("behavior_class").size()
print("\nBehavior Classification Summary:")
print(summary)
# %%
dd = df.groupby(["sub", "proba", "arrow"])["meanVelo"].mean().reset_index()
# %%
for s in dd["sub"].unique():
    behavior_value = subject_classification[subject_classification["sub"] == s][
        "behavior_class"
    ].values[0]
    dd.loc[dd["sub"] == s, "behavior"] = behavior_value

# %%
# %%
sns.lmplot(
    data=dd[(dd["arrow"] == "down")],
    x="proba",
    y="meanVelo",
    hue="behavior",
)
plt.title("Down Arrow", fontsize=30)
plt.show()
# %%
sns.lmplot(
    data=dd[(dd["arrow"] == "up")],
    x="proba",
    y="meanVelo",
    hue="behavior",
)
plt.title("Up Arrow", fontsize=30)
plt.show()

# %%

# Computing the peristance score based on the transition probabilites
# %%
conditional_probabilities.columns
# %%
dd = (
    conditional_probabilities.groupby(["sub", "transition_state"])["conditional_prob"]
    .mean()
    .reset_index()
)
# %%
dd["sub"].value_counts()
# %%
dd["transition_state"].unique()
# %%
ts = dd["transition_state"].unique()
new_rows = []
for s in dd["sub"].unique():
    existing_ts = dd[dd["sub"] == s]["transition_state"].unique()
    for t in ts:
        if t not in existing_ts:
            # Add a new row with sub = s and transition_state = t, setting transition_state to 0
            new_row = {"sub": s, "transition_state": t, "conditional_prob": 0}
            new_rows.append(new_row)

# Concatenate the new rows to the original DataFrame
dd = pd.concat([dd, pd.DataFrame(new_rows)], ignore_index=True)

print(dd)
# %%

dd = dd.groupby(["sub", "transition_state"])["conditional_prob"].mean().reset_index()
# %%
dd["transition_state"].unique()


# %%
# Function to classify transition_state as persistent or alternating
def classify_transition(state):
    return (
        "persistent"
        if state == "('up', 'up')" or state == "('down', 'down')"
        else "alternating"
    )


# Apply the classification function
dd["transition_type"] = dd["transition_state"].apply(classify_transition)
# %%

# Group by 'sub' and calculate the score
result = (
    dd.groupby("sub")
    .apply(
        lambda x: x[x["transition_type"] == "persistent"]["conditional_prob"].sum()
        - x[x["transition_type"] == "alternating"]["conditional_prob"].sum()
    )
    .reset_index(name="persistence_score")
)

print(result)

# %%
dd["transition_state"].unique()
# %%
result["persistence_score_up"] = np.nan
result["persistence_score_down"] = np.nan
print(result)
# %%
for s in dd["sub"].unique():
    up_up_prob = dd[(dd["sub"] == s) & (dd["transition_state"] == "('up', 'up')")][
        "conditional_prob"
    ].values[0]
    down_up_prob = dd[(dd["sub"] == s) & (dd["transition_state"] == "('down', 'up')")][
        "conditional_prob"
    ].values[0]
    result.loc[result["sub"] == s, "persistence_score_up"] = up_up_prob - down_up_prob
    down_down_prob = dd[
        (dd["sub"] == s) & (dd["transition_state"] == "('down', 'down')")
    ]["conditional_prob"].values[0]
    up_down_prob = dd[(dd["sub"] == s) & (dd["transition_state"] == "('up', 'down')")][
        "conditional_prob"
    ].values[0]
    result.loc[result["sub"] == s, "persistence_score_down"] = (
        down_down_prob - up_down_prob
    )
print(result)
# %%

# Group by 'sub', 'proba', and 'color' and calculate the mean of 'meanVelo'
mean_velo = df.groupby(["sub", "proba", "arrow"])["meanVelo"].mean().reset_index()

# Pivot the table to have 'proba' as columns
pivot_table = mean_velo.pivot_table(
    index=["sub", "arrow"], columns="proba", values="meanVelo"
).reset_index()

# Calculate the adaptation
pivot_table["adaptation"] = np.abs(pivot_table[0.75] - pivot_table[0.25])
# %%
print(pivot_table)
# %%

# Select the relevant columns
adaptation = pivot_table[["sub", "arrow", "adaptation"]]

print(adaptation)
# %%
adaptation.groupby("sub")["adaptation"].mean().values
# %%

result["adaptation"] = adaptation.groupby("sub")["adaptation"].mean().values
result["adaptation_down"] = adaptation[adaptation["arrow"] == "down"][
    "adaptation"
].values
result["adaptation_up"] = adaptation[adaptation["arrow"] == "up"]["adaptation"].values
# %%
result = pd.DataFrame(result)
print(result)
# %%
sns.lmplot(data=result, x="persistence_score", y="adaptation")
plt.savefig(pathFig + "/samplingBiasArrows.svg")
plt.show()
# %%
sns.lmplot(data=result, x="persistence_score_down", y="adaptation_down")
plt.savefig(pathFig + "/samplingBiasDown.svg")
plt.show()
# %%
sns.lmplot(
    data=result,
    x="persistence_score_up",
    y="adaptation_up",
)
plt.savefig(pathFig + "/samplingBiasUp.svg")

plt.show()
# %%
sns.lmplot(
    data=result,
    x="adaptation_down",
    y="adaptation_up",
)
plt.show()
# %%
sns.lmplot(
    data=result,
    x="persistence_score_down",
    y="persistence_score_up",
)
plt.show()
# %%


correlation, p_value = spearmanr(
    result["persistence_score"],
    result["adaptation"],
)
print(
    f"Spearman's correlation for the adaptation score): {correlation}, p-value: {p_value}"
)

# %%
correlation, p_value = spearmanr(
    result["persistence_score_down"],
    result["adaptation_down"],
)
print(
    f"Spearman's correlation for the adaptation score for down): {correlation}, p-value: {p_value}"
)

# %%
correlation, p_value = spearmanr(
    result["persistence_score_up"],
    result["adaptation_up"],
)
print(
    f"Spearman's correlation for the adaptation score for up): {correlation}, p-value: {p_value}"
)

# %%
model = sm.OLS.from_formula("adaptation_up~ persistence_score_up ", result).fit()

print(model.summary())
# %%
model = sm.OLS.from_formula("adaptation_down~ persistence_score_down ", result).fit()

print(model.summary())
# %%
model = sm.OLS.from_formula("adaptation~ persistence_score ", result).fit()

print(model.summary())
# %%
model = sm.OLS.from_formula("adaptation_up~ adaptation_down ", result).fit()

print(model.summary())
# %%
df.columns
# %%
df.groupby(["sub", "proba", "arrow", "up_arrow_position", "TD_prev"])[
    "meanVelo"
].mean().reset_index()
# %%
# Group by 'sub', 'proba', and 'color' and calculate the mean of 'meanVelo'
mean_velo = df.groupby(["sub", "proba", "arrow"])["meanVelo"].mean().reset_index()

# Pivot the table to have 'proba' as columns
pivot_table = mean_velo.pivot_table(
    index=["sub", "proba"], columns="arrow", values="meanVelo"
).reset_index()

# Calculate the adaptation
pivot_table["adaptation"] = (
    np.abs(pivot_table["down"]) + np.abs(pivot_table["up"])
) / 2

print(pivot_table)
# %%
sns.scatterplot(
    data=pivot_table, x="up", y="down", hue="proba", palette="viridis", s=50
)
plt.axhline(y=0, color="k", linestyle="--")  # Horizontal line at y=0
plt.axvline(x=0, color="k", linestyle="--")  # Vertical line at x=0
plt.show()
# %%
sns.boxplot(
    data=pivot_table,
    x="proba",
    y="adaptation",
)
plt.show()
# %%
sns.boxplot(
    data=pivot_table,
    x="proba",
    y="down",
)
plt.show()
# %%
sns.boxplot(
    data=pivot_table,
    x="proba",
    y="up",
)
plt.show()
# %%
# Create the plot with connected dots for each participant
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=pivot_table,
    x="up",
    y="down",
    hue="proba",
    palette="viridis",
    style="proba",
markers = ["o", "s", "D", "^", "v"],
    s=50
)
# Connect dots for each participant
for sub in pivot_table["sub"].unique():
    subset = pivot_table[pivot_table["sub"] == sub]
    plt.plot(subset["up"], subset["down"], color="gray", alpha=0.5, linestyle="--")
# Add plot formatting
plt.axhline(0, color="black", linestyle="--")
plt.axvline(0, color="black", linestyle="--")
plt.title(f"Participants adaptaion across probabilites")
plt.xlabel("up")
plt.ylabel("down")
plt.ylim(-2.5, 2.5)
plt.xlim(-2.5, 2.5)
plt.legend(title="proba")
plt.tight_layout()
plt.show()
plt.show()
# %%
# Connect dots for each participant
for sub in pivot_table["sub"].unique():
    subset = pivot_table[pivot_table["sub"] == sub]
    plt.plot(subset["up"], subset["down"], color="gray", alpha=0.5, linestyle="--")
    sns.scatterplot(
        data=pivot_table[pivot_table["sub"] == sub],
        x="up",
        y="down",
        hue="proba",
        palette="viridis",
        style="proba",
        markers=["o", "s", "D"],
    )
    # Add plot formatting
    plt.axhline(0, color="black", linestyle="--")
    plt.axvline(0, color="black", linestyle="--")
    plt.title(f"Participant:{sub}")
    plt.xlabel("up")
    plt.ylabel("down")
    plt.legend(title="proba")
    plt.tight_layout()
    plt.show()
# %%
# Group by 'sub', 'proba', and 'color' and calculate the mean of 'meanVelo'
mean_velo = df.groupby(["sub", "proba", "interaction"])["meanVelo"].mean().reset_index()
print(mean_velo)
mean_velo["interaction"] = mean_velo["interaction"].astype("str")
# %%
# Pivot the table to have 'proba' as columns
pivot_table = mean_velo.pivot_table(
    index=["sub", "proba"], columns="interaction", values="meanVelo"
).reset_index()

# Calculate the adaptation
# pivot_table["adaptation"] = (
#     np.abs(pivot_table["green"]) + np.abs(pivot_table["red"])
# ) / 2

print(pivot_table)
pivot_table = pd.DataFrame(pivot_table)
pivot_table.columns
# %%
pivot_table.columns[2]
# %%
# pivot_table.rename(
#     columns={
#         ('left', 'green'): "left_green",
#         ('left', 'red'): "left_red",
#         ('right', 'green'): "right_green",
#         ('right', 'red'): "right_red",
#     },
#     inplace=True,
# )
#
# pivot_table.columns
# %%
sns.scatterplot(
    data=pivot_table, x="('right', 'up')", y="('right', 'down')", hue="proba"
)
# Or alternatively
plt.axhline(y=0, color="k", linestyle="--")  # Horizontal line at y=0
plt.axvline(x=0, color="k", linestyle="--")  # Vertical line at x=0
plt.xlim(-2.5, 2.5)
plt.ylim(-2.5, 2.5)
plt.show()
# a %%
sns.pointplot(
    data=pivot_table,
    x="proba",
    y="adaptation",
)
plt.show()

# %%
sns.scatterplot(
    data=pivot_table, x="('left', 'red')", y="('left', 'green')", hue="proba"
)
# Or alternatively
plt.axhline(y=0, color="k", linestyle="--")  # Horizontal line at y=0
plt.axvline(x=0, color="k", linestyle="--")  # Vertical line at x=0
plt.xlim(-2.5, 2.5)
plt.ylim(-2.5, 2.5)
plt.show()
# a %%
sns.pointplot(
    data=pivot_table,
    x="proba",
    y="adaptation",
)
plt.show()

# %%
# Looking at the interaction between the position and the choice of the color

plt.hist(
    df.groupby(["sub", "proba", "color", "trial_color_UP"])["meanVelo"]
    .count()
    .reset_index(name="count")["count"]
)
plt.show()
# %%
df_inter = (
    df.groupby(["sub", "proba", "color", "trial_color_UP", "TD_prev"])["meanVelo"]
    .mean()
    .reset_index()
)

# %%
df_inter["interaction"] = list(zip(df_inter["color"], df_inter["trial_color_UP"]))
df_inter["interaction"] = df_inter["interaction"].astype("str")
# %%
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.barplot(
    x="proba",
    y="meanVelo",
    hue="interaction",
    palette=greenColorsPalette,
    data=df_inter[df_inter.color == "green"],
)
plt.legend(fontsize=20)
plt.title("Anticipatory Velocity Given the Color Position: Green ", fontsize=30)
plt.ylabel("ASEM (deg/s)", fontsize=30)
plt.xlabel("P(Left|GREEN)", fontsize=30)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.ylim(-1, 1)
# plt.savefig(pathFig + "/meanVeloGreenTD.svg")
plt.show()
# %%
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.barplot(
    x="proba",
    y="meanVelo",
    hue="interaction",
    palette=redColorsPalette,
    data=df_inter[df_inter.color == "red"],
)
plt.legend(fontsize=20)
plt.title("Anticipatory Velocity Given the Color Position: RED", fontsize=30)
plt.ylabel("ASEM (deg/s)", fontsize=30)
plt.xlabel("P(Left|GREEN)", fontsize=30)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.ylim(-1, 1)
# plt.savefig(pathFig + "/meanVeloGreenTD.svg")
plt.show()

# %%

df["interColPos"] = list(zip(df["color"], df["trial_color_UP"]))
df["interColPos"] = df["interColPos"].astype("str")

# %%

for s in df["sub"].unique():
    dfs = df[df["sub"] == s]
    sns.histplot(data=dfs, x="interColPos")
    plt.show()


# %%
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.barplot(
    x="proba",
    y="meanVelo",
    hue="interColPos",
    palette=greenColorsPalette,
    data=df[df.color == "green"],
)
plt.legend(fontsize=20)
plt.title("Anticipatory Velocity Given the Color Position: Green ", fontsize=30)
plt.ylabel("ASEM (deg/s)", fontsize=30)
plt.xlabel("P(Left|GREEN)", fontsize=30)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.ylim(-1, 1)
# plt.savefig(pathFig + "/meanVeloGreenTD.svg")
plt.show()
# %%
for s in df["sub"].unique():
    dfs = df[df["sub"] == s]
    sns.histplot(data=dfs, x="interColPos")
    plt.show()


# %%
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.barplot(
    x="proba",
    y="meanVelo",
    hue="interColPos",
    palette=redColorsPalette,
    data=df[df.color == "red"],
)
plt.legend(fontsize=20)
plt.title("Anticipatory Velocity Given the Color Position: Green ", fontsize=30)
plt.ylabel("ASEM (deg/s)", fontsize=30)
plt.xlabel("P(Left|GREEN)", fontsize=30)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.ylim(-1, 1)
# plt.savefig(pathFig + "/meanVeloGreenTD.svg")
plt.show()
# %%
