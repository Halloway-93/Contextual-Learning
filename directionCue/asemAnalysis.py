"""
Script to analyze the data from the voluntaryDirection task.
Saccades and blinks has been removed from the data. on the window -200ms(Fixation Offset) to 600ms the end of the trials
"""

import os
import pingouin as pg
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

path = "/Volumes/work/brainets/oueld.h/contextuaLearning/directionCue/results_voluntaryDirection/"
pathFig = "~/PhD/Contextual-Learning/directionCue/figures/"
jobLibData = "JobLibProcessing.csv"
allEventsFile = "/Volumes/work/brainets/oueld.h/contextuaLearning/directionCue/results_voluntaryDirection/allEvents.csv"

# %%
jlData = pd.read_csv(os.path.join(path, jobLibData))
# %%
exampleJL = jlData[
    (jlData["sub"] == 8) & (jlData["proba"] == 0.25) & (jlData["trial"] == 118)
]
exampleJL
# %%
# Plotting one example
# Plotting one example
for t in exampleJL.trial.unique():
    plt.plot(
        exampleJL[exampleJL["trial"] == t].time,
        exampleJL[exampleJL["trial"] == t].filtVelo,
        alpha=0.5,
    )
    plt.plot(
        exampleJL[exampleJL["trial"] == t].time,
        exampleJL[exampleJL["trial"] == t].velo,
        alpha=0.5,
    )
    plt.xlabel("Time in ms", fontsize=20)
    plt.ylabel("Filtered Velocity in deg/s", fontsize=20)
    plt.title(f"Filtered Velocity of trial {t} ", fontsize=30)
    plt.show()
# %%
for t in exampleJL.trial.unique():
    plt.plot(
        exampleJL[exampleJL["trial"] == t].time,
        exampleJL[exampleJL["trial"] == t].xp,
        alpha=0.5,
    )
    plt.plot(
        exampleJL[exampleJL["trial"] == t].time,
        exampleJL[exampleJL["trial"] == t].filtPos,
        alpha=0.5,
    )
    plt.xlabel("Time in ms", fontsize=20)
    plt.ylabel("Eye Position", fontsize=20)
    plt.title(f"Filtered Velocity of trial {t} ", fontsize=30)
    plt.show()


# %%
redColorsPalette = ["#e83865", "#cc3131"]
greenColorsPalette = ["#8cd790", "#285943"]
# %%
allEvents = pd.read_csv(allEventsFile)
df = pd.read_csv(
    "/Volumes/work/brainets/oueld.h/contextuaLearning/directionCue/results_voluntaryDirection/processedResults.csv"
)
# %%
df.columns
# %%
badTrials = df[(df["meanVelo"] < -11) | (df["meanVelo"] > 11)]
badTrials
# %%
# df = df[(df["meanVelo"] <= 11) & (df["meanVelo"] >= -11)]
df["meanVelo"].max()
# %%
sns.histplot(data=df, x="meanVelo")
plt.show()
# %%
print(df)
# %%
balance = df.groupby(["chosen_arrow", "sub", "proba"])["trial"].count().reset_index()
balance
# %%
for sub in balance["sub"].unique():
    sns.barplot(
        x="proba", y="trial", hue="chosen_arrow", data=balance[balance["sub"] == sub]
    )
    plt.title(f"Subject {sub}")
    plt.show()
# %%
df.columns
# %%
# getting previous TD for each trial for each subject and each proba
for sub in df["sub"].unique():
    for p in df[df["sub"] == sub]["proba"].unique():
        df.loc[(df["sub"] == sub) & (df["proba"] == p), "TD_prev"] = df.loc[
            (df["sub"] == sub) & (df["proba"] == p), "target_direction"
        ].shift(1)
        df.loc[(df["sub"] == sub) & (df["proba"] == p), "arrow_prev"] = df.loc[
            (df["sub"] == sub) & (df["proba"] == p), "chosen_arrow"
        ].shift(1)
# %%
df.columns
# %%
df_prime = df[
    [
        "sub",
        "trial",
        "proba",
        "chosen_arrow",
        "TD_prev",
        "posOffSet",
        "meanVelo",
    ]
]
learningCurve = (
    df_prime.groupby(["sub", "proba", "chosen_arrow", "TD_prev"])
    .mean()[
        [
            "posOffSet",
            "meanVelo",
        ]
    ]
    .reset_index()
)


learningCurve
# %%
df_prime.groupby(["proba", "chosen_arrow", "TD_prev"]).count()[
    ["posOffSet", "meanVelo"]
]

# %%
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.pointplot(
    x="proba",
    y="meanVelo",
    hue="chosen_arrow",
    errorbar="sd",
    data=learningCurve,
    alpha=0.7
)
plt.title("ASEM for all Participants")
plt.xlabel("P(Right|UP)")
plt.show()
# %%
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.pointplot(
    x="proba",
    y="meanVelo",
    hue="chosen_arrow",
    errorbar="se",
    data=learningCurve[~((learningCurve.proba == 0) | (learningCurve.proba == 1.0))],
    alpha=0.5,
)
plt.title("ASEM for all Participants")
plt.xlabel("P(Right|UP)")
plt.show()
# %%
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.barplot(
    x="proba",
    y="posOffSet",
    data=df[df.arrow == "up"],
)
plt.title("Position Offset: Arrow UP")
plt.xlabel("P(Right|UP)")
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
    data=df[df.arrow == "up"],
)
plt.title("Position Offset: Arrow UP")
plt.xlabel("P(Right|UP)")
plt.show()
# %%

pg.friedman(
    data=df[df["arrow"] == "up"],
    dv="meanVelo",
    within="proba",
    subject="sub",
)

# %%
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.barplot(
    x="proba",
    y="meanVelo",
    data=df[df.chosen_arrow == "up"],
)
plt.title("ASEM: Arrow UP")
plt.xlabel("P(Right|UP)")
plt.show()
# %%
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.pointplot(
    x="proba",
    y="meanFiltVelo",
    hue="TD_prev",
    errorbar="se",
    data=learningCurve[learningCurve.arrow == "up"],
)
plt.title("Anticipatory Velocity: Arrow UP")
plt.xlabel("P(Right|UP)")
plt.show()
# %%
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.barplot(
    x="proba",
    y="meanFiltVelo",
    hue="TD_prev",
    data=learningCurve[learningCurve.arrow == "up"],
)
plt.title("Anticipatory Velocity: Arrow UP")
plt.xlabel("P(Right|UP)")
plt.show()
# %%
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.barplot(
    x="proba",
    y="posOffSet",
    data=learningCurve[learningCurve.arrow == "down"],
)
plt.title("Position Offset: Arrow DOWN")
plt.xlabel("P(Left|DOWN)")
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
    data=learningCurve[learningCurve.arrow == "down"],
)
plt.title("Position Offset: Arrow DOWN")
plt.xlabel("P(Left|DOWN)")
plt.show()
# %%
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.barplot(
    x="proba",
    y="meanVelo",
    data=learningCurve[learningCurve.arrow == "down"],
)
plt.title("ASEM: Arrow DOWN")
plt.xlabel("P(Left|DOWN)")
plt.show()
# %%
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.pointplot(
    x="proba",
    y="meanVelo",
    hue="TD_prev",
    errorbar="se",
    data=learningCurve[learningCurve.arrow == "down"],
)
plt.title("Anticipatory Velocity: Arrow DOWN")
plt.xlabel("P(Right|UP)")
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
    data=learningCurve[learningCurve.arrow == "down"],
)
plt.title("meanVelo: Arrow DOWN")
plt.xlabel("P(Left|DOWN)")
plt.show()


# Interaction between previous TD and arrow chosen
# %%
df["interaction"] = list(zip(df["TD_prev"], df["arrow_prev"]))
df_prime = df[["trial", "proba", "arrow", "interaction", "posOffSet", "meanVelo"]]
# %%
valueToKeep = df_prime.interaction.unique()[1:]
valueToKeep
# %%
df_prime = df_prime[df_prime["interaction"].isin(valueToKeep)]
df_prime.interaction.unique()
# %%
learningCurveInteraction = (
    df_prime.groupby(["proba", "interaction", "arrow"])
    .mean()[["posOffSet", "meanVelo"]]
    .reset_index()
)
learningCurveInteraction
# %%
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
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
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.barplot(
    x="proba",
    y="meanVelo",
    hue="interaction",
    data=learningCurveInteraction[learningCurveInteraction.arrow == "up"],
)
plt.title("ASEM: Arrow UP")
plt.xlabel("P(Right|UP)")
plt.show()
# %%
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
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
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.barplot(
    x="proba",
    y="meanVelo",
    hue="interaction",
    data=learningCurveInteraction[learningCurveInteraction.arrow == "down"],
)
plt.title("ASEM: Arrow DOWN")
plt.xlabel("P(Left|DOWN)")
plt.show()
# %%
