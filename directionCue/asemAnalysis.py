"""
Script to analyze the data from the voluntaryDirection task.
Saccades and blinks has been removed from the data. on the window -200ms(Fixation Offset) to 600ms the end of the trials
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pingouin as pg

# %%
dirPath = "/Users/mango/contextuaLearning/directionCue/results_voluntaryDirection/"
fileName = "/Volumes/work/brainets/oueld.h/contextuaLearning/directionCue/results_voluntaryDirection/rawAndFiltereDataNoSacc.csv"
# %%
df = pd.read_csv(fileName)
allEvents = pd.read_csv(dirPath + "allEvents.csv")
allEvents
# %%
df
# %%
example = df[(df["sub"] == 6.0) & (df["proba"] == 1.0)]
example
# %%
# Plotting one example
for t in example.trial.unique():
    plt.plot(
        example[example["trial"] == t].time,
        example[example["trial"] == t].filtVelo,
        alpha=0.5,
    )
    plt.plot(
        example[example["trial"] == t].time,
        example[example["trial"] == t].velo,
        alpha=0.5,
    )
    plt.xlabel("Time in ms", fontsize=20)
    plt.ylabel("Filtered Velocity in deg/s", fontsize=20)
    plt.title(f"Filtered Velocity of trial {t} ", fontsize=30)
    plt.show()
# %%
# Getting the Position Offset for each Participant and each proba
df_results = []
filtered_data = df[(df["time"] >= 80) & (df["time"] <= 120)]
unique_sub = df["sub"].unique()
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
            # getting the region of interest
            roi = filtered_data[
                (filtered_data["trial"] == trial)
                & (filtered_data["sub"] == sub)
                & (filtered_data["proba"] == p)
            ]
            meanVelo = np.nanmean(roi["velo"])  # print(meanVelo)
            meanFiltVelo = np.nanmean(roi["filtVelo"])  # print(meanVelo)
            posOffset = (
                roi.xp.values[-1] - roi.xp.values[0]
            )  # Append the result to the DataFrame
            df_results.append(
                {
                    "trial": trial,
                    "sub": sub,
                    "proba": p,
                    "posOffSet": posOffset,
                    "meanVelo": meanVelo,
                    "meanFiltVelo": meanFiltVelo,
                }
            )
# %%
df_results = pd.DataFrame(df_results)
df_results
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
# Saving the dataframe to a csv file.
df_results.to_csv(dirPath + "resultsNoSacc.csv", index=False)
# %%
# getting the results of the anticipatory position offset and mean velocity on [80, 120] ms
df = pd.read_csv(dirPath + "resultsNoSacc.csv")
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
df.columns
# %%
df_prime = df[
    [
        "sub",
        "trial",
        "proba",
        "arrow",
        "TD_prev",
        "posOffSet",
        "meanVelo",
        "meanFiltVelo",
    ]
]
learningCurve = (
    df_prime.groupby(["sub", "proba", "arrow", "TD_prev"])
    .mean()[["posOffSet", "meanVelo", "meanFiltVelo"]]
    .reset_index()
)


learningCurve
# %%
df_prime.groupby(["proba", "arrow", "TD_prev"]).count()[
    ["posOffSet", "meanVelo", "meanFiltVelo"]
]

# %%
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.pointplot(
    x="proba",
    y="meanFiltVelo",
    hue="arrow",
    errorbar="se",
    data=learningCurve,
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
    y="meanFiltVelo",
    data=df[df.arrow == "up"],
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
