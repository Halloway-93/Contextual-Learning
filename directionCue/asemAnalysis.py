'''
Script to analyze the data from the voluntaryDirection task.
Saccades and blinks has been removed from the data. on the window -200ms(Fixation Offset) to 600ms the end of the trials
'''
import pandas as pd
import numpy as np

dirPath = (
    "/Users/mango/boubou/contextuaLearning/directionCue/results_voluntaryDirection/"
)
df = pd.read_csv(dirPath + "RawDataNoSacc.csv")
allEvents = pd.read_csv(dirPath + "allEvents.csv")
allEvents
# Getting the Position Offset for each Participant and each proba
df_results = []
filtered_data = df[(df["time"] >= -50) & (df["time"] <= 50)]
unique_sub = df.sub.unique()
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
