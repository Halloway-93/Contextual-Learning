"""
Script to analyze the data from the voluntaryDirection task.
Saccades and blinks has been removed from the data. on the window -200ms(Fixation Offset) to 600ms the end of the trials
"""

import os
from scipy import stats
from statsmodels.stats.multitest import multipletests
import scikit_posthocs as sp
from scipy.stats import friedmanchisquare, wilcoxon
from scipy.stats import spearmanr
import statsmodels.formula.api as smf
import pingouin as pg
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
import numpy as np

path = "/Volumes/work/brainets/oueld.h/contextuaLearning/directionCue/results_voluntaryDirection"
jobLibData = "JobLibProcessing.csv"

# %%
jlData = pd.read_csv(os.path.join(path, jobLibData))
# %%
exampleJL = jlData[
    (jlData["sub"] == 14)
    & (jlData["proba"] == 0.25)
    & (jlData["trial"] == 58)
    #   & (jlData["time"] <= 100)
]
# %%
# Plotting one example
# Plotting one example
for t in exampleJL.trial.unique():
    plt.plot(
        exampleJL[exampleJL["trial"] == t].time,
        exampleJL[exampleJL["trial"] == t].filtVeloFilt,
        alpha=0.5,
    )
    plt.plot(
        exampleJL[exampleJL["trial"] == t].time,
        exampleJL[exampleJL["trial"] == t].velo,
        alpha=0.5,
    )
    plt.xlabel("Time in ms", fontsize=20)
    plt.ylabel("Filteup Velocity in deg/s", fontsize=20)
    plt.title(f"Filteup Velocity of trial {t} ", fontsize=30)
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
    plt.title(f"Filteup Velocity of trial {t} ", fontsize=30)
    plt.show()


# %%
uparrowsPalette = ["#e83865", "#cc3131"]
downarrowsPalette = ["#8cd790", "#285943"]
# %%
pathFig = "PhD/Contextual-Learning/directionCue/figures/voluntaryDirection/"
allEventsFile = "/Volumes/work/brainets/oueld.h/contextuaLearning/directionCue/results_voluntaryDirection/allEvents.csv"
allEvents = pd.read_csv(allEventsFile)
df = pd.read_csv(
    "/Volumes/work/brainets/oueld.h/contextuaLearning/directionCue/results_voluntaryDirection/processedResultsWindow(80,120).csv"
)

# %%

df.columns
# %%
df = df[~((df["proba"] == 0) | (df["proba"] == 1))]
# %%
badTrials = df[(df["meanVelo"] <= -5.5) | (df["meanVelo"] >= 5.5)]
badTrials
# %%
df = df[(df["meanVelo"] <= 5.5) & (df["meanVelo"] >= -5.5)]
# %%
df[df["meanVelo"] == df["meanVelo"].max()]
# %%
sns.histplot(data=df, x="meanVelo")
plt.show()
# %%
df["arrow"] = df["chosen_arrow"].values
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
dd = (
    df.groupby(["sub", "arrow", "proba"])[["meanVelo", "posOffSet"]]
    .mean()
    .reset_index()
)
dd
# %%
np.abs(dd.meanVelo.values).max()
# a%
dd[np.abs(dd.meanVelo.values) > 1.8]
# %%
df = df[(df.trial >= 60)]
# %%
# Normalizing the data
# dd["meanVeloNorm"] = (dd["meanVelo"] - dd["meanVelo"].mean()) / dd["meanVelo"].std()
# %%
# Statisitcal test


# model = sm.OLS.from_formula("meanVelo~ C(proba)*C(arrow) ", data=dd)
# result = model.fit()
#
# print(result.summary())
# dd = dd[~((dd["proba"] == 0) | (dd["proba"] == 1))]
# %%

meanVelo = dd[dd.arrow == "up"]["meanVelo"]
proba = dd[dd.arrow == "up"]["proba"]

# Spearman's rank correlation
correlation, p_value = spearmanr(meanVelo, proba)
print(f"Spearman's correlation(up): {correlation}, p-value: {p_value}")
# %%
meanVelo = dd[dd.arrow == "down"]["meanVelo"]
proba = dd[dd.arrow == "down"]["proba"]

# Spearman's rank correlation
correlation, p_value = spearmanr(meanVelo, proba)
print(f"Spearman's correlation: {correlation}, p-value: {p_value}")

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

# Friedman test for up arrow
pg.friedman(data=dd[dd.arrow == "up"], dv="meanVelo", within="proba", subject="sub")
# %%

pg.friedman(data=dd[dd.arrow == "down"], dv="meanVelo", within="proba", subject="sub")
# %%
# Wilcoxon Test to see whether the arrow has an effect within each proba
# It is the equivalent of paiup t-test
pg.wilcoxon(
    x=dd[(dd.arrow == "down") & (dd.proba == 0.5)].meanVelo,
    y=dd[(dd.arrow == "up") & (dd.proba == 0.5)].meanVelo,
)
# %%

pg.wilcoxon(
    x=dd[(dd.arrow == "down") & (dd.proba == 0.25)].meanVelo,
    y=dd[(dd.arrow == "up") & (dd.proba == 0.25)].meanVelo,
)
# %%
pg.wilcoxon(
    x=dd[(dd.arrow == "down") & (dd.proba == 0.75)].meanVelo,
    y=dd[(dd.arrow == "up") & (dd.proba == 0.75)].meanVelo,
)
# %%
# Pivot the data for proba
pivot_proba = dd[dd.arrow == "up"].pivot(
    index="sub", columns="proba", values="meanVelo"
)
pivot_proba
# %%
# Perform the Friedman Test for proba
statistic_proba, p_value_proba = friedmanchisquare(
    pivot_proba[0.25], pivot_proba[0.5], pivot_proba[0.75]
)
print(
    f"Friedman Test for proba: Statistic(up) = {statistic_proba}, p-value = {p_value_proba}"
)


# %%
# Pivot the data for proba
pivot_proba = dd[dd.arrow == "down"].pivot(
    index="sub", columns="proba", values="meanVelo"
)
pivot_proba
# %%
# Perform the Friedman Test for proba
statistic_proba, p_value_proba = friedmanchisquare(
    pivot_proba[0.25], pivot_proba[0.5], pivot_proba[0.75]
)
print(
    f"Friedman Test for proba: Statistic(down) = {statistic_proba}, p-value = {p_value_proba}"
)


# %%
# Pivot the data for proba
pivot_arrow = dd[dd.proba == 0.25].pivot(
    index="sub", columns="arrow", values="meanVelo"
)
pivot_arrow

# a %%
# Perform the wilcoxon Test for arrow
statistic_arrow, p_value_arrow = wilcoxon(pivot_arrow["down"], pivot_arrow["up"])
print(
    f"Wilcoxon Test for arrow Statistic(P(Right|up)=0.25) = {statistic_arrow}, p-value = {p_value_arrow}"
)


# %%
# Pivot the data for proba
pivot_arrow = dd[dd.proba == 0.5].pivot(index="sub", columns="arrow", values="meanVelo")
pivot_arrow

# a %%
# Perform the wilcoxon Test for arrow
statistic_arrow, p_value_arrow = wilcoxon(pivot_arrow["down"], pivot_arrow["up"])
print(
    f"Wilcoxon Test for arrow Statistic(P(Right|up)=0.5) = {statistic_arrow}, p-value = {p_value_arrow}"
)


# %%
# Pivot the data for proba
pivot_arrow = dd[dd.proba == 0.75].pivot(
    index="sub", columns="arrow", values="meanVelo"
)
pivot_arrow

# a %%
# Perform the wilcoxon Test for arrow
statistic_arrow, p_value_arrow = wilcoxon(pivot_arrow["down"], pivot_arrow["up"])
print(
    f"Wilcoxon Test for arrow Statistic(P(Right|up)=0.75) = {statistic_arrow}, p-value = {p_value_arrow}"
)


# %%
# pos-hoc analysis

# Perform the Nemenyi Test
posthoc = sp.posthoc_nemenyi_friedman(pivot_proba.values)
print(posthoc)

# %%

# Perform the Wilcoxon Test post-hoc analysis
posthoc = sp.posthoc_wilcoxon(pivot_proba.values.T)
print(posthoc)
# %%
# Apply the Holm-Bonferroni correction to the Wilcoxon Test p-values
corrected_p_values = multipletests(posthoc.values.flatten(), method="holm")[1]
corrected_p_values = corrected_p_values.reshape(posthoc.shape)

print("Holm-Bonferroni corrected Wilcoxon Test p-values:")
print(pd.DataFrame(corrected_p_values, index=posthoc.index, columns=posthoc.columns))
# %%
model = sm.OLS.from_formula("meanVelo~ (proba) ", data=dd[dd.arrow == "up"])
result = model.fit()

print(result.summary())
# %%
model = sm.OLS.from_formula("meanVelo~ (proba) ", data=dd[dd.arrow == "down"])
result = model.fit()

print(result.summary())
# %%
sns.histplot(
    data=df[df.proba == 0.25],
    x="meanVelo",
    hue="arrow",
)
plt.show()
# %%
# Early trials
earlyTrials = 40
p = 0.75
sns.histplot(
    data=df[(df.proba == p) & (df.trial <= earlyTrials)],
    x="meanVelo",
    hue="arrow",
    alpha=0.5,
    # multiple="dodge",
)
plt.title(f"Early Trials: {earlyTrials}, P(Right|up)={proba}")
plt.show()

# %%
# Mid trials
midTrials = [60, 180]
sns.histplot(
    data=df[(df.proba == p) & (df.trial <= midTrials[1]) & (df.trial > midTrials[0])],
    x="meanVelo",
    hue="arrow",
    alpha=0.5,
    # multiple="dodge",
)
plt.title(f"Mid Trials{midTrials[0]},{midTrials[1]}: P(Right|up)={proba}")
plt.show()
# %%
# Late trials
lateTrials = 200
sns.histplot(
    data=df[(df.proba == p) & (df.trial > lateTrials)],
    x="meanVelo",
    hue="arrow",
    alpha=0.5,
    # multiple="dodge",
)
plt.title(f"Early Trials>{lateTrials}: P(Right|up)={proba}")
plt.show()
# %%
# Repeated measures ANOVA
# Perform mixed ANOVA
model = ols("meanVelo ~ C(arrow)*(proba) ", data=dd).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)
# %%
# cehcking the normality of the data
print(pg.normality(dd["meanVelo"]))
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

# Create pointplots for each sub
facet_grid.map_dataframe(sns.histplot, x="meanVelo", hue="arrow", alpha=0.3)

# Add legends
facet_grid.add_legend()

# Set titles for each subplot
for ax, p in zip(facet_grid.axes.flat, df.proba.unique()):
    ax.set_title(f"ASEM: P(Right|up)=P(Left|down)={p}")
# Adjust spacing between subplots
facet_grid.fig.subplots_adjust(
    wspace=0.2, hspace=0.2
)  # Adjust wspace and hspace as needed

# Show the plot
plt.show()

# %%
# Perform mixed repeated measures ANOVA
anova_results = pg.rm_anova(
    dv="meanVelo",
    within="proba",
    subject="sub",
    data=dd[dd.arrow == "down"],
    correction=True,
)

print(anova_results)
# %%
anova_results = pg.rm_anova(
    dv="meanVelo",
    within="proba",
    subject="sub",
    data=dd[dd.arrow == "up"],
    correction=True,
)

print(anova_results)
# %%
sns.pointplot(
    data=dd,
    x="proba",
    y="meanVelo",
    capsize=0.1,
    errorbar="sd",
    hue="arrow",
)
_ = plt.title("asem across porba")
plt.show()
# %%
sns.pointplot(
    data=df[df.arrow == "up"],
    x="proba",
    y="meanVelo",
    capsize=0.1,
    errorbar="se",
    hue="sub",
    palette="Set2",
)
_ = plt.title("ASEM  across porba: up")
plt.show()
# %%

sns.pointplot(
    data=df[df.arrow == "down"],
    x="proba",
    y="meanVelo",
    capsize=0.1,
    errorbar="se",
    hue="sub",
    palette="Set2",
)
_ = plt.title("asem across porba: down")
plt.show()
# %%
anova_results = pg.rm_anova(
    dv="meanVelo",
    within=["proba", "arrow"],
    subject="sub",
    data=dd,
)

print(anova_results)
# %%

model = smf.mixedlm(
    "meanVelo~C( arrow )",
    data=dd[dd.proba == 0.75],
    # re_formula="~proba",
    groups=dd[dd.proba == 0.75]["sub"],
).fit()
model.summary()

# %%
model = smf.mixedlm(
    "meanVelo~( proba )",
    data=dd[dd.arrow == "up"],
    # re_formula="~proba",
    groups=dd[dd.arrow == "up"]["sub"],
).fit()
model.summary()

# %%
model = smf.mixedlm(
    "meanVelo~ (proba)",
    data=dd[dd.arrow == "down"],
    # re_formula="~proba",
    groups=dd[dd.arrow == "down"]["sub"],
).fit()
model.summary()


# %%
model = smf.mixedlm(
    "meanVelo~ C(proba)*C(arrow)",
    data=dd,
    # re_formula="~proba",
    groups=dd["sub"],
).fit()
model.summary()

# %%
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.barplot(
    x="proba",
    y="meanVelo",
    hue="arrow",
    data=dd,
)
plt.title("ASEM over 3 different probabilites for down & up.")
plt.xlabel("P(Right|up)=P(Left|down)")
plt.savefig(pathFig + "/meanVeloarrows.png")
plt.show()

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
df.columns
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
uparrowsPalette = ["#e83865", "#cc3131"]
downarrowsPalette = ["#8cd790", "#285943"]
# %%
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.barplot(
    x="proba",
    y="posOffSet",
    data=df_prime[df_prime.arrow == "up"],
)
plt.title("Position Offset: arrow up", fontsize=30)
plt.xlabel("P(Right|up)")
plt.ylabel("Position Offset", fontsize=20)
plt.savefig(pathFig + "/posOffSetup.png")
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
    data=learningCurve[learningCurve.arrow == "up"],
)
plt.legend(fontsize=20)
plt.title("Position Offset: arrow up Given Previous Target Direction", fontsize=30)
plt.xlabel("P(Right|up)", fontsize=20)
plt.ylabel("Position Offset", fontsize=20)
plt.savefig(pathFig + "/posOffSetupTD.png")
plt.show()
# %%
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.barplot(
    x="proba",
    y="meanVelo",
    data=df_prime[df_prime.arrow == "up"],
)
plt.title("ASEM: arrow up", fontsize=30)
plt.xlabel("P(Right|up)", fontsize=20)
plt.ylabel("Anticipatory Smooth Eye Movement", fontsize=20)
plt.savefig(pathFig + "/meanVeloup.png")
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
    data=df_prime[df_prime.arrow == "up"],
)
plt.legend(fontsize=20)
plt.title("Anticipatory Velocity Given Previous TD: arrow up ", fontsize=30)
plt.xlabel("P(Right|up)", fontsize=20)
plt.ylabel("Anticipatory Smooth Eye Movement", fontsize=20)
plt.savefig(os.path.join(pathFig, "meanVeloupTD.png"))
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
plt.title("Position Offset: arrow down", fontsize=30)
plt.xlabel("P(Left|down)", fontsize=20)
plt.ylabel("Position Offset", fontsize=20)
plt.savefig(pathFig + "/posOffSetdown.png")
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
plt.savefig(pathFig + "/posOffSetdownTD.png")
plt.show()
# %%
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.barplot(
    x="proba",
    y="meanVelo",
    data=df[df.arrow == "down"],
)
plt.title("ASEM: arrow down", fontsize=30)
plt.xlabel("P(Left|down)", fontsize=20)
plt.ylabel("Anticipatory Smooth Eye Movement", fontsize=20)
plt.savefig(pathFig + "/meanVelodown.png")
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
    palette=downarrowsPalette,
    data=learningCurve[learningCurve.arrow == "down"],
)
plt.legend(fontsize=20)
plt.title("meanVelo: arrow down\n ", fontsize=30)
plt.ylabel("Anticipatory Smooth Eye Movement", fontsize=20)
plt.xlabel("P(Left|down)", fontsize=20)
plt.savefig(pathFig + "/meanVelodownTD.png")
plt.show()
# Adding the interacrion between  previous arrow and previous TD.
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
valueToKeep = df_prime.interaction.unique()[1:]
valueToKeep
# %%
df_prime = df_prime[df_prime["interaction"].isin(valueToKeep)]
df_prime.interaction.unique()
# %%

learningCurveInteraction = (
    df_prime.groupby(["sub", "proba", "interaction", "arrow"])
    .mean()[["posOffSet", "meanVelo"]]
    .reset_index()
)

# %%
df.columns
# %%
df_prime.groupby(["sub", "proba", "interaction", "arrow"]).count()[
    ["posOffSet", "meanVelo"]
]

# %%
learningCurveInteraction
# %%
# Create a figure and axis
fig = plt.figure()

# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()

sns.barplot(
    x="proba",
    y="posOffSet",
    palette="magma",
    hue="interaction",
    data=learningCurveInteraction[learningCurveInteraction.arrow == "up"],
)
plt.legend(fontsize=20)
plt.title(
    "Position Offset:arrow up\n Interaction of Previous Target Direction & arrow Chosen ",
    fontsize=30,
)
plt.xlabel("P(Right|up)", fontsize=20)
plt.ylabel("Position Offset", fontsize=20)
plt.savefig(pathFig + "/posOffSetUpupInteraction.png")
plt.show()
# %%
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.barplot(
    x="proba",
    y="meanVelo",
    palette="coolwarm",
    hue="interaction",
    data=learningCurveInteraction[learningCurveInteraction.arrow == "up"],
)
plt.title(
    "ASEM: arrow up\n Interaction of Previous Target Direction & arrow Chosen",
    fontsize=30,
)
plt.xlabel("P(Right|arrow)", fontsize=20)
plt.ylabel("Anticipatory Smooth Eye Movement", fontsize=20)
plt.savefig(pathFig + "/meanVeloupInteraction.png")
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
    data=learningCurveInteraction[learningCurveInteraction.arrow == "down"],
)
plt.title(
    "Position Offset: arrow down\n Interaction of Previous Target Direction & arrow Chosen",
    fontsize=30,
)
plt.xlabel("P(Left|down)", fontsize=20)
plt.ylabel("Position Offset", fontsize=20)
plt.savefig(pathFig + "/posOffSetdownInteraction.png")
plt.show()
# %%
fig = plt.figure()

# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.barplot(
    x="proba",
    y="meanVelo",
    palette="coolwarm",
    hue="interaction",
    data=learningCurveInteraction[learningCurveInteraction.arrow == "down"],
)
plt.title("ASEM:arrow down\n Interaction of Previous Target Direction & arrow Chosen")
plt.xlabel("P(Left|down)")
plt.show()
# %%
df
# %%
dd = df.groupby(["sub", "proba", "arrow", "TD_prev"])["meanVelo"].mean().reset_index()
dd
# %%
model = smf.mixedlm(
    "meanVelo~  C(arrow,Treatment('up'))*C(TD_prev)",
    data=dd[dd.proba == 0.25],
    groups=dd[dd.proba == 0.25]["sub"],
).fit()
model.summary()
# %%
model = smf.mixedlm(
    "meanVelo~  C(arrow,Treatment('up'))*C(TD_prev)",
    data=dd[dd.proba == 0.75],
    # re_formula="~proba",
    groups=dd[dd.proba == 0.75]["sub"],
).fit()
model.summary()
# %%
model = smf.mixedlm(
    "meanVelo~  C(arrow)*C(TD_prev)",
    data=dd[dd.proba == 0.5],
    # re_formula="~proba",
    groups=dd[dd.proba == 0.5]["sub"],
).fit()
model.summary()
# %%
df.groupby("interaction").count()
# %%
