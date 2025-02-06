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
from matplotlib.colors import LinearSegmentedColormap

path = "/Volumes/work/brainets/oueld.h/attentionalTask/data/"
pathFig = "/Users/mango/PhD/Contextual-Learning/attentionTask/figures/"
jobLibData = "jobLibProcessingCC.csv"
allEventsFile = "/Volumes/work/brainets/oueld.h/attentionalTask/data/allEvents.csv"


# %%
jlData = pd.read_csv(os.path.join(path, jobLibData))

# %%
jlData
# %%
exampleJL = jlData[
    (jlData["sub"] == 4) & (jlData["proba"] == 75) & (jlData["trial"] == 183)
]
exampleJL
# %%
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
allEventsFile = "/Volumes/work/brainets/oueld.h/attentionalTask/data/allEvents.csv"
pathFig = "/Users/mango/Contextual-Learning/attentionTask/figures/"
# %%
allEvents = pd.read_csv(allEventsFile)
df = pd.read_csv(
    "/Volumes/work/brainets/oueld.h/attentionalTask/data/processedResultsWindow(80,120).csv"
)
# %%
badTrials = df[(df["meanVelo"] < -8) | (df["meanVelo"] > 8)]
badTrials
# %%
df = df[df["sub"] != 13]
# %%
df = df[(df["meanVelo"] <= 8) & (df["meanVelo"] >= -8)]
df["meanVelo"].max()
# %%
sns.histplot(data=df, x="meanVelo")
plt.show()
# %%
df.columns
# %%
# df = pd.read_csv(os.path.join(path, fileName))
# [print(df[df["sub"] == i]["meanVelo"].isna().sum()) for i in range(1, 13)]
# df.dropna(inplace=True)
df["color"] = df["trial_color_imposed"].apply(lambda x: "green" if x == 0 else "red")
df["color_chosen"] = df["trial_color_chosen"].apply(
    lambda x: "green" if x == 0 else "red"
)


# df = df.dropna(subset=["meanVelo"])
# Assuming your DataFrame is named 'df' and the column you want to rename is 'old_column'
# df.rename(columns={'old_column': 'new_column'}, inplace=True)
# %%
# df.dropna(subset=["meanVelo"], inplace=True)
# df = df[(df.meanVelo <= 15) & (df.meanVelo >= -15)]
df
# %%
colors = ["green", "red"]
# %%
dd = (
    df.groupby(["sub", "color", "proba"])[["meanVelo", "posOffSet"]]
    .mean()
    .reset_index()
)

dd
# %%
np.abs(dd.meanVelo.values).max()
# %%


# model = sm.OLS.from_formula("meanVelo~ C(proba)*C(color) ", data=dd)
# result = model.fit()
#
# print(result.summary())
dd
# %%

meanVelo = dd[dd.color == "red"]["meanVelo"]
proba = dd[dd.color == "red"]["proba"]

# Spearman's rank correlation
correlation, p_value = spearmanr(meanVelo, proba)
print(f"Spearman's correlation(RED): {correlation}, p-value: {p_value}")
# %%
meanVelo = dd[dd.color == "green"]["meanVelo"]
proba = dd[dd.color == "green"]["proba"]

# Spearman's rank correlation
correlation, p_value = spearmanr(meanVelo, proba)
print(f"Spearman's correlation: {correlation}, p-value: {p_value}")

# %%
meanVelo = dd[dd.proba == 75]["meanVelo"]
color = dd[dd.proba == 75]["color"]

# Spearman's rank correlation
correlation, p_value = spearmanr(meanVelo, color)
print(f"Spearman's correlation(Proba 75): {correlation}, p-value: {p_value}")


# %%

meanVelo = dd[dd.proba == 25]["meanVelo"]
color = dd[dd.proba == 25]["color"]

# Spearman's rank correlation
correlation, p_value = spearmanr(meanVelo, color)
print(f"Spearman's correlation(Proba 25): {correlation}, p-value: {p_value}")


# %%

meanVelo = dd[dd.proba == 50]["meanVelo"]
color = dd[dd.proba == 50]["color"]

# Spearman's rank correlation
correlation, p_value = spearmanr(meanVelo, color)
print(f"Spearman's correlation(Proba 50): {correlation}, p-value: {p_value}")


# %%

# Friedman test for Red color
pg.friedman(data=df[df.color == "red"], dv="meanVelo", within="proba", subject="sub")
# %%

pg.friedman(data=df[df.color == "green"], dv="meanVelo", within="proba", subject="sub")
# %%
# Wilcoxon Test to see whether the color has an effect within each proba
# It is the equivalent of paired t-test
pg.wilcoxon(
    x=dd[(dd.color == "green") & (dd.proba == 50)].meanVelo,
    y=dd[(dd.color == "red") & (dd.proba == 50)].meanVelo,
)
# %%

pg.wilcoxon(
    x=dd[(dd.color == "green") & (dd.proba == 25)].meanVelo,
    y=dd[(dd.color == "red") & (dd.proba == 25)].meanVelo,
)
# %%
pg.wilcoxon(
    x=dd[(dd.color == "green") & (dd.proba == 75)].meanVelo,
    y=dd[(dd.color == "red") & (dd.proba == 75)].meanVelo,
)
# %%
# Pivot the data for proba
pivot_proba = dd[dd.color == "green"].pivot(
    index="sub", columns="proba", values="meanVelo"
)
pivot_proba
# %%
# Perform the Friedman Test for proba
statistic_proba, p_value_proba = friedmanchisquare(
    pivot_proba[25], pivot_proba[50], pivot_proba[75]
)
print(
    f"Friedman Test for proba: Statistic(Red) = {statistic_proba}, p-value = {p_value_proba}"
)


# %%
# Pivot the data for proba
pivot_proba = dd[dd.color == "green"].pivot(
    index="sub", columns="proba", values="meanVelo"
)
pivot_proba
# %%
# Perform the Friedman Test for proba
statistic_proba, p_value_proba = friedmanchisquare(
    pivot_proba[25], pivot_proba[50], pivot_proba[75]
)
print(
    f"Friedman Test for proba: Statistic(Green) = {statistic_proba}, p-value = {p_value_proba}"
)


# %%
# Pivot the data for proba
pivot_color = dd[dd.proba == 25].pivot(index="sub", columns="color", values="meanVelo")
pivot_color

# a %%
# Perform the wilcoxon Test for color
statistic_color, p_value_color = wilcoxon(pivot_color["green"], pivot_color["red"])
print(
    f"Wilcoxon Test for color Statistic(P(Right|Red)=25) = {statistic_color}, p-value = {p_value_color}"
)


# %%
# Pivot the data for proba
pivot_color = dd[dd.proba == 50].pivot(index="sub", columns="color", values="meanVelo")
pivot_color

# a %%
# Perform the wilcoxon Test for color
statistic_color, p_value_color = wilcoxon(pivot_color["green"], pivot_color["red"])
print(
    f"Wilcoxon Test for color Statistic(P(Right|Red)=50) = {statistic_color}, p-value = {p_value_color}"
)


# %%
# Pivot the data for proba
pivot_color = dd[dd.proba == 75].pivot(index="sub", columns="color", values="meanVelo")
pivot_color

# a %%
# Perform the wilcoxon Test for color
statistic_color, p_value_color = wilcoxon(pivot_color["green"], pivot_color["red"])
print(
    f"Wilcoxon Test for color Statistic(P(Right|Red)=75) = {statistic_color}, p-value = {p_value_color}"
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
model = sm.OLS.from_formula("meanVelo~ C(proba) ", data=dd[dd.color == "red"])
result = model.fit()

print(result.summary())
# %%
model = sm.OLS.from_formula("meanVelo~ C(proba) ", data=dd[dd.color == "green"])
result = model.fit()

print(result.summary())
# %%
colors = ["green", "red"]
sns.displot(
    data=df[df.proba == 25],
    x="meanVelo",
    hue="color",
    alpha=0.5,
    # element="step",
    kind="kde",
    fill=True,
    # multiple="dodge",
    palette=colors,
)
plt.show()
# %%
# Early trials
earlyTrials = 40
p = 75
sns.displot(
    data=df[(df.proba == p) & (df.trial <= earlyTrials)],
    x="meanVelo",
    hue="color",
    hue_order=colors,
    alpha=0.5,
    element="step",
    # multiple="dodge",
    palette=colors,
)
plt.title(f"Early Trials: {earlyTrials}, P(Right|Red)={p}")
plt.show()

# %%
# Mid trials
midTrials = [60, 180]
sns.histplot(
    data=df[(df.proba == p) & (df.trial <= midTrials[1]) & (df.trial > midTrials[0])],
    x="meanVelo",
    hue="color",
    hue_order=colors,
    alpha=0.5,
    # multiple="dodge",
    palette=colors,
)
plt.title(f"Mid Trials{midTrials[0]},{midTrials[1]}: P(Right|Red)={proba}")
plt.show()
# %%
# Late trials
lateTrials = 200
sns.histplot(
    data=df[(df.proba == p) & (df.trial > lateTrials)],
    x="meanVelo",
    hue="color",
    hue_order=colors,
    alpha=0.5,
    # multiple="dodge",
    palette=colors,
)
plt.title(f"Early Trials>{lateTrials}: P(Right|Red)={proba}")
plt.show()
# %%
# Repeated measures ANOVA
# Perform mixed ANOVA
model = ols("meanVelo ~ C(color)*(proba) ", data=dd).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)
# %%
# cehcking the normality of the data
print(pg.normality(dd["meanVelo"]))
# %%
x = dd["meanVelo"]
ax = pg.qqplot(x, dist="norm")
plt.show()
# %%
sns.histplot(data=df, x="meanVelo", hue="color", bins=20, palette=colors)
plt.show()
# %%
sns.histplot(data=df, x="meanVelo", hue="proba", bins=20, palette="viridis")
plt.show()
# %%


# Set up the FacetGrid
facet_grid = sns.FacetGrid(
    data=df,
    col="proba",
    col_wrap=3,
    height=8,
    aspect=1.5,
)

# Create pointplots for each sub
facet_grid.map_dataframe(
    sns.histplot, x="meanVelo", hue="color", palette=colors, hue_order=colors
)

# Add legends
facet_grid.add_legend()

# Set titles for each subplot
for ax, p in zip(facet_grid.axes.flat, df.proba.unique()):
    ax.set_title(f"ASEM: P(Right|Red)=P(Left|Green)={p}")

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
    data=df[df.color == "green"],
)

print(anova_results)
# %%
anova_results = pg.rm_anova(
    dv="meanVelo",
    within="proba",
    subject="sub",
    data=dd[dd.color == "red"],
)

print(anova_results)
# %%
sns.pointplot(
    data=df,
    x="proba",
    y="meanVelo",
    capsize=0.1,
    errorbar="ci",
    hue="color",
    hue_order=colors,
    palette=colors,
)
_ = plt.title("asem across porba")
plt.show()
# %%
sns.catplot(
    data=dd,
    x="proba",
    y="meanVelo",
    hue="color",
    kind="violin",
    split=True,
    palette=colors,
)
plt.show()
# %%
cmapR = LinearSegmentedColormap.from_list(
    "redCmap", ["w", "red"], N=len(df["sub"].unique())
)

# %%
cmapG = LinearSegmentedColormap.from_list(
    "redCmap", ["w", "green"], N=len(df["sub"].unique())
)
# %%
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.pointplot(
    data=df[df.color == "red"],
    x="proba",
    y="meanVelo",
    capsize=0.1,
    errorbar="ci",
    hue="sub",
    palette="tab20",
    alpha=0.8,
)
_ = plt.title("ASEM Per Subject: Color Red", fontsize=30)
plt.legend(fontsize=20)
plt.xlabel("P(Right|RED)", fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel("ASEM (deg/s)", fontsize=30)
plt.savefig(pathFig + "/individualsRed.svg")
plt.show()
# %%
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.pointplot(
    data=df[df.color == "green"],
    x="proba",
    y="meanVelo",
    capsize=0.1,
    errorbar="ci",
    hue="sub",
    palette="tab20",
    alpha=0.8,
)
_ = plt.title("ASEM Per Subject: Color Green", fontsize=30)
plt.legend(fontsize=20)
plt.xlabel("P(Right|GREEN)", fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel("ASEM (deg/s)", fontsize=30)
plt.savefig(pathFig + "/individualsGreen.svg")
plt.show()
# %%
# pg.normality(df[df.color == "red"], group="proba", dv="meanVelo")
# %%
anova_results = pg.rm_anova(
    dv="meanVelo",
    within=["proba", "color"],
    subject="sub",
    data=df,
)

print(anova_results)
# %%
model = smf.mixedlm(
    "meanVelo~C( proba,Treatment(50)) *color",
    data=df,
    # re_formula="~proba",
    groups=df["sub"],
).fit()
model.summary()

# %%
residuals = model.resid

# Q-Q plot
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("Q-Q plot of residuals")
plt.show()
# %%
pg.qqplot(residuals, dist="norm")
plt.show()
# %%
# Histogram
plt.hist(residuals, bins=50)
plt.title("Histogram of residuals")
plt.show()
# %%
# Shapiro-Wilk test for normality# Perform the KS test on the residuals
stat, p = stats.kstest(residuals, "norm")

print(f"KS test statistic: {stat:.4f}")
print(f"KS test p-value: {p:.4f}")
# %%
normaltest_result = stats.normaltest(residuals)
print(f"D'Agostino's K^2 test p-value: {normaltest_result.pvalue:.4f}")
# %%
model = smf.mixedlm(
    "meanVelo~C( proba,Treatment(50))",
    data=df[df.color == "red"],
    # re_formula="~proba",
    groups=df[df.color == "red"]["sub"],
).fit()
model.summary()

# %%
model = smf.mixedlm(
    "meanVelo~C( proba,Treatment(50))",
    data=df[df.color == "green"],
    # re_formula="~proba",
    groups=df[df.color == "green"]["sub"],
).fit()
model.summary()

# %%
residuals = model.resid

# Q-Q plot
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("Q-Q plot of residuals")
plt.show()

# Histogram
plt.hist(residuals, bins=50)
plt.title("Histogram of residuals")
plt.show()
# %%
stat, p = stats.kstest(residuals, "norm")

print(f"KS test statistic: {stat:.4f}")
print(f"KS test p-value: {p:.4f}")
normaltest_result = stats.normaltest(residuals)
print(f"D'Agostino's K^2 test p-value: {normaltest_result.pvalue:.4f}")
# %%
model = smf.mixedlm(
    "meanVelo~ C(color)",
    data=df[df.proba == 25],
    # re_formula="~color",
    groups=df[df.proba == 25]["sub"],
).fit()
model.summary()
# %%
residuals = model.resid

# Q-Q plot
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("Q-Q plot of residuals")
plt.show()

# Histogram
plt.hist(residuals, bins=50)
plt.title("Histogram of residuals")
plt.show()

normaltest_result = stats.normaltest(residuals)
print(f"D'Agostino's K^2 test p-value: {normaltest_result.pvalue:.4f}")
# %%
model = smf.mixedlm(
    "meanVelo~ C(color)",
    data=df[df.proba == 50],
    re_formula="~color",
    groups=df[df.proba == 50]["sub"],
).fit()
model.summary()
# %%
model = smf.mixedlm(
    "meanVelo~ C(color)",
    data=df[df.proba == 75],
    # re_formula="~color",
    groups=df[df.proba == 75]["sub"],
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
    hue="color",
    hue_order=colors,
    errorbar="ci",
    palette=colors,
    data=df,
)
plt.legend(fontsize=20)
plt.title("ASEM Across 3 Different Probabilities", fontsize=30)
plt.xlabel("P(Right|RED)=P(Left|GREEN)", fontsize=30)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.ylim(-0.75, 0.75)
plt.legend(fontsize=20)
plt.ylabel("ASEM (deg/s)", fontsize=30)
plt.savefig(pathFig + "/meanVeloColors.svg", transparent=True)
plt.show()

# %%

fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.barplot(
    x="proba",
    y="posOffSet",
    hue="color",
    palette=colors,
    data=df,
)
plt.title("ASEM over 3 different probabilites for Green & Red.")
plt.xlabel("P(Right|RED)=P(Left|Green)")
plt.savefig(pathFig + "/posOffSetColors.png")
plt.show()
# %%
# Adding the column of the color of the  previous trial
# %%
# getting previous TD for each trial for each subject and each proba
for sub in df["sub"].unique():
    for p in df[df["sub"] == sub]["proba"].unique():
        df.loc[(df["sub"] == sub) & (df["proba"] == p), "TD_prev"] = df.loc[
            (df["sub"] == sub) & (df["proba"] == p), "trial_direction"
        ].shift(1)
        df.loc[(df["sub"] == sub) & (df["proba"] == p), "color_prev"] = df.loc[
            (df["sub"] == sub) & (df["proba"] == p), "color"
        ].shift(1)
# %%
df.columns
# %%
df["TD_prev"] = df["TD_prev"].apply(lambda x: "left" if x == -1 else "right")
# %%
df_prime = df[
    [
        "sub",
        "trial",
        "proba",
        "color",
        "trial_direction",
        "TD_prev",
        "posOffSet",
        "meanVelo",
    ]
]
learningCurve = (
    df_prime.groupby(["sub", "proba", "color", "TD_prev"])
    .mean()[["posOffSet", "meanVelo"]]
    .reset_index()
)


learningCurve
# %%
df_prime.groupby(["sub", "proba", "color", "TD_prev"]).count()[
    ["posOffSet", "meanVelo"]
]

# %%
redColorsPalette = ["#e83865", "#cc3131"]
greenColorsPalette = ["#8cd790", "#285943"]
# %%
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.barplot(
    x="proba",
    y="posOffSet",
    color="red",
    data=df_prime[df_prime.color == "red"],
)
plt.title("Position Offset: Color Red", fontsize=30)
plt.xlabel("P(Right|RED)")
plt.ylabel("Position Offset", fontsize=20)
plt.savefig(pathFig + "/posOffSetRed.png")
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
    hue_order=["left", "right"],
    palette=redColorsPalette,
    data=df[df.color == "red"],
)
plt.legend(fontsize=20)
plt.title("Position Offset: Color Red Given Previous Target Direction", fontsize=30)
plt.xlabel("P(Right|RED)", fontsize=20)
plt.ylabel("Position Offset", fontsize=20)
plt.savefig(pathFig + "/posOffSetRedTD.png")
plt.show()
# %%
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.barplot(
    x="proba",
    y="meanVelo",
    color="red",
    data=df[df.color == "red"],
)
plt.title("Anticipatory Smooth Eye Movement: Color Red", fontsize=30)
plt.xlabel("P(Right|RED)", fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylim(-1, 1)
plt.ylabel("ASEM (deg/s)", fontsize=30)
plt.savefig(pathFig + "/meanVeloRed.svg")
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
    hue_order=["left", "right"],
    errorbar="ci",
    palette=redColorsPalette,
    data=df[df.color == "red"],
)
plt.legend(fontsize=20)
plt.title("Anticipatory Velocity Given Previous TD: Color Red ", fontsize=30)
plt.xlabel("P(Right|RED)", fontsize=30)
plt.ylabel("ASEM (deg/s)", fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylim(-1, 1)
plt.savefig(pathFig + "/meanVeloRedTD.svg")
plt.show()
# %%
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.barplot(
    x="proba",
    y="posOffSet",
    color="green",
    data=learningCurve[learningCurve.color == "green"],
)
plt.title("Position Offset: Color Green", fontsize=30)
plt.xlabel("P(Left|GREEN)", fontsize=20)
plt.ylabel("Position Offset", fontsize=20)
plt.savefig(pathFig + "/posOffSetGreen.png")
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
    hue_order=["left", "right"],
    palette=greenColorsPalette,
    data=df[df.color == "green"],
)
plt.legend(fontsize=20)
plt.title("Anticipatory Velocity Given Previous TD: Color Green ", fontsize=30)
plt.xlabel("P(Left|GREEN)", fontsize=20)
plt.ylabel("Position Offset", fontsize=20)
plt.savefig(pathFig + "/posOffSetGreenTD.png")
plt.show()
# %%
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.barplot(
    x="proba",
    y="meanVelo",
    color="green",
    errorbar="ci",
    data=df[df.color == "green"],
)
plt.title("Anticipatory Smooth Eye Movement: Color Green", fontsize=30)
plt.xlabel("P(Left|GREEN)", fontsize=30)
plt.ylabel("ASEM (deg/s)", fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylim(-1, 1)
plt.savefig(pathFig + "/meanVeloGreen.svg")
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
    hue_order=["left", "right"],
    palette=greenColorsPalette,
    data=df[df.color == "green"],
)
plt.legend(fontsize=20)
plt.title("Anticipatory Velocity Given Previous TD: Color Green ", fontsize=30)
plt.ylabel("ASEM (deg/s)", fontsize=30)
plt.xlabel("P(Left|GREEN)", fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylim(-1, 1)
plt.savefig(pathFig + "/meanVeloGreenTD.svg")
plt.show()
# %%
df["interaction"] = list(zip(df["TD_prev"], df["color_prev"]))
df_prime = df[
    [
        "sub",
        "trial",
        "proba",
        "color",
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
    df_prime.groupby(["sub", "proba", "interaction", "color"])
    .mean()[["posOffSet", "meanVelo"]]
    .reset_index()
)

# %%
df.columns
# %%
df_prime.groupby(["proba", "interaction", "color"]).count()[["posOffSet", "meanVelo"]]

# %%
learningCurveInteraction
# %%
# Cmap for green and red for the interaction plots

cmapRed = LinearSegmentedColormap.from_list("Red", ["w", "red"])
colorsRed = list(cmapRed(np.linspace(0.5, 1, 4)))
cmapGreen = LinearSegmentedColormap.from_list("Green", ["w", "green"])
colorsGreen = list(cmapGreen(np.linspace(0, 1, 4)))
# %%
redColorsPalette = ["#e83865", "#cc3131"]
greenColorsPalette = ["#8cd790", "#285943"]
colorsPalette = ["#285943", "#cc3131", "#8cd790", "#e83865"]
# %%
df_prime["interaction"].unique()
# %%
hue_order = [("right", "green"), ("right", "red"), ("left", "green"), ("left", "red")]
# %%
# Create a figure and axis
fig = plt.figure()

# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()

sns.barplot(
    x="proba",
    y="posOffSet",
    palette=colorsPalette,
    hue="interaction",
    hue_order=hue_order,
    data=df[df.color == "red"],
)
plt.legend(fontsize=20)
plt.title(
    "Position Offset:Color Red\n Interaction of Previous Target Direction & Color Chosen ",
    fontsize=30,
)
plt.xlabel("P(Right|RED)", fontsize=30)
plt.ylabel("Position Offset", fontsize=30)
plt.savefig(pathFig + "/posOffSetUpRedInteraction.png")
plt.show()
# %%

fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.barplot(
    x="proba",
    y="meanVelo",
    palette=colorsPalette,
    hue="interaction",
    hue_order=hue_order,
    data=df_prime[df_prime.color == "red"],
)
plt.title(
    "ASEM: Color Red\n Interaction of Previous Target Direction & Color Chosen",
    fontsize=30,
)
plt.legend(fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylim(-1, 1)
plt.xlabel("P(Right|Red)", fontsize=30)
plt.ylabel("ASEM(deg/s)", fontsize=30)
plt.savefig(pathFig + "/meanVeloRedInteraction.svg")
plt.show()
# %%
fig = plt.figure()

# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.barplot(
    x="proba",
    y="posOffSet",
    palette=colorsPalette,
    hue="interaction",
    hue_order=df_prime["interaction"].unique(),
    data=df[df.color == "green"],
)
plt.title(
    "Position Offset: Color Green\n Interaction of Previous Target Direction & Color Chosen",
    fontsize=30,
)
plt.xlabel("P(Left|GREEN)", fontsize=30)
plt.ylabel("Position Offset", fontsize=30)
plt.savefig(pathFig + "/posOffSetGreenInteraction.png")
plt.show()
# %%
fig = plt.figure()

# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.barplot(
    x="proba",
    y="meanVelo",
    palette=colorsPalette,
    hue="interaction",
    hue_order=hue_order,
    data=df_prime[df_prime.color == "green"],
)
plt.legend(fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylim(-1, 1)
plt.title(
    "ASEM:Color Green\n Interaction of Previous Target Direction & Color Chosen",
    fontsize=30,
)
plt.xlabel("P(Left|GREEN)", fontsize=30)
plt.ylabel("ASEM (deg/s)", fontsize=30)
plt.savefig(pathFig + "/meanVeloGreenInteraction.svg")
plt.show()
# %%
df
# %%
df.dropna(subset=["TD_prev"], inplace=True)
df.dropna(subset=["color_prev"], inplace=True)
# %%
model = smf.mixedlm(
    "meanVelo~  C(color,Treatment('red'))*C(TD_prev)",
    data=df[df.proba == 25],
    # re_formula="~color",
    groups=df[df.proba == 25]["sub"],
).fit()
model.summary()
# %%
model = smf.mixedlm(
    "meanVelo~  C(color,Treatment('red')) * C(TD_prev)",
    data=df[df.proba == 75],
    # re_formula="~color",
    groups=df[df.proba == 75]["sub"],
).fit()
model.summary()
# %%
model = smf.mixedlm(
    "meanVelo~  C(color,Treatment('red'))*C(TD_prev)",
    data=df[df.proba == 50],
    # re_formula="~color",
    groups=df[df.proba == 50]["sub"],
).fit()
model.summary()
# %%
# %%
df.color_prev
# %%
