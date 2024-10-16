import os
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

path = "/Volumes/work/brainets/oueld.h/contextuaLearning/ColorCue/data"
pathFig = "/Users/mango/PhD/Contextual-Learning/ColorCue/figures/"
fileName = "results.csv"
# %%
redColorsPalette = ["#e83865", "#cc3131"]
greenColorsPalette = ["#8cd790", "#285943"]
# %%
df = pd.read_csv(os.path.join(path, fileName))
# [print(df[df["sub_number"] == i]["meanVelo"].isna().sum()) for i in range(1, 13)]
# df.dropna(inplace=True)
df["color"] = df["trial_color_chosen"].apply(lambda x: "green" if x == 0 else "red")


# df = df.dropna(subset=["meanVelo"])
# Assuming your DataFrame is named 'df' and the column you want to rename is 'old_column'
# df.rename(columns={'old_column': 'new_column'}, inplace=True)
df = df[(df["sub_number"] != 8) & (df["sub_number"] != 11)]
# df.dropna(subset=["meanVelo"], inplace=True)
# df = df[(df.meanVelo <= 15) & (df.meanVelo >= -15)]
df
# %%
colors = ["green", "red"]
# %%
dd = (
    df.groupby(["sub_number", "color", "proba"])[["meanVelo", "posOffSet"]]
    .mean()
    .reset_index()
)

dd
# %%
np.abs(dd.meanVelo.values).max()
# %%
dd[np.abs(dd.meanVelo.values) > 1.93]
# %%
# df[(df.meanVelo > 15) | (df.meanVelo < -15)]
# %%
# Normalizing the data
# dd["meanVeloNorm"] = (dd["meanVelo"] - dd["meanVelo"].mean()) / dd["meanVelo"].std()
# %%
# Statisitcal test


# model = sm.OLS.from_formula("meanVelo~ C(proba)*C(color) ", data=dd)
# result = model.fit()
#
# print(result.summary())
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
pg.friedman(
    data=df[df.color == "red"], dv="meanVelo", within="proba", subject="sub_number"
)
# %%

pg.friedman(
    data=df[df.color == "green"], dv="meanVelo", within="proba", subject="sub_number"
)
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
pivot_proba = dd[dd.color == "red"].pivot(
    index="sub_number", columns="proba", values="meanVelo"
)
pivot_proba
# %%
# Perform the Friedman Test for proba
statistic_proba, p_value_proba = friedmanchisquare(
    pivot_proba[25], pivot_proba[50], pivot_proba[75]
)
print(
    f"Friedman Test for proba: Statistic = {statistic_proba}, p-value = {p_value_proba}"
)


# %%
# Pivot the data for proba
pivot_color = dd[dd.proba == 75].pivot(
    index="sub_number", columns="color", values="meanVelo"
)
pivot_color

# %%
# Perform the wilcoxon Test for color
statistic_color, p_value_color = wilcoxon(pivot_color["green"], pivot_color["red"])
print(
    f"Wilcoxon Test for color Statistic = {statistic_color}, p-value = {p_value_color}"
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
sns.histplot(
    data=df[df.proba == 75],
    x="meanVelo",
    hue="color",
    alpha=0.5,
    # multiple="dodge",
    palette=colors,
)
plt.show()
# %%
# Early trials
sns.histplot(
    data=df[(df.proba == 75) & (df.trial_number <= 40)],
    x="meanVelo",
    hue="color",
    alpha=0.5,
    # multiple="dodge",
    palette=colors,
)
plt.show()

# %%
# Mid trials
sns.histplot(
    data=df[(df.proba == 75) & (df.trial_number <= 180) & (df.trial_number > 60)],
    x="meanVelo",
    hue="color",
    hue_order=colors,
    alpha=0.5,
    # multiple="dodge",
    palette=colors,
)
plt.show()
# %%
# Late trials
sns.histplot(
    data=df[(df.proba == 75) & (df.trial_number > 200)],
    x="meanVelo",
    hue="color",
    hue_order=colors,
    alpha=0.5,
    # multiple="dodge",
    palette=colors,
)
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

# Create pointplots for each sub_number
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
    subject="sub_number",
    data=df[df.color == "green"],
)

print(anova_results)
# %%
anova_results = pg.rm_anova(
    dv="meanVelo",
    within="proba",
    subject="sub_number",
    data=df[df.color == "red"],
)

print(anova_results)
# %%
sns.pointplot(
    data=df,
    x="proba",
    y="meanVelo",
    capsize=0.1,
    errorbar="se",
    hue="color",
    palette=colors,
)
_ = plt.title("asem across porba")
plt.show()
# %%
sns.pointplot(
    data=df[df.color == "red"],
    x="proba",
    y="meanVelo",
    capsize=0.1,
    errorbar="se",
    hue="sub_number",
    palette="dark:salmon",
)
_ = plt.title("ASEM  across porba: Red")
plt.show()
# %%

# %%
sns.pointplot(
    data=df[df.color == "green"],
    x="proba",
    y="meanVelo",
    capsize=0.1,
    errorbar="se",
    hue="sub_number",
    palette="Set2",
)
_ = plt.title("asem across porba: Green")
plt.show()
# %%
# pg.normality(df[df.color == "red"], group="proba", dv="meanVelo")
# %%
anova_results = pg.rm_anova(
    dv="meanVelo",
    within=["proba", "color"],
    subject="sub_number",
    data=df,
)

print(anova_results)
# %%

df.sub_number.unique()
# %%
model = smf.mixedlm(
    "meanVelo~ C( proba )",
    data=df[df.color == "red"],
    # re_formula="~proba",
    groups=df[df.color == "red"]["sub_number"],
).fit()
model.summary()

# %%
model = smf.mixedlm(
    "meanVelo~ C(proba)",
    data=df[df.color == "green"],
    # re_formula="~proba",
    groups=df[df.color == "green"]["sub_number"],
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
    palette=colors,
    data=df,
)
plt.title("ASEM over 3 different probabilites for Green & Red.")
plt.xlabel("P(Right|RED)=P(Left|Green)")
plt.savefig(pathFig + "/meanVeloColors.png")
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
    data=dd,
)
plt.title("ASEM over 3 different probabilites for Green & Red.")
plt.xlabel("P(Right|RED)=P(Left|Green)")
plt.savefig(pathFig + "/posOffSetColors.png")
plt.show()
# %%
df.columns
# %%
# Adding the column of the color of the  previous trial
df.trial_direction
# %%
# getting previous TD for each trial for each subject and each proba
for sub in df["sub_number"].unique():
    for p in df[df["sub_number"] == sub]["proba"].unique():
        df.loc[(df["sub_number"] == sub) & (df["proba"] == p), "TD_prev"] = df.loc[
            (df["sub_number"] == sub) & (df["proba"] == p), "trial_direction"
        ].shift(1)
        df.loc[(df["sub_number"] == sub) & (df["proba"] == p), "color_prev"] = df.loc[
            (df["sub_number"] == sub) & (df["proba"] == p), "color"
        ].shift(1)
# %%
df.TD_prev
# %%
df_prime = df[
    [
        "sub_number",
        "trial_number",
        "proba",
        "color",
        "trial_direction",
        "TD_prev",
        "posOffSet",
        "meanVelo",
    ]
]
learningCurve = (
    df_prime.groupby(["sub_number", "proba", "color", "TD_prev"])
    .mean()[["posOffSet", "meanVelo"]]
    .reset_index()
)


learningCurve
# %%
df_prime.groupby(["proba", "color", "TD_prev"]).count()[["posOffSet", "meanVelo"]]

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
    palette=redColorsPalette,
    data=learningCurve[learningCurve.color == "red"],
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
    data=df_prime[df_prime.color == "red"],
)
plt.title("ASEM: Color Red", fontsize=30)
plt.xlabel("P(Right|RED)", fontsize=20)
plt.ylabel("Anticipatory Smooth Eye Movement", fontsize=20)
plt.savefig(pathFig + "/meanVeloRed.png")
plt.show()
# %%
df_prime[df_prime["TD_prev"].isna()]
# %%
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.barplot(
    x="proba",
    y="meanVelo",
    hue="TD_prev",
    palette=redColorsPalette,
    data=df_prime[df_prime.color == "red"],
)
plt.legend(fontsize=20)
plt.title("Anticipatory Velocity Given Previous TD: Color Red ", fontsize=30)
plt.xlabel("P(Right|RED)", fontsize=20)
plt.ylabel("Anticipatory Smooth Eye Movement", fontsize=20)
plt.savefig(pathFig + "/meanVeloRedTD.png")
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
    palette=greenColorsPalette,
    data=learningCurve[learningCurve.color == "green"],
)
plt.legend(fontsize=20)
plt.title("Position Offset:Color Green \n  ", fontsize=30)
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
    data=learningCurve[learningCurve.color == "green"],
)
plt.title("ASEM: Color Green", fontsize=30)
plt.xlabel("P(Left|GREEN)", fontsize=20)
plt.ylabel("Anticipatory Smooth Eye Movement", fontsize=20)
plt.savefig(pathFig + "/meanVeloGreen.png")
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
    palette=greenColorsPalette,
    data=learningCurve[learningCurve.color == "green"],
)
plt.legend(fontsize=20)
plt.title("meanVelo: Color Green\n ", fontsize=30)
plt.ylabel("Anticipatory Smooth Eye Movement", fontsize=20)
plt.xlabel("P(Left|GREEN)", fontsize=20)
plt.savefig(pathFig + "/meanVeloGreenTD.png")
plt.show()
# Adding the interacrion between  previous color and previous TD.
# %%
df["interaction"] = list(zip(df["TD_prev"], df["color_prev"]))
df_prime = df[
    [
        "sub_number",
        "trial_number",
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
    df_prime.groupby(["sub_number", "proba", "interaction", "color"])
    .mean()[["posOffSet", "meanVelo"]]
    .reset_index()
)

# %%
df.columns
# %%
df_prime.groupby(["sub_number", "proba", "interaction", "color"]).count()[
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
    data=learningCurveInteraction[learningCurveInteraction.color == "red"],
)
plt.legend(fontsize=20)
plt.title(
    "Position Offset:Color Red\n Interaction of Previous Target Direction & Color Chosen ",
    fontsize=30,
)
plt.xlabel("P(Right|RED)", fontsize=20)
plt.ylabel("Position Offset", fontsize=20)
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
    palette="magma_r",
    hue="interaction",
    data=learningCurveInteraction[learningCurveInteraction.color == "red"],
)
plt.title(
    "ASEM: Color Red\n Interaction of Previous Target Direction & Color Chosen",
    fontsize=30,
)
plt.xlabel("P(Right|Color)", fontsize=20)
plt.ylabel("Anticipatory Smooth Eye Movement", fontsize=20)
plt.savefig(pathFig + "/meanVeloRedInteraction.png")
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
    data=learningCurveInteraction[learningCurveInteraction.color == "green"],
)
plt.title(
    "Position Offset: Color Green\n Interaction of Previous Target Direction & Color Chosen",
    fontsize=30,
)
plt.xlabel("P(Left|GREEN)", fontsize=20)
plt.ylabel("Position Offset", fontsize=20)
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
    palette="viridis",
    hue="interaction",
    data=learningCurveInteraction[learningCurveInteraction.color == "green"],
)
plt.title("ASEM:Color Green\n Interaction of Previous Target Direction & Color Chosen")
plt.xlabel("P(Left|GREEN")
plt.show()
# %%
