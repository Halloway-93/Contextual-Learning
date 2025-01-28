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
# %%
pathFig = "/Users/mango/Contextual-Learning/ColorCue/figures/voluntaryColor/"
df=pd.read_csv("/Users/mango/dataANEMO_allSubs_activeColorCP.csv")
print(df)
# %%
df.rename(columns={'pR-Red': 'proba'}, inplace=True)
df.rename(columns={'trial_color': 'color'}, inplace=True)
print(df.columns)
# %%
badTrials = df[(df["aSPv"] <= -8) | (df["aSPv"] >= 8)]
badTrials
# %%
# df = df[(df["aSPv"] <= 8) & (df["aSPv"] >= -8)]
# df = df[(df["SPlat"] <= 150) ]
df["aSPv"].max()
# %%
sns.histplot(data=df, x="aSPv")
plt.show()
# %%
sns.histplot(data=df, x="SPlat")
plt.show()
# %%
for sub in df["sub"].unique():
    for p in df[df["sub"] == sub]["proba"].unique():
        df.loc[(df["sub"] == sub) & (df["proba"] == p), "TD_prev"] = df.loc[
            (df["sub"] == sub) & (df["proba"] == p), "target_dir"
        ].shift(1)
        df.loc[(df["sub"] == sub) & (df["proba"] == p), "color_prev"] = df.loc[
            (df["sub"] == sub) & (df["proba"] == p), "color"
        ].shift(1)
# %%
df.columns
df[(df["TD_prev"].isna())]
# %%
df = df[~(df["TD_prev"].isna())]
# %%
df.TD_prev
# %%
df["TD_prev"] = df["TD_prev"].apply(lambda x: "left" if x == -1 else "right")
# %%
df = df[(df["sub"] != 9)]
# df = df[df["sub"].isin([1, 2, 5, 7, 8, 11, 13])]
# %%
# df.dropna(subset=["aSPv"], inplace=True)
# df = df[(df.aSPv <= 15) & (df.aSPv >= -15)]
df.columns
# %%
colors = ["Green", "Red"]
# %%
dd = (
    df.groupby(["sub", "color", "proba"])[["aSPv"]]
    .mean()
    .reset_index()
)

dd
# %%
np.abs(dd.aSPv.values).max()
# %%
dd[np.abs(dd.aSPv.values) > 1]
# %%
# df[(df.aSPv > 15) | (df.aSPv < -15)]
# %%
# Normalizing the data
# dd["aSPvNorm"] = (dd["aSPv"] - dd["aSPv"].mean()) / dd["aSPv"].std()
# %%
# Statisitcal test


#samodel = sm.OLS.from_formula("aSPv~ C(proba)*C(color) ", data=dd)
# result = model.fit()
#
# print(result.summary())
dd
# %%

aSPv = dd[dd.color == "Red"]["aSPv"]
proba = dd[dd.color == "Red"]["proba"]

# Spearman's rank correlation
correlation, p_value = spearmanr(aSPv, proba)
print(f"Spearman's correlation(Red): {correlation}, p-value: {p_value}")
# %%
aSPv = dd[dd.color == "Green"]["aSPv"]
proba = dd[dd.color == "Green"]["proba"]

# Spearman's rank correlation
correlation, p_value = spearmanr(aSPv, proba)
print(f"Spearman's correlation (Green): {correlation}, p-value: {p_value}")

# %%
aSPv = dd[dd["proba"] == .75]["aSPv"]
color = dd[dd["proba"]== .75]["color"]

# Spearman's rank correlation
correlation, p_value = spearmanr(aSPv, color)
print(f"Spearman's correlation(proba 75): {correlation}, p-value: {p_value}")


# %%

aSPv = dd[dd["proba"] == .25]["aSPv"]
color = dd[dd["proba"] == .25]["color"]

# Spearman's rank correlation
correlation, p_value = spearmanr(aSPv, color)
print(f"Spearman's correlation(proba 25): {correlation}, p-value: {p_value}")


# %%

aSPv = dd[dd["proba"] == .50]["aSPv"]
color = dd[dd["proba"] == .50]["color"]

# Spearman's rank correlation
correlation, p_value = spearmanr(aSPv, color)
print(f"Spearman's correlation(proba 50): {correlation}, p-value: {p_value}")


# %%

# Friedman test for Red color
pg.friedman(data=df[df.color == "Red"], dv="aSPv", within="proba", subject="sub")
# %%

pg.friedman(data=df[df.color == "Green"], dv="aSPv", within="proba", subject="sub")
# %%
# Wilcoxon Test to see whether the color has an effect within each proba
# It is the equivalent of paiRed t-test
pg.wilcoxon(
    x=dd[(dd.color == "Green") & (dd[ "proba" ]== .50)].aSPv,
    y=dd[(dd.color == "Red") & (dd[ "proba" ] == .50)].aSPv,
)
# %%

pg.wilcoxon(
    x=dd[(dd.color == "Green") & (dd[ "proba" ]==.25)].aSPv,
    y=dd[(dd.color == "Red") & (dd[ "proba" ] == .25)].aSPv,
)
# %%
pg.wilcoxon(
    x=dd[(dd.color == "Green") & (dd["proba"] ==.75)].aSPv,
    y=dd[(dd.color == "Red") & (dd["proba"] == .75)].aSPv,
)
# %%
# Pivot the data for proba
pivot_proba = dd[dd.color == "Green"].pivot(
    index="sub", columns="proba", values="aSPv"
)
pivot_proba
# %%
# Perform the Friedman Test for proba
statistic_proba, p_value_proba = friedmanchisquare(
    pivot_proba[.25], pivot_proba[.50], pivot_proba[.75]
)
print(
    f"Friedman Test for proba: Statistic(Red) = {statistic_proba}, p-value = {p_value_proba}"
)


# %%
# Pivot the data for proba
pivot_proba = dd[dd.color == "Red"].pivot(
    index="sub", columns="proba", values="aSPv"
)
pivot_proba
# %%
# Perform the Friedman Test for proba
statpivot_proba, p_pivot_proba = friedmanchisquare(
    pivot_proba[.25], pivot_proba[.50], pivot_proba[.75]
)
print(
    f"Friedman Test for proba: Statistic(Green) = {statistic_proba}, p-value = {p_value_proba}"
)


# %%
# Pivot the data for proba
pivot_color = dd[dd["proba"]== .25].pivot(index="sub", columns="color", values="aSPv")
pivot_color

# a %%
# Perform the wilcoxon Test for color
statistic_color, p_value_color = wilcoxon(pivot_color["Green"], pivot_color["Red"])
print(
    f"Wilcoxon Test for color Statistic(P(Right|Red)=25) = {statistic_color}, p-value = {p_value_color}"
)


# %%
# Pivot the data for proba
pivot_color = dd[dd["proba"]== .50].pivot(index="sub", columns="color", values="aSPv")
pivot_color

# a %%
# Perform the wilcoxon Test for color
statistic_color, p_value_color = wilcoxon(pivot_color["Green"], pivot_color["Red"])
print(
    f"Wilcoxon Test for color Statistic(P(Right|Red)=50) = {statistic_color}, p-value = {p_value_color}"
)


# %%
# Pivot the data for proba
pivot_color = dd[dd["proba"] == .75].pivot(index="sub", columns="color", values="aSPv")
pivot_color

# a %%
# Perform the wilcoxon Test for color
statistic_color, p_value_color = wilcoxon(pivot_color["Green"], pivot_color["Red"])
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
model = sm.OLS.from_formula("aSPv~ (proba) ", data=dd[dd.color == "Red"])
result = model.fit()

print(result.summary())
# %%
model = sm.OLS.from_formula("aSPv~ (proba) ", data=dd[dd.color == "Green"])
result = model.fit()

print(result.summary())
# %%
colors = ["Green", "Red"]
sns.displot(
    data=df[df.proba == .75],
    x="aSPv",
    hue="color",
    hue_order=colors,
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
p = .75
sns.displot(
    data=df[(df.proba == p) & (df.trial <= earlyTrials)],
    x="aSPv",
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
    x="aSPv",
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
    x="aSPv",
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
model = ols("aSPv ~ C(color)*(proba) ", data=dd).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)
# %%
# cehcking the normality of the data
print(pg.normality(dd["aSPv"]))
# %%
stat, p = stats.kstest(
    dd["aSPv"], "norm", args=(dd["aSPv"].mean(), dd["aSPv"].std(ddof=1))
)
print(f"Statistic: {stat}, p-value: {p}")
# %%
x = dd["aSPv"]
ax = pg.qqplot(x, dist="norm")
plt.show()
# %%
sns.histplot(data=df, x="aSPv", hue="color", bins=20, palette=colors)
plt.show()
# %%
sns.histplot(data=df, x="aSPv", hue="proba", bins=20, palette="viridis")
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
    sns.histplot, x="aSPv", hue="color", palette=colors, hue_order=colors
)

# Add legends
facet_grid.add_legend()

# Set titles for each subplot
for ax, p in zip(facet_grid.axes.flat, df.proba.unique()):
    ax.set_title(f"ASEM: P(Right|Red)=P(Left|Green)={p}")
    ax.legend(["Red", "Green"])
# Adjust spacing between subplots
facet_grid.figure.subplots_adjust(
    wspace=0.2, hspace=0.2
)  # Adjust wspace and hspace as needed

# Show the plot
plt.show()

# %%
for s in df["sub"].unique():
    df_s = df[df["sub"] == s]

    # Set up the FacetGrid
    facet_grid = sns.FacetGrid(
        data=df_s,
        col="proba",
        col_wrap=3,
        height=8,
        aspect=1.5,
    )

    # Create pointplots for each sub
    facet_grid.map_dataframe(
        sns.histplot, x="aSPv", hue="color", palette=colors, hue_order=colors
    )

    # Add legends
    facet_grid.add_legend()

    # Set titles for each subplot
    for ax, p in zip(facet_grid.axes.flat, df_s.proba.unique()):
        ax.set_title(f"ASEM Subject {s}: P(Right|Red)=P(Left|Green)={p}")
        ax.legend(["Red", "Green"])
    # Adjust spacing between subplots
    facet_grid.figure.subplots_adjust(
        wspace=0.2, hspace=0.2
    )  # Adjust wspace and hspace as needed

    # Show the plot
    plt.show()

# %%
# Perform mixed repeated measures ANOVA
anova_results = pg.rm_anova(
    dv="aSPv",
    within="proba",
    subject="sub",
    data=df[df.color == "Green"],
)

print(anova_results)
# %%
anova_results = pg.rm_anova(
    dv="aSPv",
    within="proba",
    subject="sub",
    data=dd[dd.color == "Red"],
)

print(anova_results)
# %%
dd
# %%
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.pointplot(
    data=dd,
    x="proba",
    y="aSPv",
    capsize=0.1,
    errorbar="ci",
    hue="color",
    hue_order=colors,
    palette=colors,
)
_ = plt.title("ASEM Across probabilities", fontsize=30)
plt.legend(fontsize=20)
plt.xlabel("P(Right|Red)=P(Left|Green)", fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel("ASEM (deg/s)", fontsize=30)
plt.savefig(pathFig + "/asemAcrossprobapp.svg")
plt.show()
# %%
sns.catplot(
    data=dd,
    x="proba",
    y="aSPv",
    hue="color",
    kind="violin",
    split=True,
    palette=colors,
)
plt.show()
# %%
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.pointplot(
    data=df[df.color == "Red"],
    x="proba",
    y="aSPv",
    capsize=0.1,
    errorbar="ci",
    hue="sub",
    palette="tab20",
    alpha=0.8,
)
_ = plt.title("ASEM Per Subject: color Red", fontsize=30)
plt.legend(fontsize=20)
plt.xlabel("P(Right|Red)", fontsize=30)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.ylabel("ASEM (deg/s)", fontsize=30)
plt.savefig(pathFig + "/individualsRed.svg")
plt.show()
# %%
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.pointplot(
    data=df[df.color == "Green"],
    x="proba",
    y="aSPv",
    capsize=0.1,
    errorbar="ci",
    hue="sub",
    palette="tab20",
    alpha=0.8,
)
_ = plt.title("ASEM Per Subject: color Green", fontsize=30)
plt.legend(fontsize=20)
plt.xlabel("P(Right|Green)", fontsize=30)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.ylabel("ASEM (deg/s)", fontsize=30)
plt.savefig(pathFig + "/individualsGreen.svg")
plt.show()
# %%
# pg.normality(df[df.color == "Red"], group="proba", dv="aSPv")
# %%
anova_results = pg.rm_anova(
    dv="aSPv",
    within=["proba", "color"],
    subject="sub",
    data=dd,
)

print(anova_results)
# %%
model = smf.mixedlm(
    "aSPv~C( proba,Treatment(.50)) *color",
    data=df,
    re_formula="~proba",
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
# a %%
normaltest_result = stats.normaltest(residuals)
print(f"D'Agostino's K^2 test p-value: {normaltest_result.pvalue:.4f}")
# %%
model = smf.mixedlm(
    "aSPv~C( proba,Treatment(.50))",
    data=df[df.color == "Red"],
    # re_formula="~proba",
    groups=df[df.color == "Red"]["sub"],
).fit()
model.summary()

# %%
model = smf.mixedlm(
    "aSPv~C( proba,Treatment(.50))",
    data=df[df.color == "Green"],
    # re_formula="~proba",
    groups=df[df.color == "Green"]["sub"],
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
    "aSPv~ C(color)",
    data=df[df.proba == .25],
    re_formula="~color",
    groups=df[df.proba == .25]["sub"],
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
    "aSPv~ C(color)",
    data=df[df.proba == .50],
    re_formula="~color",
    groups=df[df.proba == .50]["sub"],
).fit()
model.summary()
# %%
model = smf.mixedlm(
    "aSPv~ C(color)",
    data=df[df.proba == .75],
    re_formula="~color",
    groups=df[df.proba == .75]["sub"],
).fit()
model.summary()

# %%
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.barplot(
    x="proba",
    y="aSPv",
    hue="color",
    errorbar="se",
    palette=colors,
    hue_order=colors,
    data=dd,
)
plt.legend(fontsize=20)
plt.title("ASEM across 3 different probabilites", fontsize=30)
plt.xlabel("P(Right|Red)=P(Left|Green)", fontsize=30)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.ylim(-0.75, 0.75)
plt.legend(fontsize=20)
plt.ylabel("ASEM (deg/s)", fontsize=30)
plt.savefig(pathFig + "/aSPvcolors.svg")
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
    hue_order=colors,
    palette=colors,
    data=df,
)
plt.title("ASEM over 3 different probabilites for Green & Red.")
plt.xlabel("P(Right|Red)=P(Left|Green)")
plt.savefig(pathFig + "/posOffSetcolors.png")
plt.show()
# %%
# Adding the column of the color of the  previous trial
# %%
# getting previous TD for each trial for each subject and each proba
# %%
df_prime = df[
    [
        "sub",
        "trial",
        "proba",
        "color",
        "target_dir",
        "TD_prev",
        "posOffSet",
        "aSPv",
    ]
]
learningCurve = (
    df_prime.groupby(["sub", "proba", "color", "TD_prev"])
    .mean()[["posOffSet", "aSPv"]]
    .reset_index()
)


learningCurve
# %%
df_prime.groupby(["sub", "proba", "color", "TD_prev"]).count()[
    ["posOffSet", "aSPv"]
]

# %%
RedcolorsPalette = ["#e83865", "#cc3131"]
GreencolorsPalette = ["#8cd790", "#285943"]
# %%
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.barplot(
    x="proba",
    y="posOffSet",
    color="Red",
    data=df_prime[df_prime.color == "Red"],
)
plt.title("Position Offset: color Red", fontsize=30)
plt.xlabel("P(Right|Red)")
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
    palette=RedcolorsPalette,
    data=df[df.color == "Red"],
)
plt.legend(fontsize=20)
plt.title("Position Offset: color Red Given Previous Target Direction", fontsize=30)
plt.xlabel("P(Right|Red)", fontsize=20)
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
    y="aSPv",
    color="Red",
    data=dd[dd[ "color" ] == "Red"],
)
plt.title("Anticipatory Smooth Eye Movement: color Red", fontsize=30)
plt.xlabel("P(Right|Red)", fontsize=30)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.ylim(-0.75, 0.75)
plt.ylabel("ASEM (deg/s)", fontsize=30)
plt.savefig(pathFig + "/aSPvRed.svg")
plt.show()
# %%
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.barplot(
    x="proba",
    y="aSPv",
    hue="TD_prev",
    hue_order=["left", "right"],
    errorbar="ci",
    palette=RedcolorsPalette,
    data=df[df.color == "Red"],
)
plt.legend(fontsize=20)
plt.title("Anticipatory Velocity Given Previous TD: color Red ", fontsize=30)
plt.xlabel("P(Right|Red)", fontsize=30)
plt.ylabel("ASEM (deg/s)", fontsize=30)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.ylim(-0.75, 0.75)
plt.savefig(pathFig + "/aSPvRedTD.svg")
plt.show()
# %%
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.barplot(
    x="proba",
    y="posOffSet",
    color="Green",
    data=learningCurve[learningCurve.color == "Green"],
)
plt.title("Position Offset: color Green", fontsize=30)
plt.xlabel("P(Left|Green)", fontsize=20)
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
    palette=GreencolorsPalette,
    data=df[df.color == "Green"],
)
plt.legend(fontsize=20)
plt.title("Anticipatory Velocity Given Previous TD: color Green ", fontsize=30)
plt.xlabel("P(Left|Green)", fontsize=20)
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
    y="aSPv",
    color="Green",
    errorbar="ci",
    data=df[df.color == "Green"],
)
plt.title("Anticipatory Smooth Eye Movement: color Green", fontsize=30)
plt.xlabel("P(Left|Green)", fontsize=30)
plt.ylabel("ASEM (deg/s)", fontsize=30)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.ylim(-0.75, 0.75)
plt.savefig(pathFig + "/aSPvGreen.svg")
plt.show()
# %%
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.barplot(
    x="proba",
    y="aSPv",
    hue="TD_prev",
    hue_order=["left", "right"],
    palette=GreencolorsPalette,
    data=df[df.color == "Green"],
)
plt.legend(fontsize=20)
plt.title("Anticipatory Velocity Given Previous TD: color Green ", fontsize=30)
plt.ylabel("ASEM (deg/s)", fontsize=30)
plt.xlabel("P(Left|Green)", fontsize=30)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.ylim(-1, 1)
plt.savefig(pathFig + "/aSPvGreenTD.svg")
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
        "aSPv",
    ]
]
df_prime
# %%

learningCurveInteraction = (
    df_prime.groupby(["sub", "proba", "interaction", "color"])
    .mean()[["posOffSet", "aSPv"]]
    .reset_index()
)

# %%
df.columns
# %%
df_prime.groupby(["proba", "interaction", "color"]).count()[["posOffSet", "aSPv"]]

# %%
learningCurveInteraction
# %%
# Cmap for Green and Red for the interaction plots

cmapRed = LinearSegmentedcolormap.from_list("Red", ["w", "Red"])
colorsRed = list(cmapRed(np.linspace(0.5, 1, 4)))
cmapGreen = LinearSegmentedcolormap.from_list("Green", ["w", "Green"])
colorsGreen = list(cmapGreen(np.linspace(0, 1, 4)))
# %%
RedcolorsPalette = ["#e83865", "#cc3131"]
GreencolorsPalette = ["#8cd790", "#285943"]
colorsPalette = ["#285943", "#cc3131", "#e83865", "#8cd790"]
# %%
df_prime["interaction"].unique()
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
    hue_order=df_prime["interaction"].unique(),
    data=df[df.color == "Red"],
)
plt.legend(fontsize=20)
plt.title(
    "Position Offset:color Red\n Interaction of Previous Target Direction & color Chosen ",
    fontsize=30,
)
plt.xlabel("P(Right|Red)", fontsize=30)
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
    y="aSPv",
    palette=colorsPalette,
    hue="interaction",
    hue_order=df_prime["interaction"].unique(),
    data=df_prime[df_prime.color == "Red"],
)
plt.title(
    "ASEM: color Red\n Interaction of Previous Target Direction & color Chosen",
    fontsize=30,
)
plt.legend(fontsize=20)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.ylim(-1.25, 1.25)
plt.xlabel("P(Right|Red)", fontsize=30)
plt.ylabel("ASEM(deg/s)", fontsize=30)
plt.savefig(pathFig + "/aSPvRedInteraction.svg")
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
    data=df[df.color == "Green"],
)
plt.title(
    "Position Offset: color Green\n Interaction of Previous Target Direction & color Chosen",
    fontsize=30,
)
plt.xlabel("P(Left|Green)", fontsize=30)
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
    y="aSPv",
    palette=colorsPalette,
    hue="interaction",
    hue_order=df_prime["interaction"].unique(),
    data=df_prime[df_prime.color == "Green"],
)
plt.legend(fontsize=20)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.ylim(-1.25, 1.25)
plt.title(
    "ASEM:color Green\n Interaction of Previous Target Direction & color Chosen",
    fontsize=30,
)
plt.xlabel("P(Left|Green)", fontsize=30)
plt.ylabel("ASEM (deg/s)", fontsize=30)
plt.savefig(pathFig + "/aSPvGreenInteraction.svg")
plt.show()
# %%
df
# %%
model = smf.mixedlm(
    "aSPv~  C(color,Treatment('Red'))*C(TD_prev)",
    data=df[df.proba == 25],
    re_formula="~color",
    groups=df[df.proba == 25]["sub"],
).fit(method="lbfgs")
model.summary()
# %%
model = smf.mixedlm(
    "aSPv~  C(color,Treatment('Red')) * C(TD_prev)",
    data=df[df.proba == 75],
    re_formula="~color",
    groups=df[df.proba == 75]["sub"],
).fit(method="lbfgs")
model.summary()
# %%
model = smf.mixedlm(
    "aSPv~  C(color,Treatment('Red'))*C(TD_prev)",
    data=df[df.proba == 50],
    re_formula="~color",
    groups=df[df.proba == 50]["sub"],
).fit(method=["lbfgs"])
model.summary()
# %%
df.color_prev
# %%
# Sampling Bias analysis

# Define transition counts for previous state = Green
Green_transitions = (
    df[df["color_prev"] == "Green"]
    .groupby(["sub", "proba", "color"])["aSPv"]
    .count()
    .reset_index(name="count")
)
Green_transitions["total"] = Green_transitions.groupby(["sub", "proba"])[
    "count"
].transform("sum")

Green_transitions["conditional_prob"] = (
    Green_transitions["count"] / Green_transitions["total"]
)

Green_transitions = Green_transitions.rename(columns={"color": "current_state"})
Green_transitions["previous_state"] = "Green"


# Define transition counts for previous state = Red
Red_transitions = (
    df[df["color_prev"] == "Red"]
    .groupby(["sub", "proba", "color"])["aSPv"]
    .count()
    .reset_index(name="count")
)
Red_transitions["total"] = Red_transitions.groupby(["sub", "proba"])["count"].transform(
    "sum"
)
Red_transitions["conditional_prob"] = (
    Red_transitions["count"] / Red_transitions["total"]
)
Red_transitions = Red_transitions.rename(columns={"color": "current_state"})
Red_transitions["previous_state"] = "Red"
# %%
# Combine results
conditional_probabilities = pd.concat([Green_transitions, Red_transitions])
conditional_probabilities
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
conditional_probabilities


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
        ax.set_ylabel("Conditional probability")
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
                group[group["transition_state"] == "('Green', 'Green')"][
                    "conditional_prob"
                ]
            )
            > 0
        ):
            Green_to_Green = group[group["transition_state"] == "('Green', 'Green')"][
                "conditional_prob"
            ].values[0]
        else:
            Green_to_Green = 0

        if (
            len(
                group[group["transition_state"] == "('Red', 'Red')"]["conditional_prob"]
            )
            > 0
        ):
            Red_to_Red = group[group["transition_state"] == "('Red', 'Red')"][
                "conditional_prob"
            ].values[0]
        else:
            Red_to_Red = 0

        if (
            len(
                group[group["transition_state"] == "('Red', 'Green')"][
                    "conditional_prob"
                ]
            )
            > 0
        ):
            Green_to_Red = group[group["transition_state"] == "('Red', 'Green')"][
                "conditional_prob"
            ].values[0]
        else:
            Green_to_Red = 0

        if len(
            group[group["transition_state"] == "('Green', 'Red')"]["conditional_prob"]
        ):

            Red_to_Green = group[group["transition_state"] == "('Green', 'Red')"][
                "conditional_prob"
            ].values[0]
        else:
            Red_to_Green = 0

        # Persistent: high probability of staying in the same state
        if Green_to_Green > 0.6 and Red_to_Red > 0.6:
            return "Persistent"

        # Alternating: high probability of switching states
        if Green_to_Red > 0.6 and Red_to_Green > 0.6:
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
    plt.title("Subject Behavior Classification\n(Consistent Across probabilities)")
    plt.show()

    # Print detailed results
    print(subject_classification)

    # Additional detailed view
    detailed_behavior = subject_proba_behavior.pivot_table(
        index="sub", columns="proba", values="behavior", aggfunc="first"
    )
    print("\nDetailed Behavior Across probabilities:")
    print(detailed_behavior)

    return subject_classification


subject_classification = classify_subject_behavior(conditional_probabilities)
# %%
subject_classification

# %%
# Perform classification
# Optional: Create a more detailed summary
summary = subject_classification.groupby("behavior_class").size()
print("\nBehavior Classification Summary:")
print(summary)
# %%
dd = df.groupby(["sub", "proba", "color"])["aSPv"].mean().reset_index()
# %%
for s in dd["sub"].unique():
    behavior_value = subject_classification[subject_classification["sub"] == s][
        "behavior_class"
    ].values[0]
    dd.loc[dd["sub"] == s, "behavior"] = behavior_value

# %%
dd
# %%
sns.lmplot(
    data=dd[(dd["color"] == "Green")],
    x="proba",
    y="aSPv",
    hue="behavior",
)
plt.show()
# %%
sns.lmplot(
    data=dd[(dd["color"] == "Red")],
    x="proba",
    y="aSPv",
    hue="behavior",
)
plt.show()

# %%
dd
# %%
# Computing the peristance score based on the transition probabilites
# One should expect a U shape fit here
for p in df["proba"].unique():
    ddp = dd[dd["proba"] == p].copy()
    sns.regplot(
        data=ddp,
        x=ddp[ddp["color"] == "Green"]["aSPv"].values,
        y=ddp[ddp["color"] == "Red"]["aSPv"].values,
    )
    plt.ylabel("adaptation_Red", fontsize=20)
    plt.xlabel("adaptation_Green", fontsize=20)
    plt.title(f"proba={p}")
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()
# %%
conditional_probabilities
# %%
dd = (
    conditional_probabilities.groupby(["sub", "transition_state", "proba"])[
        "conditional_prob"
    ]
    .mean()
    .reset_index()
)
dd
# %%
dd["sub"].value_counts()
# %%
ts = dd["transition_state"].unique()
new_rows = []
for s in dd["sub"].unique():
    for p in dd[dd["sub"] == s]["proba"].unique():
        existing_ts = dd[(dd["sub"] == s) & (dd["proba"] == p)][
            "transition_state"
        ].unique()
        for t in ts:
            if t not in existing_ts:
                # Add a new row with sub = s and transition_state = t, setting transition_state to 0
                new_row = {
                    "sub": s,
                    "proba": p,
                    "transition_state": t,
                    "conditional_prob": 0,
                }
                new_rows.append(new_row)

# Concatenate the new rows to the original DataFrame
dd = pd.concat([dd, pd.DataFrame(new_rows)], ignore_index=True)

print(dd)
# %%

dd = (
    dd.groupby(["sub", "proba", "transition_state"])["conditional_prob"]
    .mean()
    .reset_index()
)
dd
# %%
dd["transition_state"].unique()

# %%
dd["sub"].value_counts()


# %%
# Function to classify transition_state as persistent or alternating
def classify_transition(state):
    return (
        "persistent"
        if state == "('Red', 'Red')" or state == "('Green', 'Green')"
        else "alternating"
    )


# Apply the classification function
dd["transition_type"] = dd["transition_state"].apply(classify_transition)
dd
# %%
adaptation = (
    dd.groupby(["sub", "transition_type"])["conditional_prob"].mean().reset_index()
)
adaptation
# %%
# Group by 'sub' and calculate the score
result = pd.DataFrame()
result["sub"] = df["sub"].unique()
# %%
result["persistence_score"] = (
    adaptation[adaptation["transition_type"] == "persistent"]["conditional_prob"].values
    - adaptation[adaptation["transition_type"] == "alternating"][
        "conditional_prob"
    ].values
)
result
# %%
dd["transition_state"].unique()
# %%
result["persistence_score_Red"] = np.nan
result["persistence_score_Green"] = np.nan
result
# %%
dd
# %%

for s in dd["sub"].unique():
    Red_Red_prob = np.mean(
        dd[(dd["sub"] == s) & (dd["transition_state"] == "('Red', 'Red')")][
            "conditional_prob"
        ]
    )
    Green_Red_prob = np.mean(
        dd[(dd["sub"] == s) & (dd["transition_state"] == "('Green', 'Red')")][
            "conditional_prob"
        ]
    )

    result.loc[result["sub"] == s, "persistence_score_Red"] = (
        Red_Red_prob - Green_Red_prob
    )

    Green_Green_prob = np.mean(
        dd[(dd["sub"] == s) & (dd["transition_state"] == "('Green', 'Green')")][
            "conditional_prob"
        ]
    )

    Red_Green_prob = np.mean(
        dd[(dd["sub"] == s) & (dd["transition_state"] == "('Red', 'Green')")][
            "conditional_prob"
        ]
    )
    result.loc[result["sub"] == s, "persistence_score_Green"] = (
        Green_Green_prob - Red_Green_prob
    )
result
# %%

# Group by 'sub', 'proba', and 'color' and calculate the mean of 'aSPv'
mean_velo = df.groupby(["sub", "proba", "color"])["aSPv"].mean().reset_index()

# Pivot the table to have 'proba' as columns
pivot_table = mean_velo.pivot_table(
    index=["sub", "color"], columns="proba", values="aSPv"
).reset_index()

# Calculate the adaptation
pivot_table["adaptation"] = (
    np.abs(pivot_table[75] + pivot_table[25] - 2 * pivot_table[50]) / 2
)
# %%
print(pivot_table)
# %%

# Select the relevant columns
adaptation = pivot_table[["sub", "color", "adaptation"]]

print(adaptation)
# %%

result["adaptation"] = adaptation.groupby("sub")["adaptation"].mean().values
result["adaptation_Green"] = adaptation[adaptation["color"] == "Green"][
    "adaptation"
].values
result["adaptation_Red"] = adaptation[adaptation["color"] == "Red"]["adaptation"].values
# %%
result = pd.DataFrame(result)
result
# %%
sns.lmplot(data=result, x="persistence_score", y="adaptation", height=10)
plt.ylabel("adaptation", fontsize=20)
plt.xlabel("persistence_score", fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig(pathFig + "/samplingBiasColours.svg")
plt.show()
# %%
sns.lmplot(
    data=result,
    x="persistence_score_Green",
    y="adaptation_Green",
    height=10,
    scatter_kws={"color": "seaGreen"},
    line_kws={"color": "seaGreen"},
)
plt.ylabel("adaptation_Green", fontsize=20)
plt.xlabel("persistence_score", fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig(pathFig + "/samplingBiasGreen.svg")
plt.show()
# %%

sns.lmplot(
    data=result,
    x="persistence_score_Red",
    y="adaptation_Red",
    height=10,
    scatter_kws={"color": "salmon"},
    line_kws={"color": "salmon"},
)
plt.ylabel("adaptation_Red", fontsize=20)
plt.xlabel("persistence_score", fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig(pathFig + "/samplingBiasRed.svg")
plt.show()
# %%
sns.lmplot(data=result, x="adaptation_Green", y="adaptation_Red")
plt.show()
# %%
sns.scatterplot(
    data=result,
    x="adaptation_Green",
    y="adaptation_Red",
    hue="sub",
    s=100,
    palette="tab20",
)
plt.show()
# %%

sns.lmplot(
    data=result,
    x="persistence_score_Green",
    y="persistence_score_Red",
)
plt.show()

# %%

correlation, p_value = spearmanr(
    result["persistence_score"],
    result["adaptation"],
)
print(
    f"Spearman's correlation for the adaptation score: {correlation}, p-value: {p_value}"
)

# %%
correlation, p_value = spearmanr(
    result["persistence_score_Green"],
    result["adaptation_Green"],
)
print(
    f"Spearman's correlation for the adaptation score for Green: {correlation}, p-value: {p_value}"
)

# %%
correlation, p_value = spearmanr(
    result["persistence_score_Red"],
    result["adaptation_Red"],
)
print(
    f"Spearman's correlation for the adaptation score for Red: {correlation}, p-value: {p_value}"
)

# %%
correlation, p_value = spearmanr(
    result["adaptation_Green"],
    result["adaptation_Red"],
)
print(
    f"Spearman's correlation for the adaptation score for Red): {correlation}, p-value: {p_value}"
)

# %%

correlation, p_value = spearmanr(
    result["persistence_score_Red"],
    result["persistence_score_Green"],
)
print(
    f"Spearman's correlation for the adaptation score for Red): {correlation}, p-value: {p_value}"
)

# %%
model = sm.OLS.from_formula("adaptation_Red~ persistence_score_Red ", result).fit()

print(model.summary())
# %%
model = sm.OLS.from_formula("adaptation_Green~ persistence_score_Green ", result).fit()

print(model.summary())
# %%

model = sm.OLS.from_formula("adaptation_Red~ adaptation_Green ", result).fit()

print(model.summary())
# %%
df.columns
# %%
df.groupby(["sub", "proba", "color", "trial_color_UP", "TD_prev"])[
    "aSPv"
].mean().reset_index()
# %%
# Group by 'sub', 'proba', and 'color' and calculate the mean of 'aSPv'
mean_velo = df.groupby(["sub", "proba", "color"])["aSPv"].mean().reset_index()

# Pivot the table to have 'proba' as columns
pivot_table = mean_velo.pivot_table(
    index=["sub", "proba"], columns="color", values="aSPv"
).reset_index()

# Calculate the adaptation
pivot_table["adaptation"] = (
    np.abs(pivot_table["Green"]) + np.abs(pivot_table["Red"])
) / 2

print(pivot_table)
# %%
sns.scatterplot(
    data=pivot_table, x="Red", y="Green", hue="proba", palette="viridis", s=50
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
    y="Green",
)
plt.show()
# %%
sns.boxplot(
    data=pivot_table,
    x="proba",
    y="Green",
)
plt.show()
# %%
# Create the plot with connected dots for each participant
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=pivot_table,
    x="Red",
    y="Green",
    hue="proba",
    palette="viridis",
    style="proba",
    markers=["o", "s", "D"],
)
# Connect dots for each participant
for sub in pivot_table["sub"].unique():
    subset = pivot_table[pivot_table["sub"] == sub]
    plt.plot(subset["Red"], subset["Green"], color="gray", alpha=0.5, linestyle="--")
# Add plot formatting
plt.axhline(0, color="black", linestyle="--")
plt.axvline(0, color="black", linestyle="--")
plt.title(f"Participants adaptaion across probabilites")
plt.xlabel("Red")
plt.ylabel("Green")
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
    plt.plot(subset["Red"], subset["Green"], color="gray", alpha=0.5, linestyle="--")
    sns.scatterplot(
        data=pivot_table[pivot_table["sub"] == sub],
        x="Red",
        y="Green",
        hue="proba",
        palette="viridis",
        style="proba",
        markers=["o", "s", "D"],
    )
    # Add plot formatting
    plt.axhline(0, color="black", linestyle="--")
    plt.axvline(0, color="black", linestyle="--")
    plt.title(f"Participant:{sub}")
    plt.xlabel("Red")
    plt.ylabel("Green")
    plt.legend(title="proba")
    plt.tight_layout()
    plt.show()
# %%
# Group by 'sub', 'proba', and 'color' and calculate the mean of 'aSPv'
mean_velo = df.groupby(["sub", "proba", "interaction"])["aSPv"].mean().reset_index()
print(mean_velo)
mean_velo["interaction"] = mean_velo["interaction"].astype("str")
# %%
# Pivot the table to have 'proba' as columns
pivot_table = mean_velo.pivot_table(
    index=["sub", "proba"], columns="interaction", values="aSPv"
).reset_index()

# Calculate the adaptation
# pivot_table["adaptation"] = (
#     np.abs(pivot_table["Green"]) + np.abs(pivot_table["Red"])
# ) / 2

print(pivot_table)
pivot_table = pd.DataFrame(pivot_table)
pivot_table.columns
# %%
pivot_table.columns[2]
# %%
# pivot_table.rename(
#     columns={
#         ('left', 'Green'): "left_Green",
#         ('left', 'Red'): "left_Red",
#         ('right', 'Green'): "right_Green",
#         ('right', 'Red'): "right_Red",
#     },
#     inplace=True,
# )
#
# pivot_table.columns
# %%
sns.scatterplot(
    data=pivot_table, x="('right', 'Red')", y="('right', 'Green')", hue="proba"
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
    data=pivot_table, x="('left', 'Red')", y="('left', 'Green')", hue="proba"
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
    df.groupby(["sub", "proba", "color", "trial_color_UP"])["aSPv"]
    .count()
    .reset_index(name="count")["count"]
)
plt.show()
# %%
df_inter = (
    df.groupby(["sub", "proba", "color", "trial_color_UP", "TD_prev"])["aSPv"]
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
    y="aSPv",
    hue="interaction",
    palette=GreencolorsPalette,
    data=df_inter[df_inter.color == "Green"],
)
plt.legend(fontsize=20)
plt.title("Anticipatory Velocity Given the color Position: Green ", fontsize=30)
plt.ylabel("ASEM (deg/s)", fontsize=30)
plt.xlabel("P(Left|Green)", fontsize=30)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.ylim(-1, 1)
# plt.savefig(pathFig + "/aSPvGreenTD.svg")
plt.show()
# %%
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.barplot(
    x="proba",
    y="aSPv",
    hue="interaction",
    palette=RedcolorsPalette,
    data=df_inter[df_inter.color == "Red"],
)
plt.legend(fontsize=20)
plt.title("Anticipatory Velocity Given the color Position: Red", fontsize=30)
plt.ylabel("ASEM (deg/s)", fontsize=30)
plt.xlabel("P(Left|Green)", fontsize=30)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.ylim(-1, 1)
# plt.savefig(pathFig + "/aSPvGreenTD.svg")
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
    y="aSPv",
    hue="interColPos",
    palette=GreencolorsPalette,
    data=df[df.color == "Green"],
)
plt.legend(fontsize=20)
plt.title("Anticipatory Velocity Given the color Position: Green ", fontsize=30)
plt.ylabel("ASEM (deg/s)", fontsize=30)
plt.xlabel("P(Left|Green)", fontsize=30)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.ylim(-1, 1)
# plt.savefig(pathFig + "/aSPvGreenTD.svg")
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
    y="aSPv",
    hue="interColPos",
    palette=RedcolorsPalette,
    data=df[df.color == "Red"],
)
plt.legend(fontsize=20)
plt.title("Anticipatory Velocity Given the color Position: Green ", fontsize=30)
plt.ylabel("ASEM (deg/s)", fontsize=30)
plt.xlabel("P(Left|Green)", fontsize=30)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.ylim(-1, 1)
# plt.savefig(pathFig + "/aSPvGreenTD.svg")
plt.show()
# %%
