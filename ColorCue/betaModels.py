import os.path as op
from types import CellType
import numpy as np
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import beta, linregress, pearsonr, bernoulli
import matplotlib.pyplot as plt
import random
import seaborn as sns
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import statsmodels.api as sm
import statsmodels.formula.api as smf

# from frites import set_mpl_style
# set_mpl_style()

from statsmodels.graphics.factorplots import interaction_plot
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from sklearn.model_selection import permutation_test_score

# import warnings
# warnings.filterwarnings('ignore')
import researchpy as rp


# ## 0. Define methods

# In[2]:


class BetaModel:
    def __init__(self, leak_factor=1, a=1, b=1, vt=11):

        self.a = a  # the number of times this color went to the right
        self.b = b  # the number of times this color went to the left
        self.n = 0
        # Leak factor
        self.omega = leak_factor
        self.vt = vt

    def outcome(self, q):
        # return the outcome with the predifined Contengency
        return np.random.random() < q

    def update(self, reward):

        self.n += 1
        if reward == 1:
            self.a += 1
        else:
            self.b += 1

        self.a *= self.omega
        self.b *= self.omega

    def reset(self, a=1, b=1):
        self.n = 0
        self.a = a
        self.b = b

    def sample(self):
        # return a value sampled from the beta distribution
        return np.random.beta(self.a, self.b)

    def mean(self):
        # return a value sampled from the beta distribution
        return beta.mean(self.a, self.b)

    def velocity(self):
        p = self.mean()
        return 0.1 * self.vt * 1 / 2 * np.log((p) / (1 - p))


def plot_distributions(betamodels, true_probas, title=""):
    x = np.linspace(0.0, 1.0, 200)
    trials = sum([betamodel.n for betamodel in betamodels])

    colors = ["red", "green"]

    c_index = 0

    for i in range(len(true_probas)):

        c = colors[c_index]

        y = beta(betamodels[i].a, betamodels[i].b)
        plt.plot(x, y.pdf(x), lw=2, color=c, label=f"P(Right|{c})")
        plt.fill_between(x, y.pdf(x), 0, color=c, alpha=0.2)
        plt.vlines(
            true_probas[i], 0, 5, colors=c, linestyles="--", lw=2
        )  # y.pdf(true_probas[i])
        plt.autoscale(tight="True")
        plt.title(title + f"{trials} Trials")
        plt.legend()
        plt.autoscale(tight=True)
        c_index += 1
    # plt.savefig('distributions.png')
    plt.show()


# %%

allEventsFile = (
    "/Volumes/work/brainets/oueld.h/contextuaLearning/ColorCue/data/allEvents.csv"
)
allEvents = pd.read_csv(allEventsFile)
df = pd.read_csv(
    "/Volumes/work/brainets/oueld.h/contextuaLearning/ColorCue/data/processedResultsWindow(80,120).csv"
)
badTrials = df[(df["meanVelo"] <= -8) | (df["meanVelo"] >= 8)]
df = df[(df["meanVelo"] <= 8) & (df["meanVelo"] >= -8)]
df["meanVelo"].max()
# df = pd.read_csv(os.path.join(path, fileName))
# [print(df[df["sub"] == i]["meanVelo"].isna().sum()) for i in range(1, 13)]
# df.dropna(inplace=True)
df["color"] = df["trial_color_chosen"].apply(lambda x: "red" if x == 1 else "green")
df["trial_color_UP"] = df["trial_color_UP"].apply(
    lambda x: "red" if x == 1 else "green"
)


# df = df.dropna(subset=["meanVelo"])
# Assuming your DataFrame is named 'df' and the column you want to rename is 'old_column'
# df.rename(columns={'old_column': 'new_column'}, inplace=True)
for sub in df["sub"].unique():
    for p in df[df["sub"] == sub]["proba"].unique():
        df.loc[(df["sub"] == sub) & (df["proba"] == p), "TD_prev"] = df.loc[
            (df["sub"] == sub) & (df["proba"] == p), "trial_direction"
        ].shift(1)
        df.loc[(df["sub"] == sub) & (df["proba"] == p), "color_prev"] = df.loc[
            (df["sub"] == sub) & (df["proba"] == p), "color"
        ].shift(1)
df[(df["TD_prev"].isna())]
df = df[~(df["TD_prev"].isna())]

df["TD_prev"] = df["TD_prev"].apply(lambda x: "left" if x == -1 else "right")
df = df[(df["sub"] != 9)]
# df = df[df["sub"].isin([1, 2, 5, 7, 8, 11, 13])]
# %%
aVeloGreen = []
aVeloRed = []
for s in df["sub"].unique():
    for p in df[df["sub"] == s]["proba"].unique():
        print(p)
        observerGreen = BetaModel(leak_factor=0.5)
        observerRed = BetaModel(leak_factor=0.5)
        dfsp = df[(df["sub"] == s) & (df["proba"] == p)]

        print(dfsp)
        for t in dfsp["trial"].unique():
            td = dfsp[dfsp["trial"] == t]["trial_direction"].values[0]
            color = dfsp[dfsp["trial"] == t]["color"].iloc[0]
            if color == "green":
                observerGreen.update(td)
            else:
                observerRed.update(td)

        plot_distributions(
            [observerRed, observerGreen],
            [p / 100, (100 - p) / 100],
            f"Sub {s}, Proba {p} :",
        )

        aVeloGreen.append((s, p, observerGreen.velocity()))
        aVeloRed.append((s, p, observerRed.velocity()))


# %%

redColorsPalette = ["#e83865", "#cc3131"]
greenColorsPalette = ["#8cd790", "#285943"]
dfRed = pd.DataFrame(np.reshape(aVeloRed, (45, 3)), columns=["sub", "proba", "asemRed"])
dfGreen = pd.DataFrame(
    np.reshape(aVeloGreen, (45, 3)), columns=["sub", "proba", "asemGreen"]
)
# %%
df = dfRed.merge(dfGreen, on=["sub", "proba"])
# %%
# Melt the DataFrame to have a single 'asem' column and a 'color' column
df_melted = pd.melt(
    df,
    id_vars=["sub", "proba"],  # Columns to keep as is
    value_vars=["asemRed", "asemGreen"],  # Columns to melt
    var_name="color",  # New column name for the variable
    value_name="asem",  # New column name for the values
)

# Map color column to more descriptive labels
df_melted["color"] = df_melted["color"].replace(
    {"asemRed": "red", "asemGreen": "green"}
)

# Display the result
print(df_melted)
# %%
sns.barplot(
    data=df_melted,
    x="proba",
    y="asem",
    hue="color",
    palette=["#e83865", "#8cd790"],
)
plt.show()


# %%
