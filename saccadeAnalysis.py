import numpy as np
from scipy.signal import butter, sosfiltfilt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %%
sacc = pd.read_csv(
    "/Volumes/work/brainets/oueld.h/contextuaLearning/ColorCue/data/saccades.csv"
)
sacc.columns
# %%
mess = pd.read_csv(
    "/Volumes/work/brainets/oueld.h/contextuaLearning/ColorCue/data/allMessages.csv"
)
mess
mess["trial"] = mess["trial"].values - 1
# %%
mess["sub"]
# %%
events = pd.read_csv(
    "/Volumes/work/brainets/oueld.h/contextuaLearning/ColorCue/data/allEvents.csv"
)
events.columns
# %%

all_rois = pd.DataFrame()
for  s in sacc.subject.unique():
    for p in sacc[sacc.subject==s]["probability"].unique():
        for t in sacc[(sacc.subject==s)&(sacc.probability==p)]["trial"].unique():
            decision=mess[(mess[ "trial" ] == t) & (mess[ "sub" ] == s) & (mess[ "proba" ] == p)]["color_chosen_time"].values[0]
            rt=events[(events[ "trial" ] == t) & (events[ "sub" ] == s) & (events[ "proba" ] == p)]["trial_RT_colochoice"].values[0]

            roi = sacc[(sacc.trial == t) & (sacc.subject == s) & (sacc.probability == p)&(sacc.end<decision)&(sacc.start>decision-rt*100)]
            # print(roi.count) shj
            print(f"      ROI count: {len(roi)}")
            all_rois = pd.concat([all_rois, roi], ignore_index=True)
# %%

def create_displacement_heatmap(data, bins=30):
    """
    Create a 2D heatmap of x and y displacements
    
    Parameters:
    data (DataFrame): DataFrame containing x_displacement and y_displacement columns
    bins (int): Number of bins for the 2D histogram
    """
    
    # Create 2D histogram
    hist2d, xedges, yedges = np.histogram2d(
        data['x_displacement'],
        data['y_displacement'],
        bins=bins
    )
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    
    # Create heatmap using seaborn
    sns.heatmap(
        hist2d.T,  # Transpose to match conventional xy orientation
        cmap='YlOrRd',  # Yellow-Orange-Red colormap
        xticklabels=np.round(xedges[:-1], 1)[::bins//10],  # Show fewer tick labels
        yticklabels=np.round(yedges[:-1], 1)[::bins//10],
        cbar_kws={'label': 'Count'}
    )
    
    # Customize the plot
    plt.title('2D Heatmap of Displacements')
    plt.xlabel('X Displacement')
    plt.ylabel('Y Displacement')
    
    # Adjust layout
    plt.tight_layout()
    
    return plt

# Example usage:
# Assuming your data is in a DataFrame called 'df'
create_displacement_heatmap(all_rois)
plt.show()

# To also include a scatter plot overlay:
def create_displacement_plot_with_scatter(data, bins= 50 ):
    """
    Create a 2D heatmap with scatter plot overlay of x and y displacements
    
    Parameters:
    data (DataFrame): DataFrame containing x_displacement and y_displacement columns
    bins (int): Number of bins for the 2D histogram
    """
    
    # Create figure with two subplots
    fig,  ax2 = plt.subplots(1, 1, figsize=(15, 15))
    
    
    # Scatter plot
    ax2.scatter(
        data['x_displacement'],
        data['y_displacement'],
        alpha=0.5,
        s=20
    )
    ax2.set_title('Displacement Scatter Plot')
    ax2.set_xlabel('X Displacement')
    ax2.set_ylabel('Y Displacement')
    ax2.grid(True)
    
    # Adjust layout
    plt.tight_layout()
    
    return plt

create_displacement_plot_with_scatter(all_rois)
plt.show()
# %%
def plot_displacement_density(data):
    """
    Create a continuous density plot of x and y displacements using kernel density estimation
    
    Parameters:
    data (DataFrame): DataFrame containing x_displacement and y_displacement columns
    """
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Create kernel density plot using seaborn
    sns.kdeplot(
        data=data,
        x='x_displacement',
        y='y_displacement',
        cmap='viridis',  # Using viridis colormap for better density visualization
        fill=True,
        levels=20,  # Number of contour levels
        thresh=0,  # Show full range of density
    )
    
    # Add scatter points with low opacity to show actual data points
    plt.scatter(
        data['x_displacement'],
        data['y_displacement'],
        alpha=0.2,  # Low opacity to show density
        color='white',
        s=10  # Point size
    )
    
    # Customize the plot
    plt.title('Displacement Density Plot')
    plt.xlabel('X Displacement')
    plt.ylabel('Y Displacement')
    
    # Add colorbar
    plt.colorbar(label='Density')
    
    # Adjust layout
    plt.tight_layout()
    
    return plt

# Example usage:
plot_displacement_density(all_rois)
plt.show()
# %%
def plot_saccade_histogram(data):
    """
    Create a histogram showing the distribution of saccades across trials
    
    Parameters:
    data (DataFrame): DataFrame containing 'trial' column
    """
    # Count saccades per trial
    saccade_counts = data['trial'].value_counts().sort_index()
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Create histogram
    sns.histplot(
        data=saccade_counts,
        bins=50,
        color='skyblue',
        edgecolor='black'
    )
    
    # Customize the plot
    plt.title('Distribution of Saccades per Trial')
    plt.xlabel('Trial Number')
    plt.ylabel('Number of Saccades')
    
    # Add a grid for better readability
    plt.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    return plt

# Example usage:
plot_saccade_histogram(all_rois)
plt.show()

def plot_saccade_distribution(data):
    """
    Create a distribution plot showing saccades across trials with density estimate
    
    Parameters:
    data (DataFrame): DataFrame containing 'trial' column
    """
    # Count saccades per trial
    saccade_counts = data['trial'].value_counts().sort_index()
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Create distribution plot with rug plot
    sns.displot(
        data=saccade_counts,
        kind='kde',
        rug=True,
        fill=True,
        color='skyblue',
        rug_kws={'color': 'navy', 'alpha': 0.5}
    )
    
    # Customize the plot
    plt.title('Distribution of Saccades per Trial')
    plt.xlabel('Trial Number')
    plt.ylabel('Density')
    
    # Add a grid for better readability
    plt.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    return plt# Example usage:
plot_saccade_distribution(all_rois)
plt.show()


