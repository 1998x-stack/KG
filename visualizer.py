import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd  # Needed for rolling mean
import os
os.makedirs('figures', exist_ok=True)

def visualize_data_list(data_list, title, additional_info='', smooth_rate=1):
    # Set the style of seaborn for more sophisticated visuals
    sns.set(style='whitegrid')

    # Create a plot with larger size for better visibility
    plt.figure(figsize=(10, 6))

    # Checking if smoothing is required
    if smooth_rate > 1:
        # Converting list to DataFrame for rolling operation
        data_list_df = pd.DataFrame(data_list, columns=[title])
        # Calculating rolling mean
        smooth_data = data_list_df.rolling(window=smooth_rate, center=True).mean()
    else:
        smooth_data = pd.DataFrame(data_list, columns=[title])

    # Plotting the datalist with or without smoothing
    sns.lineplot(data=smooth_data, color='blue', linewidth=2.5)

    # Setting labels and title with enhanced font properties
    plt.xlabel('Episodes', fontsize=14, labelpad=15)
    plt.ylabel(title, fontsize=14, labelpad=15)
    plt.title(f'{title} vs Episodes', fontsize=16, pad=20)

    # Enhancing tick parameters for better readability
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Saving the plot with high resolution
    plt.savefig(f'figures/{title}_{additional_info}_sm{smooth_rate}.png', dpi=300)

    # Closing the plot to free up memory
    plt.close()