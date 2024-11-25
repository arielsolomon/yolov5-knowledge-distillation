import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

def plot_datasets(df,x_data, column_names, figsize=(15,10)):

    fig, axes = plt.subplots(2,3, figsize=figsize)

    axes_flat = axes.flatten()

    for idx, (ax, col_name) in enumerate(zip(axes_flat, column_names)):

        ax.plot(x_data, df[col_name], '-o', linewidth=1, linestyle='--', color='red', markersize=4, markerfacecolor='green', markeredgecolor='green')
        ax.set_title(f'{col_name}')
        ax.set_xlabel('Num epochs')
        ax.set_ylabel(str(col_name))

    plt.tight_layout()
    plt.savefig('results_csv_plot.png')
    plt.show()

print(f'\n{os.getcwd()}\n')
csv_path = 'results_cont_353.csv'
df = pd.read_csv(csv_path)
features = list(df.keys())
print(f'\n{features}\n')
sublist = features[5:11]
desired_features = features[0],sublist[0], sublist[1],sublist[2],sublist[3],sublist[4],sublist[5]
x = df['epoch']
plot_datasets(df,x, desired_features[1::], figsize=(15,10))
