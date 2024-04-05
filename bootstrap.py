import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

np.random.seed(40)

def bootstrap_mean_iterations(filename, n, step):
    
    data = pd.read_csv(filename)
    num_rows = len(data)

    means_per_iteration = []
    for iteration in range(step):

        sampled_rows = np.random.choice(num_rows, size=n, replace=True)
        
        iteration_means = []
        for column in data.columns:

            sampled_data = data.iloc[sampled_rows][column].dropna()
            
            mean_value = np.mean(sampled_data)
            iteration_means.append(mean_value)
        
        means_per_iteration.append(iteration_means[:25])

    return means_per_iteration

def calculate_mean_and_std(data):
    array_data = np.array(data)

    mean_values = np.mean(array_data, axis=1)
    std_values = np.std(array_data, axis=1)

    return mean_values, std_values

def plot_mean_and_std(means, stds, name, filter_num = 25):
    fig = plt.figure(figsize=(15, 4), dpi=360)
    x_values = np.arange(filter_num)
    # labels = ['Levin et al., 2021', 'POT', 'conv5']
    labels = ['70%', '80%', '85%', '90%', '95%']
    ylabels = ['incorrect class confidence change', 'correct class confidence change', 'percentage of corrected samples']

    for i in range(3):
        ax = fig.add_subplot((131 + i))
        for j in range(5):
            ax.plot(x_values, means[i][j][:filter_num] * 100, label=labels[j])
            ax.fill_between(x_values, (np.array(means[i][j]) - np.array(stds[i][j]))[:filter_num] * 100, (np.array(means[i][j]) + np.array(stds[i][j]))[:filter_num] * 100, alpha=0.2)
            ax.set_ylabel(ylabels[i], fontsize=12)
            ax.set_xlabel('#finetuned filters', fontsize=12)
            ax.legend()
        
    name = os.path.join(name) + '.png'
    fig.savefig(name)


path = 'Figs/resnet50/Imagenet'
sample_num = 50000
step_num = 100
filter_num = 25

ip = bootstrap_mean_iterations(os.path.join(path, 'POT', 'df_i'), sample_num, step_num)
cp = bootstrap_mean_iterations(os.path.join(path, 'POT', 'df_c'), sample_num, step_num)
fp = bootstrap_mean_iterations(os.path.join(path, 'POT', 'df_f'), sample_num, step_num)
i7 = bootstrap_mean_iterations(os.path.join(path, '0.7', 'POT', 'df_i'), sample_num, step_num)
c7 = bootstrap_mean_iterations(os.path.join(path, '0.7', 'POT', 'df_c'), sample_num, step_num)
f7 = bootstrap_mean_iterations(os.path.join(path, '0.7', 'POT', 'df_f'), sample_num, step_num)
i8 = bootstrap_mean_iterations(os.path.join(path, '0.8', 'POT', 'df_i'), sample_num, step_num)
c8 = bootstrap_mean_iterations(os.path.join(path, '0.8', 'POT', 'df_c'), sample_num, step_num)
f8 = bootstrap_mean_iterations(os.path.join(path, '0.8', 'POT', 'df_f'), sample_num, step_num)
i85 = bootstrap_mean_iterations(os.path.join(path, '0.85', 'POT', 'df_i'), sample_num, step_num)
c85 = bootstrap_mean_iterations(os.path.join(path, '0.85', 'POT', 'df_c'), sample_num, step_num)
f85 = bootstrap_mean_iterations(os.path.join(path, '0.85', 'POT', 'df_f'), sample_num, step_num)
i95 = bootstrap_mean_iterations(os.path.join(path, '0.95', 'POT', 'df_i'), sample_num, step_num)
c95 = bootstrap_mean_iterations(os.path.join(path, '0.95', 'POT', 'df_c'), sample_num, step_num)
f95 = bootstrap_mean_iterations(os.path.join(path, '0.95', 'POT', 'df_f'), sample_num, step_num)

print(len(ib))
print(len(ip))
print(len(ic))
print(len(cb))
print(len(cp))
print(len(cc))
print(len(fb))
print(len(fp))
print(len(fc))

incorrect = [i7, i8, i85, ip, i95]
correct = [c7, c8, c85, cp, c95]
flipped = [f7, f8, f85, fp, f95]

mi, si = calculate_mean_and_std(incorrect)
mc, sc = calculate_mean_and_std(correct)
mf, sf = calculate_mean_and_std(flipped)

means = [mi, mc, mf]
stds = [si, sc, sf]

path = os.path.join('Figs/resnet50/Imagenet', 'result')
plot_mean_and_std(means, stds, path, filter_num)



