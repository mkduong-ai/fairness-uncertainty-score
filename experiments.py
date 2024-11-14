#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

# plot
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

import itertools


# ### Experiments on our Uncertainty Framework

# According to our definitions, we compare decision-makers. A decision-maker can exhibit discrimination through different `treatments` of groups. Each `group can also consist of different sample sizes.` We first want to create groups of different sizes with different treatments.

# In[2]:


import uncertainty
import utility


# In[3]:


def create_decision_maker(treatment_i, treatment_j, n_i, n_j):
    y_i = np.zeros(n_i, dtype=int)
    y_i[:int(treatment_i * n_i)] = 1
    y_j = np.zeros(n_j, dtype=int)
    y_j[:int(treatment_j * n_j)] = 1

    z = np.zeros(n_i + n_j, dtype=int)
    z[n_i:] = 1
    # Create Sample DataFrame for Decision-Maker 1
    data = {'y': np.concatenate((y_i, y_j)),
            'z': z}
    df = pd.DataFrame(data)
    
    # Defining Events (for statistical parity)
    i = (df['z'] == 0)
    j = (df['z'] == 1)
    E1 = (df['y'] == 1)

    return uncertainty.decision_maker(i=i, j=j, E1=E1)


# In[4]:


sample_sizes = [1, 5, 10, 50]
samples_ij = list(itertools.product(sample_sizes, sample_sizes))


# In[5]:


treatments = []
for sample_ij in samples_ij:
    # if sample_ij[0] <= 10 and sample_ij[1] <= 10:
    #     treatments.append((np.linspace(0, 1, sample_ij[0]+1), np.linspace(0, 1, sample_ij[1]+1)))
    # else:
    #     treatments.append((np.linspace(0, 1, 11), np.linspace(0, 1, 11)))
    treatments.append((np.linspace(0, 1, sample_ij[0]+1), np.linspace(0, 1, sample_ij[1]+1)))


# In[6]:


def create_decision_makers(treatments, samples_ij):
    results_df = []
    for treatment, sample_size in zip(treatments, samples_ij):
        treatment_is, treatment_js = treatment
        for treatment_i in treatment_is:
            for treatment_j in treatment_js:
                decision_maker = create_decision_maker(treatment_i, treatment_j, sample_size[0], sample_size[1])
                results_df.append({'n_i': sample_size[0],
                                   'k_i': int(treatment_i * sample_size[0]),
                                'n_j': sample_size[1],
                                'k_j': int(treatment_j * sample_size[1]),
                                'Treatment_i': treatment_i,
                                'Treatment_j': treatment_j,
                                'Decision_Maker': decision_maker,
                                'Utility': utility.utilityTopsis(decision_maker[0], decision_maker[1])})

    return pd.DataFrame(results_df)


# In[7]:


results_df = create_decision_makers(treatments, samples_ij)


# In[8]:


# Format DataFrame
def format_array(arr):
    return '[' + ', '.join([f'{x:.3f}' for x in arr]) + ']'

# Create decision makers and calculate utility
results_df = create_decision_makers(treatments, samples_ij)

# Format array elements
results_df['Decision_Maker'] = results_df['Decision_Maker'].apply(format_array)

# Round treatments
results_df[['Treatment_i', 'Treatment_j']] = results_df[['Treatment_i', 'Treatment_j']].round(3).astype(str)

# Display the DataFrame
print(results_df)


# In[9]:


results_df_sorted = results_df.sort_values('Utility', ascending=False).reset_index(drop=True)
results_df_sorted.index += 1
results_df_sorted.index.name = 'Rank'


# In[10]:


rename_dict = {'Treatment_i': r'$\hat{p}_i$',
              'Treatment_j': r'$\hat{p}_j$',
              'k_i': r'$k_i$',
              'k_j': r'$k_j$',
              'n_i': r'$n_i$',
              'n_j': r'$n_j$',
              'Decision_Maker': 'DM'}


# In[11]:


results_df_sorted = results_df_sorted.rename(columns=rename_dict)


# In[12]:


results_df_sorted['Utility']


# ## Save some results

# In[13]:


# Round utility column to 3 decimal places
results_df_latex = results_df_sorted.copy()
results_df_latex['Utility'] = results_df_latex['Utility'].round(3).astype(str)


# In[14]:
def ensure_folder_exists(folder_path):
    """
    Checks if a folder exists at the given path, and creates it if it does not.

    Parameters
    ----------
    folder_path: str
        The path of the folder to check or create.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    else:
        print(f"Folder '{folder_path}' already exists.")

def save_df(df, buf):
    ensure_folder_exists(buf.split('/')[0])
    df.style.to_latex(buf=buf,
                          position='tb',
                          multirow_align='t',
                          multicol_align='l',
                          hrules=True,
                          clines='skip-last;data',)


# In[15]:
save_df(results_df_latex[:4], buf='LaTeX/top4.txt')


# In[16]:


results_df_sorted[-4:]


# In[17]:


save_df(results_df_latex[-4:], buf='LaTeX/last4.txt')


# ## Plots

# ### Histogram

# In[18]:


plt.figure(figsize=(3.5, 1.25)) # Paper
sns.histplot(results_df, bins=21, x='Utility', stat='percent')
#plt.legend(fontsize=6)
plt.xticks(fontsize=6)
plt.yticks(fontsize=6)
plt.xlabel('Utility', fontsize=6)
plt.ylabel('Percent', fontsize=6)
plt.savefig('histogram.pdf', bbox_inches='tight', pad_inches=0)
plt.show()


# ### 2D Decision-Maker Plot

# In[19]:


def dmplot(points, utilityFunction=utility.utilityTopsis, print_distances=False, plot_lines=False):
    """
    points: ndarray (num_points, 2)
    """
    num_points = points.shape[0]
    corners = np.array([[0, 0],  # best case
                      [0, 1],
                      [1, 1],
                      [1, 0]])  # worst case
    # Calculate distances of the points to edges
    distances = cdist(points, corners)

    # Calculate preference for each point
    preferences = [utilityFunction(points[i, 0], points[i, 1]) for i in range(num_points)]
    
    # Determine the preferred point
    # preferred_index = np.argmax(preferences)
    preferred_indices = np.argwhere(preferences == np.max(preferences))

    # Plotting
    plt.figure(figsize=(3.5, 2.5)) # Paper
    # plt.figure(figsize=(8, 8))

    # Plot corners
    plt.scatter(corners[:, 0], corners[:, 1],
                color='black',
                label='Trivial Cases',
                s=10)

    # Plot points
    plt.scatter(points[:, 0], points[:, 1],
                color='red',
                label='Decision-Makers',
                s=5)

    # Plot Preference
    for i, preferred_index in enumerate(preferred_indices):
        label = 'Optimal DM' if i == 0 else None
        plt.scatter(points[preferred_index, 0], points[preferred_index, 1],
                color='red',
                s=20)
        plt.scatter(points[preferred_index, 0], points[preferred_index, 1],
                    marker='*',
                    color='blue',
                    s=20,
                    label=label)
        

    # Draw lines from points to edges
    if plot_lines:
        for i in range(len(corners)):
            for j in range(num_points):
                if i in [0, 3]:
                    plt.plot([points[j, 0], corners[i, 0]], [points[j, 1], corners[i, 1]],
                            color='gray',
                            linestyle='--',
                            linewidth=0.5)
                if print_distances:
                    plt.text((points[j, 0] + corners[i, 0]) / 2, (points[j, 1] + corners[i, 1]) / 2,
                            f'{distances[j,i]:.2f}',
                            color='black',
                            fontsize=8)

    # Set plot properties
    # plt.title('Decision-Makers and Corner Points')
    plt.xlabel('Disparity', fontsize=6)
    plt.ylabel('Uncertainty', fontsize=6)
    plt.legend(fontsize=6)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.grid(True)
    # Save the plot with specified options
    plt.savefig('decisionmakers.pdf', bbox_inches='tight', pad_inches=0)
    plt.show()


# In[324]:


results_df = create_decision_makers(treatments, samples_ij)


# In[358]:


points = np.vstack(results_df['Decision_Maker'].to_numpy())


# In[360]:


dmplot(points,
       print_distances=False,
       plot_lines=False)


