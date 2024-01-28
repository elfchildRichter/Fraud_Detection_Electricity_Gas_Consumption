## Module for plots in the EDA part ##

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# plt.style.use('bmh')
# plt.rcParams['font.size'] = 10


def feature_target_distribution(df, feature, target):
    
    grouped = df.groupby([feature, target]).size().unstack(fill_value=0)

    grouped['ratio_0'] = grouped[0] / (grouped[1] + grouped[0]) 
    grouped['ratio_1'] = grouped[1] / (grouped[1] + grouped[0])
    
    print('The table means the number and the ratio of non-fraudulent and fraudulent cases for each feature value.')
    print(grouped)
    print('')
    
    # for i in range(len(grouped)):
    #     print(f'For {feature} = {grouped.index[i]}, the ratio of {target} = 1 is {grouped["ratio_1"].iloc[i]:.2%}')
    #     print(f'For {feature} = {grouped.index[i]}, the ratio of {target} = 0 is {grouped["ratio_0"].iloc[i]:.2%}')
    #     print('')
    
    length = len(grouped)
    ax = grouped[[0, 1]].plot(kind='bar', stacked=True, figsize=(length, 6)) # , figsize=(10, 6)

    for i in range(len(grouped)):
        
        height_1 = grouped.iloc[i, 1]  
        height_0 = grouped.iloc[i, 0]  
        label_position_1 = height_1 / 2 + height_0  
        ax.annotate(f'{grouped["ratio_1"].iloc[i]:.2%}', (i-0.2, label_position_1), va='center')

        height_0 = grouped.iloc[i, 0] 
        label_position_0 = height_0 / 2 
        ax.annotate(f'{grouped["ratio_0"].iloc[i]:.2%}', (i-0.2, label_position_0), va='center')
        
        
    plt.style.use('bmh')
    plt.rcParams['font.size'] = 10
    plt.title(f'Percentage of Fraudulent and Non-fraudulent cases vs {feature.capitalize()}')
    plt.xticks(rotation=0)
    plt.xlabel(f'{feature.capitalize()}')
    plt.yticks()
    plt.ylabel('Number of cases')
    plt.grid()
    plt.legend(title=f'{target.capitalize()}', loc='upper right')
    plt.show()
    
    sorted_group = grouped.sort_values(by='ratio_1', ascending=False)
    
    if len(sorted_group) >=1: 
        index_name_0 = sorted_group.iloc[0].name
        ratio_1_value_0 = sorted_group.iloc[0]['ratio_1']
        print(f'The highest rate of fraudulent is {feature} = {index_name_0} with rate {ratio_1_value_0:.2%}')
    
    if len(sorted_group) >=2:
        index_name_1 = sorted_group.iloc[1].name
        ratio_1_value_1 = sorted_group.iloc[1]['ratio_1']
        print(f'The second highest rate of fraudulent is tarif_type = {index_name_1} with rate {ratio_1_value_1:.2%}')

    if len(sorted_group) >=3:
        index_name_2 = sorted_group.iloc[2].name
        ratio_1_value_2 = sorted_group.iloc[2]['ratio_1']
        print(f'The third highest rate of fraudulent is tarif_type = {index_name_2} with rate {ratio_1_value_2:.2%}')
    
    #return sorted_group


def fraud_ratio(df):
    
    # show the counts of fraudulent and non-fraudulent clients
    print(df.target.value_counts())
    print('')

    # calculate the percentage of the fraudulent clients
    fraud_rate = df.target.mean() * 100
    non_fraud_rate = 100 - fraud_rate

    print(f'Target has {fraud_rate:.2f} % of 1, which is detected as fraud.')
    print('This is a highly imbalanced dataset.')

    # plot the percentage of the fraudulent and non-fraudulent clients
    plt.figure(figsize=(6, 6))
    ax = sns.barplot(x=['non-fraud', 'fraud'], y=[non_fraud_rate, fraud_rate])

    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}%', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center', 
                    xytext = (0, 5), 
                    textcoords = 'offset points')

    plt.style.use('bmh')
    plt.rcParams['font.size'] = 10
    plt.title('Percentage of Fraudulent vs Non-Fraudulent Clients')
    plt.xlabel('Target')
    plt.ylabel('Percentage (%)')
    plt.show()