import pandas as pd
import uncertainty
from utility import utilityTopsis

# Create Sample DataFrame for Decision-Maker 1 (Nearly everybody has the same outcome except of 1 candidate)
data = {'y': [1, 1, 1, 1, 1, 0], # y is label
        'z': [0, 0, 0, 1, 1, 1]} # z is protected attribute
df = pd.DataFrame(data)

# Defining Events (for statistical parity)
i = (df['z'] == 0) # group i
j = (df['z'] == 1) # group j
E1 = (df['y'] == 1) # outcome event

# Create a decision-maker (Definition 6 in the paper)
dA = uncertainty.decision_maker(i=i, j=j, E1=E1)

# Create Sample DataFrame for Decision-Maker 2 (Nearly everybody has the same outcome except of 1 candidate)
data = {'y': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        'z': [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]}
df = pd.DataFrame(data)

# Defining Events (for statistical parity)
i = (df['z'] == 0)
j = (df['z'] == 1)
E1 = (df['y'] == 1)

# Create a decision-maker (Definition 6 in the paper)
dB = uncertainty.decision_maker(i=i, j=j, E1=E1)

print('Decision-Maker A is represented with: ', dA)
print('Decision-Maker B is represented with: ', dB)

# Use TOPSIS utility (Example 4 in the paper)
utilitydA = utilityTopsis(dA[0], dA[1])
utilitydB = utilityTopsis(dB[0], dB[1])

print('The utility value of DM A is: ', utilitydA)
print('The utility value of DM B is: ', utilitydB)
print('Higher utility is better, therefore we prefer DB.')
print('This makes sense as we have more data about decision-maker B.')
