import pandas as pd
import uncertainty
from utility import utilityTopsis

# Create Sample DataFrame for Decision-Maker 1
data = {'y': [0, 0, 0, 1, 1, 1],
        'z': [0, 0, 0, 1, 1, 1]}
df = pd.DataFrame(data)

# Defining Events (for statistical parity)
i = (df['z'] == 0)
j = (df['z'] == 1)
E1 = (df['y'] == 1)

dA = uncertainty.decision_maker(i=i, j=j, E1=E1)

# Create Sample DataFrame for Decision-Maker 2
data = {'y': [0, 1],
        'z': [0, 1]}
df = pd.DataFrame(data)

# Defining Events (for statistical parity)
i = (df['z'] == 0)
j = (df['z'] == 1)
E1 = (df['y'] == 1)

dB = uncertainty.decision_maker(i=i, j=j, E1=E1)

print('Decision-Maker A is represented with: ', dA)
print('Decision-Maker B is represented with: ', dB)

utilitydA = utilityTopsis(dA[0], dA[1])
utilitydB = utilityTopsis(dB[0], dB[1])

print('The utility value of DM A is: ', utilitydA)
print('The utility value of DM B is: ', utilitydB)
print('Higher utility is better, therefore we prefer DB.')
