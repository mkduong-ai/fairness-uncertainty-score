# Fairness Is Not All You Need!

Python implementation of "(Un)certainty of (Un)fairness: Preference-Based Selection of Certainly Fair Decision-Makers" by Manh Khoi Duong and Stefan Conrad. In ECAI 2024: 27th European Conference on Artificial Intelligence, volume 392 of Frontiers in Artificial Intelligence and Applications. IOS Press, 2024.

This repository provides the code for evaluating decision-makers (humans or machine learning models) about their fairness. Our scores are based on group fairness metrics but incorporate the uncertainty that occurs when measuring for fairness.

## Example
```python
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
```

## File Overview

- **Demo.py**: A simple walkthrough of how to use our proposed score.
- **Figure1.py**: Contains the code to replicate the experiment in Figure 1 of the paper.
- **uncertainty.py**: Contains all the functions necessary for performing Bayesian analysis, calculating treatment effects and disparities, plotting prior and posterior distributions, and estimating uncertainty.
- **experiments.py**: Contains the code of the synthetic experiments as found in Table 3 in the paper.
- **utility.py**: A set of functions, of which some are utility functions and some are not.

## Features

- **Bayesian Inference**: 
  - Functions for computing posterior distributions using a beta prior, useful for binary outcome data.
  - Includes functions to visualize prior and posterior distributions.
  
- **Frequentist and Bayesian Treatment Effects**: 
  - Functions to calculate treatment/success probabilities using both frequentist and Bayesian approaches.
  
- **Disparity Calculation**:
  - Measures disparity between two groups based on their success probabilities, using both frequentist and Bayesian approaches.
  
- **Uncertainty Estimation**:
  - Functions to calculate the uncertainty of disparities, including normalized variance of the beta distribution as a metric of uncertainty.

- **Decision-Maker**:
  - A `decision_maker` function that returns both disparity and uncertainty, representing a decision-maker as in the paper.

## Setup

This package relies on:
- `numpy`
- `scipy`
- `pandas`
- `matplotlib`

These libraries are required to perform calculations and generate plots. To install the dependencies, perform

```bash
pip install -r requirements.txt
```
## Citation

When using our score in your work, cite our paper:

```BibTeX
@inproceedings{duong2024uncertain,
  title        = {{(Un)certainty of (Un)fairness: Preference-Based Selection of Certainly Fair Decision-Makers}},
  author       = {Manh Khoi Duong and Stefan Conrad},
  booktitle    = {ECAI 2024 - 27th European Conference on Artificial Intelligence},
  series       = {Frontiers in Artificial Intelligence and Applications},
  volume       = {392},
  pages        = {882--889},
  year         = {2024},
  publisher    = {IOS Press},
}
```
