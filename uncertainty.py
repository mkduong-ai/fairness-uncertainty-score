import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt


def beta_posterior_data(n_success, n_total, return_prior=False):
    """
    Parameters
    ----------
    n_success: int
        Number of samples with a binary outcomes (e.g., success or failure)
    n_total: int
        Number of total samples

    Returns
    -------
    posterior_density: np.array
    """
    # parameters for the prior distribution
    alpha_prior = 1
    beta_prior = 1
    
    # parameters for the posterior distribution
    alpha_posterior = alpha_prior + n_success
    beta_posterior = beta_prior + n_total - n_success
    
    # create a range of possible p values
    p_values = np.linspace(0, 1, 1000)

    # compute the density of the prior at each p value
    # The beta distribution is often used to model probabilities or proportions,
    # especially when dealing with binary outcomes
    prior_density = stats.beta.pdf(p_values, alpha_prior, beta_prior)

    # compute the density of the posterior at each p value
    posterior_density = stats.beta.pdf(p_values, alpha_posterior, beta_posterior)
    
    if return_prior:
        return posterior_density, prior_density


def beta_posterior(n_success, n_total, return_prior=False):
    """
    Parameters
    ----------
    n_success: int
        Number of samples with a binary outcomes (e.g., success or failure)
    n_total: int
        Number of total samples

    Returns
    -------
    posterior: scipy.stats.beta
    """
    if n_success > n_total:
        raise ValueError("n_success cannot be greater than n_total.")
        
    # set uninformative parameters for the prior distribution
    alpha_prior = 1
    beta_prior = 1
    
    # parameters for the posterior distribution
    alpha_posterior = alpha_prior + n_success
    beta_posterior = beta_prior + n_total - n_success

    # compute the the prior
    # The beta distribution is often used to model probabilities or proportions,
    # especially when dealing with binary outcomes
    prior = stats.beta(alpha_prior, beta_prior)

    # compute the posterior
    posterior = stats.beta(alpha_posterior, beta_posterior)
    
    if return_prior:
        return posterior, prior
    return posterior


def plot_pdf(prior_density, posterior_density):
    """
    Parameters
    ----------
    prior_density: np.array
    posterior_density: np.array

    Returns
    -------
    posterior_density: np.array
    """
    # create a range of possible p values
    p_values = np.linspace(0, 1, 1000)
    
    plt.figure(figsize=(8, 6))
    plt.plot(p_values, prior_density, label='Prior', color='blue')
    plt.plot(p_values, posterior_density, label='Posterior', color='red')
    plt.xlabel('p', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.title('Prior and Posterior Distributions', fontsize=16)
    plt.legend()
    plt.show()


def plot_pdfs(p, q):
    """
    Parameters
    ----------
    p: Distribution class from scipy.stats
    q: Distribution class from scipy.stats
    
    Returns
    -------
    None
    """
    # create a range
    xn = np.linspace(0, 1, 1000)
    
    plt.figure(figsize=(8, 6))
    plt.plot(xn, p.pdf(xn), label='PDF P', color='blue')
    plt.plot(xn, q.pdf(xn), label='PDF Q', color='red')
    plt.xlabel('Values', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.title('Comparing PDFs', fontsize=16)
    plt.legend()
    plt.show()


def treatment_frequentist(i, E1, E2=True):
    """
    Parameters
    ----------
    i: pandas.Series
        boolean mask
    E1: pandas.Series
        boolean mask
    E2: pandas.Series
        boolean mask
    
    Returns
    -------
    float
        frequentist treatment/success probability"""
    denominator = (i & E2).sum()
    nominator = (E1 & i & E2).sum()
    return nominator/denominator


def treatment_bayesian(i, E1, E2=True):
    """
    Parameters
    ----------
    i: pandas.Series
        boolean mask
    E1: pandas.Series
        boolean mask
    E2: pandas.Series
        boolean mask
    
    Returns
    -------
    float
        Bayesian treatment/success probability/posterior mean"""
    alpha_prior = 1
    beta_prior = 1
    n_success = (E1 & i & E2).sum()
    alpha_post = alpha_prior + n_success
    beta_post = beta_prior + (i & E2).sum() - n_success
    return beta_expected_value(alpha_post, beta_post)


def disparity(i, j, E1, E2=True):
    """
    Parameters
    ----------
    i: pandas.Series
        boolean mask
    j: pandas.Series
        boolean mask
    E1: pandas.Series
        boolean mask
    E2: pandas.Series
        boolean mask
        
    Returns
    -------
    float
        disparity between two groups i and j"""
    return np.abs(treatment_frequentist(i=i, E1=E1, E2=E2) - treatment_frequentist(i=j, E1=E1, E2=E2))
    
    
def bayesian_disparity(i, j, E1, E2=True):
    """
    Parameters
    ----------
    i: pandas.Series
        boolean mask
    j: pandas.Series
        boolean mask
    E1: pandas.Series
        boolean mask
    E2: pandas.Series
        boolean mask
        
    Returns
    -------
    float
        Bayesian disparity between two groups i and j"""
    treatment_bay_i = treatment_bayesian(i=i, E1=E1, E2=E2)
    treatment_bay_j = treatment_bayesian(i=j, E1=E1, E2=E2)
    return np.abs(treatment_bay_i - treatment_bay_j)
    
    
def beta_expected_value(alpha, beta):
    """
    Parameters
    ----------
    alpha: float
        alpha parameter of the beta distribution
    beta: float
        beta parameter of the beta distribution
    
    Returns
    -------
    float
        expected value of the beta distribution"""
    return alpha/(alpha+beta)


def beta_variance(alpha, beta):
    """
    Parameters
    ----------
    alpha: float
        alpha parameter of the beta distribution
    beta: float
        beta parameter of the beta distribution
    
    Returns
    -------
    float
        variance of the beta distribution"""
    return (alpha*beta)/((alpha+beta)**2*(alpha+beta+1))


def beta_normalized_variance(alpha, beta):
    """
    Parameters
    ----------
    alpha: float
        alpha parameter of the beta distribution
    beta: float
        beta parameter of the beta distribution
    
    Returns
    -------
    float
        normalized variance [0, 1] of the beta distribution
        (variance divided by the max variance)"""
    return beta_variance(alpha, beta)/beta_variance(1, 2)
    

def uncertainty(i, j, E1, E2):
    """
    Parameters
    ----------
    i: pandas.Series
        boolean mask
    j: pandas.Series
        boolean mask
    E1: pandas.Series
        boolean mask
    E2: pandas.Series
        boolean mask
    
    Returns
    -------
    float
        uncertainty of the disparity between two groups i and j"""
    alpha_prior = 1
    beta_prior = 1
    # group i
    n_success_i = (E1 & i & E2).sum()
    alpha_post_i = alpha_prior + n_success_i
    beta_post_i = beta_prior + (i & E2).sum() - n_success_i
    norm_var_i = beta_normalized_variance(alpha_post_i, beta_post_i)
    # group j
    n_success_j = (E1 & j & E2).sum()
    alpha_post_j = alpha_prior + n_success_j
    beta_post_j = beta_prior + (j & E2).sum() - n_success_j
    norm_var_j = beta_normalized_variance(alpha_post_j, beta_post_j)
    
    return (norm_var_i + norm_var_j)/2


def decision_maker(i, j, E1, E2=True):
    """
    Creates a decision-maker based on treatments of group i and j

    Parameters
    ----------
    i: pandas.Series
        boolean mask
    j: pandas.Series
        boolean mask
    E1: pandas.Series
        boolean mask
    E2: pandas.Series
        boolean mask
    
    Returns
    -------
    np.array (1, 2)
        disparity and uncertainty between two groups i and j"""
    return np.array([disparity(i=i, j=j, E1=E1, E2=E2), uncertainty(i=i, j=j, E1=E1, E2=E2)])