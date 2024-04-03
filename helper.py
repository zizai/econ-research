# file for helper functions for econ

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn import linear_model
from scipy.stats import norm
from copy import deepcopy

def summarize(col, weight=None):
    '''summarizes a column of data, including mean, median, standard deviation, variance, 25th, 50th, and 75th percentiles, skewness, kurtosis, min, and max. If weight is provided, it will also calculate the weighted mean, weighted standard deviation, and weighted variance.
    '''
    # convert to numpy array
    col = np.array(col).astype(float)
    if weight is not None:
        weight = np.array(weight).astype(float)

    # drop all NaNs
    if weight is None:
        col = col[~np.isnan(col)]
    else:
        col_new = col[~np.isnan(col) & ~np.isnan(weight)]
        weight_new = weight[~np.isnan(col) & ~np.isnan(weight)]
        # make sure col and weight are the same length
        assert len(col_new) == len(weight_new)
        col = col_new
        weight = weight_new

    print('Num observations: ', len(col))
    if type(col) != pd.Series:
        col = pd.Series(col)
    if weight is not None:
        print("Weighted Mean: ", np.average(col, weights=weight))
    else:
        print("Mean: ", col.mean())
        print("Median: ", col.median())
    if weight is not None:
        print("Weighted Standard Deviation: ", np.sqrt(np.average((col - np.average(col, weights=weight))**2, weights=weight)))
        print("Weighted Variance: ", np.average((col - np.average(col, weights=weight))**2, weights=weight))
    else:
        print("Standard Deviation: ", col.std())
        print("Variance: ", col.var())

    # else:
    print("1st percentile: ", col.quantile(0.01))
    print("5th percentile: ", col.quantile(0.05))
    print("10th percentile: ", col.quantile(0.10))
    print("25th percentile: ", col.quantile(0.25))
    print("50th percentile: ", col.quantile(0.50))
    print("75th percentile: ", col.quantile(0.75))
    print("90th percentile: ", col.quantile(0.90))
    print("95th percentile: ", col.quantile(0.95))
    print("99th percentile: ", col.quantile(0.99))

    # get skewness
    print("Skewness: ", col.skew())

    # get kurtosis
    print("Kurtosis: ", col.kurtosis())

    # get min and max
    print("Min: ", col.min())
    print("Max: ", col.max())

def prepare_data(dataframe, condition, X_cols, Y_col, W_col_dep, W_col_indep, return_dataframe=False):
    '''Creates X, y, and w for a weighted least squares regression.

    Params:
        dataframe: pd.DataFrame
        condition: np.array
        X_cols: list
        Y_col: str
        W_col_dep: str for weights for the dependent variable
        W_col_indep: str for weights for the independent variables
        return_dataframe: bool, whether to return the dataframe with the condition applied

    '''

    y = deepcopy(dataframe[Y_col])
    X = deepcopy(dataframe[X_cols])
    w_dep = deepcopy(dataframe[W_col_dep])
    w_indep = deepcopy(dataframe[W_col_indep])

    y = y.astype(float)
    X = X.astype(float)
    w_dep = w_dep.astype(float)
    w_indep = w_indep.astype(float)

    # make sure no NaN values
    if condition is not None:
        y = y[condition]
        X = X[condition]
        w_dep = w_dep[condition]
        w_indep = w_indep[condition]

    y = y[(~np.isnan(w_dep)) & (~np.isnan(w_indep))& (~np.isnan(y)) & (~np.isnan(X).any(axis=1))]
    X = X[(~np.isnan(w_dep)) & (~np.isnan(w_indep)) & (~np.isnan(y)) & (~np.isnan(X).any(axis=1))]
    w_dep = w_dep[(~np.isnan(w_dep)) & (~np.isnan(w_indep)) & (~np.isnan(y)) & (~np.isnan(X).any(axis=1))]
    w_indep = w_indep[(~np.isnan(w_dep)) & (~np.isnan(w_indep)) & (~np.isnan(y)) & (~np.isnan(X).any(axis=1))]

    if return_dataframe:
        if condition is not None:
            return X, y, w_dep, w_indep, dataframe[condition]
        else:
            return X, y, w_dep, w_indep, dataframe
    else:
        return X, y, w_dep, w_indep

def run_WLS(X, y, w, print_summary=True):
    '''runs a weighted least squares regression'''
    # Fitting the model
    X = sm.add_constant(X)

    model = sm.WLS(y, X, weights=w).fit()

    # Using robust standard errors (e.g., HC1)
    robust_model = model.get_robustcov_results(cov_type='HC1')

    # Viewing the summary with robust standard errors
    if print_summary:
        print(robust_model.summary())
    return robust_model

def oaxaca_blinder(X1, X2, y1, y2, w1_dep, w1_indep, w2_dep, w2_indep, comparison=0):
    '''performs an Oaxaca-Blinder decomposition of the mean differences in characteristics and the mean differences in coefficients between two groups.

    Params:
        X1: independent variables for group 1
        X2: independent variables for group 2
        y1: dependent variable for group 1
        y2: dependent variable for group 2
        w1_dep: weights for dependent vars group 1
        w1_indep: weights for independent vars group 2
        w2_dep: weights for dependent vars group 2
        w2_indep: weights for independent vars group 2
        comparison: int, 0 for group 1 as the reference group, 1 for group 2 as the reference group, and 2 for the average of the two groups as the reference group
    
    
    '''

    # weighted mean differences in characteristics
    # add constant
    X1 = sm.add_constant(X1)
    X2 = sm.add_constant(X2)
    X1_mean = np.average(X1, weights=w1_indep, axis=0)
    X2_mean = np.average(X2, weights=w2_indep, axis=0)
    mean_diff = X1_mean - X2_mean
    

    # run WLS
    model1 = run_WLS(X1, y1, w1_dep, print_summary=False)
    model2 = run_WLS(X2, y2, w2_dep, print_summary=False)

    # Coefficients
    coeff1 = model1.params
    coeff2 = model2.params

    print('Coeff1: ', coeff1)
    print('Coeff2: ', coeff2)

    # Decomposition
    if comparison == 0:
        explained = mean_diff.dot(coeff2)
        unexplained = (X1_mean.dot(coeff1 - coeff2))
        print(coeff1 - coeff2)
        print('-----')
        print(X1_mean)

    elif comparison == 1:
        explained = mean_diff.dot(coeff1)
        unexplained = (X2_mean.dot(coeff1 - coeff2))

    elif comparison == 2:
        explained = mean_diff.dot((coeff1 + coeff2) / 2)
        unexplained = ((X1_mean + X2_mean).dot(coeff1 - coeff2)/2)

    # Print results
    print(f"Explained component: {explained}")
    print(f"Unexplained component: {unexplained}")

    # total gap
    print(f"Total gap: {explained + unexplained}")
        

def compute_marginal_effects(model, X, eps=1e-6):
    '''computes marginal effects manually for a model'''
    # Calculate predicted probabilities
    pred_probs = model.predict(X)

    avg_marginal_effects = {}

    # Iterate over each column, calculate marginal effects, and take the average
    for column in X.columns:
        X_plus_delta = X.copy()
        X_plus_delta[column] = X_plus_delta[column] + eps
        pred_probs_delta = model.predict(X_plus_delta)
        marginal_effects = (pred_probs_delta - pred_probs) / eps
        avg_marginal_effects[column] = np.mean(marginal_effects)

    return avg_marginal_effects

def run_logit(y, X, w, print_summary=True, return_marginal_effects=True):
    '''Assumes the error terms follow a logistic distribution. Uses the logistic function to model the probability that the dependent variable equals 1.

    Params:
        y: dependent variable
        X: independent variables
        w: weights
        print_summary: whether to print the summary of the model
        return_marginal_effects: whether to return the marginal effects of the model

    Returns:
        logit_result_glm.predict(X): predicted probabilities
        logit_marginal_effects: marginal effects of the model (if return_marginal_effects is True)
    
    '''

    logit_link = sm.genmod.families.links.Logit()
    logit_model_glm = sm.GLM(y, X, family=sm.families.Binomial(link=logit_link), var_weights=w)
    logit_result_glm = logit_model_glm.fit()

    # Compute marginal effects for the weighted logit model
    logit_marginal_effects = compute_marginal_effects(logit_result_glm, X)
    
    if print_summary:
        print("Weighted Logit Model Summary (GLM):")
        print(logit_result_glm.summary())
        
        print("Marginal Effects - Weighted Logit Model:")
        print(logit_marginal_effects)

    if return_marginal_effects:
        return logit_result_glm.predict(X), logit_marginal_effects
    else:
        return logit_result_glm.predict(X)
    
def run_probit(y, X, w, print_summary=True, return_marginal_effects=True):
    '''Assumes the error terms are normally distributed. Uses the cumulative distribution function (CDF) of the standard normal distribution (a probit function) to model the probability.

    Params:
        y: dependent variable
        X: independent variables
        w: weights
        print_summary: whether to print the summary of the model
        return_marginal_effects: whether to return the marginal effects of the model

    Returns:
        probit_result_glm.predict(X): predicted probabilities
        probit_marginal_effects: marginal effects of the model (if return_marginal_effects is True)
        
    '''
   # Fit a weighted probit model using GLM
    probit_link = sm.genmod.families.links.Probit()
    probit_model_glm = sm.GLM(y, X, family=sm.families.Binomial(link=probit_link), var_weights=w)
    probit_result_glm = probit_model_glm.fit()

    # Compute marginal effects for the weighted probit model
    probit_marginal_effects = compute_marginal_effects(probit_result_glm, X)

    if print_summary:
        print("Weighted Probit Model Summary (GLM):")
        print(probit_result_glm.summary())

        print("Marginal Effects - Weighted Probit Model:")
        print(probit_marginal_effects)

    if return_marginal_effects:
        return probit_result_glm.predict(X), probit_marginal_effects
    else:
        return probit_result_glm.predict(X)

def run_DFL(X, y, w, print_summary=True, return_marginal_effects=False):
    '''runs dinardo fortin lemieux decomposition to make group

    Params:
        X: independent variables
        y: dependent variable
        w: weights
        print_summary: whether to print the summary of the model
        return_marginal_effects: whether to return the marginal effects of the model

    Returns:
        DFL_weight: reweighting factor
        Psi_x: reweighting factor
    
    '''
    # run probit
    if return_marginal_effects:
        prob_y1_given_X, probit_marginal_effects = run_probit(y, X, w, print_summary=print_summary, return_marginal_effects=return_marginal_effects)
    else:
        prob_y1_given_X = run_logit(y, X, w, print_summary=print_summary, return_marginal_effects=return_marginal_effects)

    prob_y2_given_X = 1-prob_y1_given_X
    
    # get probability of y1 and y2
    prob_y1 = np.average(y, weights=w)
    prob_y2 = 1-prob_y1
    
    # get psi_x reweighting factor
    Psi_x = np.where(y==0, prob_y1_given_X / prob_y2_given_X * prob_y2 / prob_y1, 1)

    DFL_weight = w * Psi_x 
    return DFL_weight, Psi_x


if __name__ == "__main__":
    pass