# lab.py


from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def after_purchase():
    return ['NMAR', 'MD', 'MAR', 'NMAR', 'MAR']


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def multiple_choice():
    return ['MAR', 'NMAR', 'MAR', 'NMAR', 'MCAR']


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------



def first_round():
    payments_fp = Path(__file__).resolve().parent / 'data' / 'payment.csv'
    payments = pd.read_csv(payments_fp)
    dob = pd.to_datetime(payments['date_of_birth'], format='%d-%b-%Y', errors='coerce')
    ages = 2024 - dob.dt.year
    indicators = payments['credit_card_number'].isna()
    data = pd.DataFrame({'age': ages, 'missing': indicators}).dropna(subset=['age'])
    groups = data['missing'].to_numpy()
    values = data['age'].to_numpy()
    observed = abs(values[groups].mean() - values[~groups].mean())
    rng = np.random.default_rng(0)
    stats_samples = []
    for _ in range(1000):
        shuffled = rng.permutation(values)
        diff = abs(shuffled[groups].mean() - shuffled[~groups].mean())
        stats_samples.append(diff)
    p_value = float(np.mean(np.array(stats_samples) >= observed))
    decision = 'R' if p_value < 0.05 else 'NR'
    return [p_value, decision]


def second_round():
    payments_fp = Path(__file__).resolve().parent / 'data' / 'payment.csv'
    payments = pd.read_csv(payments_fp)
    dob = pd.to_datetime(payments['date_of_birth'], format='%d-%b-%Y', errors='coerce')
    ages = 2024 - dob.dt.year
    indicators = payments['credit_card_number'].isna()
    data = pd.DataFrame({'age': ages, 'missing': indicators}).dropna(subset=['age'])
    group_true = data.loc[data['missing'], 'age']
    group_false = data.loc[~data['missing'], 'age']
    observed_stat = float(stats.ks_2samp(group_true, group_false, alternative='two-sided', method='auto').statistic)
    groups = data['missing'].to_numpy()
    values = data['age'].to_numpy()
    rng = np.random.default_rng(0)
    stats_samples = []
    for _ in range(1000):
        shuffled = rng.permutation(values)
        sample_true = shuffled[groups]
        sample_false = shuffled[~groups]
        stat = stats.ks_2samp(sample_true, sample_false, alternative='two-sided', method='auto').statistic
        stats_samples.append(stat)
    p_value = float(np.mean(np.array(stats_samples) >= observed_stat))
    decision = 'R' if p_value < 0.05 else 'NR'
    conclusion = 'D' if decision == 'R' else 'ND'
    return [p_value, decision, conclusion]


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def verify_child(heights):
    father = heights['father']
    results = {}
    for column in heights.columns:
        if column.startswith('child_'):
            mask = heights[column].isna()
            group_true = father[mask].dropna()
            group_false = father[~mask].dropna()
            results[column] = stats.ks_2samp(group_true, group_false, alternative='two-sided', method='auto').pvalue
    return pd.Series(results)


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def cond_single_imputation(new_heights):
    bins = pd.qcut(new_heights['father'], 4, duplicates='drop')
    means = new_heights.groupby(bins, observed=False)['child'].transform('mean')
    return new_heights['child'].fillna(means)


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def quantitative_distribution(child, N):
    observed = child.dropna().to_numpy()
    if observed.size == 0 or N == 0:
        return np.array([])
    densities, edges = np.histogram(observed, bins=10, density=True)
    probs = densities * np.diff(edges)
    probs = probs / probs.sum()
    rng = np.random.default_rng()
    choices = rng.choice(len(probs), size=N, p=probs)
    lowers = edges[:-1][choices]
    uppers = edges[1:][choices]
    return rng.uniform(lowers, uppers)


def impute_height_quant(child):
    result = child.copy()
    missing = result.isna()
    count = missing.sum()
    if count == 0:
        return result
    imputations = quantitative_distribution(child, count)
    result.loc[missing] = imputations
    return result


# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------


def answers():
    mc = [1, 2, 2, 1]
    sites = ['https://xkcd.com', 'https://www.instagram.com']
    return mc, sites
