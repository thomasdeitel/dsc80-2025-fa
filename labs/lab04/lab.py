# lab.py


import pandas as pd
import numpy as np
import io
from pathlib import Path
import os


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def prime_time_logins(login):
    login_dt = login.copy()
    login_dt['Time'] = pd.to_datetime(login_dt['Time'])
    prime_mask = (login_dt['Time'].dt.hour >= 16) & (login_dt['Time'].dt.hour < 20)
    counts = login_dt.assign(prime=prime_mask).groupby('Login Id')['prime'].sum().astype(int)
    return counts.to_frame(name='Time')


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def count_frequency(login):
    login_dt = login.assign(Time=pd.to_datetime(login['Time']))
    ref_time = pd.Timestamp('2024-01-31 23:59:00')
    aggregated = login_dt.groupby('Login Id')['Time'].agg(['min', 'count'])
    days = (ref_time - aggregated['min']).dt.days
    return (aggregated['count'] / days).rename('Frequency')


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def cookies_null_hypothesis():
    return [1, 2]
                         
def cookies_p_value(N):
    sims = np.random.binomial(250, 0.04, size=N)
    return float(np.mean(sims >= 15))


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def car_null_hypothesis():
    return [1, 4]

def car_alt_hypothesis():
    return [2, 6]

def car_test_statistic():
    return [1, 4]

def car_p_value():
    return 4


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def superheroes_test_statistic():
    return [1, 2]
    
def bhbe_col(heroes):
    hair = heroes['Hair color'].str.contains('blond', case=False, na=False)
    eyes = heroes['Eye color'].str.contains('blue', case=False, na=False)
    return hair & eyes

def superheroes_observed_statistic(heroes):
    mask = bhbe_col(heroes)
    good = heroes['Alignment'].str.lower() == 'good'
    return good[mask].mean()

def simulate_bhbe_null(heroes, N):
    mask = bhbe_col(heroes)
    n = mask.sum()
    p_good = (heroes['Alignment'].str.lower() == 'good').mean()
    samples = np.random.binomial(n, p_good, size=N)
    return samples / n

def superheroes_p_value(heroes):
    observed = superheroes_observed_statistic(heroes)
    sims = simulate_bhbe_null(heroes, 100000)
    p_val = float(np.mean(sims >= observed))
    decision = 'Reject' if p_val < 0.01 else 'Fail to reject'
    return [p_val, decision]


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def diff_of_means(data, col='orange'):
    means = data.groupby('Factory')[col].mean()
    return abs(means.loc['Yorkville'] - means.loc['Waco'])


def simulate_null(data, col='orange'):
    shuffled = np.random.permutation(data['Factory'])
    permuted = data.assign(Factory=shuffled)
    return diff_of_means(permuted, col=col)


def color_p_value(data, col='orange'):
    observed = diff_of_means(data, col=col)
    sims = np.array([simulate_null(data, col=col) for _ in range(1000)])
    return float(np.mean(sims >= observed))


# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------


def ordered_colors():
    return [
        ('yellow', 0.0),
        ('orange', 0.042),
        ('red', 0.214),
        ('green', 0.467),
        ('purple', 0.965),
    ]


# ---------------------------------------------------------------------
# QUESTION 8
# ---------------------------------------------------------------------


    
def same_color_distribution():
    return (0.011, 'Fail to Reject')

# ---------------------------------------------------------------------
# QUESTION 9
# ---------------------------------------------------------------------


def perm_vs_hyp():
    return ['P', 'P', 'H', 'H', 'P']
