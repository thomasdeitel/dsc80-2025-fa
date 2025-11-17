# project.py


import pandas as pd
import numpy as np
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pd.options.plotting.backend = 'plotly'

from IPython.display import display

# DSC 80 preferred styles
pio.templates["dsc80"] = go.layout.Template(
    layout=dict(
        margin=dict(l=30, r=30, t=30, b=30),
        autosize=True,
        width=600,
        height=400,
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True),
        title=dict(x=0.5, xanchor="center"),
    )
)
pio.templates.default = "simple_white+dsc80"
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def clean_loans(loans):

    cleaned = loans.copy()

    cleaned['issue_d'] = pd.to_datetime(cleaned['issue_d'], format='%b-%Y')

    cleaned['term'] = cleaned['term'].str.extract(r'(\d+)').astype(int)

    emp_title = cleaned['emp_title'].str.lower().str.strip()
    cleaned['emp_title'] = emp_title.where(emp_title != 'rn', 'registered nurse')

    cleaned['term_end'] = (
        cleaned['issue_d'].dt.to_period('M') + cleaned['term']
    ).dt.to_timestamp()

    return cleaned


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------



def correlations(df, pairs):
    indices = []
    values = []

    for col1, col2 in pairs:
        indices.append(f'r_{col1}_{col2}')
        values.append(df[col1].corr(df[col2]))

    return pd.Series(values, index=indices, dtype='float64')



# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def create_boxplot(loans):
    loans = loans.copy()

    bins = [580, 670, 740, 800, 851]
    labels = [
        '[580, 670)',
        '[670, 740)',
        '[740, 800)',
        '[800, 850)',
    ]

    credit_bins = pd.cut(
        loans['fico_range_low'],
        bins=bins,
        labels=labels,
        right=False,
        include_lowest=True,
    )
    loans = loans.assign(credit_score_bin=credit_bins).dropna(subset=['credit_score_bin'])
    loans['term_str'] = loans['term'].astype(str)

    fig = px.box(
        loans,
        x='credit_score_bin',
        y='int_rate',
        color='term_str',
        category_orders={'credit_score_bin': labels, 'term_str': ['36', '60']},
        color_discrete_map={'36': '#7d3c98', '60': '#f1c40f'},
        labels={'credit_score_bin': 'Credit Score Range', 'int_rate': 'Interest Rate (%)'},
        title='Interest Rate vs. Credit Score',
    )

    fig.update_layout(legend_title_text='Loan Length (Months)')
    fig.update_xaxes(title='Credit Score Range')
    fig.update_yaxes(title='Interest Rate (%)')

    return fig


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def ps_test(loans, N):
    loans = loans.copy()
    loans['has_ps'] = loans['desc'].notna()

    with_ps = loans['has_ps']
    rates = loans['int_rate'].to_numpy()
    observed = rates[with_ps].mean() - rates[~with_ps].mean()

    n_with = with_ps.sum()
    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(N):
        shuffled = rng.permutation(rates)
        diff = shuffled[:n_with].mean() - shuffled[n_with:].mean()
        diffs.append(diff)

    diffs = np.array(diffs)
    p_value = np.mean(diffs >= observed)
    return p_value
    
def missingness_mechanism():
    return 2
    
def argument_for_nmar():
    return (
        "Applicants facing financial hardship might skip personal statements to avoid disclosing sensitive circumstances, making the missingness depend on unobserved borrower traits."
    )


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def tax_owed(income, brackets):
    tax = 0.0
    for i, (rate, lower) in enumerate(brackets):
        if income <= lower:
            break

        if i + 1 < len(brackets):
            upper = brackets[i + 1][1]
            taxable = min(income, upper) - lower
        else:
            taxable = income - lower

        taxable = max(taxable, 0)
        tax += rate * taxable

    return tax


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def clean_state_taxes(state_taxes_raw): 
    df = state_taxes_raw.copy()
    df = df.dropna(how='all')

    import json
    state_mapping_path = Path('data') / 'state_mapping.json'
    with open(state_mapping_path, 'r') as f:
        state_mapping = json.load(f)

    valid_states = set(state_mapping.keys())
    df['State'] = df['State'].where(df['State'].isin(valid_states))
    df['State'] = df['State'].ffill()

    rate_series = (
        df['Rate']
        .fillna('0%')
        .str.lower()
        .str.replace('%', '', regex=False)
        .replace({'none': '0'})
    )
    df['Rate'] = pd.to_numeric(rate_series, errors='coerce').fillna(0) / 100
    df['Rate'] = df['Rate'].round(2)

    lower_series = (
        df['Lower Limit']
        .fillna('$0')
        .str.replace('[$,]', '', regex=True)
        .replace('', '0')
    )
    df['Lower Limit'] = pd.to_numeric(lower_series, errors='coerce').fillna(0).astype(int)

    return df[['State', 'Rate', 'Lower Limit']]


# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------


def state_brackets(state_taxes):
    grouped = (
        state_taxes
        .sort_values(['State', 'Lower Limit'])
        .groupby('State')
        [['Rate', 'Lower Limit']]
        .apply(lambda df: list(df.itertuples(index=False, name=None)))
    )
    return grouped.to_frame(name='bracket_list')
    
def combine_loans_and_state_taxes(loans, state_taxes):
    import json
    state_mapping_path = Path('data') / 'state_mapping.json'
    with open(state_mapping_path, 'r') as f:
        state_mapping = json.load(f)
        
    bracket_df = state_brackets(state_taxes).reset_index()
    bracket_df['State'] = bracket_df['State'].map(state_mapping)

    loans_with_state = loans.rename(columns={'addr_state': 'State'})
    merged = loans_with_state.merge(bracket_df, on='State', how='left')

    return merged


# ---------------------------------------------------------------------
# QUESTION 8
# ---------------------------------------------------------------------


def find_disposable_income(loans_with_state_taxes):
    FEDERAL_BRACKETS = [
     (0.1, 0), 
     (0.12, 11000), 
     (0.22, 44725), 
     (0.24, 95375), 
     (0.32, 182100),
     (0.35, 231251),
     (0.37, 578125)
    ]
    df = loans_with_state_taxes.copy()

    df['federal_tax_owed'] = df['annual_inc'].apply(lambda inc: tax_owed(inc, FEDERAL_BRACKETS))
    df['state_tax_owed'] = df.apply(lambda row: tax_owed(row['annual_inc'], row['bracket_list']), axis=1)
    df['disposable_income'] = df['annual_inc'] - df['federal_tax_owed'] - df['state_tax_owed']

    return df


# ---------------------------------------------------------------------
# QUESTION 9
# ---------------------------------------------------------------------


def aggregate_and_combine(loans, keywords, quantitative_column, categorical_column):
    emp_titles = loans['emp_title'].fillna('')
    categories = set()
    keyword_columns = {}

    for keyword in keywords:
        mask = emp_titles.str.contains(keyword, na=False)
        subset = loans[mask]
        group_means = subset.groupby(categorical_column)[quantitative_column].mean()
        categories.update(group_means.index.tolist())
        col_name = f'{keyword}_mean_{quantitative_column}'
        keyword_columns[col_name] = {
            'group_means': group_means,
            'overall': subset[quantitative_column].mean()
        }

    categories = sorted(categories)
    index = categories + ['Overall']
    result = pd.DataFrame(index=index)

    for col_name, values in keyword_columns.items():
        group_means = values['group_means'].reindex(categories)
        result.loc[categories, col_name] = group_means.values
        result.loc['Overall', col_name] = values['overall']

    result.index.name = categorical_column
    return result


# ---------------------------------------------------------------------
# QUESTION 10
# ---------------------------------------------------------------------


def exists_paradox(loans, keywords, quantitative_column, categorical_column):
    combined = aggregate_and_combine(loans, keywords, quantitative_column, categorical_column)
    col_a = f'{keywords[0]}_mean_{quantitative_column}'
    col_b = f'{keywords[1]}_mean_{quantitative_column}'

    per_group = combined.iloc[:-1][[col_a, col_b]].dropna()
    if per_group.empty:
        return False

    if not bool((per_group[col_a] > per_group[col_b]).all()):
        return False

    overall = combined.iloc[-1]
    return bool(overall[col_a] < overall[col_b])
    
def paradox_example(loans):
    return {
        'loans': loans,
        'keywords': ['teacher', 'service'],
        'quantitative_column': 'int_rate',
        'categorical_column': 'verification_status'
    }
