# lab.py


import os
import io
from pathlib import Path
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def read_linkedin_survey(dirname):
    """
    Combine all LinkedIn survey CSVs in dirname into a single cleaned DataFrame.
    """
    directory = Path(dirname)
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {dirname}")

    target_cols = [
        "first name",
        "last name",
        "current company",
        "job title",
        "email",
        "university",
    ]

    def _standardize_columns(cols):
        standardized = []
        for col in cols:
            cleaned = col.strip().lower().replace("_", " ")
            cleaned = " ".join(cleaned.split())
            standardized.append(cleaned)
        return standardized

    frames = []
    for csv_path in sorted(directory.glob("survey*.csv")):
        df = pd.read_csv(csv_path)
        df.columns = _standardize_columns(df.columns)
        df = df.reindex(columns=target_cols)
        for col in target_cols:
            if df[col].dtype == object:
                df[col] = df[col].str.strip()
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=target_cols)

    combined = pd.concat(frames, ignore_index=True)
    return combined[target_cols]


def com_stats(df):
    ohio_mask = df["university"].str.contains("Ohio", case=False, na=False)
    programmer_mask = df["job title"].str.contains("Programmer", case=False, na=False)
    ohio_programmers = (ohio_mask & programmer_mask).sum()
    ohio_total = ohio_mask.sum()
    ohio_prop = ohio_programmers / ohio_total if ohio_total else np.nan
    if ohio_total:
        ohio_prop = float(ohio_prop)

    job_titles = df["job title"].dropna()
    engineer_titles = job_titles[job_titles.str.endswith("Engineer")]
    engineer_count = int(engineer_titles.nunique())

    longest_title = job_titles.loc[job_titles.str.len().idxmax()]

    manager_count = int(df["job title"].str.contains("manager", case=False, na=False).sum())

    return [ohio_prop, engineer_count, longest_title, manager_count]



# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def read_student_surveys(dirname):
    directory = Path(dirname)
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {dirname}")

    frames = []
    for csv_path in sorted(directory.glob("favorite*.csv")):
        df = pd.read_csv(csv_path).set_index("id")
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, axis=1).sort_index()
    combined.index.name = "id"
    return combined


def check_credit(df):
    surveys = df.drop(columns=["name"])

    valid = surveys.notna()
    if "genre" in surveys.columns:
        genre_clean = surveys["genre"].fillna("").str.strip()
        valid["genre"] = genre_clean.ne("") & genre_clean.ne("(no genres listed)")

    response_counts = valid.sum(axis=1)
    num_questions = surveys.shape[1]

    individual_bonus = np.where(response_counts >= num_questions / 2, 5, 0)
    question_completion = valid.mean(axis=0)
    class_bonus = min((question_completion >= 0.9).sum(), 2)

    result = pd.DataFrame({"name": df["name"], "ec": individual_bonus + class_bonus}, index=df.index)
    result["ec"] = result["ec"].astype(int)
    result.index.name = df.index.name
    return result


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def most_popular_procedure(pets, procedure_history):
    valid_pet_ids = pets["PetID"]
    filtered = procedure_history[procedure_history["PetID"].isin(valid_pet_ids)]
    counts = filtered["ProcedureType"].value_counts()
    return counts.idxmax() if not counts.empty else None

def pet_name_by_owner(owners, pets):
    owners_indexed = owners.set_index("OwnerID")["Name"]
    pet_lists = pets.groupby("OwnerID")["Name"].agg(list)

    combined = owners_indexed.to_frame("OwnerName").join(
        pet_lists.rename("PetNames"), how="left"
    )

    def _format(names):
        if isinstance(names, list):
            if len(names) == 1:
                return names[0]
            if len(names) > 1:
                return names
        return np.nan

    pet_values = combined["PetNames"].apply(_format)
    pet_values.index = combined["OwnerName"].values
    pet_values.index.name = None
    return pet_values


def total_cost_per_city(owners, pets, procedure_history, procedure_detail):
    owners_city = owners[["OwnerID", "City"]]
    pets_owner = pets[["PetID", "OwnerID"]]

    history_with_price = procedure_history.merge(
        procedure_detail, on=["ProcedureType", "ProcedureSubCode"], how="left"
    )
    history_with_owner = history_with_price.merge(pets_owner, on="PetID", how="inner")
    history_with_city = history_with_owner.merge(owners_city, on="OwnerID", how="inner")

    history_with_city = history_with_city.dropna(subset=["Price"])
    cost_by_city = history_with_city.groupby("City")["Price"].sum()

    cities = owners_city["City"].dropna().drop_duplicates()
    return cost_by_city.reindex(cities, fill_value=0)


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def average_seller(sales):
    return sales.groupby("Name")["Total"].mean().to_frame("Average Sales")

def product_name(sales):
    return sales.pivot_table(
        index="Name",
        columns="Product",
        values="Total",
        aggfunc="sum",
    )

def count_product(sales):
    pivot = sales.pivot_table(
        index=["Product", "Name"],
        columns="Date",
        values="Total",
        aggfunc="count",
        fill_value=0,
    )
    return pivot.astype(int)

def total_by_month(sales):
    months = pd.to_datetime(sales["Date"], format="%m.%d.%Y", errors="coerce").dt.month_name()
    sales_with_month = sales.assign(Month=months).dropna(subset=["Month"])
    return sales_with_month.pivot_table(
        index=["Name", "Product"],
        columns="Month",
        values="Total",
        aggfunc="sum",
        fill_value=0,
    )
