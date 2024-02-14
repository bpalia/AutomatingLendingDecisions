# Last updated January 17, 2024
# Version 0.1.0

from numpy import dtype
import polars as pl
from IPython.display import display
from typing import List
from datetime import datetime
from polars import col as c
import polars.selectors as cs
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler


def rejected_data_load(path: str, target: str = "target") -> pl.DataFrame:
    """Function to load in Polars and display the head of the rejected loan
    data for accepted/rejected classification. To facilitate homogenity,
    `dti_joint`, `target` columns are added, `date` is rounded to months and
    `dti` is casted to float. 'title' text cleaned and made consistent with
    `purpose` categories. Risk score divided in 5 levels."""
    new_columns = [
        "loan_amnt",
        "date",
        "purpose",
        "risk_scoreX",
        "dti",
        "zip_code",
        "state",
        "emp_length",
        "policy_code",
    ]
    df = pl.read_csv(path, new_columns=new_columns)
    df = df.with_columns([
        pl.col("date").str.to_date("%Y-%m-%d"),
        pl.col("dti").str.strip_suffix("%").cast(pl.Float64),
        pl.col("purpose")
        .str.replace("_", " ")
        .str.to_lowercase()
        .str.strip_chars(".!? \n\t")
        .fill_null(""),
        pl.lit(None).alias("dti_joint").cast(pl.Float64),
    ])
    new_column = (  # same categories as in accepted df
        pl.when(pl.col("purpose").str.contains("business|equipment|inventory"))
        .then(pl.lit("small_business"))
        .when(
            pl.col("purpose").str.contains(
                "education|school|student|college|tuition|learning"
            )
        )
        .then(pl.lit("educational"))
        .when(pl.col("purpose").str.contains("medical|surgery|dental"))
        .then(pl.lit("medical"))
        .when(pl.col("purpose").str.contains("vacation"))
        .then(pl.lit("vacation"))
        .when(pl.col("purpose").str.contains("moving|relocat|move"))
        .then(pl.lit("moving"))
        .when(pl.col("purpose").str.contains("wedding"))
        .then(pl.lit("wedding"))
        .when(pl.col("purpose").str.contains(r"renewable|\bgreen\b"))
        .then(pl.lit("renewable_energy"))
        .when(
            (
                pl.col("purpose").str.contains("house|home")
                & pl.col("purpose").str.contains(
                    "fix|repair|upgrade|inprov|imp|remodel|renov|project|emprov"
                )
            )
            | pl.col("purpose").str.contains(
                "furni|pool|kitchen|roof|household|improvement"
            )
        )
        .then(pl.lit("home_improvement"))
        .when(pl.col("purpose").str.contains("home|house|home buying"))
        .then(pl.lit("house"))
        .when(
            pl.col("purpose").str.contains(
                r"\bcar\b|motorcycle|auto|truck|vehicle|motor cycle"
            )
            & ~pl.col("purpose").str.contains("repair|fix")
        )
        .then(pl.lit("car"))
        .when(pl.col("purpose").str.contains("purchase|boat|engagement ring"))
        .then(pl.lit("major_purchase"))
        .when(
            pl.col("purpose").str.contains(r"credit card|\bcc\b|credit|card")
        )
        .then(pl.lit("credit_card"))
        .when(
            pl.col("purpose").str.contains(
                "debt|consolidat|consolodat|payoff|pay off|freedom|refi"
            )
        )
        .then(pl.lit("debt_consolidation"))
        .otherwise(pl.lit("other"))
        .alias("purpose")
        .cast(str)
    )
    df = df.with_columns(new_column)
    df1 = df.filter(pl.col("date") < datetime(2013, 11, 5))  # FICO score
    new_column = (
        pl.when(pl.col("risk_scoreX") < 300)
        .then(None)
        .when(pl.col("risk_scoreX") <= 579)
        .then(pl.lit("F"))
        .when(pl.col("risk_scoreX") <= 669)
        .then(pl.lit("D"))
        .when(pl.col("risk_scoreX") <= 739)
        .then(pl.lit("C"))
        .when(pl.col("risk_scoreX") <= 799)
        .then(pl.lit("B"))
        .when(pl.col("risk_scoreX") <= 850)
        .then(pl.lit("A"))
        .otherwise(None)
        .alias("risk_score")
    )
    df1 = df1.with_columns(new_column)
    df2 = df.filter(pl.col("date") >= datetime(2013, 11, 5))  # Vantage Score
    new_column = (
        pl.when(pl.col("risk_scoreX") < 300)
        .then(None)
        .when(pl.col("risk_scoreX") <= 599)
        .then(pl.lit("F"))
        .when(pl.col("risk_scoreX") <= 699)
        .then(pl.lit("D"))
        .when(pl.col("risk_scoreX") <= 799)
        .then(pl.lit("C"))
        .when(pl.col("risk_scoreX") <= 899)
        .then(pl.lit("B"))
        .when(pl.col("risk_scoreX") <= 990)
        .then(pl.lit("A"))
        .otherwise(None)
        .alias("risk_score")
    )
    df2 = df2.with_columns(new_column)
    df = df1.vstack(df2)
    df = df.with_columns([
        pl.col("date").dt.month_start(),
        pl.col("dti").replace(-1, None),
        pl.lit(0).alias(target).cast(pl.Int8),
    ])
    df = df.drop(columns=["risk_scoreX"])
    display(df.head())
    return df


def accepted_data_load(path: str, target: str = "target") -> pl.DataFrame:
    """Function to load in Polars and display the head of the accepted loan
    data for accepted/rejected classification. To facilitate homogenity,
    `target` column is added. Fico low range is divided in 5 levels."""
    columns = [
        "loan_amnt",
        "issue_d",
        "purpose",
        "fico_range_low",
        "dti",
        "zip_code",
        "addr_state",
        "emp_length",
        "policy_code",
        "dti_joint",
    ]
    df = pl.read_csv(path, columns=columns)
    df = df.drop_nulls(subset="loan_amnt")
    df = df.rename({"issue_d": "date", "addr_state": "state"})
    df = df.with_columns([
        pl.col("date").str.to_date("%b-%Y"),
        pl.col("dti").replace(-1, None),
        pl.lit(1).alias(target).cast(pl.Int8),
    ])
    new_column = (
        pl.when(pl.col("fico_range_low") < 300)
        .then(None)
        .when(pl.col("fico_range_low") <= 579)
        .then(pl.lit("F"))
        .when(pl.col("fico_range_low") <= 669)
        .then(pl.lit("D"))
        .when(pl.col("fico_range_low") <= 739)
        .then(pl.lit("C"))
        .when(pl.col("fico_range_low") <= 799)
        .then(pl.lit("B"))
        .when(pl.col("fico_range_low") <= 850)
        .then(pl.lit("A"))
        .otherwise(None)
        .alias("risk_score")
    )
    df = df.with_columns(new_column)
    ordered_columns = [
        "loan_amnt",
        "date",
        "purpose",
        "dti",
        "zip_code",
        "state",
        "emp_length",
        "policy_code",
        "dti_joint",
        "risk_score",
        "target",
    ]
    df = df.select(ordered_columns)
    display(df.head())
    return df


def accepted_rejected_cleaning(df: pl.DataFrame) -> pl.DataFrame:
    """Function for cleaning Lending Club data in Polars related to
    classifying loans to accepted and rejected. Performs `emp_lenght`
    conversion from `str` to `int`, `title` cleaning, incorporating `dti_joint`
    into `dti`, dropping `dti_joint` and `policy_code`, creating `dti_missing`
    feature."""
    df = df.with_columns([
        (pl.col("loan_amnt").cast(pl.UInt32)),
        (pl.col("risk_score").cast(pl.Categorical(ordering="lexical"))),
        (pl.col("purpose").cast(pl.Categorical)),
        (pl.col("state").cast(pl.Categorical)),
        (pl.col("zip_code").cast(pl.Categorical)),
        (
            pl.col("emp_length")
            .str.replace(r"\+* year.*", "")
            .str.replace(r"< 1", "0")
            .cast(pl.Float32)
        ),
    ])
    new_column = (
        pl.when(
            ((df["dti_joint"] < df["dti"]) | df["dti"].is_null())
            & df["dti_joint"].is_not_null()
        )
        .then(df["dti_joint"])
        .otherwise(df["dti"])
        .clip(upper_bound=100)
        .alias("dti")
        .cast(pl.Float32)
    )
    df = df.with_columns(new_column)
    df = df.drop(columns=["dti_joint", "policy_code"])
    return df


def initial_accepted_loan_cleaning(df: pl.DataFrame) -> pl.DataFrame:
    """Function to drop unrelevant columns or columns related to information
    unavailable at loan origination. FICO scores is taken as a middle of the
    range. Correlated columns are also removed."""
    df = df.drop(
        cs.contains((
            "id",
            "url",
            "title",  # free-form text provided by the applicant
            "desc",  # free-form description
            "zip_code",  # should not affect the decision
            "policy_code",
            "loan_status",
            "initial_list_status",
            "funded_amnt",
            "last_credit",
            "last_fico",
            "last_pymnt",
            "next_pymnt",
            "recover",
            "out_prncp",
            "pymnt_plan",
            "total_pymnt",
            "total_rec",
            "deferral_",
            "hardship_",
            "payment_",
            "settlement_",
            "additional_",
            "disbursement",
            "installment",  # includes interest rate
            "earliest_cr_line",  # correlated with "mo_sin_old_rev_tl_op"
            "open_il_24m",  # correlated with "open_il_12m"
            "num_rev_tl_bal_gt_0",  # correlated with "num_actv_rev_tl"
            "open_rv_12m",  # correlated with "num_tl_op_past_12m"
            "open_rv_24m",  # correlated with "num_tl_op_past_12m"
            "total_bc_limit",  # correlated with "bc_open_to_buy"
            "tot_cur_bal",  # correlated with "tot_hi_cred_lim"
            "revol_util",  # correlated with "bc_util"
            "il_util",  # correlated with "all_util"
            "inq_fi",  # correlated with "inq_last_12m"
            "percent_bc_gt_75",  # correlated with "bc_util"
            "open_acc_6m",  # correlated with "num_tl_op_past_12m"
        ))
    )
    df = df.with_columns(
        c.date.str.to_date("%b-%Y"),
        c.emp_length.str.replace(r"\+* year.*", "")
        .str.replace(r"< 1", "0")
        .cast(pl.Float32),
        c.dti.replace(-1, None).clip(upper_bound=100),
        c.dti_joint.replace(-1, None).clip(upper_bound=100),
        c.revol_bal_joint.cast(pl.Float32),
        c.sec_app_fico_range_low.cast(pl.Float32),
        c.sec_app_fico_range_high.cast(pl.Float32),
        c.sec_app_inq_last_6mths.cast(pl.Float32),
        c.sec_app_mort_acc.cast(pl.Float32),
        c.sec_app_open_acc.cast(pl.Float32),
        c.sec_app_open_act_il.cast(pl.Float32),
        c.sec_app_num_rev_accts.cast(pl.Float32),
        c.sec_app_chargeoff_within_12_mths.cast(pl.Float32),
        c.sec_app_collections_12_mths_ex_med.cast(pl.Float32),
        c.sec_app_mths_since_last_major_derog.cast(pl.Float32),
        c.grade.cast(pl.Categorical(ordering="lexical")),
        c.sub_grade.cast(pl.Categorical(ordering="lexical")),
        c.verification_status.cast(pl.Categorical),
        c.verification_status_joint.cast(pl.Categorical),
        c.purpose.cast(pl.Categorical),
        c.state.cast(pl.Categorical),
        c.application_type.cast(pl.Categorical),
        c.term.str.lstrip().cast(pl.Categorical),
    )
    df = df.with_columns(
        ((c.fico_range_low + c.fico_range_high) / 2).round().alias("fico"),
        ((c.sec_app_fico_range_low + c.sec_app_fico_range_high) / 2)
        .round()
        .alias("sec_app_fico"),
        pl.when(c.home_ownership.is_in(["MORTGAGE", "RENT", "OWN"]))
        .then(c.home_ownership)
        .otherwise(pl.lit("OTHER"))
        .alias("home_ownership")
        .cast(pl.Categorical),
    )
    df = df.drop(cs.contains("fico_range"))
    return df


def joint_conditions(col: str) -> pl.Expr:
    """Create expression to change input for joint loan."""
    if col == "dti":  # because of the requirements
        expr = (
            pl.when(
                (c.application_type == "Individual")
                | pl.col(f"{col}_joint").is_null()
                | (pl.col(f"{col}_joint") > 50)
            )
            .then(pl.col(f"{col}"))
            .otherwise(pl.col(f"{col}_joint"))
            .alias(col)
        )
    else:
        expr = (
            pl.when(
                (c.application_type == "Individual")
                | pl.col(f"{col}_joint").is_null()
            )
            .then(pl.col(f"{col}"))
            .otherwise(pl.col(f"{col}_joint"))
            .alias(col)
        )
    return expr


def avg_conditions(col: str) -> pl.Expr:
    """Create expression to average inputs from two borrowers in joint loan."""
    return (
        pl.when(
            (c.application_type == "Individual")
            | pl.col(f"sec_app_{col}").is_null()
        )
        .then(pl.col(f"{col}"))
        .otherwise(((pl.col(f"{col}") + pl.col(f"sec_app_{col}")) / 2).round())
        .alias(col)
    )


def joint_loan_integration(df: pl.DataFrame) -> pl.DataFrame:
    """Function for integrating `dti_joint`, `verification_status_joint`,
    `sec_app_inq_last_6mths`, `sec_app_mort_acc`, `sec_app_fico` columns."""
    df = df.with_columns(
        joint_conditions("dti"),
        joint_conditions("verification_status"),
        joint_conditions("revol_bal"),
        joint_conditions("annual_inc"),
        avg_conditions("inq_last_6mths"),
        avg_conditions("mort_acc"),
        avg_conditions("open_acc"),
        avg_conditions("open_act_il"),
        avg_conditions("num_rev_accts"),
        avg_conditions("chargeoff_within_12_mths"),
        avg_conditions("collections_12_mths_ex_med"),
        avg_conditions("mths_since_last_major_derog"),
        avg_conditions("fico"),
    )
    df = df.drop(cs.contains(("_joint", "sec_app_")))
    return df


def sample_balanced_polars(
    df: pl.DataFrame, target: str = "target", sort: str = None
) -> pl.DataFrame:
    """Function to downsample majority (negative) class in Polars. If necessary,
    the return dataframe can be sorted."""
    pos = df.filter(pl.col(target) == 1)
    neg = df.filter(pl.col(target) == 0).sample(n=pos.shape[0], seed=42)
    balanced_df = pos.vstack(neg)
    if sort:
        balanced_df = balanced_df.sort(sort)
    return balanced_df


def sample_balanced_pandas(
    df: pd.DataFrame, target: str = "target"
) -> pd.DataFrame:
    """Function to downsample majority classes in Pandas."""
    subsample, y = RandomUnderSampler(random_state=42).fit_resample(
        df.drop(columns=target), df[target]
    )
    subsample[target] = y
    return subsample


def display_basic_info(df: pl.DataFrame) -> None:
    """Function to display date, size and duplicate information."""
    print(
        f"Loans are from {df.select(pl.min('date')).item()} to"
        f" {df.select(pl.max('date')).item()}."
    )
    print("Number of loans:", df.shape[0])
    print("Number of duplicates:", df.is_duplicated().sum())
    return
