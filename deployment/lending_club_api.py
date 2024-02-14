from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import (
    BaseModel,
    NonNegativeInt,
    PositiveInt,
    NonNegativeFloat,
)
from typing import Literal
from enum import Enum


# Pydantic classes for input and output
class PurposeEnum(str, Enum):
    debt_consolidation = "debt_consolidation"
    other = "other"
    house = "house"
    educational = "educational"
    credit_card = "credit_card"
    home_improvement = "home_improvement"
    major_purchase = "major_purchase"
    medical = "medical"
    moving = "moving"
    car = "car"
    small_business = "small_business"
    vacation = "vacation"
    wedding = "wedding"
    renewable_energy = "renewable_energy"


class VerificationEnum(str, Enum):
    not_verified = "Not Verified"
    source_verified = "Source Verified"
    verified = "Verified"


class TermEnum(str, Enum):
    mnths30 = "36 months"
    mnths60 = "60 months"


class TypeEnum(str, Enum):
    individual = "Individual"
    joint = "Joint App"


class HomeEnum(str, Enum):
    mortgage = "MORTGAGE"
    rent = "RENT"
    own = "OWN"
    other = "OTHER"


class GradeEnum(str, Enum):
    a = "A"
    b = "B"
    c = "C"
    d = "D"
    e = "E"
    f = "F"
    g = "G"


class SubgradeEnum(str, Enum):
    a1 = "A1"
    a2 = "A2"
    a3 = "A3"
    a4 = "A4"
    a5 = "A5"
    b1 = "B1"
    b2 = "B2"
    b3 = "B3"
    b4 = "B4"
    b5 = "B5"
    c1 = "C1"
    c2 = "C2"
    c3 = "C3"
    c4 = "C4"
    c5 = "C5"
    d1 = "D1"
    d2 = "D2"
    d3 = "D3"
    d4 = "D4"
    d5 = "D5"
    e1 = "E1"
    e2 = "E2"
    e3 = "E3"
    e4 = "E4"
    e5 = "E5"
    f1 = "F1"
    f2 = "F2"
    f3 = "F3"
    f4 = "F4"
    f5 = "F5"
    g1 = "G1"
    g2 = "G2"
    g3 = "G3"
    g4 = "G4"
    g5 = "G5"


class AcceptanceInfo(BaseModel):
    loan_amnt: PositiveInt = 1000
    purpose: PurposeEnum = PurposeEnum.debt_consolidation
    dti: float
    emp_length: NonNegativeInt | float = 0
    risk_score: Literal["A", "B", "C", "D", "F"] | float


class GradeInfo(BaseModel):
    loan_amnt: PositiveInt = 1000
    term: TermEnum = TermEnum.mnths60
    verification_status: VerificationEnum = VerificationEnum.verified
    purpose: PurposeEnum = PurposeEnum.debt_consolidation
    dti: NonNegativeFloat = 0
    inq_last_6mths: float
    all_util: float
    total_rev_hi_lim: float
    acc_open_past_24mths: float
    bc_open_to_buy: float
    bc_util: float
    mo_sin_old_il_acct: float
    mo_sin_old_rev_tl_op: float
    mths_since_recent_inq: float
    num_tl_op_past_12m: float
    tot_hi_cred_lim: float
    fico: PositiveInt = 600


class SubgradeInfo(BaseModel):
    loan_amnt: PositiveInt = 1000
    term: TermEnum = TermEnum.mnths60
    grade: GradeEnum = GradeEnum.c
    home_ownership: HomeEnum = HomeEnum.mortgage
    verification_status: VerificationEnum = VerificationEnum.verified
    purpose: PurposeEnum = PurposeEnum.debt_consolidation
    dti: NonNegativeFloat = 0
    inq_last_6mths: float
    mths_since_last_record: float
    application_type: TypeEnum = TypeEnum.individual
    open_il_12m: float
    all_util: float
    total_rev_hi_lim: float
    inq_last_12m: float
    acc_open_past_24mths: float
    bc_open_to_buy: float
    bc_util: float
    mo_sin_old_il_acct: float
    mo_sin_old_rev_tl_op: float
    mort_acc: float
    mths_since_recent_inq: float
    num_actv_rev_tl: float
    num_tl_op_past_12m: float
    pct_tl_nvr_dlq: float
    tot_hi_cred_lim: float
    fico: PositiveInt = 600


class AcceptPredictionOut(BaseModel):
    accept: Literal["yes", "no"]


class GradePredictionOut(BaseModel):
    grade: GradeEnum


class SubgradeRatePredictionOut(BaseModel):
    subgrade: SubgradeEnum
    int_rate: float


# Load the models
acceptance_model = joblib.load("acceptance_model_deploy.joblib")
grade_model = joblib.load("grade_model_deploy.joblib")
subgrade_model = joblib.load("subgrade_model.joblib")
rate_model = joblib.load("rate_model.joblib")

# Start the app
app = FastAPI()


# Home page
@app.get("/")
def home():
    return {"message": "Lending Club Loan Prediction App"}


# Prediction endpoints
@app.post("/predict/accept", response_model=AcceptPredictionOut)
def predict(data: AcceptanceInfo):
    df = pd.DataFrame([data.model_dump()])
    prediction = acceptance_model.predict(df)
    if prediction == 1:
        output = "yes"
    else:
        output = "no"
    result = {"accept": output}
    return result


@app.post("/predict/grade", response_model=GradePredictionOut)
def predict(data: GradeInfo):
    grades = ["A", "B", "C", "D", "E", "F", "G"]
    df = pd.DataFrame([data.model_dump()])
    df["term"] = df["term"].astype("category")
    df["verification_status"] = df["verification_status"].astype("category")
    df["purpose"] = df["purpose"].astype("category")
    prediction = grade_model.predict(df)
    output = grades[prediction.item()]
    result = {"grade": output}
    return result


@app.post("/predict/subgrade_rate", response_model=SubgradeRatePredictionOut)
def predict(data: SubgradeInfo):
    subgrades = [
        "A1",
        "A2",
        "A3",
        "A4",
        "A5",
        "B1",
        "B2",
        "B3",
        "B4",
        "B5",
        "C1",
        "C2",
        "C3",
        "C4",
        "C5",
        "D1",
        "D2",
        "D3",
        "D4",
        "D5",
        "E1",
        "E2",
        "E3",
        "E4",
        "E5",
        "F1",
        "F2",
        "F3",
        "F4",
        "F5",
        "G1",
        "G2",
        "G3",
        "G4",
        "G5",
    ]
    df = pd.DataFrame([data.model_dump()])
    df["term"] = df["term"].astype("category")
    df["verification_status"] = df["verification_status"].astype("category")
    df["purpose"] = df["purpose"].astype("category")
    df["grade"] = df["grade"].astype("category")
    df["home_ownership"] = df["home_ownership"].astype("category")
    df["application_type"] = df["application_type"].astype("category")
    subgrade_prediction = subgrade_model.predict(df)
    subgrade_output = subgrades[subgrade_prediction.item()]
    rate_prediction = rate_model.predict(df)
    result = {"subgrade": subgrade_output, "int_rate": rate_prediction}
    return result
