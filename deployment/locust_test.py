from locust import HttpUser, task, constant_throughput

# Define test json requests
test_acceptance = {
    "loan_amnt": 40000,
    "purpose": "debt_consolidation",
    "dti": 35.61,
    "emp_length": 0.0,
    "risk_score": "nan",
}
test_grade = {
    "loan_amnt": 16000.0,
    "term": "36 months",
    "verification_status": "Verified",
    "purpose": "debt_consolidation",
    "dti": 23.21,
    "inq_last_6mths": 0.0,
    "all_util": 68.0,
    "total_rev_hi_lim": 27700.0,
    "acc_open_past_24mths": 10.0,
    "bc_open_to_buy": 92.0,
    "bc_util": 98.9,
    "mo_sin_old_il_acct": 197.0,
    "mo_sin_old_rev_tl_op": 331.0,
    "mths_since_recent_inq": 10.0,
    "num_tl_op_past_12m": 6.0,
    "tot_hi_cred_lim": 169944.0,
    "fico": 672.0,
}
test_subgrade_rate = {
    "loan_amnt": 28000.0,
    "term": "60 months",
    "grade": "B",
    "home_ownership": "MORTGAGE",
    "verification_status": "Not Verified",
    "purpose": "debt_consolidation",
    "dti": 19.51,
    "inq_last_6mths": 0.0,
    "mths_since_last_record": "nan",
    "application_type": "Individual",
    "open_il_12m": 1.0,
    "all_util": 53.0,
    "total_rev_hi_lim": 41100.0,
    "inq_last_12m": 1.0,
    "acc_open_past_24mths": 3.0,
    "bc_open_to_buy": 32463.0,
    "bc_util": 21.0,
    "mo_sin_old_il_acct": 155.0,
    "mo_sin_old_rev_tl_op": 340.0,
    "mort_acc": 3.0,
    "mths_since_recent_inq": 10.0,
    "num_actv_rev_tl": 2.0,
    "num_tl_op_past_12m": 2.0,
    "pct_tl_nvr_dlq": 88.5,
    "tot_hi_cred_lim": 271024.0,
    "fico": 727.0,
}


class Loan(HttpUser):
    # Means that a user will send 1 request per second
    wait_time = constant_throughput(1)

    # Task to be performed (send data & get response)
    @task
    def predict(self):
        self.client.post(
            "/predict/accept",
            json=test_acceptance,
            timeout=1,
        )
        self.client.post(
            "/predict/grade",
            json=test_grade,
            timeout=1,
        )
        self.client.post(
            "/predict/subgrade_rate",
            json=test_subgrade_rate,
            timeout=1,
        )
