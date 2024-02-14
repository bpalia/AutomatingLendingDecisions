# LendingClub: Automating Lending Decisions

LendingClub wants to automate their lending decisions fully and has provided old loan data from the middle of 2007 to the end of 2018 to build machine learning models for that. The process of lending automation is expected to happen in 3 steps:
1. loan classification into accepted and rejected;
2. prediction of the loan grade;
3. prediction of the loan subgrade and interest rate.

Lending club data on rejected and accepted loans can be found [here](https://www.kaggle.com/datasets/wordsforthewise/lending-club). The data dictionary explaining columns in the tables can be found [here](https://www.kaggle.com/datasets/jonchan2003/lending-club-data-dictionary).

In notebook `Part1_EDA_evaluation_accepted_loans.ipynb`, only the **1st step** is addressed. The company wants a model accurate for both loan classes: accepted and rejected. Therefore, balanced accuracy is aimed to improve. The last month of data is used for testing to better evaluate possible model's performance in the future. Validation is performed on the month before the last.

In notebook `Part2_EDA_evaluation_loan_acceptance.ipynb`, **steps 2 and 3** are addressed. Since grading is an ordered classification task, macro-averaged mean absolute error is tried to minimize to penalize far misclassification of loan grades. Loan subgrade and interest prediction is the last step, which uses loan grades as an input. For subgrades, F1 score is the target metric, while root mean squared error is being minimized in interest rate prediction. The last 6 months of data is used for testing to better evaluate possible model's performance in the future. Validation is performed on the 6 months before the last 6 months.

The **aim** is to analyze accepted and rejected loans and provide insights for the Lending Club on loan acceptance, their grading and interest rates.

The **objectives** are as follows:
* Explore the dataset to identify important features;
* Provide insights for joint loan applications;
* Build models to help automate loan acceptance, loan grading as well as interest rate determination;
* Deploy models to the Google Cloud Platform.

---
Additional notebooks concern initially trying out different algorithms (`lazy_prediction_***.ipynb`) and hyperparameter tuning (`tuning_***.ipynb`).

---
Models are deployed at https://lending-club-service-c4gigp2h5q-lm.a.run.app. 
1. Loan acceptance model can be tested out at https://lending-club-service-c4gigp2h5q-lm.a.run.app/docs#/default/predict_predict_accept_post with example request: `{"loan_amnt":1000,"purpose":"vacation","dti":1.6399999857,"emp_length":3.0,"risk_score":"nan"}`
2. Grade model can be tested out at https://lending-club-service-c4gigp2h5q-lm.a.run.app/docs#/default/predict_predict_grade_post with example request: `{"loan_amnt":25000.0,"term":"60 months","verification_status":"Not Verified","purpose":"credit_card","dti":29.99,"inq_last_6mths":1.0,"all_util":42.0,"total_rev_hi_lim":106200.0,"acc_open_past_24mths":11.0,"bc_open_to_buy":57768.0,"bc_util":30.6,"mo_sin_old_il_acct":230.0,"mo_sin_old_rev_tl_op":128.0,"mths_since_recent_inq":5.0,"num_tl_op_past_12m":4.0,"tot_hi_cred_lim":160293.0,"fico":712.0}`
3. Subgrade and interest rate models canbe tested out at https://lending-club-service-c4gigp2h5q-lm.a.run.app/docs#/default/predict_predict_subgrade_rate_post with example request: `{"loan_amnt":10000.0,"term":"36 months","grade":"C","home_ownership":"MORTGAGE","verification_status":"Source Verified","purpose":"credit_card","dti":26.45,"inq_last_6mths":1.0,"mths_since_last_record":"nan","application_type":"Individual","open_il_12m":1.0,"all_util":86.0,"total_rev_hi_lim":13400.0,"inq_last_12m":5.0,"acc_open_past_24mths":7.0,"bc_open_to_buy":1006.0,"bc_util":89.7,"mo_sin_old_il_acct":218.0,"mo_sin_old_rev_tl_op":39.0,"mort_acc":1.0,"mths_since_recent_inq":2.0,"num_actv_rev_tl":5.0,"num_tl_op_past_12m":3.0,"pct_tl_nvr_dlq":100.0,"tot_hi_cred_lim":290806.0,"fico":662.0}`

The deployment can be also tested with locust by running `locust -f deployment/locust_test.py`.

