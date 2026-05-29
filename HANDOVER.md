# Loan Default Prediction Handover

## Project Context

This project uses `loan_processed_data.csv` to build a credit risk modeling workflow for loan default prediction.

The intended business framing is account-opening/origination decisioning. The model should estimate default risk at the time of application and could support approval, decline, pricing, or manual-review decisions. It should not be framed as an account-management or post-origination portfolio-monitoring model.

Target definition:

- Source column: `loan_status`
- Positive class: `Charged Off = 1`
- Negative class: `Fully Paid = 0`

The project should emphasize the full modeling pipeline more than raw model performance: data understanding, leakage control, feature selection, validation design, model comparison, optimization, explanation, and governance awareness.

## Dataset Facts Learned

Dataset shape:

- Rows: `987,548`
- Columns: `57`

Target balance:

| Loan status | Count | Share |
| --- | ---: | ---: |
| Fully Paid | 790,652 | 80.06% |
| Charged Off | 196,896 | 19.94% |

The positive class is moderately imbalanced, but not extremely rare.

Date fields use values like `Dec-2015`. Avoid `format="%b-%Y"` on machines where the OS locale is not English, because `%b` depends on localized month names. Use the notebook helper `parse_english_month_year()` instead:

```python
df["issue_date"] = parse_english_month_year(df["issue_d"])
```

Known `issue_d` range:

- Earliest: August 2012
- Latest: December 2018

High-cardinality fields observed:

| Field | Unique values | Initial treatment |
| --- | ---: | --- |
| `emp_title` | 291,220 | Drop initially |
| `title` | 31,747 | Drop initially; overlaps with `purpose` |
| `zip_code` | 938 | Use cautiously; proxy/fair-lending concern |
| `earliest_cr_line` | 717 | Convert to credit history age |

No obvious missingness was found in the checked columns during initial EDA, but the modeling pipeline should still include explicit missing-value checks.

## Important EDA Findings

Default rates by grade were strongly monotonic:

| Grade | Default rate |
| --- | ---: |
| A | 5.72% |
| B | 12.90% |
| C | 22.09% |
| D | 30.17% |
| E | 38.75% |
| F | 45.41% |
| G | 50.41% |

Default rates by term:

| Term | Default rate |
| --- | ---: |
| 36 months | 15.66% |
| 60 months | 32.42% |

Higher-risk purposes included:

- `small_business`
- `moving`
- `medical`
- `other`
- `debt_consolidation`

General borrower/loan patterns:

- Charged-off loans tend to have higher `int_rate`.
- Charged-off loans tend to have higher `dti`.
- Charged-off loans tend to have larger loan amounts.
- Charged-off loans tend to have lower annual income.
- Charged-off loans tend to have lower FICO scores.
- Charged-off loans tend to have higher revolving utilization.

These are useful descriptive findings, but final model features must still pass the origination-availability and governance checks.

## Leakage Lessons

For an origination model, only use features known at application/account opening. Post-origination performance fields must be excluded from the final model.

Drop these from the clean origination model:

- `total_pymnt`
- `total_rec_int`
- `total_rec_late_fee`
- `recoveries`
- `last_pymnt_d`
- `last_pymnt_amnt`
- `last_credit_pull_d`
- `last_fico_range_high`
- `debt_settlement_flag`

Important leakage observations:

- `debt_settlement_flag = Y` is almost always associated with default, so it is not a valid origination feature.
- `last_fico_range_high`, `recoveries`, and payment-related fields are observed after loan performance unfolds.
- These fields can produce unrealistically strong model performance and should only be used in a clearly labeled leakage demonstration.

Additional caution:

- `int_rate`, `grade`, and `sub_grade` are highly predictive, but may be lender underwriting or pricing outputs rather than raw applicant inputs.
- Exclude them from the primary applicant-only origination model.
- Optionally include them in a diagnostic benchmark to show how much predictive signal is embedded in the lender's existing decisioning/pricing process.

Recommended model variants:

| Model variant | Purpose | Valid for final origination decisioning? |
| --- | --- | --- |
| Clean origination-only model | Main model | Yes |
| Model including `int_rate`, `grade`, `sub_grade` | Diagnostic benchmark | Maybe, depending on use case |
| Model including post-origination leakage fields | Leakage demonstration only | No |

## Feature Engineering Plan

Recommended first-pass transformations:

- Convert `term` to numeric months.
- Convert `emp_length` to ordinal numeric values.
- Create `credit_history_months_at_issue` from `issue_d - earliest_cr_line`.
- One-hot encode low-cardinality categorical variables.
- Drop `emp_title` and `title` initially.
- Treat `addr_state` and `zip_code` carefully because they may act as geographic or protected-class proxies.

Candidate low-cardinality categoricals for encoding:

- `term`
- `home_ownership`
- `verification_status`
- `purpose`
- `initial_list_status`
- `application_type`
- `disbursement_method`

Fields to challenge before final use:

- `addr_state`
- `zip_code`
- `grade`
- `sub_grade`
- `int_rate`

## Validation Strategy

Prefer time-aware validation using `issue_d`.

Recommended split logic:

- Train on older vintages.
- Validate on the next time period.
- Test on the most recent mature period.

Why:

- Credit risk changes over time.
- Random splits can overstate performance by mixing old and new vintages.
- Recent loans may be censored, especially 60-month loans that have not had enough time to fully perform.

Validation checks:

- Plot default rate by issue year and issue month.
- Check counts by vintage.
- Check whether recent vintages have artificially low default rates.
- Compare random-split performance against time-split performance as a diagnostic, but do not use random split as the primary final evidence.

## Modeling Plan

Build at least two clean model types:

1. Logistic Regression
   - Interpretable baseline.
   - Useful for coefficient-based explanation.
   - Helps show disciplined credit-risk modeling practice.

2. Gradient boosting or tree-based model
   - Nonlinear benchmark.
   - Candidate libraries: scikit-learn HistGradientBoosting, LightGBM, XGBoost, or CatBoost, depending on availability.
   - Useful for stronger predictive performance and interaction effects.

Imbalance approach:

- Start with class weighting.
- Do not use SMOTE by default.
- If SMOTE is tested, apply it only inside training folds, never before train/test splitting.
- Compare SMOTE against class weighting using PR-AUC, calibration, and time-split validation.

Optimization approach:

- Establish baselines first.
- Tune only the most promising model family.
- Prefer randomized search or time-aware cross-validation over exhaustive grid search.
- Tune parameters such as learning rate, tree depth, number of estimators, regularization, minimum leaf size, and class weights.

## Evaluation Plan

Primary metrics:

- ROC-AUC
- PR-AUC
- Precision
- Recall
- F1
- Confusion matrix
- Calibration curve
- Brier score

Avoid relying on accuracy alone because the target is imbalanced and the business cost of errors is asymmetric.

Threshold analysis:

- Evaluate multiple thresholds rather than assuming `0.5`.
- Tie thresholds to business scenarios:
  - Conservative approval
  - Balanced approval/review strategy
  - High-risk decline or manual-review queue

Stability checks:

- Evaluate performance by issue year.
- Evaluate performance by term.
- Evaluate performance by purpose.
- Evaluate performance by grade band as a diagnostic, even if grade is excluded from the primary model.

Above-and-beyond business analysis:

- Rank applicants by predicted risk.
- Create default rate by predicted-risk decile.
- Add lift/gains chart.
- Simulate approval cutoffs and show the tradeoff between approval volume and expected default rate.

This is likely to impress the audience more than focusing only on a single AUC number because it connects the model to actual credit decisioning.

## Explainability And Governance

Include model explanation at the end of the modeling notebook or presentation.

Recommended explanation tools:

- Logistic Regression coefficients for the interpretable baseline.
- Permutation importance for model-agnostic feature importance.
- SHAP values if available for the tree/boosting model.

Explain results in business language:

- Which features increase estimated default risk?
- Which features decrease estimated default risk?
- Are the top drivers plausible?
- Are any top drivers risky from a regulatory, fairness, or proxy-variable perspective?

Adverse-action discussion:

- In a real credit decisioning context, explanations may need to support adverse action reasons.
- Reason codes should be specific, stable, and understandable.
- Avoid relying on opaque or hard-to-justify features without governance review.

Regulatory and model-risk topics to be prepared to discuss:

- ECOA / Regulation B
- FCRA
- Fair-lending review
- Protected-class proxy risk
- Data lineage
- Feature rationale
- Model validation
- Calibration
- Monitoring and drift
- Performance stability by segment and vintage
- Change management and periodic review

Suggested model documentation contents:

- Model purpose and intended use
- Target definition
- Population and exclusions
- Data sources and lineage
- Feature list and feature rationale
- Leakage review
- Validation design
- Model methodology
- Hyperparameter tuning approach
- Performance results
- Calibration results
- Explainability outputs
- Fair-lending/proxy-variable assessment
- Limitations
- Monitoring plan

## Presentation Plan

Recommended short presentation structure:

1. Business problem and target
2. Data understanding and important statistics
3. Leakage and feature availability
4. Feature engineering
5. Time-aware validation strategy
6. Model comparison and optimization
7. Final model performance and calibration
8. Model interpretation and key drivers
9. Business decisioning / lift analysis
10. Regulatory and model-risk considerations
11. Outstanding questions and future work

Suggested storyline:

- Start with the business use case: origination default risk.
- Show that the target is understandable and moderately imbalanced.
- Demonstrate strong data understanding with grade, term, purpose, and numeric-risk patterns.
- Emphasize leakage discipline as a major modeling judgment.
- Show why time-aware validation is more credible than a random split.
- Compare models, but avoid over-indexing on raw performance.
- End with decisioning thresholds, explainability, governance, and future improvements.

## Outstanding Questions

Questions to resolve or mention as limitations:

- Are `grade`, `sub_grade`, and `int_rate` available before the model decision, or are they outputs of an existing underwriting/pricing process?
- Was the dataset filtered to completed loans only?
- Are recent vintages fully mature, especially 60-month loans?
- What exact business objective should threshold selection optimize?
- Are geographic variables allowed under the intended governance policy?
- Are there costs available for false approvals vs false declines?
- Is the final use case approval/decline, pricing, or manual-review triage?

## Immediate Next Steps

1. Extend the notebook from EDA into modeling.
2. Add clean feature-preparation pipeline.
3. Create time-aware train/validation/test split.
4. Train Logistic Regression baseline.
5. Train one tree/boosting benchmark.
6. Compare clean, diagnostic, and leakage-demo model variants.
7. Add calibration, threshold, and lift/gains analysis.
8. Add explainability section.
9. Prepare the short presentation using the structure above.

## Verification Checklist

Before presenting or handing off:

- Confirm all numeric facts match the notebook outputs.
- Confirm date parsing uses the locale-safe `parse_english_month_year()` helper.
- Confirm post-origination leakage fields are excluded from the final model.
- Confirm the document clearly distinguishes valid features, diagnostic features, and invalid leakage features.
- Confirm validation uses chronological splits for final evidence.
- Confirm SMOTE, if tested, is used only inside training folds.
- Confirm model explanations are included.
- Confirm regulatory and model-risk considerations are discussed, but not overstated as a full model-risk-management document.
