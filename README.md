# Experian Loan Default Risk Case Study

This project develops a credit risk modeling workflow for predicting loan default risk at origination/account opening.

The focus is the full modeling pipeline rather than maximizing a single performance metric:

- data understanding and EDA
- target definition
- leakage review
- origination-safe feature engineering
- time-aware validation
- model comparison and tuning
- calibration and threshold analysis
- model explanation
- governance and regulatory considerations

## Files

- `notebook.ipynb` - EDA and modeling workflow notebook.
- `HANDOVER.md` - project handover notes, findings, assumptions, and next steps.
- `.gitignore` - excludes local data, virtual environments, and generated artifacts.

## Data

The source file is expected locally as:

```text
loan_processed_data.csv
```

The CSV is intentionally excluded from Git because it is large and should not be committed to GitHub.

## Target

The target column is `loan_status`:

- `Charged Off = 1`
- `Fully Paid = 0`

The intended modeling context is account-opening/origination decisioning, not post-origination account management.

## Run

Open `notebook.ipynb` and run the cells in order. The notebook includes an optional `MODEL_SAMPLE_SIZE` setting in the modeling section for faster iteration.

```python
MODEL_SAMPLE_SIZE = None
```

Set it to an integer such as `250_000` if you want a quicker demo run.

