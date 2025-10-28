import pandas as pd
import numpy as np


class GroupEstimate:
    def __init__(self, estimate="mean"):
        if estimate not in ["mean", "median"]:
            raise ValueError("estimate must be either 'mean' or 'median'")
        self.estimate = estimate
        self.group_estimates = None

    
    def fit(self, X, y, default_category=None):
        # Combine X and y
        df = X.copy()
        df["target"] = y

        # Store the column names for later
        self.columns = list(X.columns)
        self.default_category = default_category

        # Group by all categorical columns
        grouped = df.groupby(self.columns, observed=True)["target"]

        # Compute mean or median
        if self.estimate == "mean":
            self.group_estimates = grouped.mean()
        else:
            self.group_estimates = grouped.median()

        # If default_category is provided, calculate fallback group
        if default_category:
            self.category_defaults = (
                df.groupby(default_category)["target"]
                .mean() if self.estimate == "mean" else
                df.groupby(default_category)["target"].median()
            )
        else:
            self.category_defaults = None
        return self

    def predict(self, X_):
        # Convert to DataFrame if needed
        if not isinstance(X_, pd.DataFrame):
            X_ = pd.DataFrame(X_, columns=self.columns)

        preds = []
        missing_count = 0

        for _, row in X_.iterrows():
            key = tuple(row[col] for col in self.columns)

            # Check if exact group exists
            if key in self.group_estimates.index:
                preds.append(self.group_estimates.loc[key])
            else:
                # Try using default category if available
                if self.default_category and row[self.default_category] in self.category_defaults.index:
                    preds.append(self.category_defaults.loc[row[self.default_category]])
                else:
                    preds.append(np.nan)
                    missing_count += 1

        if missing_count > 0:
            print(f"{missing_count} missing group(s) encountered. Returned NaN for them.")
        return np.array(preds)
        

