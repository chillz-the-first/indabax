import pandas as pd

def report(df):
    """
    Print a quick diagnostic summary of a DataFrame.

    Shows shape, duplicate count, missing-value counts per column,
    the first/last 5 rows, and the column dtypes via df.info().
    Useful as a "before and after" snapshot when cleaning.
    """
    print(f"Total rows: {df.shape[0]}, Total columns: {df.shape[1]}")

    total_dupes = df.duplicated().sum()
    print(f"\nDuplicate rows: {total_dupes}")

    null_values = df.isnull().sum()
    null_values = null_values[null_values != 0]
    if len(null_values) > 0:
        print("\nMissing values:")
        for col, val in null_values.items():
                print(f"{col}: {val} missing values")
    else:
        print("\nNo missing values")

    print()
    df.info()

def find_outliers(df, col):
    """
    Identify outliers in a numeric column using the IQR method.

    An outlier is any value below Q1 - 1.5*IQR or above Q3 + 1.5*IQR.
    Returns the subset of rows that are outliers (callers can decide what
    to do with them) and prints mean/median/count for quick comparison.
    """
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].reset_index(drop=True)

    print(f"Mean: {df[col].mean()}")
    print(f"Median: {df[col].median()}")
    print(f"Outliers found: {len(outliers)}")