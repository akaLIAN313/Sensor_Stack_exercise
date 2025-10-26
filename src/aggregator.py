import datetime
import pandas as pd

agg_functions = {
    "value": ["count", "mean", "min", "max", "std"],
}

def aggregate_data(data: pd.DataFrame,
    group_by: list[str]) -> pd.DataFrame:
    """
    Aggregate the data from the source.
    """
    grouped_data = data.groupby(group_by, observed=True)
    agg_data = grouped_data.agg(agg_functions)
    agg_data.columns = ['_'.join(col).strip() for col in agg_data.columns.values]
    agg_data.reset_index(inplace=True)
    return grouped_data, agg_data