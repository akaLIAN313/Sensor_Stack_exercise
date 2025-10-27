import datetime
import pandas as pd
import numpy as np


def aggregate_data(data: pd.DataFrame,
    group_by: list[str]) -> pd.DataFrame:
    """
    Aggregate the data from the source.
    includes count, mean, min, max, std, and sum of squares.
    Args:
        data: The data to aggregate.
        group_by: List of columns to group by
    Returns:
        Grouped data and aggregated data
    """
    grouped_data = data.groupby(group_by, observed=True)
    agg_data = grouped_data.agg(
        value_count=('value', 'count'),
        value_mean=('value', 'mean'),
        value_min=('value', 'min'),
        value_max=('value', 'max'),
        value_std=('value', 'std'),
        value_sum_sq=('value', lambda x: (x ** 2).sum())
    ).reset_index()
    
    return grouped_data, agg_data

def merge_aggregates(agg1: pd.DataFrame, agg2: pd.DataFrame, group_by: list[str]) -> pd.DataFrame:
    """
    Merge two aggregated dataframes by computing combined statistics.
    Uses sum of squares to correctly compute mean and std for merged data.
    
    Args:
        agg1: First aggregated dataframe
        agg2: Second aggregated dataframe  
        group_by: List of columns to group by
    
    Returns:
        Merged aggregated dataframe with correct statistics
    """
    # Outer merge to get all combinations
    merged = pd.merge(
        agg1.fillna(0), 
        agg2.fillna(0), 
        on=group_by, 
        how='outer', 
        suffixes=('_1', '_2')
    )
    merged = merged.fillna(0)
    # Combine counts
    merged['value_count'] = merged['value_count_1'] + merged['value_count_2']
    # Combine means using weighted average: (n1*mean1 + n2*mean2) / (n1+n2)
    total_count = merged['value_count']
    merged['value_mean'] = (
        merged['value_count_1'] * merged['value_mean_1'] + 
        merged['value_count_2'] * merged['value_mean_2']
    ) / total_count
    merged.loc[total_count == 0, 'value_mean'] = 0
    # Combine min and max
    merged['value_min'] = merged[['value_min_1', 'value_min_2']].min(axis=1, skipna=True)
    merged['value_max'] = merged[['value_max_1', 'value_max_2']].max(axis=1, skipna=True)
    # Combine sum of squares for proper std calculation
    merged['value_sum_sq'] = merged['value_sum_sq_1'] + merged['value_sum_sq_2']
    # Compute std from sum of squares: sqrt((sum(x^2) - n*mean^2) / (n-1))
    # Using ddof=1 for sample standard deviation
    merged['value_std'] = np.sqrt(
        (merged['value_sum_sq'] - total_count * merged['value_mean']**2) / np.maximum(total_count - 1, 1)
    )
    # Keep only the final columns
    result_cols = group_by + ['value_count', 'value_mean', 'value_min', 'value_max', 'value_std', 'value_sum_sq']
    result = merged[result_cols].copy()
    return result