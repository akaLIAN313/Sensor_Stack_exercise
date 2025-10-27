import datetime
import pandas as pd

class FilterType():
    """
    A class to represent a filter. Provides a way to trnsform arguments from
    the command line into a filter object to be packed into a query string.
    """
    def __init__(self, key: str, value: any, value_type: type, \
        compare_str: str = None):
        """
        Initialize a FilterType object.
        Args:
            key: The key to filter by.
            value: The value to filter by.
            value_type: The type of the value.
            compare_str: The compare string to use.
        """
        self.key = key
        self.value = value
        self.value_type = value_type
        if compare_str is not None:
            self.compare_str = compare_str
        else:
            self.compare_str = "=="

def filter_data(data: pd.DataFrame,
    filter_by: list[FilterType]) -> pd.DataFrame:
    """
    Filter the data from the source.
    Args:
        data: The data to filter.
        filter_by: The filters to apply.
    Returns:
        The filtered data.
    """
    if len(filter_by) == 0:
        return data
    filter_conditions = []
    query_locals = {}
    for filter in filter_by:
        if filter.key not in data.columns:
            raise ValueError(f"Filter key {filter.key} not found in data")
        elif filter.compare_str is None:
            raise ValueError(f"Filter compare string is not set for {filter.key}")
        else:
            # Handle different value types for proper query string formatting
            if isinstance(filter.value, pd.Timestamp):
                # For Timestamps, use @ to reference the value variable
                value_str = f"@filter_{filter.key}"
                query_locals[f"filter_{filter.key}"] = filter.value
            else:
                # Default to quoted string for other types
                value_str = f"\"{filter.value}\""
            
            filter_conditions.append(
                f"{filter.key} {filter.compare_str} {value_str}")
    filter_query = " & ".join(filter_conditions)
    data = data.query(filter_query, local_dict=query_locals)
    return data