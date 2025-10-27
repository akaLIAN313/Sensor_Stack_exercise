import argparse
import datetime
import pandas as pd
from src.filter import FilterType, filter_data
from src.aggregator import aggregate_data, merge_aggregates, normalize_metric_names

def chunk_reader(file_path: str, chunk_size: int) -> pd.DataFrame:
    """
    Read the data file in chunks.
    Args:
        file_path: The path to the data file.
        chunk_size: The number of rows to read in each chunk.
    Returns:
        A generator of chunks.
    """
    chunks = pd.read_csv(
        file_path,
        dtype={"site": 'category', "device": 'category', "metric": 'category',
            "value": 'float64'
        },
        parse_dates=["time"],
        date_format="%Y-%m-%d %H:%M:%S %z UTC",
        chunksize=chunk_size
    )
    return chunks

def main():
    """
    Main function to run the program.
    Parses the command line arguments, read the data file, filter the data
    based on the arguments, aggregate the data, and save the results to files.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=False,
        default="data/sample_data.csv")
    parser.add_argument("--output_prefix", type=str, required=False,
        default="data/output_")
    parser.add_argument("--site", type=str, required=False)
    parser.add_argument("--device", type=str, required=False)
    parser.add_argument("--metric", type=str, required=False)
    parser.add_argument("--time_start", type=str, required=False)
    parser.add_argument("--time_end", type=str, required=False)
    parser.add_argument("--chunk_size", type=int, required=False,
        default=10000)
    args = parser.parse_args()

    # set up filters
    filter_by = []
    if args.site:
        filter_by.append(FilterType(key="site", value=args.site, value_type=str))
    if args.device:
        filter_by.append(FilterType(key="device", value=args.device,
        value_type=str))
    if args.metric:
        filter_by.append(FilterType(key="metric", value=args.metric,
        value_type=str))
    if args.time_start:
        # Remove " UTC" suffix if present (pandas doesn't parse this format)
        time_str = args.time_start.replace(" UTC", "").strip()
        time_start = pd.to_datetime(time_str, utc=True)
        filter_by.append(FilterType(key="time", value=time_start, 
        value_type=pd.Timestamp, compare_str=">="))
    if args.time_end:
        # Remove " UTC" suffix if present (pandas doesn't parse this format)
        time_str = args.time_end.replace(" UTC", "").strip()
        time_end = pd.to_datetime(time_str, utc=True)
        filter_by.append(FilterType(key="time", value=time_end,
        value_type=pd.Timestamp, compare_str="<="))
        # Read and process data in chunks for memory efficiency
    # for memory efficiency, use category types for the columns that are not
    # converted to datetime, and datetime uses Pandas's parse_dates which is
    # datetime64[ns] under the hood.
    chunks = chunk_reader(args.input, args.chunk_size)
    # Process each chunk: filter and collect filtered data
    agg_data = None
    for chunk_num, chunk in enumerate(chunks):
        print(f"Processing chunk {chunk_num + 1}...")
        # Filter the chunk
        filtered_chunk = filter_data(chunk, filter_by)
        if len(filtered_chunk) > 0:
            # Normalize metric names before aggregation (e.g., 'temp' -> 'temperature')
            filtered_chunk = normalize_metric_names(filtered_chunk, metric_col='metric')
            _, agg_chunk = aggregate_data(filtered_chunk, ["site", "device", "metric"])
        else:
            # No data after filtering
            continue
        if agg_data is None:
            agg_data = agg_chunk
        else:
            agg_data = merge_aggregates(agg_data, agg_chunk, ["site", "device", "metric"])
    agg_data = agg_data.sort_values(by=["site", "device", "metric"])
    # Check if we have aggregated data before saving
    if agg_data is None or len(agg_data) == 0:
        print("No data to save")
        return
    agg_data.to_csv(f"{args.output_prefix}aggregated.csv", index=False)
    top10_avg = agg_data.sort_values(by="value_mean", ascending=False).head(10)
    top10_avg.to_csv(f"{args.output_prefix}top10_avg.csv", index=False)
    top10_std = agg_data.sort_values(by="value_std", ascending=False).head(10)
    top10_std.to_csv(f"{args.output_prefix}top10_std.csv", index=False)
    
    # Detect outliers by re-reading the file
    # Outliers: readings whose value deviates from the mean by more than 3 standard deviations
    print("Detecting outliers...")
    outliers_list = []
    chunks_for_outliers = chunk_reader(args.input, args.chunk_size)
    for chunk in chunks_for_outliers:
        filtered_chunk = filter_data(chunk, filter_by)
        if len(filtered_chunk) > 0:
            filtered_chunk = normalize_metric_names(filtered_chunk, metric_col='metric')
            merged = pd.merge(
                filtered_chunk, 
                agg_data[['site', 'device', 'metric', 'value_mean', 'value_std']], 
                on=["site", "device", "metric"], 
                how="inner"
            )
            # Find outliers where |value - mean| > 3*std
            outlier_mask = abs(merged['value'] - merged['value_mean']) > 3 * merged['value_std']
            outlier_chunk = merged[outlier_mask]
            if len(outlier_chunk) > 0:
                outliers_list.append(outlier_chunk[['time', 'site', 'device', 'metric', 'value', 'unit']])
    if len(outliers_list) > 0:
        outliers = pd.concat(outliers_list, ignore_index=True)
        outliers.to_csv(f"{args.output_prefix}outliers.csv", index=False)
        print(f"Found {len(outliers)} outlier readings")
    else:
        print("No outliers found")


if __name__ == "__main__":
    main()
