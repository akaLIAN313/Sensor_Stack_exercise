import argparse
import datetime
import pandas as pd
from src.filter import FilterType, filter_data
from src.aggregator import aggregate_data

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
    args = parser.parse_args()

    # read the data
    # for memory efficiency, use category types for the columns that are not
    # converted to datetime, and datatime user Pandas's parse_dates which is
    # datetime64[ns] under the hood.
    data = pd.read_csv(
        args.input,
        dtype={"site": 'category', "device": 'category', "metric": 'category',
            "value": 'float64'
        },
        parse_dates=["time"],
        date_format="%Y-%m-%d %H:%M:%S %z UTC"
    )
    # set up filters and filter the data
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
        filter_by.append(FilterType(key="time_start", value=args.time_start,
        value_type=datetime.datetime, compare_str="<="))
    if args.time_end:
        filter_by.append(FilterType(key="time_end", value=args.time_end,
        value_type=datetime.datetime, compare_str=">="))
    filtered_data = filter_data(data, filter_by)
    # aggregate and savethe data
    _, agg_data = aggregate_data(filtered_data, ["site", "device", "metric"])
    agg_data.to_csv(f"{args.output_prefix}aggregated.csv", index=False)
    top10_avg = agg_data.sort_values(by="value_mean", ascending=False).head(10)
    top10_avg.to_csv(f"{args.output_prefix}top10_avg.csv", index=False)
    top10_std = agg_data.sort_values(by="value_std", ascending=False).head(10)
    top10_std.to_csv(f"{args.output_prefix}top10_std.csv", index=False)


if __name__ == "__main__":
    main()
