import unittest
import pandas as pd
import sys
import os
import numpy as np

# Add the src directory to the path so we can import aggregator
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from aggregator import aggregate_data


class TestAggregator(unittest.TestCase):
    
    def generate_sensor_data(self, n_sites: int, n_timestamps: int) -> pd.DataFrame:
        """
        Generate sensor data for testing.
        2*n devices in n sites, each have m data timed evenly from 
        2025-01-01 00:00 +0000UTC to 2025-11-01 00:00 +0000UTC 
        with mean value same as the device id for for each metric:
        (temp and humidity)
        Args:
            n_sites: Number of sites (will create 2*n_sites devices)
            n_timestamps: Number of timestamps per device (evenly spaced)
        
        Returns:
            DataFrame with columns: time, site, device, metric, unit, value
        """
        # Generate time range
        start_time = pd.Timestamp('2025-01-01 00:00:00', tz='UTC')
        end_time = pd.Timestamp('2025-11-01 00:00:00', tz='UTC')
        timestamps = pd.date_range(start=start_time, end=end_time, periods=n_timestamps)
        
        # Set seed for reproducibility
        np.random.seed(42)
        
        df = pd.DataFrame(columns=['time', 'site', 'device', 'metric', 'unit', 'value'])
        expected_values = {}
        for site_num in range(n_sites):
            site_name = f"site_{site_num}"
            # Create 2 devices per site
            for device_in_site in range(2):
                device_id = 2*site_num + device_in_site
                expected_values[device_id] = {}
                for metric_id in range(2):
                    device_name = f"device_{device_id:03d}"
                    device_timestamps = timestamps.copy()
                    device_times = [t.strftime('%Y-%m-%d %H:%M:%S +0000 UTC') for t in device_timestamps]
                    # Generate values with mean = device_id * metric_id
                    device_values = np.random.normal(
                        loc=device_id * metric_id,
                        scale=1.0,
                        size=n_timestamps
                    )
                    expected_values[device_id][metric_id] = {
                        'count': n_timestamps,
                        'mean': device_values.mean(),
                        'min': device_values.min(),
                        'max': device_values.max(),
                        'std': device_values.std()
                    }
                    device_data = pd.DataFrame({
                        'time': device_times,
                        'site': [site_name] * n_timestamps,
                        'device': [device_name] * n_timestamps,
                        'metric': [f"m{metric_id}"] * n_timestamps,
                        'unit': ['unit'] * n_timestamps,
                        'value': device_values
                    })
                    df = pd.concat([df, device_data], ignore_index=True)
        # Sort by time, site, device, metric for cleaner output
        df = df.sort_values(['time', 'site', 'device', 'metric']).reset_index(drop=True)
        return expected_values, df

    def test_basic_aggregation(self):
        """
        Test basic aggregation functionality with multiple groups and values.
        """
        n_sites = 2
        n_timestamps = 10
        expected_values, data = self.generate_sensor_data(n_sites=n_sites, n_timestamps=n_timestamps)
        grouped_data, agg_data = aggregate_data(data, ['site', 'device', 'metric'])
        print(agg_data)
        for site in range(n_sites):
            site_name = f"site_{site}"
            for device_in_site in range(2):
                device_id = 2*site + device_in_site
                device_name = f"device_{device_id:03d}"
                for metric_id in range(2):
                    metric = f"m{metric_id}"
                    self.assertEqual(
                        agg_data[
                            (agg_data['site'] == site_name) 
                            & (agg_data['device'] == device_name)
                            & (agg_data['metric'] == metric)].iloc[0]['value_count'],
                        n_timestamps
                    )
                    self.assertAlmostEqual(
                        agg_data[
                            (agg_data['site'] == site_name) 
                            & (agg_data['device'] == device_name)
                            & (agg_data['metric'] == metric)].iloc[0]['value_mean'],
                        expected_values[device_id][metric_id]['mean'],
                        places=7
                    )
                    self.assertAlmostEqual(
                        agg_data[
                            (agg_data['site'] == site_name) 
                            & (agg_data['device'] == device_name)
                            & (agg_data['metric'] == metric)].iloc[0]['value_min'],
                        expected_values[device_id][metric_id]['min'],
                        places=7
                    )
                    self.assertAlmostEqual(
                        agg_data[
                            (agg_data['site'] == site_name) 
                            & (agg_data['device'] == device_name)
                            & (agg_data['metric'] == metric)].iloc[0]['value_max'],
                        expected_values[device_id][metric_id]['max'],
                        places=7
                    )
    


if __name__ == '__main__':
    unittest.main()
