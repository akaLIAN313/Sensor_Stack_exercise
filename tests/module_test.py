import unittest
import pandas as pd
import sys
import os

# Add the src directory to the path so we can import from src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestChunkConsistency(unittest.TestCase):
    """
    Test that chunked aggregation produces consistent results regardless of chunk size.
    """
    
    def test_chunk_consistency(self):
        """
        Test that results with different chunk sizes produce identical aggregated data.
        """
        os.system("python ../main.py --output_prefix ../data/chunk_1000_ \
            --input ../data/sample_data.csv --chunk_size 1000 > /dev/null 2>&1")
        os.system("python ../main.py --output_prefix ../data/chunk_100000_ \
            --input ../data/sample_data.csv --chunk_size 100000 > /dev/null 2>&1")
        # Read the two aggregated CSV files
        file1 = "../data/chunk_1000_aggregated.csv"
        file2 = "../data/chunk_100000_aggregated.csv"
        # Read both CSV files
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)
        # Sort both dataframes by grouping columns for comparison
        sort_cols = ['site', 'device', 'metric']
        if all(col in df1.columns for col in sort_cols):
            df1 = df1.sort_values(by=sort_cols).reset_index(drop=True)
        if all(col in df2.columns for col in sort_cols):
            df2 = df2.sort_values(by=sort_cols).reset_index(drop=True)
        # Check that both dataframes have the same shape
        self.assertEqual(df1.shape, df2.shape, 
                        f"Dataframes have different shapes: {df1.shape} vs {df2.shape}")
        # Compare each row
        for idx, (row1, row2) in enumerate(zip(df1.iterrows(), df2.iterrows())):
            _, data1 = row1
            _, data2 = row2
            # Compare each column value
            for col in df1.columns:
                val1 = data1[col]
                val2 = data2[col]
                # Use assertAlmostEqual for numeric columns (handles floating point precision)
                if pd.api.types.is_numeric_dtype(df1[col]) and not pd.api.types.is_integer_dtype(df1[col]):
                    self.assertAlmostEqual(
                        val1, val2, delta=1e-6,
                        msg=f"Row {idx}, column {col}: {val1} != {val2}"
                    )
                else:
                    # Use assertEqual for non-numeric columns
                    self.assertEqual(
                        val1, val2,
                        msg=f"Row {idx}, column {col}: {val1} != {val2}"
                    )
    
if __name__ == '__main__':
    unittest.main()
