#!/usr/bin/env python3

import argparse
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser(
        description="Generate a test Parquet file with u32 and f32 columns"
    )
    parser.add_argument(
        "output_file",
        type=str,
        nargs="?",
        default="test_data.parquet",
        help="Output Parquet file path (default: test_data.parquet)"
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=268_435_456,
        help="Number of rows to generate (default: 268,435,456)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    args = parser.parse_args()

    print(f"Generating Parquet file with {args.rows:,} rows...")
    print(f"  Random seed: {args.seed}")

    # Set random seed for reproducibility
    np.random.seed(args.seed)
    rng = np.random.default_rng()
    numbers = rng.integers(0, 64, size=args.rows, dtype=np.uint32)
    float_values = np.arange(64, dtype=np.float32) / 10.0
    floats = rng.choice(float_values, size=args.rows).astype(np.float32)

    # Create PyArrow table
    table = pa.table({
        'numbers': pa.array(numbers, type=pa.uint32()),
        'floats': pa.array(floats, type=pa.float32())
    })

    # Write to Parquet
    print(f"\nWriting to {args.output_file}...")
    pq.write_table(table, args.output_file)

    # Verify the file
    file_size_mb = pa.parquet.ParquetFile(args.output_file).metadata.serialized_size / (1024 * 1024)
    print(f"✓ File created successfully!")
    print(f"  File size: {file_size_mb:.2f} MB")
    print(f"  Rows: {args.rows:,}")
    print(f"  Columns: numbers (uint32), floats (float32)")

    # Show sample data
    print("\nSample data (first 5 rows):")
    sample_table = pq.read_table(args.output_file, columns=['numbers', 'floats'])
    print(sample_table.to_pandas().head())


if __name__ == "__main__":
    main()
