import pytest
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader

from pathlib import Path
import sys
import psutil
import time

from src.parquet_dataset import ParquetDataset

@pytest.fixture
def sample_parquet_file(tmp_path):
    """Create a sample parquet file for testing"""
    # Create sample data
    n_rows = 10000
    df = pd.DataFrame({
        'feature1': np.random.randn(n_rows),
        'feature2': np.random.randn(n_rows),
        'target': np.random.randint(0, 2, n_rows)
    })
    
    # Save to parquet
    parquet_path = tmp_path / "test.parquet"
    df.to_parquet(parquet_path)
    return str(parquet_path)

def test_dataset_length(sample_parquet_file):
    dataset = ParquetDataset(
        sample_parquet_file,
        feature_columns=['feature1', 'feature2'],
        target_column='target'
    )
    assert len(dataset) == 10000
    


@pytest.mark.parametrize("target_size", [
    pytest.param(100_000, id="100K-rows"),
    pytest.param(1_000_000, id="1M-rows"),
    pytest.param(10_000_000, id="10M-rows"),
    pytest.param(890_000_000, id="890M-rows")
])
def test_larger_than_memory_handling(tmp_path, target_size):
    """Test handling of a file larger than available RAM"""
    target_n_rows = target_size
    chunk_size = min(100_000, target_size)
    parquet_path = tmp_path / "large_test.parquet"
    
    print("Creating parquet file in chunks...")
    
    # Initialize writer first (don't write first chunk separately)
    first_chunk = pd.DataFrame({
        'feature1': np.random.randn(chunk_size),
        'feature2': np.random.randn(chunk_size),
        'target': np.random.randint(0, 2, chunk_size)
    })
    
    # Convert to PyArrow table to get schema
    table = pa.Table.from_pandas(first_chunk)
    
    with pq.ParquetWriter(parquet_path, table.schema) as writer:
        writer.write_table(table)
    
        # Free memory
        del first_chunk, table
        
        # Write remaining chunks
        rows_written = chunk_size
        
        while rows_written < target_n_rows:
            remaining_rows = target_n_rows - rows_written
            chunk_size_current = min(chunk_size, remaining_rows)
            
            chunk_df = pd.DataFrame({
                'feature1': np.random.randn(chunk_size_current),
                'feature2': np.random.randn(chunk_size_current),
                'target': np.random.randint(0, 2, chunk_size_current)
            })
            
            # Convert to PyArrow table and write
            table = pa.Table.from_pandas(chunk_df)
            writer.write_table(table)
            
            # Update progress
            rows_written += chunk_size_current
            
            # Free memory
            del chunk_df, table
            
            # Print progress every 10%
            if rows_written % (target_n_rows // 10) < chunk_size:
                print(f"Progress: {rows_written / target_n_rows * 100:.1f}%")
    
    print(f"Finished writing {rows_written:,} rows")
    actual_size_mb = parquet_path.stat().st_size / (1024 * 1024)
    print(f"Actual file size: {actual_size_mb:.1f}MB")
    
    print("\nTesting complete DataLoader iteration...")
    
    # Create dataset and dataloader
    dataset = ParquetDataset(
        str(parquet_path),
        feature_columns=['feature1', 'feature2'],
        target_column='target',
        batch_size=1000
    )
    
    batch_size = 1000
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        # shuffle=True,
        num_workers=4,
        persistent_workers=True
    )
    
    start_time = time.time()
    total_rows_processed = 0
    total_batches_processed = 0
    
    # Iterate through the entire dataset
    for batch_idx, (features, targets) in enumerate(dataloader):

        if batch_idx > 10:
            break

        # Verify batch properties
        assert features.shape[0] <= batch_size
        assert features.shape[1] == 2
        assert targets.shape[0] <= batch_size
        assert targets.shape[1] == 1
        
        # Verify data integrity
        assert not torch.isnan(features).any()
        assert not torch.isnan(targets).any()
        
        total_rows_processed += len(features)
        total_batches_processed += 1
        
        # Print progress every 10%
        if batch_idx % (len(dataloader) // 25) == 0:
            elapsed_time = time.time() - start_time
            progress = total_rows_processed / target_n_rows * 100
            print(f"Progress: {progress:.1f}% | "
                  f"Rows: {total_rows_processed:,}/{target_n_rows:,} | "
                  f"Time: {elapsed_time:.1f}s")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Verify we processed all rows
    assert total_rows_processed == batch_size * total_batches_processed, (
            f"Processed {total_rows_processed:,} rows, "
            f"expected {batch_size * total_batches_processed:,}"
        )
    
    # Calculate and print statistics    
    print("\nTest Results:")
    print(f"Total rows processed: {total_rows_processed:,}")
    print(f"Total batches processed: {total_batches_processed:,}")
    print(f"Total time: {total_time:.2f}s")