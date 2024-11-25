import torch
from torch.utils.data import Dataset
import duckdb
import pyarrow.parquet as pq
from typing import List, Union, Optional

class ParquetDataset(Dataset):
    def __init__(
        self,
        parquet_path: str,
        feature_columns: List[str],
        target_column: Optional[str] = None,
        batch_size: int = 1000,
        transform=None
    ):
        """
        Initialize ParquetDataset for handling large Parquet files
        
        Args:
            parquet_path: Path to the parquet file
            feature_columns: List of column names to use as features
            target_column: Column name to use as target (optional)
            batch_size: Number of rows to load at once
            transform: Optional transform to apply to features
        """
        self.parquet_path = parquet_path
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.batch_size = batch_size
        self.transform = transform
        
        # Initialize DuckDB connection
        self.conn = duckdb.connect()
        
        # Get total number of rows without loading entire file
        self.total_rows = self._get_total_rows()
        
    def _get_total_rows(self) -> int:
        """Get total number of rows in parquet file"""
        return self.conn.execute(
            f"SELECT COUNT(*) FROM read_parquet('{self.parquet_path}')"
        ).fetchone()[0]
    
    def __len__(self) -> int:
        return self.total_rows
    
    def __getitem__(self, idx: int) -> Union[torch.Tensor, tuple]:
        # Calculate batch boundaries
        start_idx = (idx // self.batch_size) * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.total_rows)
        
        # Create SQL query to fetch specific rows
        columns = ", ".join(self.feature_columns)
        if self.target_column:
            columns += f", {self.target_column}"
            
        query = f"""
        SELECT {columns}
        FROM read_parquet('{self.parquet_path}')
        LIMIT {self.batch_size}
        OFFSET {start_idx}
        """
        
        # Execute query and convert to tensors
        result = self.conn.execute(query).fetch_arrow_table()
        
        # Convert features to tensor
        features = torch.tensor(
            result.select(self.feature_columns).to_pandas().values,
            dtype=torch.float32
        )
        
        # Apply transform if specified
        if self.transform:
            features = self.transform(features)
        
        # If target column specified, return features and target
        if self.target_column:
            targets = torch.tensor(
                result.select([self.target_column]).to_pandas().values,
                dtype=torch.float32
            )
            return features[idx - start_idx], targets[idx - start_idx]
        
        return features[idx - start_idx]
    
    def __del__(self):
        """Clean up DuckDB connection"""
        if hasattr(self, 'conn'):
            self.conn.close()