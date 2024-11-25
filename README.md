# ParquetDataset for PyTorch

A PyTorch Dataset implementation for efficiently handling large Parquet files.

## Features
- Memory-efficient loading of large Parquet files
- Support for multiple feature columns
- Configurable batch sizes
- Efficient data streaming

## Setup

1. Clone your repository
```bash
git clone https://github.com/yourusername/parquet-dataset.git
cd parquet-dataset
```
2. Install dependencies
```bash
pip install -r requirements.txt
```
3. Use the code directly from the src directory or run the tests using the commands in the section - Testing.

## Testing
Run tests with:

# Run all tests, but use only 10K parameter for test_larger_than_memory_handling
```bash
pytest tests/test_parquet_dataset.py -v -s -k "not test_larger_than_memory_handling or 10K"
```

Replace 10k in the above command with 890M to run the test with a file that is bigger than memory [assuming less than 8GB RAM is available on your machine].