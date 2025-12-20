"""
Utility to load parquet datasets and display first few rows.
"""

import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
import json
from typing import Union, Optional, List, Dict, Any
import os

def load_parquet_dataset(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load a parquet dataset from local directory.
    
    Args:
        file_path: Path to the parquet file
        
    Returns:
        DataFrame containing the loaded data
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file is not a valid parquet file
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {file_path}")
    
    try:
        # Load the parquet file
        df = pd.read_parquet(path)
        # 输出数据集中的列
        # print(f"Columns: {df.columns.tolist()}")
        print(f"Rows: {len(df)}")
        return df
    except Exception as e:
        raise ValueError(f"Error loading parquet file: {e}")


def display_first_rows(
    df: pd.DataFrame, 
    n_rows: int = 5,
    columns: Optional[list] = None
) -> None:
    """
    Display first few rows of the dataset.
    
    Args:
        df: DataFrame to display
        n_rows: Number of rows to display (default: 5)
        columns: Specific columns to display (default: all columns)
    """
    print(f"Dataset shape: {df.shape}")
    
    if columns:
        df = df[columns]
    
    print(f"\nFirst {n_rows} rows:")
    print("=" * 50)
    print(df.head(n_rows).to_string(index=False))


def load_and_display_parquet(
    file_path: Union[str, Path],
    n_rows: int = 5,
    columns: Optional[list] = None
) -> pd.DataFrame:
    """
    Load a parquet dataset and display first few rows.
    
    Args:
        file_path: Path to the parquet file
        n_rows: Number of rows to display (default: 5)
        columns: Specific columns to display (default: all columns)
        
    Returns:
        DataFrame containing the loaded data
    """
    # Load the dataset
    df = load_parquet_dataset(file_path)
    
    # Display information about the dataset
    print(f"Loaded dataset from: {file_path}")
    
    # Display first few rows
    display_first_rows(df, n_rows, columns)
    
    return df


def extract_columns_to_json(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    output_path: Union[str, Path] = None,
    orient: str = "records"
) -> Union[List[Dict[str, Any]], str]:
    """
    Extract specified columns from dataframe and convert to JSON.
    
    Args:
        df: Input DataFrame
        columns: List of column names to extract (None for all columns)
        output_path: Path to save JSON file (if None, returns JSON string)
        orient: Format of JSON string (default: "records")
        
    Returns:
        Either JSON string/list or path to saved file
    """
    # Select columns if specified
    if columns:
        # Check if all columns exist
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Columns not found in dataset: {missing_cols}")
        df_selected = df[columns]
    else:
        df_selected = df.copy()
    
    # Convert to JSON
    json_data = df_selected.to_json(orient=orient, force_ascii=False, indent=2)
    
    # Save to file if output_path is provided
    if output_path:
        output_path = Path(output_path)
        # Create directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(json_data)
        return str(output_path)
    else:
        # Return parsed JSON data
        return json.loads(json_data)

if __name__ == "__main__":
    base_dir="/mnt/n0/datasets/ragbench"
    # 输入路径列表：基础路径下的每个一级子文件夹内后缀为parquet的二级子文件
    input_paths = [
        f"{base_dir}/{folder}/{sub_folder}"
        for folder in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, folder))
                for sub_folder in os.listdir(os.path.join(base_dir, folder))
                    if sub_folder.endswith(".parquet")
    ]


    # 输出路径列表：遍历输入路径列表，将后缀改为json
    output_paths = [
        f"{path.replace('.parquet', '.json')}"
        for path in input_paths
    ]

    print(f"Input paths: {input_paths}")
    print(f"Output paths: {output_paths}")

    for input_path, output_path in zip(input_paths, output_paths):
        print(f"Processing {input_path}")
        if not Path(input_path).exists():
            raise FileNotFoundError(f"Parquet file not found: {input_path}")
        df=load_parquet_dataset(input_path)
        extract_columns_to_json(df=df, output_path=output_path, columns=["question", "documents", "response"])
