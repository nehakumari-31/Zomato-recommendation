"""
Data loader for Zomato restaurant dataset.

Handles loading from Hugging Face Datasets, cleaning, and normalization.
"""

import pandas as pd
import numpy as np
from datasets import load_dataset
from typing import Optional, Dict, List, Any
import re


class ZomatoDataLoader:
    """
    Loads and processes the Zomato restaurant dataset.
    
    Handles:
    - Loading from Hugging Face Datasets
    - Column name normalization
    - Data cleaning and validation
    - Price/cost normalization
    - Rating parsing
    """
    
    def __init__(self, dataset_name: str = "ManikaSaini/zomato-restaurant-recommendation"):
        """
        Initialize the data loader.
        
        Args:
            dataset_name: Hugging Face dataset identifier
        """
        self.dataset_name = dataset_name
        self.raw_data: Optional[pd.DataFrame] = None
        self.processed_data: Optional[pd.DataFrame] = None
        
    def load_dataset(self, split: str = "train", cache_dir: Optional[str] = None) -> pd.DataFrame:
        """
        Load dataset from Hugging Face.
        
        Args:
            split: Dataset split to load (default: "train")
            cache_dir: Optional cache directory
            
        Returns:
            Raw DataFrame with original column names
        """
        print(f"Loading dataset: {self.dataset_name} (split: {split})...")
        dataset = load_dataset(self.dataset_name, split=split, cache_dir=cache_dir)
        
        # Convert to pandas DataFrame
        self.raw_data = dataset.to_pandas()
        print(f"Loaded {len(self.raw_data)} rows, {len(self.raw_data.columns)} columns")
        
        return self.raw_data
    
    def normalize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize column names for code-friendly access.
        
        Converts:
        - `listed_in(city)` -> `listed_in_city`
        - `approx_cost(for two people)` -> `approx_cost_for_two`
        
        Args:
            df: DataFrame with original column names
            
        Returns:
            DataFrame with normalized column names
        """
        df = df.copy()
        
        # Map problematic column names
        column_mapping = {
            'listed_in(city)': 'listed_in_city',
            'approx_cost(for two people)': 'approx_cost_for_two',
            'listed_in(type)': 'listed_in_type'
        }
        
        df.rename(columns=column_mapping, inplace=True)
        
        return df
    
    def parse_rate(self, rate_str: str) -> Optional[float]:
        """
        Parse rating from string format like "4.1/5" or "NEW" or "-".
        
        Args:
            rate_str: Rating string
            
        Returns:
            Numeric rating (0-5) or None if invalid
        """
        if pd.isna(rate_str) or rate_str in ["NEW", "-", "", "nan"]:
            return None
        
        # Extract numeric part (support optional leading minus)
        # Examples: "4.1/5" -> 4.1, "-1/5" -> -1.0
        match = re.search(r'(-?\d+\.?\d*)', str(rate_str))
        if match:
            try:
                rating = float(match.group(1))
                # Clamp to 0-5 range
                return min(max(rating, 0.0), 5.0)
            except ValueError:
                return None
        
        return None
    
    def parse_cost(self, cost_str: str) -> Optional[float]:
        """
        Parse cost from string to numeric value.
        
        Handles:
        - Numeric strings: "500" -> 500.0
        - Ranges: "500-1000" -> 750.0 (midpoint) or None
        - Non-numeric: None
        
        Args:
            cost_str: Cost string
            
        Returns:
            Numeric cost value or None if invalid
        """
        if pd.isna(cost_str) or cost_str in ["", "nan"]:
            return None
        
        cost_str = str(cost_str).strip()
        
        # Remove commas and currency symbols
        cost_str = re.sub(r'[,\s₹$]', '', cost_str)
        
        # Try to extract numeric value(s)
        numbers = re.findall(r'\d+', cost_str)
        
        if not numbers:
            return None
        
        try:
            if len(numbers) == 1:
                # Single value
                return float(numbers[0])
            elif len(numbers) == 2:
                # Range: take midpoint
                return (float(numbers[0]) + float(numbers[1])) / 2.0
            else:
                # Multiple numbers: take first
                return float(numbers[0])
        except ValueError:
            return None
    
    def normalize_cuisines(self, cuisines_str: str) -> List[str]:
        """
        Parse and normalize comma-separated cuisines.
        
        Args:
            cuisines_str: Comma-separated cuisine string
            
        Returns:
            List of normalized cuisine names
        """
        if pd.isna(cuisines_str) or cuisines_str == "":
            return []
        
        # Split by comma and clean
        cuisines = [c.strip() for c in str(cuisines_str).split(',')]
        # Remove empty strings
        cuisines = [c for c in cuisines if c]
        
        return cuisines
    
    def clean_and_validate(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Clean and validate the dataset.
        
        Performs:
        - Column name normalization
        - Rate parsing
        - Cost parsing
        - Cuisine normalization
        - City normalization
        
        Args:
            df: Optional DataFrame (uses self.raw_data if None)
            
        Returns:
            Cleaned DataFrame
        """
        if df is None:
            if self.raw_data is None:
                raise ValueError("No data loaded. Call load_dataset() first.")
            df = self.raw_data
        
        print("Cleaning and validating data...")
        
        # Normalize column names
        df = self.normalize_column_names(df)
        
        # Parse ratings
        if 'rate' in df.columns:
            df['rating_numeric'] = df['rate'].apply(self.parse_rate)
            print(f"  - Parsed ratings: {df['rating_numeric'].notna().sum()} valid ratings")
        
        # Parse costs
        if 'approx_cost_for_two' in df.columns:
            df['cost_numeric'] = df['approx_cost_for_two'].apply(self.parse_cost)
            print(f"  - Parsed costs: {df['cost_numeric'].notna().sum()} valid costs")
        
        # Normalize cuisines (store as list)
        if 'cuisines' in df.columns:
            df['cuisines_list'] = df['cuisines'].apply(self.normalize_cuisines)
            print(f"  - Parsed cuisines: {df['cuisines_list'].apply(len).sum()} total cuisine entries")
        
        # Normalize city (trim and handle case)
        if 'listed_in_city' in df.columns:
            df['city_normalized'] = df['listed_in_city'].astype(str).str.strip()
            unique_cities = df['city_normalized'].nunique()
            print(f"  - Found {unique_cities} unique cities")
        
        # Ensure votes is numeric
        if 'votes' in df.columns:
            df['votes'] = pd.to_numeric(df['votes'], errors='coerce').fillna(0).astype(int)
        
        self.processed_data = df
        print(f"Cleaning complete. Final dataset: {len(df)} rows")
        
        return df
    
    def get_processed_data(self) -> pd.DataFrame:
        """
        Get the processed dataset.
        
        Returns:
            Processed DataFrame
            
        Raises:
            ValueError: If data hasn't been loaded and processed
        """
        if self.processed_data is None:
            raise ValueError("Data not processed. Call load_dataset() and clean_and_validate() first.")
        
        return self.processed_data
    
    def get_unique_cities(self) -> List[str]:
        """Get list of unique areas/localities in the dataset."""
        if self.processed_data is None:
            raise ValueError("Data not processed.")
        
        # Return the actual areas/localities from the 'city_normalized' column
        # These are the real values that users should select from
        if 'city_normalized' in self.processed_data.columns:
            areas = self.processed_data['city_normalized'].dropna().unique().tolist()
            # Filter out any non-area entries and sort
            valid_areas = [area for area in areas if isinstance(area, str) and len(area.strip()) > 0]
            return sorted(valid_areas)
        
        return ['Bangalore']  # Fallback
    
    def get_unique_cuisines(self) -> List[str]:
        """Get list of unique cuisines in the dataset."""
        if self.processed_data is None:
            raise ValueError("Data not processed.")
        
        all_cuisines = set()
        for cuisine_list in self.processed_data['cuisines_list'].dropna():
            all_cuisines.update(cuisine_list)
        
        return sorted(list(all_cuisines))
    
    def get_price_ranges(self) -> Dict[str, Any]:
        """
        Get statistics about price ranges in the dataset.
        
        Returns:
            Dictionary with min, max, mean, median prices
        """
        if self.processed_data is None:
            raise ValueError("Data not processed.")
        
        cost_col = 'cost_numeric'
        if cost_col not in self.processed_data.columns:
            return {}
        
        valid_costs = self.processed_data[cost_col].dropna()
        
        if len(valid_costs) == 0:
            return {}
        
        return {
            'min': float(valid_costs.min()),
            'max': float(valid_costs.max()),
            'mean': float(valid_costs.mean()),
            'median': float(valid_costs.median()),
            'count': int(len(valid_costs))
        }
