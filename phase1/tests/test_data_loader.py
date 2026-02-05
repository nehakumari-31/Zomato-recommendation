"""
Test cases for ZomatoDataLoader.

Tests cover:
- Dataset loading
- Column name normalization
- Rate parsing
- Cost parsing
- Cuisine normalization
- Data cleaning and validation
"""

import pytest
import pandas as pd
import numpy as np
from phase1.data_loader import ZomatoDataLoader


class TestZomatoDataLoader:
    """Test suite for ZomatoDataLoader."""
    
    def test_init(self):
        """Test initialization."""
        loader = ZomatoDataLoader()
        assert loader.dataset_name == "ManikaSaini/zomato-restaurant-recommendation"
        assert loader.raw_data is None
        assert loader.processed_data is None
    
    def test_normalize_column_names(self):
        """Test column name normalization."""
        loader = ZomatoDataLoader()
        
        # Create test DataFrame with problematic column names
        df = pd.DataFrame({
            'name': ['Restaurant A'],
            'listed_in(city)': ['Bangalore'],
            'approx_cost(for two people)': ['500'],
            'listed_in(type)': ['Casual Dining']
        })
        
        normalized = loader.normalize_column_names(df)
        
        assert 'listed_in_city' in normalized.columns
        assert 'approx_cost_for_two' in normalized.columns
        assert 'listed_in_type' in normalized.columns
        assert 'listed_in(city)' not in normalized.columns
        assert 'approx_cost(for two people)' not in normalized.columns
    
    def test_parse_rate_valid(self):
        """Test parsing valid rating strings."""
        loader = ZomatoDataLoader()
        
        assert loader.parse_rate("4.1/5") == 4.1
        assert loader.parse_rate("4/5") == 4.0
        assert loader.parse_rate("5.0/5") == 5.0
        assert loader.parse_rate("0/5") == 0.0
        assert loader.parse_rate("3.75/5") == 3.75
    
    def test_parse_rate_invalid(self):
        """Test parsing invalid rating strings."""
        loader = ZomatoDataLoader()
        
        assert loader.parse_rate("NEW") is None
        assert loader.parse_rate("-") is None
        assert loader.parse_rate("") is None
        assert loader.parse_rate(None) is None
        assert loader.parse_rate(np.nan) is None
    
    def test_parse_rate_edge_cases(self):
        """Test edge cases for rate parsing."""
        loader = ZomatoDataLoader()
        
        # Rating above 5 should be clamped
        assert loader.parse_rate("6/5") == 5.0
        # Rating below 0 should be clamped
        assert loader.parse_rate("-1/5") == 0.0
    
    def test_parse_cost_single_value(self):
        """Test parsing single cost value."""
        loader = ZomatoDataLoader()
        
        assert loader.parse_cost("500") == 500.0
        assert loader.parse_cost("1000") == 1000.0
        assert loader.parse_cost("1,500") == 1500.0
        assert loader.parse_cost("₹500") == 500.0
        assert loader.parse_cost("$50") == 50.0
    
    def test_parse_cost_range(self):
        """Test parsing cost range."""
        loader = ZomatoDataLoader()
        
        # Range should return midpoint
        assert loader.parse_cost("500-1000") == 750.0
        assert loader.parse_cost("300-600") == 450.0
        assert loader.parse_cost("1,000-2,000") == 1500.0
    
    def test_parse_cost_invalid(self):
        """Test parsing invalid cost strings."""
        loader = ZomatoDataLoader()
        
        assert loader.parse_cost("") is None
        assert loader.parse_cost(None) is None
        assert loader.parse_cost(np.nan) is None
        assert loader.parse_cost("NEW") is None
        assert loader.parse_cost("N/A") is None
    
    def test_normalize_cuisines(self):
        """Test cuisine normalization."""
        loader = ZomatoDataLoader()
        
        # Single cuisine
        assert loader.normalize_cuisines("North Indian") == ["North Indian"]
        
        # Multiple cuisines
        result = loader.normalize_cuisines("North Indian, Chinese, Italian")
        assert result == ["North Indian", "Chinese", "Italian"]
        
        # With spaces
        result = loader.normalize_cuisines("North Indian, Chinese , Italian")
        assert result == ["North Indian", "Chinese", "Italian"]
        
        # Empty/invalid
        assert loader.normalize_cuisines("") == []
        assert loader.normalize_cuisines(None) == []
        assert loader.normalize_cuisines(np.nan) == []
    
    def test_clean_and_validate(self):
        """Test data cleaning and validation."""
        loader = ZomatoDataLoader()
        
        # Create test DataFrame
        df = pd.DataFrame({
            'name': ['Restaurant A', 'Restaurant B', 'Restaurant C'],
            'rate': ['4.1/5', 'NEW', '3.5/5'],
            'votes': [100, 50, 200],
            'approx_cost(for two people)': ['500', '1000-1500', ''],
            'listed_in(city)': ['Bangalore', 'Mumbai', 'Delhi'],
            'cuisines': ['North Indian', 'Chinese, Italian', ''],
            'url': ['http://example.com/a', 'http://example.com/b', 'http://example.com/c']
        })
        
        loader.raw_data = df
        cleaned = loader.clean_and_validate()
        
        # Check normalized columns exist
        assert 'listed_in_city' in cleaned.columns
        assert 'approx_cost_for_two' in cleaned.columns
        
        # Check parsed fields
        assert 'rating_numeric' in cleaned.columns
        assert 'cost_numeric' in cleaned.columns
        assert 'cuisines_list' in cleaned.columns
        assert 'city_normalized' in cleaned.columns
        
        # Check parsed values
        assert cleaned.loc[0, 'rating_numeric'] == 4.1
        assert cleaned.loc[0, 'cost_numeric'] == 500.0
        # Pandas will store missing floats as NaN (not Python None)
        assert pd.isna(cleaned.loc[1, 'rating_numeric'])  # "NEW"
        assert cleaned.loc[1, 'cost_numeric'] == 1250.0  # midpoint of range
        assert pd.isna(cleaned.loc[2, 'cost_numeric'])  # empty string
    
    def test_get_unique_cities(self):
        """Test getting unique cities."""
        loader = ZomatoDataLoader()
        
        df = pd.DataFrame({
            'city_normalized': ['Bangalore', 'Mumbai', 'Bangalore', 'Delhi']
        })
        loader.processed_data = df
        
        cities = loader.get_unique_cities()
        assert 'Bangalore' in cities
        assert 'Mumbai' in cities
        assert 'Delhi' in cities
        assert len(cities) == 3
    
    def test_get_unique_cuisines(self):
        """Test getting unique cuisines."""
        loader = ZomatoDataLoader()
        
        df = pd.DataFrame({
            'cuisines_list': [
                ['North Indian', 'Chinese'],
                ['Chinese', 'Italian'],
                ['North Indian']
            ]
        })
        loader.processed_data = df
        
        cuisines = loader.get_unique_cuisines()
        assert 'North Indian' in cuisines
        assert 'Chinese' in cuisines
        assert 'Italian' in cuisines
        assert len(cuisines) == 3
    
    def test_get_price_ranges(self):
        """Test getting price range statistics."""
        loader = ZomatoDataLoader()
        
        df = pd.DataFrame({
            'cost_numeric': [500.0, 1000.0, 1500.0, 2000.0, np.nan]
        })
        loader.processed_data = df
        
        stats = loader.get_price_ranges()
        assert stats['min'] == 500.0
        assert stats['max'] == 2000.0
        assert stats['mean'] == 1250.0
        assert stats['median'] == 1250.0
        assert stats['count'] == 4
    
    def test_get_processed_data_error(self):
        """Test error when processed data not available."""
        loader = ZomatoDataLoader()
        
        with pytest.raises(ValueError, match="Data not processed"):
            loader.get_processed_data()
    
    def test_get_unique_cities_error(self):
        """Test error when data not processed."""
        loader = ZomatoDataLoader()
        
        with pytest.raises(ValueError, match="Data not processed"):
            loader.get_unique_cities()
    
    def test_clean_and_validate_no_data(self):
        """Test error when no data loaded."""
        loader = ZomatoDataLoader()
        
        with pytest.raises(ValueError, match="No data loaded"):
            loader.clean_and_validate()


@pytest.mark.integration
class TestZomatoDataLoaderIntegration:
    """Integration tests that require actual dataset loading."""
    
    @pytest.mark.skip(reason="Requires internet connection and Hugging Face access")
    def test_load_real_dataset(self):
        """Test loading actual dataset from Hugging Face."""
        loader = ZomatoDataLoader()
        df = loader.load_dataset(split="train")
        
        assert df is not None
        assert len(df) > 0
        assert 'name' in df.columns
        assert 'listed_in(city)' in df.columns or 'listed_in_city' in df.columns
    
    @pytest.mark.skip(reason="Requires internet connection and Hugging Face access")
    def test_full_pipeline(self):
        """Test full data loading and cleaning pipeline."""
        loader = ZomatoDataLoader()
        
        # Load
        loader.load_dataset(split="train")
        
        # Clean
        cleaned = loader.clean_and_validate()
        
        # Verify structure
        assert len(cleaned) > 0
        assert 'rating_numeric' in cleaned.columns
        assert 'cost_numeric' in cleaned.columns
        assert 'cuisines_list' in cleaned.columns
        
        # Get statistics
        cities = loader.get_unique_cities()
        cuisines = loader.get_unique_cuisines()
        price_stats = loader.get_price_ranges()
        
        assert len(cities) > 0
        assert len(cuisines) > 0
        assert price_stats['count'] > 0
