"""
Example usage of Phase 1 data loader.

This script demonstrates how to use ZomatoDataLoader to load and process
the Zomato restaurant dataset.
"""

from phase1.data_loader import ZomatoDataLoader


def main():
    """Example usage of the data loader."""
    
    # Initialize loader
    print("Initializing ZomatoDataLoader...")
    loader = ZomatoDataLoader()
    
    # Load dataset from Hugging Face
    print("\n" + "="*50)
    print("Loading dataset from Hugging Face...")
    print("="*50)
    try:
        loader.load_dataset(split="train")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Note: This requires internet connection and Hugging Face access.")
        return
    
    # Clean and validate
    print("\n" + "="*50)
    print("Cleaning and validating data...")
    print("="*50)
    cleaned_df = loader.clean_and_validate()
    
    # Display summary statistics
    print("\n" + "="*50)
    print("Dataset Summary")
    print("="*50)
    print(f"Total restaurants: {len(cleaned_df)}")
    print(f"Total columns: {len(cleaned_df.columns)}")
    
    # Get unique values
    cities = loader.get_unique_cities()
    cuisines = loader.get_unique_cuisines()
    price_stats = loader.get_price_ranges()
    
    print(f"\nUnique cities: {len(cities)}")
    print(f"Sample cities: {cities[:5]}")
    
    print(f"\nUnique cuisines: {len(cuisines)}")
    print(f"Sample cuisines: {cuisines[:10]}")
    
    print(f"\nPrice statistics:")
    if price_stats:
        print(f"  Min: ₹{price_stats['min']:.0f}")
        print(f"  Max: ₹{price_stats['max']:.0f}")
        print(f"  Mean: ₹{price_stats['mean']:.0f}")
        print(f"  Median: ₹{price_stats['median']:.0f}")
        print(f"  Valid prices: {price_stats['count']}")
    
    # Display sample data
    print("\n" + "="*50)
    print("Sample Data (first 3 rows)")
    print("="*50)
    sample_cols = ['name', 'city_normalized', 'rating_numeric', 'cost_numeric', 'cuisines_list']
    available_cols = [col for col in sample_cols if col in cleaned_df.columns]
    print(cleaned_df[available_cols].head(3).to_string())
    
    print("\n" + "="*50)
    print("Phase 1 complete! Data is ready for Phase 2.")
    print("="*50)


if __name__ == "__main__":
    main()
