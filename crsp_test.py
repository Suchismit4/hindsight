"""
Script to test if CRSP additonal data loaders work
"""

from src import DataManager

def main():
    """
    Main function to load CRSP datasets
    """
    data_manager = DataManager()
    dataset_collection = data_manager.get_data("crsp_test.yaml")
    crsp_dataset = dataset_collection["wrds/equity/crsp"]
    print(crsp_dataset)
    print('______________________meow______________________')
    crsp_time_series = crsp_dataset.sel(asset=14593).dt.to_time_indexed()
    print(crsp_time_series)

if __name__ == "__main__":
    main()