"""
Script to test if compustat loaders work
"""

from src import DataManager

def main():
    """
    Main function to load copmpustat datasets
    """
    data_manager = DataManager()
    dataset_collection = data_manager.get_data("crsp_test.yaml")
    comp_dataset = dataset_collection["wrds/equity/compustat"]
    print(comp_dataset)

if __name__ == "__main__":
    main()