# hindsight/main.py

from src import DataManager

def main():
    data_manager = DataManager()
    
    print(data_manager.list_available_data_paths())
    
    tree = data_manager.get_data('load.yaml')

    
if __name__ == "__main__":
    main()
