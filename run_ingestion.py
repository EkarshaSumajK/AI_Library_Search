from data_ingestion import DataIngestion

def run_ingestion():
    print("🚀 Initializing Ingestion for Simulated Data...")
    
    # Initialize the existing class
    ingestion = DataIngestion()
    
    # CSV file path
    csv_file = "./new_data.csv"
    
    print(f"📖 Loading data from {csv_file}...")
    
    # Load data
    count = ingestion.load_data_from_csv(csv_file)
    
    if count > 0:
        print(f"✅ Successfully ingested {count} documents from simulation.")
        stats = ingestion.get_collection_stats()
        print(f"📊 Current Collection Stats: {stats}")
    else:
        print("❌ Ingestion failed or no data found.")

if __name__ == "__main__":
    run_ingestion()
