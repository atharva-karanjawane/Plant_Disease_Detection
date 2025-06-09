import os
import requests
import zipfile
import shutil
from tqdm import tqdm

def download_plantvillage_dataset():
    os.makedirs('data/plantvillage', exist_ok=True)
    
   
    dataset_url = "https://data.mendeley.com/public-files/datasets/tywbtsjrjv/files/d5652a28-c1d8-4b76-97f3-72fb80f94efc/file_downloaded"
    
    download_path = "data/plantvillage_dataset.zip"
    
    print("Downloading PlantVillage dataset...")
    response = requests.get(dataset_url, stream=True)
    
    # Check if the request was successful
    if response.status_code == 200:
        total_size = int(response.headers.get('content-length', 0))
        
        with open(download_path, 'wb') as f, tqdm(
            desc="Downloading",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                bar.update(size)
        
        print("Download complete. Extracting files...")
        
        with zipfile.ZipFile(download_path, 'r') as zip_ref:
            zip_ref.extractall("data/")
        
      
        source_dir = "data/PlantVillage"
        if os.path.exists(source_dir):
            for item in os.listdir(source_dir):
                s = os.path.join(source_dir, item)
                d = os.path.join("data/plantvillage", item)
                shutil.move(s, d)
            
            shutil.rmtree(source_dir, ignore_errors=True)
        
        os.remove(download_path)
        
        print("Dataset extraction complete.")
    else:
        print(f"Failed to download dataset. Status code: {response.status_code}")
        print("Please download the PlantVillage dataset manually and place it in the 'data/plantvillage' directory.")

if __name__ == "__main__":
    download_plantvillage_dataset()