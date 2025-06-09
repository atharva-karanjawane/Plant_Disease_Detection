import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import shutil
from sklearn.model_selection import train_test_split

def preprocess_plantvillage_dataset(dataset_path, output_path, img_size=224):
    """Preprocess the PlantVillage dataset."""
    
    os.makedirs(output_path, exist_ok=True)
    
    class_dirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    
    print(f"Found {len(class_dirs)} classes.")
    
    for class_dir in class_dirs:
        class_path = os.path.join(dataset_path, class_dir)
        output_class_path = os.path.join(output_path, class_dir)
        
        os.makedirs(output_class_path, exist_ok=True)
        
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"Processing {len(images)} images for class {class_dir}...")
        
        
        for img_name in tqdm(images, desc=class_dir):
            img_path = os.path.join(class_path, img_name)
            output_img_path = os.path.join(output_class_path, img_name)
            
            try:
                img = Image.open(img_path)
                img = img.resize((img_size, img_size))
                
                img.save(output_img_path)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    
    print("Preprocessing complete.")

def split_dataset(dataset_path, train_path, val_path, test_path, train_split=0.8, val_split=0.1):
    """Split the dataset into train, validation, and test sets."""
   
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    
    class_dirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    
    print(f"Splitting {len(class_dirs)} classes...")
    
    for class_dir in class_dirs:
        class_path = os.path.join(dataset_path, class_dir)
        
        train_class_path = os.path.join(train_path, class_dir)
        val_class_path = os.path.join(val_path, class_dir)
        test_class_path = os.path.join(test_path, class_dir)
        
        os.makedirs(train_class_path, exist_ok=True)
        os.makedirs(val_class_path, exist_ok=True)
        os.makedirs(test_class_path, exist_ok=True)
        
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        train_images, temp_images = train_test_split(
            images, train_size=train_split, random_state=42
        )
        
        val_size = val_split / (1 - train_split)
        val_images, test_images = train_test_split(
            temp_images, train_size=val_size, random_state=42
        )
        
        print(f"Class {class_dir}: {len(train_images)} train, {len(val_images)} val, {len(test_images)} test")
        
        for img_name in train_images:
            shutil.copy(
                os.path.join(class_path, img_name),
                os.path.join(train_class_path, img_name)
            )
        
        for img_name in val_images:
            shutil.copy(
                os.path.join(class_path, img_name),
                os.path.join(val_class_path, img_name)
            )
        
        for img_name in test_images:
            shutil.copy(
                os.path.join(class_path, img_name),
                os.path.join(test_class_path, img_name)
            )
    
    print("Dataset splitting complete.")

if __name__ == "__main__":
   
    preprocess_plantvillage_dataset(
        dataset_path="data/plantvillage",
        output_path="data/processed_plantvillage",
        img_size=224
    )
    
    split_dataset(
        dataset_path="data/processed_plantvillage",
        train_path="data/plantvillage_train",
        val_path="data/plantvillage_val",
        test_path="data/plantvillage_test",
        train_split=0.8,
        val_split=0.1
    )