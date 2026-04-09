from datasets import load_dataset
import kaggle
import os

def load_hf_dataset(dataset_name):
    print(f"Descargando dataset {dataset_name} desde hugging Face...")
    dataset = load_dataset(dataset_name, split="train")
    return dataset
    
def load_kaggle_dataset(dataset_slug, download_path="./data"):
    print(f"Descargando dataset {dataset_slug} desde Kaggle...")
    os.makedirs(download_path, exist_ok=True)
    kaggle.api.dataset_download_files(dataset_slug, path=download_path, unzip=True)
    return download_path