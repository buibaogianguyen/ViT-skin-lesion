import kagglehub

def download_dataset():
    dataset_path = kagglehub.dataset_download("andrewmvd/isic-2019")
    print(f"Dataset downloaded to: {dataset_path}")

    return dataset_path

if __name__ == "__main__":
    try:
        path = download_dataset()
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Ensure you have ~10GB free disk space and an internet connection.")