import kagglehub
import shutil
import os

# Download dataset to KaggleHub cache
cache_path = kagglehub.dataset_download("mczielinski/bitcoin-historical-data")

# Destination: the same folder as this script (src/data)
dest = os.path.join(os.path.dirname(__file__), "bitcoin-historical-data")

# Copy dataset into project directory
shutil.copytree(cache_path, dest, dirs_exist_ok=True)

print("Dataset copied to:", dest)

# Delete the cached dataset (Comment as per need)
#shutil.rmtree(cache_path, ignore_errors=True)
#print("Deleted cache at:", cache_path)