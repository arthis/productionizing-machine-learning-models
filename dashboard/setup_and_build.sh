#!/bin/bash

set -e

echo "Creating dataset cache folder: ./hf_cache"
mkdir -p hf_cache

echo "Downloading IMDB dataset into hf_cache..."
python3 -c "
from datasets import load_dataset
load_dataset('imdb', cache_dir='./hf_cache')
"
echo "Dataset downloaded."

echo "Rebuilding Docker images..."
docker compose build

echo "Build complete. Run with: docker compose up"
