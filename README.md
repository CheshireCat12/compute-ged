# Compute GED

This folder contains utils to compute GED on graphs from the TUDataset repository.

## Intallation

#### Prerequisites
 - Python 3.9

#### Install
```bash
# Create virtual environment
python3.9 -m venv venv
sourve venv/bin/activate

# Install required package
pip install -r requirements.txt
```

## How to use

#### Retrieve Graphs from TUDataset repository.
```bash
python retrieve_graphs.py --dataset MUTAG --root-dataset ./data --folder-results ./graphs/MUTAG --graph-format pkl
```

#### Compute GED
```bash
python main.py --root-dataset ./graphs/MUTAG --graph-format pkl --alpha 0.4 --n-cores 8
```
