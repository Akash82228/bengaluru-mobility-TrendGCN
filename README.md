## TrendGCN – Traffic Flow Prediction

This repository contains a PyTorch implementation of a Trend-aware Graph Convolutional Network (TrendGCN) for short-term traffic flow prediction on road networks.

### Requirements

- Python 3.8+
- PyTorch
- Other dependencies listed in `requirements.yml`

Install dependencies (recommended in a virtual environment):

```bash
conda env create -f requirements.yml
conda activate TrendGCN
```

### Datasets & Configuration

Configuration files for supported datasets (e.g., PEMS, METR-LA) are in the `config` directory. Adjust paths and settings there as needed.

### Training

Run training with:

```bash
python main.py --config config/PEMS04.conf
```

Replace the config file with the one matching your dataset.

### Inference

For simple inference or testing with preprocessed inputs:

```bash
python infer.py
```

You can customize input files in the `inference_testing` directory.

### License

This project is licensed under the terms specified in `LICENSE`.

