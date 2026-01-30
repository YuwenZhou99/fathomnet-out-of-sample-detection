# Out-of-Distribution Detection for Marine Species Classification

This project studies out-of-distribution (OOD) detection for deep learning models trained on underwater imagery from the FathomNet dataset from https://www.kaggle.com/competitions/fathomnet-out-of-sample-detection

## Setup

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run model
```bash
python main.py [resnet|vit_b_16]
```

## Authors
- Yuwen Zhou (y.zhou.74@student.rug.nl)
- Tjeerd Morsch (t.p.r.morsch@student.rug.nl)
- Ebe Kort (e.kort.3@student.rug.nl)
