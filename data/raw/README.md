# Raw dataset (Phase 1)

Place the Kaggle **Customer Support on Twitter** files here.

**Source:** https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter  
**Kaggle slug:** `thoughtvector/customer-support-on-twitter`

## What to download

Kaggle Version 10 (~516.53 MB) contains:

- `twcs/` (main corpus; usually `twcs/twcs.csv`)
- `sample.csv` (tiny preview; **not** enough for brand selection)

You need the **full** `twcs.csv`, not only `sample.csv`.

## Where to put it

Any of these layouts is fine; the inspection script searches this folder recursively:

```
data/raw/twcs.csv
data/raw/twcs/twcs.csv
data/raw/sample.csv          # optional extra file
```

If you download the zip, extract it into `data/raw/` so the CSV files land under this directory.

## How to download

### Option A — Kaggle website

1. Sign in to Kaggle.
2. Open the dataset page above.
3. Click **Download**.
4. Unzip into `data/raw/`.

### Option B — Kaggle CLI

1. Create an API token: Kaggle → Account → **Create New Token** (`kaggle.json`).
2. On Windows, put it at `%USERPROFILE%\.kaggle\kaggle.json`.
3. From the project root:

```powershell
pip install kaggle
kaggle datasets download -d thoughtvector/customer-support-on-twitter -p data/raw
Expand-Archive -Path data\raw\customer-support-on-twitter.zip -DestinationPath data\raw -Force
```

Do not commit the CSV. It is gitignored.
