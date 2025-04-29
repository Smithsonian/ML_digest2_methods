# NEOCP Filter Tools

A set of tools to analyze [digest2](https://www.minorplanetcenter.net/iau/info/digest.html) output and distinguish **NEOs** (Near-Earth Objects) from **non-NEOs**.

The analysis is based on:

> **Cloete, R. & Vereš, P.**  
> *Near-Earth Object Discovery Enhancement with Machine Learning Methods*

---

## Data Files

### `neocp.obs`
- NEOCP data from **2019–2024** in [obs80/MPC1992 format](https://minorplanetcenter.net/iau/info/OpticalObs.html).
- `trksub` designations were modified:
  - Last 6 characters = `trkid`
  - First character = `'0'` (NEO) or `'1'` (non-NEO)

---

### `digest_data_19-24.csv`
- Digest2 scores for NEOCP data from **2019–2023**.
- Columns:
  - `trksub` designation
  - Digest2 numerical scores for orbit classes
  - Final column `class` = numerical orbit class (see `src/find_filter.py`)

### `digest_data_24.csv`
- Same format as above, but for **2024** data only.

---

### `MPC.config`
- Configuration file for **digest2**, including:
  - Digest2 keywords (see `OPERATION.md`)
  - Observatory codes and expected astrometric uncertainties (arcseconds)

---

### `optimal_thresholds.json`
- Output of `find_filter.py` with threshold model (`limit = 0`).
- Format:
  ```json
  {
    "class": {
      "threshold": str,
      "non_neo_count": int,
      "neo_count": int
    }
  }
  ```
- Used to filter/classify NEOs from digest2 CSV outputs.

---

## Source Code

### `find_filter.py`
- Analyzes digest2 CSV data and builds a filter model.

**Example usage:**
```bash
python3 find_filter.py digest_data_19-24.csv
```

---

### `neocp_filter.py`
- Applies JSON model to classify new observations.

**Example usage:**
```bash
python3 neocp_filter.py digest_data_24.csv optimal_thresholds.json
```

**Training models**

By default, the code trains four models: GBM, RF, SGD, and a NN. All will be saved in the same 'model' directory.

To train, execute the training_pipeline.py script, found in src:
```bash
python training_pipeline.py \
        --train_csv /path/to/training_data.csv \
        --eval_csv /path/to/eval_data.csv \
        --model_save_dir /path/to/save/models \
        --features_file /path/to/cols_of_interest.txt \
        --test_size 0.25
```

`train_csv`: path to your training data
`eval_csv`: path to your evaluation data
`model_save_dir`: where you want the models trained saved
`features_file`: a file listing the features/columns you'd like to train your models on. If not provided, all features/columns will be used.
`test_size`: the size of the train/test split

**Testing models**

To test, execute the testing_pipeline.py, found in src:

```bash
python testing_pipeline.py \
        --model_dir /path/to/models/directory \
        --test_data /path/to/test_data.csv \
        --save_results /path/to/save/results
```

`model_dir`: path to where all models are stored
`test_data`: path to your testing data
`save_results`: where you want the results to be saved

---

