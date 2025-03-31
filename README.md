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

---

