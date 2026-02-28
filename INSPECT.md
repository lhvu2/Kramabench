# 🔍 KramaBench — Inspection Notes

> A deep-dive Q&A session exploring the benchmark structure, input modalities, and lakehouse architecture options.

---

## 📋 Table of Contents

1. [What is KramaBench?](#1-what-is-kramabench)
2. [One Complete Example](#2-one-complete-example)
3. [Input Modalities — Does it include documents?](#3-input-modalities--does-it-include-documents)
4. [Does the input include images?](#4-does-the-input-include-images)
5. [Lakehouse Architecture — Is it feasible?](#5-lakehouse-architecture--is-it-feasible)
6. [Lakehouse Design — What goes in, what stays out, how do they join?](#6-lakehouse-design--what-goes-in-what-stays-out-how-do-they-join)

---

## 1. What is KramaBench?

**KramaBench** is an open-source benchmark for **end-to-end data-science agents**.

Unlike question-answer–only corpora, each task asks a system to build a *complete data pipeline*:

```
Load raw files → Clean → Transform → Compute → Final Answer
```

> Ground-truth code is provided, so KramaBench can evaluate both the **quality of the final answer** *and* the **correctness of intermediate steps**.

### Domains & Scale

| Domain      | #Tasks | #Sub-tasks | % Hard | File count | Raw size |
|-------------|-------:|-----------:|---------:|-----------:|---------:|
| Archaeology |     12 |         71 |     50% |          5 |   7.5 MB |
| Astronomy   |     12 |         68 |     50% |      1,556 |   486 MB |
| Biomedical  |      9 |         38 |     67% |          7 |   175 MB |
| Environment |     20 |        148 |     70% |         37 |    31 MB |
| Legal       |     30 |        188 |     53% |        136 |   1.3 MB |
| Wildfire    |     21 |        120 |     71% |         23 |     1 GB |
| **Total**   | **104**| **633**    | **61%** |  **1,764** | **1.7 GB** |

---

## 2. One Complete Example

### 🗂️ Task Definition (`workload/environment-tiny.json`)

```json
{
  "id": "environment-easy-1",
  "query": "What percentage (to 3 decimal places) of water samples collected from Massachusetts beaches during the 2013 bathing season exceeded bacterial standards, leading to temporary closures?",
  "answer": 4.796,
  "answer_type": "numeric_exact",
  "data_sources": ["water-body-testing-2013.csv"],
  "subtasks": [
    {
      "id": "environment-easy-1-1",
      "query": "What is the name of the file containing the 2013 beach sampling data?",
      "answer": "water-body-testing-2013.csv",
      "answer_type": "string_approximate"
    },
    {
      "id": "environment-easy-1-2",
      "query": "How many rows are marked to have 'Violation's?",
      "answer": 738,
      "answer_type": "numeric_exact"
    },
    {
      "id": "environment-easy-1-3",
      "query": "What is the total number of samples?",
      "answer": 15388,
      "answer_type": "numeric_exact"
    },
    {
      "id": "environment-easy-1-4",
      "query": "What is the percentage of water samples with violations? Round to 3 decimal places.",
      "answer": 4.796,
      "answer_type": "numeric_exact"
    }
  ]
}
```

### 📄 Input Data (`data/environment/input/water-body-testing-2013.csv`)

```
Community Code, Community,  County Code, County Description, Year, Sample Date,          Beach Name,             Beach Type, Organism,    Indicator Level, Violation
242,            Provincetown, 001,       Barnstable,         2013, 2013-07-15 00:00:00,  333 Commercial Street,  Marine,     Enterococci, 4,               no
242,            Provincetown, 001,       Barnstable,         2013, 2013-07-08 00:00:00,  333 Commercial Street,  Marine,     Enterococci, 2,               no
242,            Provincetown, 001,       Barnstable,         2013, 2013-06-18 00:00:00,  333 Commercial Street,  Marine,     Enterococci, 290,             yes  ← violation
...
```

### 🤖 What the system must do

1. Load `water-body-testing-2013.csv`
2. Count rows where `Violation == "yes"` → **738**
3. Count total rows → **15,388**
4. Compute `(738 / 15388) × 100`, rounded to 3 dp → **`4.796`**

### ✅ Expected Output

The system returns:

```python
# pipeline_code (returned by the SUT)
import pandas as pd
df = pd.read_csv("data/environment/input/water-body-testing-2013.csv")
pct = round((df['Violation'].str.lower() == 'yes').sum() / len(df) * 100, 3)
print(pct)  # → 4.796
```

And the JSON response:

```json
{
  "explanation": { "answer": 4.796 },
  "pipeline_code": "<python code string above>"
}
```

### 📊 Scoring

| Metric | Method |
|--------|--------|
| `numeric_exact` | Mean Relative Absolute Error (MRAE) vs ground truth `4.796` |
| Sub-task scores | Each intermediate step evaluated independently |
| LLM code eval | Auto-generated unit tests run against `pipeline_code` |

---

## 3. Input Modalities — Does it include documents?

> **Yes — the input goes well beyond CSV files.**

### Complete file format inventory

| Format | Domain(s) | Description |
|--------|-----------|-------------|
| **`.csv`** | All | Tabular data (water quality, wildfire stats, fraud reports, etc.) |
| **`.xlsx`** (Excel) | Archaeology, Biomedical, Wildfire | Multi-sheet spreadsheets (climate measurements, radiocarbon database, biomedical assay data) |
| **`.txt`** (plain-text bulletins) | Astronomy, Environment | NOAA space weather forecast reports — free-form structured text requiring regex parsing |
| **`.html`** (web page) | Legal | Saved Wikipedia article on Metropolitan Statistical Areas — contains HTML tables with population data |
| **`.npz`** (NumPy binary) | Astronomy | 3D atmospheric grid data (`mock_tiegcm_grid_sept2019.npz`) |
| **`.cdf`** (NASA Common Data Format) | Astronomy | Swarm satellite accelerometer calibration data |
| **`.sp3` / `.HDR`** (satellite orbit) | Astronomy | GNSS precise orbit determination files |
| **`.lst` / `.dat`** (OMNI solar wind) | Astronomy | Fixed-width columnar space physics data |
| **`.tle`** (Two-Line Element) | Astronomy | Orbital elements for satellite propagation |
| **`.gpkg`** (GeoPackage) | Wildfire | Geospatial vector data (US geographic area boundaries) |
| **`.json`** (lookup tables) | Wildfire | State abbreviation → state name mapping |
| **`.py`** (helper loader) | Wildfire | Domain-specific data loading script |

### Example: Plain-text bulletin (Astronomy)

```
:Product: 0309geomag_forecast.txt
:Issued: 2025 Mar 09 2205 UTC
# Prepared by the U.S. Dept. of Commerce, NOAA, Space Weather Prediction Center

NOAA Ap Index Forecast
Observed Ap 08 Mar 021
Estimated Ap 09 Mar 034
Predicted Ap 10 Mar-12 Mar 025-020-012    ← extracted via regex

NOAA Kp index forecast 10 Mar - 12 Mar
             Mar 10    Mar 11    Mar 12
00-03UT        4.00      3.67      3.67
...
```

### Example: HTML document (Legal)

`metropolitan_statistics.html` is a **saved Wikipedia page** containing tables of US Metropolitan Statistical Area populations. The system must parse the HTML, extract population figures, and join them with CSV identity theft data.

---

## 4. Does the input include images?

> **No. KramaBench contains no image inputs of any kind.**

After searching all files in `data/`, `dr-input/`, and all workload JSON files:

- ❌ No `.png`, `.jpg`, `.jpeg`, `.gif`, `.tiff`, `.svg`, or `.webp` files in any input directory
- ❌ No `data_sources` field in any workload JSON references an image file
- ℹ️ The `.png` files in `quickstart/` are **documentation screenshots only** (README illustrations)
- ℹ️ The word "figure" in workload JSON refers to a *computed population figure* (a number), not an image

---

## 5. Lakehouse Architecture — Is it feasible?

> **Yes — architecturally feasible and fits naturally into the SUT interface.**

### The SUT interface

```python
class System:
    def process_dataset(self, dataset_directory) -> None:
        """Called once: load/register all files in the domain."""
        ...

    def serve_query(self, query: str, query_id: str, subset_files: list) -> dict:
        """Called per task: return answer + pipeline_code."""
        return {
            "explanation": {"answer": <final_answer>},
            "pipeline_code": "<python code string>"
        }
```

The benchmark evaluates only the **final answer** and the returned `pipeline_code` — it does not mandate how data is loaded internally.

### Proposed two-stage pipeline

```
┌─────────────────────────────────────────────────────────┐
│  process_dataset()                                       │
│                                                          │
│  CSV / XLSX / HTML / JSON files                          │
│       │                                                  │
│       ▼                                                  │
│  Spark / Presto ingestion                                │
│  (register as named tables in catalog)                   │
│       │                                                  │
│       ▼                                                  │
│  Schema catalog: table_name → [col: type, ...]           │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│  serve_query()                                           │
│                                                          │
│  Stage 1: LLM sees query + DDL schemas                   │
│           → generates SQL                               │
│           → executes on Spark/Presto                    │
│           → returns intermediate DataFrame(s)           │
│                                                          │
│  Stage 2: LLM sees query + DataFrame schema + file paths │
│           → generates Python                            │
│           → loads Tier 2 files if needed                │
│           → joins / transforms / computes               │
│           → returns final answer                        │
└─────────────────────────────────────────────────────────┘
```

---

## 6. Lakehouse Design — What goes in, what stays out, how do they join?

### 🟢 Tier 1 — IN the Lakehouse (~85% of all files)

| File type | Examples | Ingestion method |
|-----------|----------|-----------------|
| **CSV** | `water-body-testing-*.csv`, all legal CSVs, wildfire CSVs | `spark.read.csv(..., inferSchema=True, header=True)` |
| **Excel (.xlsx)** | `climateMeasurements.xlsx`, biomedical mmc files | `pandas.read_excel()` → Spark DataFrame |
| **HTML tables** | `metropolitan_statistics.html` | `pandas.read_html()` → extract table → Spark DataFrame |
| **GeoPackage (.gpkg)** | `nifc_geographic_areas.gpkg` | `geopandas.read_file()` → register with Apache Sedona or as WKT |
| **JSON lookup** | `state_abbreviation_to_state.json` | `spark.read.json()` |

### 🔴 Tier 2 — OUTSIDE the Lakehouse (~15% of files, mostly Astronomy)

| File type | Examples | Why not in lakehouse | Storage |
|-----------|----------|---------------------|---------|
| **Plain-text bulletins (.txt)** | `0309geomag_forecast.txt`, `SB_DNS_POD_*.txt` | Free-form text; values extracted via regex | Object store (S3/ADLS/GCS) as raw files |
| **NumPy binary (.npz)** | `mock_tiegcm_grid_sept2019.npz` | Multi-dimensional array; no row/column structure | Object store; loaded with `numpy.load()` |
| **NASA CDF (.cdf)** | `SW_OPER_ACCACAL_2__*.cdf` | Binary scientific format; requires `spacepy.pycdf` | Object store; loaded with `cdflib` |
| **Satellite orbit (.sp3)** | `SW_OPER_SP3ACOM_2__*.sp3` | Fixed-width GNSS standard; requires specialized parser | Object store; loaded with `georinex` |
| **OMNI solar wind (.lst/.dat)** | `omni2_Kp_Index.lst` | Fixed-width columnar with metadata header | Object store; loaded with `pandas.read_fwf()` |
| **TLE orbital elements (.tle)** | `43180.tle` | Two-line element format; requires `sgp4` library | Object store; loaded in Python |

### 🔗 How joins happen at query time

Joins between Tier 1 and Tier 2 data happen **in Python code (Stage 2), not in SQL**.

#### Pattern A — SQL only (Legal, Environment, Archaeology, Biomedical)

```sql
-- All data is in the lakehouse; SQL handles everything
SELECT m.metro_name, AVG(i.reports) AS avg_reports
FROM metropolitan_stats m
JOIN identity_theft_reports i
  ON normalize(m.name) = normalize(i.metro)
WHERE m.population_2023 > 1000000
GROUP BY m.metro_name
```

#### Pattern B — Python only (Astronomy — no SQL needed)

```python
# All data is in object store; pure Python with domain libraries
import re

forecast = open("geomag_forecast/0309geomag_forecast.txt").read()
predicted = [int(x) for x in re.search(
    r"Predicted Ap.*?(\d+)-(\d+)-(\d+)", forecast).groups()]
# → [25, 20, 12]

observed = []
for fname in ["0311...", "0312...", "0313..."]:
    txt = open(fname).read()
    obs = int(re.search(r"Observed Ap \d+ \w+ (\d+)", txt).group(1))
    observed.append(obs)
# → [10, 10, 32]

mae = sum(abs(p - o) for p, o in zip(predicted, observed)) / len(predicted)
# → 15
```

#### Pattern C — SQL → Python join (Wildfire)

```python
# Stage 1: SQL pulls tabular data from lakehouse
df_fires = spark.sql("""
    SELECT state, year, SUM(acres) as total_acres
    FROM nifc_wildfires
    GROUP BY state, year
""").toPandas()

# Stage 2: Python loads geospatial file from object store
import geopandas as gpd
gdf = gpd.read_file("nifc_geographic_areas.gpkg")

# Join happens in Python
merged = gdf.merge(df_fires, on="state")
result = merged.groupby("region")["total_acres"].sum()
```

### 📊 Domain summary

| Domain | Lakehouse | Object Store | Join location |
|--------|-----------|--------------|---------------|
| **Environment** | ✅ All files | ❌ None | SQL only |
| **Legal** | ✅ All files (CSV + parsed HTML) | ❌ None | SQL only |
| **Archaeology** | ✅ All files (parsed XLSX) | ❌ None | SQL only |
| **Biomedical** | ✅ All files (XLSX + CSV) | ❌ None | SQL only |
| **Wildfire** | ✅ CSV / XLSX / JSON | ⚠️ `.gpkg` (or use Sedona) | SQL → Python merge |
| **Astronomy** | ✅ STORM-AI CSVs, SILSO CSV | ✅ `.txt` `.npz` `.cdf` `.sp3` `.lst` `.tle` | Python only |

> **Key insight:** The Astronomy domain is the outlier — most tasks use only Tier 2 binary/text files and require pure Python with domain-specific scientific libraries. For all other domains, the lakehouse handles the heavy lifting and Python only performs final computation.

---

*Generated from KramaBench inspection session — 2026-02-28*