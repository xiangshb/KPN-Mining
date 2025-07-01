# KPN-Mining
Source code for knowledge precedence netowrk mining

## To construct co-citing network
`ccn.py` is a Python toolkit for **constructing and analyzing Co-Citing Networks (CCN)**, designed for large-scale bibliometric research, scientific knowledge mapping, and scholarly community evolution studies. It focuses on the automated batch generation of author-level Co-Citing networks, community detection, semantic similarity analysis, and efficient data processing with parallelization.

To generate Co-Citing networks in parallel, run:

```
python ./network/ccn.py --min_papers 10 --n_workers 16 --parallel True
```

## To calculate and visualize the APYD distribution
`apyd.py` provides tools for **analyzing temporal differences between research communities** within Co-Citing networks.

```
python ./network/apyd.py
```
Or, from within Python:
```python
from network.apyd import APYDAnalyzer

analyzer = APYDAnalyzer()
apyd_values = analyzer.calculate_APYD(show_distribution=True)
print(f"Mean APYD: {apyd_values.mean():.2f} years")
```

## To generate a KPM with collective selection mode and WCR threshold
`kpm.py` implements the construction of the **Knowledge Precedence Matrix (KPM)** from research trajectories, capturing how knowledge (represented by concepts) flows or progresses between research communities over time.

```python
from network.kpm import KPM
from utils.params import TCPParams

kpm = KPM()
params = TCPParams(concept_level=1, less_than=True, select_mode='collective', wcr=0.9, parallel=False)
df_kpm, params_str = kpm.get_kpm(params)
print(df_kpm)
```
Or from the command line:
```
python ./network/kpm.py
```
## To construct and visualize the KPN for disciplines
`kpn.py` provides a framework for constructing and analyzing **Knowledge Precedence Networks (KPNs)** from research trajectories.

```python
from network.kpn import KPN
from utils.params import TCPParams, KPNParams

params = TCPParams(concept_level=1, less_than=True, select_mode='all')
kpn_parms = KPNParams(
    disciplines=['Mathematics', 'Computer science'],
    row_normalize=False,
    threshold_mode='value_count_limit',
    cvm_threshold=0.95,
    cfm_threshold=0.2,
    graph_show=True,
    save_kpn=True
)

kpn = KPN()
G_kpn, df_edges = kpn.kpn_analysis(params=params, kpn_parms=kpn_parms)
```

---


## Sample Data

To facilitate code reproduction and testing, we provide sample datasets that replicate the data structure and workflow of our analysis pipeline.

### Sample Data Files

The following sample data files are included in the `./sample_data` directory:

| File | Description | Original Source |
|------|-------------|-----------------|
| `sample_author_ids.csv` | Sample author IDs used as input | Corresponds to `author_batch` variable |
| `sample_author_authors_works.parquet` | Author-work relationships data | Output of `self.query_author_data(author_batch)` - `df_authors_works` |
| `sample_works_concepts.csv` | Work-concept mappings data | Output of `self.query_author_data(author_batch)` - `df_works_concepts` |

### Usage Instructions

#### For `direct_kmp.py`

**Original code (database query):**
```python
df_authors_works, df_works_concepts = self.query_author_data(author_batch)
```
#### For `direct_kpm_ave_degree_resolution_all_sensitivity.py`

**Original code (database query):**
```python
df_authors_works, df_works_concepts = self.query_author_data_with_manager(author_batch, local_db_manager)
```

## Using sample data
```
import pandas as pd
import os.path as op

author_batch = pd.read_csv('./sample_data/sample_author_ids.csv')
df_authors_works = pd.read_parquet('./sample_data/sample_author_authors_works.parquet')
df_works_concepts = pd.read_csv('./sample_data/sample_works_concepts.csv')
```



