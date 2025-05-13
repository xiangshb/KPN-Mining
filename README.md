# KPN-Mining
Source code for knowledge precedence netowrk mining

## To construct co-citing network
`ccn.py` is a Python toolkit for **constructing and analyzing Co-Citation Networks (CCN)**, designed for large-scale bibliometric research, scientific knowledge mapping, and scholarly community evolution studies. It focuses on the automated batch generation of author-level co-citation networks, community detection, semantic similarity analysis, and efficient data processing with parallelization.

To generate co-citation networks in parallel, run:

```
python ./network/ccn.py --min_papers 10 --n_workers 16 --parallel True
```

## To calculate and visualize the APYD distribution:
`apyd.py` provides tools for **analyzing temporal differences between research communities** within co-citation networks.

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
    cvm_threshold=0.9,
    cfm_threshold=0.15,
    graph_show=True,
    save_kpn=True
)

kpn = KPN()
G_kpn, df_edges = kpn.kpn_analysis(params=params, kpn_parms=kpn_parms)
```

---
