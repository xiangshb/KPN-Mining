from typing import Union
import pandas as pd
import numpy as np
import networkx as nx
from typing import List

class BaseParams:
    """Base parameter class, providing serialization and deserialization functionality"""
    
    def to_dict(self):
        """Convert parameter object to dictionary"""
        return {k: v for k, v in self.__dict__.items()}
    
    def __repr__(self):
        """Provide readable object representation"""
        params_str = ", ".join(f"{k}={repr(v)}" for k, v in self.__dict__.items())
        return f"{self.__class__.__name__}({params_str})"
    
    @classmethod
    def from_dict(cls, params_dict):
        """Create parameter object from dictionary"""
        return cls(**params_dict)
    
    def save_to_json(self, filepath):
        """Save parameters to JSON file"""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_json(cls, filepath):
        """Load parameters from JSON file"""
        import json
        with open(filepath, 'r') as f:
            params_dict = json.load(f)
        return cls.from_dict(params_dict)

class TCPParams(BaseParams):
    def __init__(self, df_author_communities: Union[pd.DataFrame, None] = None, concept_level: int = 1, less_than: bool = True, select_mode: str = 'collective', 
                 wcr: float = 0.8, work_coverage_filter_ratio: float = 0.8, top_ratio: float= 0.3, community_method = 'louvain', min_community_size: int = 2, 
                 min_community_filter_size: int = 8, min_concept_size: int = 1, max_concept_size: int = 4, min_cites = 0, debugging = False, parallel = False,
                 test = False, print_size = 500):
         # min_community_size: int = 8 for selection of desired authors
        if select_mode not in ['collective', 'respective', 'cumulative', 'all']:
             raise ValueError(f"Invalid select_mode: {select_mode}")
        if not 0 <= wcr <= 1:
             raise ValueError(f"work_coverage_ratio must be between 0 and 1, got {wcr}")
        
        self.df_author_communities = df_author_communities
        self.concept_level = concept_level
        self.less_than = less_than
        self.select_mode = select_mode
        self.wcr = wcr
        self.work_coverage_filter_ratio = work_coverage_filter_ratio
        self.top_ratio = top_ratio
        self.min_community_size = min_community_size
        self.min_community_filter_size = min_community_filter_size
        self.min_concept_size = min_concept_size
        self.max_concept_size = max_concept_size
        self.community_method = community_method
        self.min_cites = min_cites
        self.debugging = debugging
        self.parallel = parallel
        self.wcr_ranges = [1, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6]
        self.test = test
        self.print_size = print_size

class VisualParams(BaseParams):
    def __init__(self, G: Union[nx.Graph, nx.DiGraph, None] = None, layout: str = 'graphviz', node_labels: Union[dict, None] = None, representive_concepts: Union[np.array, list, None] = None, 
                 mean_pub_years: Union[np.array, list, None] = None, dpi: int = 600, pad_inches: float = -0.1, figsize: tuple = (24, 16), rescale_layout: bool = False, show_community: bool = False, 
                 show_n_pubs_by_community: bool = False, bbox_to_anchor: tuple = (0.03, 1), save_pdf: bool = False, save_png: bool = False, file_name: str = None, override: bool = False, 
                 bbox_inches: str = 'tight', axis_off: bool = False, n_col: int = 5, subplots_adjust: dict = {'effective': False, 'left':0.1, 'right':0.8, 'top':0.9, 'bottom':0.1}, 
                 paddings_adjust: dict = {'effective': False, 'w_pad':-2, 'h_pad':-2}, coordinate_adjust: dict = {'dx': -5, 'dy': -5}, show_legend: bool = True):
        self.G = G
        self.show_community = show_community
        self.show_n_pubs_by_community = show_n_pubs_by_community
        self.layout = layout
        self.figsize = figsize
        self.node_labels = node_labels
        self.representive_concepts = representive_concepts
        self.mean_pub_years = mean_pub_years
        self.rescale_layout = rescale_layout
        self.bbox_to_anchor = bbox_to_anchor
        self.save_pdf = save_pdf
        self.save_png = save_png
        self.file_name = file_name
        self.override: bool = override
        self.bbox_inches: str = bbox_inches
        self.pad_inches: float = pad_inches
        self.subplots_adjust = subplots_adjust
        self.paddings_adjust = paddings_adjust
        self.coordinate_adjust = coordinate_adjust # seems doesn't work
        self.dpi = dpi
        self.axis_off = axis_off
        self.n_col = n_col
        self.show_legend = show_legend

class KPNParams(BaseParams):
     def __init__(self, disciplines: List[str] = ['Mathematics'], row_normalize: bool = True, matrix_top_ratio: int = 0.2, graph_show: bool = False, colorbar_shrink: float = 0.9, draw_legend:bool = True,
                  layout: str = 'graphviz', matrix_imshow: bool = False, save_kpn: bool = False, save_kpn_nodes_csv: bool = False, matrix_filter: bool = False, old_matrix_version: bool = False, 
                  params_str_old: bool = False, cvm_threshold: float = 0.9, cfm_threshold: float = 0.1, quantile_threshold = 0.6, threshold_mode: str = 'value_count_limit', save_pdf: bool = False): 
        self.disciplines = disciplines
        self.row_normalize = row_normalize
        self.matrix_top_ratio = matrix_top_ratio
        self.graph_show = graph_show
        self.matrix_imshow = matrix_imshow
        self.layout = layout
        self.save_kpn  = save_kpn
        self.save_kpn_nodes_csv = save_kpn_nodes_csv
        self.cvm_threshold = cvm_threshold
        self.cfm_threshold = cfm_threshold
        self.quantile_threshold = quantile_threshold
        self.matrix_filter = matrix_filter
        self.old_matrix_version = old_matrix_version
        self.params_str_old = params_str_old
        self.save_pdf = save_pdf
        self.colorbar_shrink = colorbar_shrink
        self.threshold_mode = threshold_mode
        self.draw_legend = draw_legend
