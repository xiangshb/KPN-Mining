import os, sys
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.append(project_dir)
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['PYTHONUNBUFFERED'] = '1'

import networkx as nx
import pandas as pd
import numpy as np
import logging
import os.path as op
from network.kpn import KPN
from utils.params import KPNParams
import pickle
from utils.database import DatabaseManager
from utils.config import PathManager
from utils.smart_cache import cache_results
from utils.concept import Concept
from network.apyd import APYDAnalyzer

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


class KPMEvaluator:
    """
    Integrated class for generating Knowledge Precedence Matrix (KPM) with sensitivity analysis
    """
    path_manager = PathManager()
    
    def __init__(self, kpn_parms = False, sub_discipline = False):
        self.KPN = KPN()
        self.sub_discipline = sub_discipline
        self.logger = logging.getLogger(__name__)
        self.db_manager = DatabaseManager()
        self.APYD = APYDAnalyzer()
        self.kpn_parms = kpn_parms
        self.sciconnav_embedding, self.valid_concepts = self.load_concept_embedding()
        self.concepts_table = Concept.discipline_category_classification_llm(with_abbreviation=True)
        if self.sub_discipline:
            self.llm_pair_id_dicts, self.llm_pair_dicts, self.positive_id_gt_pairs, self.positive_gt_pairs = self.load_sub_llm_annotation_results()
        else:
            self.llm_pair_dict, self.positive_gt_pairs = self.load_llm_annotation_results()
        if self.sub_discipline:
            self.target_concepts = pd.read_csv(op.join('./llm_annotation/df_selected_top_concepts.csv'))
        else: 
            self.target_concepts = pd.read_csv(op.join('./llm_cross_annotation/df_selected_cross_concepts.csv'))

        self.discipline_to_concepts = self.target_concepts.groupby('llm_annotation')['display_name'].apply(list).to_dict()

        self.target_concepts = self.target_concepts.loc[self.target_concepts.display_name.isin(self.valid_concepts)]
        self.target_concept_ids = self.target_concepts.id.tolist()
        self.target_concept_names = self.target_concepts.display_name.tolist()
        self.concept_id_to_name = self.target_concepts.set_index('id')['display_name'].to_dict()
        self.combinations = None
        self.degree_sensitivity_kpm_dict = {}
        self.resolution_sensitivity_kpm_dict = {}
        self.citation_based_kmp = None
        self.fixed_degrees = [4, 5, 6, 7, 8, 9, 10, 11, 12, None] # None for dynamic degree
        self.resolutions = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        self.periods = ['early', 'middle', 'later']
        self.wcrs = [1, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6]

    def load_concept_embedding(self):
        from gensim.models import Word2Vec
        
        model_path = op.join(self.path_manager.ccns_dir, 'Word2Vec_dim_24_epoch_100.model')
        
        try:
            model = Word2Vec.load(model_path)
            valid_concepts = set()
            for concept in model.wv.index_to_key:
                vec = model.wv[concept]
                if not np.isnan(vec).any():
                    valid_concepts.add(concept)
            self.logger.info(f"Successfully loaded Word2Vec model with dimension: {model.vector_size}")
            return model, valid_concepts
        except Exception as e:
            self.logger.info(f"Error loading model: {str(e)}")
            return None, set()

    def load_llm_annotation_results(self):
        llm_pair_dict_path = op.join(self.path_manager.base_file_dir, 'llm_pair_dict.pkl')
        
        if not op.exists(llm_pair_dict_path):
            llm_file_path = op.join(self.path_manager.base_file_dir, 'llm_prerequisite_relations_comparison.csv')
            
            level_1_concepts = self.concepts_table.loc[self.concepts_table.level <= 1].reset_index(drop=True)
            level_1_concepts_dict = dict(level_1_concepts['display_name'])
            llm_prereq_relations = pd.read_csv(llm_file_path)
            llm_prereq_relations['concept_a'] = llm_prereq_relations['index_i'].map(level_1_concepts_dict)
            llm_prereq_relations['concept_b'] = llm_prereq_relations['index_j'].map(level_1_concepts_dict)
            llm_prereq_relations = llm_prereq_relations[['concept_a','concept_b','majority_vote']].reset_index(drop=True)
            
            llm_pair_dict = {}
            for _, row in llm_prereq_relations.iterrows():
                concept_a = row['concept_a']
                concept_b = row['concept_b']
                vote = row['majority_vote']
                
                llm_pair_dict[(concept_a, concept_b)] = vote
                
                if vote in [0, -2, 2]:
                    llm_pair_dict[(concept_b, concept_a)] = vote
                elif vote in [1, -1]:
                    llm_pair_dict[(concept_b, concept_a)] = -vote

            with open(llm_pair_dict_path, 'wb') as f:
                pickle.dump(llm_pair_dict, f)
        else:
            import pickle
            with open(llm_pair_dict_path, 'rb') as f:
                llm_pair_dict = pickle.load(f)
                
        positive_gt_pairs = {pair for pair, vote in llm_pair_dict.items() if vote > 0}
        return llm_pair_dict, positive_gt_pairs

    def load_sub_llm_annotation_results(self):
        
        llm_pair_dict_path = op.join(self.path_manager.base_file_dir, 'llm_sub_pair_dicts.pkl')
        llm_pair_id_dict_path = op.join(self.path_manager.base_file_dir, 'llm_sub_id_pair_dicts.pkl')

        if not op.exists(llm_pair_dict_path) or not op.exists(llm_pair_id_dict_path):
            llm_file_path = op.join(self.path_manager.base_file_dir, 'llm_sub_concepts_prerequisite_relations_comparison.csv')

            llm_prereq_relations = pd.read_csv(llm_file_path)

            llm_prereq_relations = llm_prereq_relations[['discipline', 'concept_a_id','concept_b_id','concept_a','concept_b','majority_vote']]

            llm_pair_dicts = {}
            llm_pair_id_dicts = {}
            for discipline, llm_prereq_relation_discipline in llm_prereq_relations.groupby('discipline'):
                llm_pair_dict = {}
                llm_pair_id_dict = {}
                # Build dictionary with forward and reverse relationships
                for _, row in llm_prereq_relation_discipline.iterrows():
                    concept_a_id = row['concept_a_id']
                    concept_b_id = row['concept_b_id']
                    concept_a = row['concept_a']
                    concept_b = row['concept_b']
                    vote = row['majority_vote']
                    
                    # Add original direction (A,B)
                    llm_pair_dict[(concept_a, concept_b)] = vote
                    llm_pair_id_dict[(concept_a_id, concept_b_id)] = vote

                    # Add reverse relationship (B,A) based on vote value
                    if vote in [0, -2, 2]:  # Direction doesn't matter
                        llm_pair_dict[(concept_b, concept_a)] = vote
                        llm_pair_id_dict[(concept_b_id, concept_a_id)] = vote
                    elif vote in [1, -1]:  # A is prerequisite for B, reverse: B depends on A
                        llm_pair_dict[(concept_b, concept_a)] = -vote
                        llm_pair_id_dict[(concept_b_id, concept_a_id)] = -vote
                llm_pair_dicts[discipline] = llm_pair_dict
                llm_pair_id_dicts[discipline] = llm_pair_id_dict

            with open(llm_pair_dict_path, 'wb') as f:
                pickle.dump(llm_pair_dicts, f)
            with open(llm_pair_id_dict_path, 'wb') as f:
                pickle.dump(llm_pair_id_dicts, f)
        else:
            with open(llm_pair_dict_path, 'rb') as f:
                llm_pair_dicts = pickle.load(f)
            with open(llm_pair_id_dict_path, 'rb') as f:
                llm_pair_id_dicts = pickle.load(f)
        positive_id_gt_pairs = {}
        for discipline, llm_pair_id_dict in llm_pair_id_dicts.items():
            positive_id_gt_pairs[discipline] = {pair for pair, vote in llm_pair_id_dict.items() if vote > 0}
            
        positive_gt_pairs = {}
        for discipline, llm_pair_dict in llm_pair_dicts.items():
            positive_gt_pairs[discipline] = {pair for pair, vote in llm_pair_dict.items() if vote > 0}
        return llm_pair_id_dicts, llm_pair_dicts, positive_id_gt_pairs, positive_gt_pairs

    def parse_param_key(self, param_key):
        """Parse parameter key to extract fixed_degree and resolution"""
        # param_key format: "degree_8_res_1.0"
        if param_key == 'citation_baseline':
            return None, None
        parts = param_key.split('_')
        # Handle dynamic degree case
        if parts[1] == 'dynamic':
            fixed_degree = None
        else:
            fixed_degree = int(parts[1])
        resolution = float(parts[3])
        return fixed_degree, resolution

    def parse_param_key_(self, param_key):
        """Parse parameter key to extract fixed_degree and resolution"""
        # param_key format: "degree_8_res_1.0"
        if param_key == 'citation_baseline':
            return None, None
        parts = param_key.split('_')
        # Handle dynamic degree case
        if parts[1] == 'dynamic':
            fixed_degree = 'log'
        else:
            fixed_degree = int(parts[1])
        resolution = float(parts[3])
        return fixed_degree, resolution
    
    def get_kpm_filename(self, fixed_ave_degree=None, resolution=1.0):
        """Generate filename for KPM based on parameters"""
        if fixed_ave_degree is not None:
            return f"df_kpm_ave_degree_{fixed_ave_degree}_res_{resolution}.csv"
        else:
            return f"df_kpm_dynamic_degree_res_{resolution}.csv"

    def load_existing_kpm_dict(self, fixed_degree = 5, fixed_resolution = 1.4):
        """Generate parameter combinations for sensitivity analysis"""
        
        self.degree_sensitivity_kpm_dict = {}
        # Fixed degrees with resolution=1.0
        for degree in self.fixed_degrees:
            if degree is None:
                param_key = f"degree_dynamic_res_{fixed_resolution}"
            else:
                 param_key = f"degree_{degree}_res_{fixed_resolution}"
            kpm_filename = self.get_kpm_filename(degree, resolution = fixed_resolution) # fix resolution = 1.0
            if self.sub_discipline:
                kpm_path = op.join(self.path_manager.sub_kpm_dir, kpm_filename)
            else:
                kpm_path = op.join(self.path_manager.kpm_dir_all_deg_res, kpm_filename)
            if op.exists(kpm_path):
                self.degree_sensitivity_kpm_dict[param_key] = pd.read_csv(kpm_path, index_col=0)
            else:
                self.degree_sensitivity_kpm_dict[param_key] = pd.DataFrame(0, index=self.target_concept_names, columns=self.target_concept_names)
        
        self.resolution_sensitivity_kpm_dict = {}
        # Different resolutions with fixed_degree=8
        for resolution in self.resolutions:
            param_key = f"degree_{fixed_degree}_res_{resolution}"
            kpm_filename = self.get_kpm_filename(fixed_ave_degree = fixed_degree, resolution = resolution)
            if self.sub_discipline:
                kpm_path = op.join(self.path_manager.sub_kpm_dir, kpm_filename)
            else:
                kpm_path = op.join(self.path_manager.kpm_dir_all_deg_res, kpm_filename)
            if op.exists(kpm_path):
                self.resolution_sensitivity_kpm_dict[param_key] = pd.read_csv(kpm_path, index_col=0)
            else:
                self.resolution_sensitivity_kpm_dict[param_key] = pd.DataFrame(0, index=self.target_concept_names, columns=self.target_concept_names)
    
        if self.sub_discipline:
            citation_kmp_path = op.join(self.path_manager.sub_kpm_dir, 'df_kpm_citation_based_final.csv')
        else:
            citation_kmp_path = op.join(self.path_manager.kpm_dir, 'df_kpm_citation_based_final.csv')

        self.citation_based_kmp = pd.read_csv(citation_kmp_path, index_col=0)

    def load_existing_kpm_dict_period(self):
        """Generate parameter combinations for sensitivity analysis"""
        
        self.period_kpm_dict = {}
        # Fixed degrees with resolution=1.0
        for period in self.periods:
            if self.sub_discipline:
                df_period_kpm_path = op.join(self.path_manager.sub_kpm_dir, f'''df_kpm_period_{period}.csv''')
            else: df_period_kpm_path = op.join(self.path_manager.kpm_dir, f'''df_kpm_period_{period}.csv''')
            
            df_kpm_period = pd.read_csv(df_period_kpm_path, index_col=0)
            self.period_kpm_dict[period] = df_kpm_period.rename(
                index=self.concept_id_to_name,
                columns=self.concept_id_to_name
            )
            
        if self.sub_discipline:
            citation_kmp_path = op.join(self.path_manager.sub_kpm_dir, 'df_kpm_citation_based_final.csv')
        else:
            citation_kmp_path = op.join(self.path_manager.kpm_dir, 'df_kpm_citation_based_final.csv')
        
        self.citation_based_kmp = pd.read_csv(citation_kmp_path, index_col=0)
        
    def load_existing_kpm_dict_wcr(self):
        """Generate parameter combinations for sensitivity analysis"""
        
        self.wcr_kpm_dict = {}
        # Fixed degrees with resolution=1.0
        for wcr in self.wcrs:
            wcr_str = f"{wcr:.1f}" if wcr == 1 else str(wcr)
            if self.sub_discipline:
                df_wcr_kpm_path = op.join(self.path_manager.sub_kpm_dir, f'''df_kpm_wcr_{wcr_str}.csv''')
            else:
                df_wcr_kpm_path = op.join(self.path_manager.kpm_dir, f'''df_kpm_wcr_{wcr_str}.csv''')
            
            df_kpm_wcr = pd.read_csv(df_wcr_kpm_path, index_col=0)
            self.wcr_kpm_dict[wcr] = df_kpm_wcr.rename(
                index=self.concept_id_to_name,
                columns=self.concept_id_to_name
            )
            
        if self.sub_discipline:
            citation_kmp_path = op.join(self.path_manager.sub_kpm_dir, 'df_kpm_citation_based_final.csv')
        else:
            citation_kmp_path = op.join(self.path_manager.kpm_dir, 'df_kpm_citation_based_final.csv')

        self.citation_based_kmp = pd.read_csv(citation_kmp_path, index_col=0)
    
    def load_previous_existing_kpm_dict(self):
        pass
    
    def load_existing_kpm_dict_all_degree_resolution(self):
        self.all_degree_resolution_sensitivity_kpm_dict = {}
        self.fixed_degrees = [4, 5, 6, 7, 8, 9, 10, 11, 12, None]
        self.resolutions = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]

        for degree in self.fixed_degrees:
            degree_str = "dynamic" if degree is None else str(degree)
            for resolution in self.resolutions:
                param_key = f"degree_{degree_str}_res_{resolution}"
                kpm_filename = self.get_kpm_filename(degree, resolution) # fix resolution = 1.0

                if self.sub_discipline:
                    kpm_path = op.join(self.path_manager.kpm_dir_all_deg_res, kpm_filename)
                else:
                    kpm_path = op.join(self.path_manager.kpm_dir_all_deg_res, kpm_filename)
                if op.exists(kpm_path):
                    self.all_degree_resolution_sensitivity_kpm_dict[param_key] = pd.read_csv(kpm_path, index_col=0)
                else:
                    self.all_degree_resolution_sensitivity_kpm_dict[param_key] = pd.DataFrame(0, index=self.target_concept_names, columns=self.target_concept_names)
        if self.sub_discipline:
            citation_kmp_path = op.join(self.path_manager.sub_kpm_dir, 'df_kpm_citation_based_final.csv')
        else:
            citation_kmp_path = op.join(self.path_manager.kpm_dir, 'df_kpm_citation_based_final.csv')

        self.citation_based_kmp = pd.read_csv(citation_kmp_path, index_col=0)
    
    def get_max_symmetric_elements(self, matrix, symmetry_preference='equal_priority'):
        """
        Create a symmetric matrix by keeping the larger values between symmetric positions
        
        Args:
            matrix: Input matrix
            symmetry_preference: How to handle equal values in symmetric positions
                - 'lower_priority': When values are equal, prefer values from lower triangle (row > column)
                - 'upper_priority': When values are equal, prefer values from upper triangle (column > row)
                - 'equal_priority': Keep both elements when they are equal (default)
                
        Returns:
            Matrix with retained larger symmetrical elements
        """
        matrix_tril, matrix_triu = np.tril(matrix, k=-1), np.triu(matrix, k=1)
        
        if symmetry_preference == 'lower_priority':
            matrix_maximum_symmetric = matrix_tril*(matrix_tril >= matrix_triu.T) + matrix_triu*(matrix_tril.T < matrix_triu)
        elif symmetry_preference == 'upper_priority':
            matrix_maximum_symmetric = matrix_tril*(matrix_tril > matrix_triu.T) + matrix_triu*(matrix_tril.T <= matrix_triu)
        elif symmetry_preference == 'equal_priority':
            matrix_maximum_symmetric = matrix_tril*(matrix_tril >= matrix_triu.T) + matrix_triu*(matrix_tril.T <= matrix_triu)
        else:
            raise ValueError('Invalid symmetry preference. Use "lower_priority", "upper_priority", or "equal_priority"')
            
        return matrix_maximum_symmetric

    def plot_degree_resolution_sensitivity(self, df_results_resolution):
        import matplotlib.pyplot as plt
        import numpy as np
        import matplotlib.cm as cm
        
        # 筛选 citation=False 的数据
        df_plot = df_results_resolution[df_results_resolution['citation'] == False].copy()
        df_plot_true = df_results_resolution[df_results_resolution['citation'] == True].copy()
        if df_plot.empty:
            print("No data for citation=False")
            return
        
        unique_degrees = ['4', '5', '6', '7', '8', '9', '10', '11', '12', 'log']
        unique_resolutions = np.sort(df_plot['resolution'].unique())
        
        # 获取 Citation 的结果（应该是唯一值）
        citation_precision = None
        citation_recall = None
        if not df_plot_true.empty:
            citation_precision = df_plot_true['precision'].iloc[0]
            citation_recall = df_plot_true['recall'].iloc[0]

        # 设置颜色映射
        colors = cm.get_cmap('tab10', 10)
        
        # 第一张图：横轴degree，每条线是不同resolution
        plt.figure(figsize=(5, 4.5))
        
        # 先绘制所有 Precision 线（包括各参数和Citation）
        for idx, resolution in enumerate(unique_resolutions):
            group = df_plot[df_plot['resolution'] == resolution].copy()
            if group.empty:
                continue
            
            # 按照 unique_degrees 的顺序排序
            group['degree_order'] = group['degree'].astype(str).map(lambda x: unique_degrees.index(x) if x in unique_degrees else 999)
            group = group.sort_values('degree_order')
            
            color = colors(idx)
            plt.plot(group['degree'].astype(str), group['precision'], marker='o', color=color, 
                    label=f'Precision (Res.={resolution:.1f})', linestyle='-')
        
        # Citation Precision 最后绘制
        if citation_precision is not None:
            plt.axhline(y=citation_precision, color='black', linestyle='-', linewidth=2,
                    label='Precision (Citation)')
        
        # 再绘制所有 Recall 线（包括各参数和Citation）
        for idx, resolution in enumerate(unique_resolutions):
            group = df_plot[df_plot['resolution'] == resolution].copy()
            if group.empty:
                continue
            
            # 按照 unique_degrees 的顺序排序
            group['degree_order'] = group['degree'].astype(str).map(lambda x: unique_degrees.index(x) if x in unique_degrees else 999)
            group = group.sort_values('degree_order')
            
            color = colors(idx)
            plt.plot(group['degree'].astype(str), group['recall'], marker='x', color=color, 
                    label=f'Recall (Res.={resolution:.1f})', linestyle='--')
        
        # Citation Recall 最后绘制
        if citation_recall is not None:
            plt.axhline(y=citation_recall, color='black', linestyle='--', linewidth=2,
                    label='Recall (Citation)')

        plt.xlabel('Average Degree', fontsize=16)
        plt.ylabel('Score', fontsize=16)
        
        # 手动设置 x 轴刻度顺序
        plt.xticks(unique_degrees, fontsize=16)
        
        plt.legend(prop={'size': 8}, loc='upper left', ncol=2, bbox_to_anchor=(0.09, 0.75))
        plt.yticks(fontsize=16)
        plt.grid(True)
        plt.tight_layout()
        
        file_path = op.join(self.path_manager.base_figures_dir, 'sensitivity_degree_resolution.pdf')
        self.path_manager.save_pdf_file(plt, abs_file_path=file_path, override=True, pad_inches = 0.02)
        plt.show()
        
        # 第二张图：横轴resolution，每条线是不同degree
        plt.figure(figsize=(5, 4.5))
        
        # 先绘制所有 Precision 线（包括各参数和Citation）
        for idx, degree in enumerate(unique_degrees):
            group = df_plot[df_plot['degree'] == degree].sort_values('resolution')
            if group.empty:
                continue
            color = colors(idx)
            plt.plot(group['resolution'], group['precision'], marker='o', color=color, 
                    label=f'Precision (Deg.={degree})', linestyle='-')
        
        # Citation Precision 最后绘制
        if citation_precision is not None:
            plt.axhline(y=citation_precision, color='black', linestyle='-', linewidth=2,
                    label='Precision (Citation)')
        
        # 再绘制所有 Recall 线（包括各参数和Citation）
        for idx, degree in enumerate(unique_degrees):
            group = df_plot[df_plot['degree'] == degree].sort_values('resolution')
            if group.empty:
                continue
            color = colors(idx)
            plt.plot(group['resolution'], group['recall'], marker='x', color=color, 
                    label=f'Recall (Deg.={degree})', linestyle='--')
        
        # Citation Recall 最后绘制
        if citation_recall is not None:
            plt.axhline(y=citation_recall, color='black', linestyle='--', linewidth=2,
                    label='Recall (Citation)')

        plt.xlabel('Louvain Resolution', fontsize=16)
        plt.ylabel('Score', fontsize=16)
        plt.legend(prop={'size': 8}, loc='upper left', ncol=2, bbox_to_anchor=(0.09, 0.772))
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.grid(True)
        plt.tight_layout()
        file_path = op.join(self.path_manager.base_figures_dir, 'sensitivity_resolution_degree.pdf')
        self.path_manager.save_pdf_file(plt, abs_file_path=file_path, override=True, pad_inches = 0.02)
        plt.show()

    def plot_degree_resolution_f1_sensitivity(self, df_results_resolution):
        import matplotlib.pyplot as plt
        import numpy as np
        import matplotlib.cm as cm
        
        # 筛选 citation=False 的数据
        df_plot = df_results_resolution[df_results_resolution['citation'] == False].copy()
        df_plot_true = df_results_resolution[df_results_resolution['citation'] == True].copy()
        if df_plot.empty:
            print("No data for citation=False")
            return
        
        # 计算 F1 score
        df_plot['f1_score'] = 2 * (df_plot['precision'] * df_plot['recall']) / (df_plot['precision'] + df_plot['recall'])
        df_plot['f1_score'] = df_plot['f1_score'].fillna(0)  # 处理除零情况
        
        unique_degrees = ['4', '5', '6', '7', '8', '9', '10', '11', '12', 'log']
        unique_resolutions = np.sort(df_plot['resolution'].unique())
        
        # 获取 Citation 的 F1 score
        citation_f1 = None
        if not df_plot_true.empty:
            citation_precision = df_plot_true['precision'].iloc[0]
            citation_recall = df_plot_true['recall'].iloc[0]
            if citation_precision + citation_recall > 0:
                citation_f1 = 2 * (citation_precision * citation_recall) / (citation_precision + citation_recall)

        # 设置颜色映射
        colors = cm.get_cmap('tab10', 10)
        
        # 第一张图：横轴degree，每条线是不同resolution
        plt.figure(figsize=(5, 4.5))
        
        # 绘制所有 F1 score 线
        for idx, resolution in enumerate(unique_resolutions):
            group = df_plot[df_plot['resolution'] == resolution].copy()
            if group.empty:
                continue
            
            # 按照 unique_degrees 的顺序排序
            group['degree_order'] = group['degree'].astype(str).map(lambda x: unique_degrees.index(x) if x in unique_degrees else 999)
            group = group.sort_values('degree_order')
            
            color = colors(idx)
            plt.plot(group['degree'].astype(str), group['f1_score'], marker='o', color=color, 
                    label=f'F1 (Res.={resolution:.1f})', linestyle='-')
        
        # Citation F1 最后绘制
        if citation_f1 is not None:
            plt.axhline(y=citation_f1, color='black', linestyle='-', linewidth=2,
                    label='F1 (Citation)')

        plt.xlabel('Average Degree', fontsize=16)
        plt.ylabel('F1 Score', fontsize=16)
        plt.ylim(0.1865, 0.2115)

        # 手动设置 x 轴刻度顺序
        plt.xticks(unique_degrees, fontsize=16)
        
        plt.legend(prop={'size': 8}, loc='upper left', ncol=2, bbox_to_anchor=(0.15, 0.26))
        plt.yticks(fontsize=16)
        plt.grid(True)
        plt.tight_layout()
        
        file_path = op.join(self.path_manager.base_figures_dir, 'sensitivity_degree_resolution_f1.pdf')
        self.path_manager.save_pdf_file(plt, abs_file_path=file_path, override=True, pad_inches=0.02)
        plt.show()
        
        # 第二张图：横轴resolution，每条线是不同degree
        plt.figure(figsize=(5, 4.5))
        
        # 绘制所有 F1 score 线
        for idx, degree in enumerate(unique_degrees):
            group = df_plot[df_plot['degree'] == degree].sort_values('resolution')
            if group.empty:
                continue
            color = colors(idx)
            plt.plot(group['resolution'], group['f1_score'], marker='o', color=color, 
                    label=f'F1 (Deg.={degree})', linestyle='-')
        
        # Citation F1 最后绘制
        if citation_f1 is not None:
            plt.axhline(y=citation_f1, color='black', linestyle='-', linewidth=2,
                    label='F1 (Citation)')

        plt.xlabel('Louvain Resolution', fontsize=16)
        plt.ylabel('F1 Score', fontsize=16)
        plt.legend(prop={'size': 8}, loc='upper left', ncol=2, bbox_to_anchor=(0.3, 0.35))
        plt.xticks([0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0], fontsize=16)
        plt.yticks(fontsize=16)
        plt.grid(True)
        plt.tight_layout()
        
        file_path = op.join(self.path_manager.base_figures_dir, 'sensitivity_resolution_degree_f1.pdf')
        self.path_manager.save_pdf_file(plt, abs_file_path=file_path, override=True, pad_inches=0.02)
        plt.show()

    @cache_results(
        cache_dir=path_manager.base_data_dir, 
        # filename_pattern="precision_recall_results_{type}.csv",
        filename_pattern="precision_recall_results_{type}_v{version}.csv",
        version=1
    )
    def calculate_precision_recall(self, type = 'degree', fixed_degree = 5, fixed_resolution = 1.4, skip_cache = False):
        """Evaluate KPM performance across different degrees for each discipline separately"""
        
        metrics_data = []
        if type == 'degree':
            self.load_existing_kpm_dict(fixed_degree=fixed_degree, fixed_resolution=fixed_resolution)
            target_kpm_dict = self.degree_sensitivity_kpm_dict
        elif type == 'resolution':
            self.load_existing_kpm_dict(fixed_degree=fixed_degree, fixed_resolution=fixed_resolution)
            target_kpm_dict = self.resolution_sensitivity_kpm_dict
        elif type == 'wcr':
            self.load_existing_kpm_dict_wcr()
            target_kpm_dict = self.wcr_kpm_dict
        elif type == 'period':
            self.load_existing_kpm_dict_period()
            target_kpm_dict = self.period_kpm_dict
        elif type == 'all_degree_resolution':
            self.load_existing_kpm_dict_all_degree_resolution()
            target_kpm_dict = self.all_degree_resolution_sensitivity_kpm_dict

        else: raise ValueError('Undefined type')
        target_kpm_dict['citation_baseline'] = self.citation_based_kmp
        # Evaluate each degree
        for param_key, df_kpm_original in target_kpm_dict.items():
            # Extract degree from param_key
            
            degree_, resolution_ = self.parse_param_key(param_key)

            df_kpm = pd.DataFrame(
                self.get_max_symmetric_elements(df_kpm_original.values), 
                index=df_kpm_original.index, 
                columns=df_kpm_original.columns
            )
            df_edges_roc = self.KPN.kpn_edges(df_kpm, self.kpn_parms, optimize=True)
            df_edges_roc['ground_truth'] = df_edges_roc[['source', 'target']].apply(
                tuple, axis=1
            ).map(self.llm_pair_dict)
            df_edges_roc['true_label'] = df_edges_roc['ground_truth'].isin([1, 2])
            
            precision = df_edges_roc['true_label'].sum() / len(df_edges_roc)
            recall = df_edges_roc['true_label'].sum() / len(self.positive_gt_pairs)

            # Add precision row
            if type == 'all_degree_resolution':
                if degree_ is None and resolution_ is not None:
                    degree_ = 'log'
                metrics_data.append({
                    'citation':param_key == 'citation_baseline',
                    'degree': degree_,
                    'resolution': resolution_,
                    'precision': precision,
                    'recall': recall
                })
            else:
                metrics_data.append({
                    'param_key': param_key,
                    'precision': precision,
                    'recall': recall
                })
        # Convert to DataFrame
        
        if type != 'all_degree_resolution':
            df_metrics = pd.DataFrame(metrics_data)
        # Pivot to get the desired structure
            df_final = (
                df_metrics
                .melt(id_vars='param_key', value_vars=['precision', 'recall'],
                    var_name='metric', value_name='value')
                .pivot(index='metric', columns='param_key', values='value')
                .reset_index()
            )
            df_final.columns.name = None 
            if type == 'degree':
                columns = ['metric'] + [f'''degree_{'dynamic' if degree is None else degree}_res_{fixed_resolution}''' for degree in self.fixed_degrees] + ['citation_baseline']
            elif type == 'resolution':
                columns = ['metric'] + [f'degree_{fixed_degree}_res_{resolution}' for resolution in self.resolutions] + ['citation_baseline']
            elif type == 'wcr':
                columns = df_final.columns.to_list()
            elif type == 'period':
                columns = ['metric', 'early', 'middle', 'later', 'citation_baseline']
            else: raise ValueError('Undefined type')
            
            df_final = df_final[columns]
            numeric_cols = df_final.columns[1:]
            df_final[numeric_cols] = (df_final[numeric_cols] * 100).round(2)
            for col in numeric_cols:
                df_final[col] = df_final[col].astype(str) + '%'
        else: df_final = pd.DataFrame(metrics_data)
        return df_final


    def plot_cvm_cfm_curves(self, df_metrics, type_filter=None):
        import matplotlib.pyplot as plt
        import numpy as np
        import matplotlib.cm as cm

        """
        绘制 Precision/Recall 关于 CVM 和 CFM 的曲线图
        :param df_metrics: 包含 type, cvm, cfm, precision, recall 列的 DataFrame
        :param type_filter: 可选，筛选 type（如 'wcr_1' 或 'citation'），不传则画全部
        """
        df_plot = df_metrics.copy()
        if type_filter is not None:
            df_plot = df_plot[df_plot['type'] == type_filter]
            if df_plot.empty:
                print(f"No data for type {type_filter}")
                return

        unique_cfm = np.sort(df_plot['cfm'].unique())
        unique_cvm = np.sort(df_plot['cvm'].unique())
        colors = cm.get_cmap('tab10', 10)

        # 第一张图：横轴cvm，每条线是不同cfm
        plt.figure(figsize=(5, 4.5))
        for idx, cfm in enumerate(unique_cfm):
            group = df_plot[df_plot['cfm'] == cfm].sort_values('cvm')
            color = colors(idx)
            plt.plot(group['cvm'], group['precision'], marker='o', color=color, label=f'Precision, CFM={cfm:.2f}', linestyle='-')
            plt.plot(group['cvm'], group['recall'], marker='x', color=color, label=f'Recall, CFM={cfm:.2f}', linestyle='--')
        plt.xlabel('CVM', fontsize=16)
        plt.ylabel('Score', fontsize=16)
        # plt.title('Precision/Recall vs CVM', fontsize=16)
        plt.legend(prop={'size': 12}, loc = 'upper left')
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.grid(True)
        plt.tight_layout()
        file_path = op.join(self.path_manager.base_figures_dir, 'cvm_sensitivity.pdf')
        self.path_manager.save_pdf_file(plt, abs_file_path=file_path, override=True, pad_inches = 0.02)
        plt.show()

        # 第二张图：横轴cfm，每条线是不同cvm
        plt.figure(figsize=(5, 4.5))
        for idx, cvm in enumerate(unique_cvm):
            group = df_plot[df_plot['cvm'] == cvm].sort_values('cfm')
            color = colors(idx)
            plt.plot(group['cfm'], group['precision'], marker='o', color=color, label=f'Precision, CVM={cvm:.2f}', linestyle='-')
            plt.plot(group['cfm'], group['recall'], marker='x', color=color, label=f'Recall, CVM={cvm:.2f}', linestyle='--')
        plt.xlabel('CFM', fontsize=16)
        plt.ylabel('Score', fontsize=16)
        # plt.title('Precision/Recall vs CFM', fontsize=16)
        plt.legend(prop={'size': 12}, loc = 'upper left')
        plt.xticks([0.1, 0.2, 0.3, 0.4, 0.5], fontsize=16)
        plt.yticks(fontsize=16)
        plt.grid(True)
        plt.tight_layout()
        file_path = op.join(self.path_manager.base_figures_dir, 'cfm_sensitivity.pdf')
        self.path_manager.save_pdf_file(plt, abs_file_path=file_path, override=True, pad_inches = 0.02)
        plt.show()


    def cvm_cfm_sensitivity_analysis(self):
        self.load_existing_kpm_dict_wcr()
        df_kpm_wcr_1 = self.wcr_kpm_dict[1.0]
        df_kpm_citation = self.citation_based_kmp
        metrics_data = []
        for idx, df_kpm_ in enumerate([df_kpm_wcr_1, df_kpm_citation]):
            df_kpm = pd.DataFrame(
                self.get_max_symmetric_elements(df_kpm_.values), 
                index=df_kpm_.index, 
                columns=df_kpm_.columns
            )

            type = 'wcr_1' if idx == 0 else 'citation'

            for cvm_ in range(90, 96):
                for cfm_ in range(10, 55, 10):
                    cvm = cvm_ / 100
                    cfm = cfm_ / 100
                    self.kpn_parms.cvm_threshold = cvm_ / 100
                    self.kpn_parms.cfm_threshold = cfm_ / 100
                    print(f"CVM={cvm}, CFM={cfm}")

                    df_edges_roc = self.KPN.kpn_edges(df_kpm, self.kpn_parms, optimize=True)
                    df_edges_roc['ground_truth'] = df_edges_roc[['source', 'target']].apply(
                        tuple, axis=1
                    ).map(self.llm_pair_dict)
                    df_edges_roc['true_label'] = df_edges_roc['ground_truth'].isin([1, 2])
                    
                    precision = df_edges_roc['true_label'].sum() / len(df_edges_roc)
                    recall = df_edges_roc['true_label'].sum() / len(self.positive_gt_pairs)
                    if precision + recall > 0:
                        f1 = 2 * precision * recall / (precision + recall)
                    else:
                        f1 = 0.0
                    # Add precision row
                    metrics_data.append({
                        'type': type,
                        'cvm': cvm,
                        'cfm': cfm,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1
                    })
        df_metrics = pd.DataFrame(metrics_data)
        self.plot_cvm_cfm_curves(df_metrics, type_filter='wcr_1')
        # self.plot_cvm_cfm_curves(df_metrics, type_filter='citation')

    def get_sub_graph(self, G, source, target):
        target_parents = set()
        for t in target:
            target_parents.update(G.predecessors(t))

        source_children = set()
        for s in source:
            source_children.update(G.successors(s))

        inter_nodes = target_parents & source_children
        # 可选：是否包含 source 和 target 节点本身
        inter_nodes.update(set(source).union(target))
        # 构建诱导子图
        disciplines = self.target_concepts.loc[self.target_concepts.display_name.isin(set(source).union(target))].llm_annotation.drop_duplicates().tolist()
        inter_nodes = self.target_concepts.loc[self.target_concepts.display_name.isin(inter_nodes) & self.target_concepts.llm_annotation.isin(disciplines)].display_name.tolist()
        subG = G.subgraph(inter_nodes).copy()
        
        return subG
        
    def learning_recommendations(self, ):
        # 生成学习子图
        self.load_existing_kpm_dict_wcr()
        df_kpm_wcr_1 = self.wcr_kpm_dict[1.0]
        df_kpm = pd.DataFrame(
            self.get_max_symmetric_elements(df_kpm_wcr_1.values), 
            index=df_kpm_wcr_1.index, 
            columns=df_kpm_wcr_1.columns
        )
        self.kpn_parms.cvm_threshold = 0.95
        df_edges_roc = self.KPN.kpn_edges(df_kpm, self.kpn_parms, optimize=True)
        non_level_0 = self.target_concepts.loc[(self.target_concepts.level > 0)].display_name.tolist()
        df_edges_roc = df_edges_roc.loc[df_edges_roc.source.isin(non_level_0) & df_edges_roc.target.isin(non_level_0)].copy()
        
        G = nx.DiGraph()
        edges = df_edges_roc[['source', 'target', 'weight']].values.tolist()
        G.add_edges_from([(src, tgt, {'weight': w}) for src, tgt, w in edges])
        # transition_dict = {'field_1': ['Statistics'], 'field_2': ['Artificial intelligence']}
        # transition_dict = {'field_1': ['Physics'], 'field_2': ['Artificial intelligence']}
        transition_dict = {'field_1': ['Machine learning'], 'field_2': ['Quantum computer', 'Quantum', 'Optics', 'Quantum mechanics']}
        # transition_dict = {'field_1': ['Optics', 'Astrophysics', 'Wavelength', 'Dispersion (optics)'], 'field_2': ['Computer vision', 'Computer graphics (images)']}
        # transition_dict = {'field_1': ['Computational physics'], 'field_2': ['Machine learning', 'Artificial intelligence', 'Deep learning']}
        # transition_dict = {'field_1': ['Computational physics', 'Quantum mechanics', 'Quantum', 'Astronomy'], 'field_2': ['Computer vision', 'Object detection', 'Segmentation', 'Image quality', 'Visualization']}
        
        subG = self.get_sub_graph(G, transition_dict['field_1'], transition_dict['field_2'])
        # subG = self.get_sub_graph(G, ['Statistics'], ['Artificial intelligence'])
        df_edges_subG = pd.DataFrame(list(subG.edges()), columns=['source', 'target'])
        df_subG_edges = pd.merge(
            df_edges_subG,
            df_edges_roc,
            on=['source', 'target'],
            how='left'
        )
        edge_file_name = op.join(self.path_manager.base_edges_dir, f"{'_'.join([transition_dict['field_1'][0], transition_dict['field_2'][0]]).replace(' ', '_')}_edge.csv")
        # edge_file_name = op.join(self.path_manager.base_edges_dir, f"{'_'.join(['Statistics', 'Artificial intelligence']).replace(' ', '_')}_edge.csv")
        self.path_manager.save_csv_file(variable=df_subG_edges, file_name=edge_file_name, override=True)
        df_subG_nodes = self.target_concepts.loc[self.target_concepts.display_name.isin(subG.nodes)][['display_name','llm_annotation','multiple_level_0_ancestors', 'level', 'gpt_abbrev']]
        color_dict = {
            "Mathematics": "#1f77b4",         # 蓝色
            "Engineering": "#ff7f0e",         # 橙色
            "Physics": "#2ca02c",             # 绿色
            "Computer science": "#d62728",    # 红色
            # "Chemical engineering": "#9467bd" # 紫色
        }
        
        df_subG_nodes['node_color'] = df_subG_nodes['llm_annotation'].map(color_dict)
        # node_file_name = op.join(self.path_manager.base_edges_dir, f"{'_'.join([transition_dict['field_1'][0], transition_dict['field_2'][0]]).replace(' ', '_')}_node.csv")
        node_file_name = op.join(self.path_manager.base_edges_dir, f"{'_'.join([transition_dict['field_1'][0], transition_dict['field_2'][0]]).replace(' ', '_')}_node.csv")
        self.path_manager.save_csv_file(variable=df_subG_nodes, file_name=node_file_name, override=True)
        
        return G
    
    def calculate_performance_metrics(self, df_edges):
        
        # Generate percentile thresholds for weight (0 to 100, step 0.1)
        weight_percentiles = np.arange(0, 100.1, 0.1)
        weight_thresholds = [np.percentile(df_edges['weight'], p) for p in weight_percentiles]
        
        # Use unique values as thresholds for max_count, sorted in ascending order
        max_count_thresholds = sorted(df_edges['max_count'].unique())
        
        evaluations = []
        # Calculate performance metrics for weight thresholds
        for threshold in weight_thresholds:
            # Filter samples predicted as positive based on threshold
            predicted_positive = df_edges['weight'] >= threshold
            
            # Calculate TP, FP, FN
            true_positive = (predicted_positive & df_edges['true_label']).sum()
            false_positive = (predicted_positive & ~df_edges['true_label']).sum()
            false_negative = (~predicted_positive & df_edges['true_label']).sum()
            
            evaluations.append(['weight', threshold, int(true_positive), int(false_positive), int(false_negative)])
            
        
        # Calculate performance metrics for max_count thresholds
        for threshold in max_count_thresholds:
            # Filter samples predicted as positive based on threshold
            predicted_positive = df_edges['max_count'] >= threshold
            
            # Calculate TP, FP, FN
            true_positive = (predicted_positive & df_edges['true_label']).sum()
            false_positive = (predicted_positive & ~df_edges['true_label']).sum()
            false_negative = (~predicted_positive & df_edges['true_label']).sum()
            
            evaluations.append(['count', threshold, int(true_positive), int(false_positive), int(false_negative)])
        
        df_results = pd.DataFrame(evaluations, columns = ['value', 'threshold' ,'tp', 'fp', 'fn'])

        return df_results

    def plot_degree_resolution_auc(self, df_all_results, value_type='weight'):
        import matplotlib.pyplot as plt
        import numpy as np
        import matplotlib.cm as cm
        import pandas as pd
        from sklearn.metrics import auc
        
        # 筛选指定的value类型
        df_filtered = df_all_results[df_all_results['value'] == value_type].copy()
        
        if df_filtered.empty:
            print(f"No data for value_type={value_type}")
            return
        
        # 分离citation=False和citation=True的数据
        df_plot = df_filtered[df_filtered['citation'] == False].copy()
        df_plot_true = df_filtered[df_filtered['citation'] == True].copy()
        
        if df_plot.empty:
            print("No data for citation=False")
            return
        
        unique_degrees = ['4', '5', '6', '7', '8', '9', '10', '11', '12', 'log']
        unique_resolutions = np.sort(df_plot['resolution'].unique())
        
        # 计算各组合的AUC
        def calculate_auc(group):
            if len(group) < 2:
                return np.nan
            
            # 按threshold排序
            group = group.sort_values('threshold')
            
            # 计算总的正样本和负样本数量
            total_positives = group['tp'].max()
            total_negatives = group['fp'].max()
            
            if total_positives == 0 or total_negatives == 0:
                return np.nan
            
            # 计算TPR和FPR
            tpr = group['tp'] / total_positives
            fpr = group['fp'] / total_negatives
            
            # 使用sklearn的auc函数计算
            try:
                auc_score = auc(fpr, tpr)
                return auc_score
            except:
                return np.nan
        
        # 为citation=False的数据计算AUC
        auc_results = []
        for degree in unique_degrees:
            for resolution in unique_resolutions:
                group = df_plot[(df_plot['degree'] == degree) & (df_plot['resolution'] == resolution)]
                if not group.empty:
                    auc_score = calculate_auc(group)
                    auc_results.append({
                        'degree': degree,
                        'resolution': resolution,
                        'auc': auc_score,
                        'citation': False
                    })
        
        # 为citation=True的数据计算AUC
        citation_auc = None
        if not df_plot_true.empty:
            citation_auc = calculate_auc(df_plot_true)
        
        df_auc = pd.DataFrame(auc_results)
        
        # 设置颜色映射
        colors = cm.get_cmap('tab10', 10)
        
        # 第一张图：横轴degree，每条线是不同resolution
        plt.figure(figsize=(5, 4.5))
        
        for idx, resolution in enumerate(unique_resolutions):
            group = df_auc[df_auc['resolution'] == resolution].copy()
            if group.empty:
                continue
            
            # 按照 unique_degrees 的顺序排序
            group['degree_order'] = group['degree'].astype(str).map(lambda x: unique_degrees.index(x) if x in unique_degrees else 999)
            group = group.sort_values('degree_order')
            
            # 过滤掉NaN值
            group = group.dropna(subset=['auc'])
            if group.empty:
                continue
            
            color = colors(idx)
            plt.plot(group['degree'].astype(str), group['auc'], marker='o', color=color, 
                    label=f'AUC (Res.={resolution:.1f})', linestyle='-')
        
        # Citation AUC
        if citation_auc is not None and not np.isnan(citation_auc):
            plt.axhline(y=citation_auc, color='black', linestyle='-', linewidth=2,
                    label='AUC (Citation)')
        
        plt.xlabel('Degree', fontsize=16)
        plt.ylabel('AUC', fontsize=16)
        plt.title(f'AUC Sensitivity Analysis ({value_type.capitalize()})', fontsize=14)
        
        # 手动设置 x 轴刻度顺序
        plt.xticks(unique_degrees, fontsize=16)
        
        plt.legend(prop={'size': 8}, loc='upper left', ncol=1, bbox_to_anchor=(0.09, 0.95))
        plt.yticks(fontsize=16)
        plt.grid(True)
        plt.tight_layout()
        
        file_path = op.join(self.path_manager.base_figures_dir, f'auc_sensitivity_degree_resolution_{value_type}.pdf')
        self.path_manager.save_pdf_file(plt, abs_file_path=file_path, override=True, pad_inches=0.02)
        plt.show()
        
        # 第二张图：横轴resolution，每条线是不同degree
        plt.figure(figsize=(5, 4.5))
        
        for idx, degree in enumerate(unique_degrees):
            group = df_auc[df_auc['degree'] == degree].sort_values('resolution')
            if group.empty:
                continue
            
            # 过滤掉NaN值
            group = group.dropna(subset=['auc'])
            if group.empty:
                continue
            
            color = colors(idx)
            plt.plot(group['resolution'], group['auc'], marker='o', color=color, 
                    label=f'AUC (Deg.={degree})', linestyle='-')
        
        # Citation AUC
        if citation_auc is not None and not np.isnan(citation_auc):
            plt.axhline(y=citation_auc, color='black', linestyle='-', linewidth=2,
                    label='AUC (Citation)')
        
        plt.xlabel('Resolution', fontsize=16)
        plt.ylabel('AUC', fontsize=16)
        plt.title(f'AUC Sensitivity Analysis ({value_type.capitalize()})', fontsize=14)
        plt.legend(prop={'size': 8}, loc='upper left', ncol=1, bbox_to_anchor=(0.09, 0.95))
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.grid(True)
        plt.tight_layout()
        
        file_path = op.join(self.path_manager.base_figures_dir, f'auc_sensitivity_resolution_degree_{value_type}.pdf')
        self.path_manager.save_pdf_file(plt, abs_file_path=file_path, override=True, pad_inches=0.02)
        plt.show()
        
        return df_auc


    @cache_results(
        cache_dir=path_manager.base_data_dir, 
        filename_pattern="kpm_degree_resolution_sensitivity_evaluation.csv",
        version=1
    )
    def kpm_degree_resolution_sensitivity_evaluation(self, kpn_parms=None, debugging=False, skip_cache = False):
        """Evaluate KPM performance across different degrees and generate ROC curves"""
        if kpn_parms is None:
            kpn_parms = KPNParams(
                disciplines=[],
                row_normalize=False,
                threshold_mode='value_count_limit',
                quantile_threshold=0.65,
                cvm_threshold=0.95,
                cfm_threshold=0.2,
                matrix_filter=True,
                matrix_imshow=False,
                graph_show=True,
                layout="graphviz",
                save_kpn=True,
                old_matrix_version=False
            )
        
        self.load_existing_kpm_dict_all_degree_resolution()
        evaluation_results_degree = []
        
        self.all_degree_resolution_sensitivity_kpm_dict['citation_baseline'] = self.citation_based_kmp
        # Evaluate each degree
        for param_key, df_kpm_original in self.all_degree_resolution_sensitivity_kpm_dict.items():
            # Extract degree from param_key

            degree, resolution = self.parse_param_key_(param_key)
            
            df_kpm = pd.DataFrame(
                self.get_max_symmetric_elements(df_kpm_original.values), 
                index=df_kpm_original.index, 
                columns=df_kpm_original.columns
            )

            df_edges_roc = self.KPN.kpn_edges(df_kpm, kpn_parms, optimize=False)
            df_edges_roc['ground_truth'] = df_edges_roc[['source', 'target']].apply(tuple, axis=1).map(self.llm_pair_dict)
            df_edges_roc['true_label'] = df_edges_roc['ground_truth'].isin([1, 2])
            
            df_evaluation_result = self.calculate_performance_metrics(df_edges_roc)
            
            df_evaluation_result['degree'] = degree
            df_evaluation_result['resolution'] = resolution
            df_evaluation_result['citation'] = (param_key == 'citation_baseline')
            evaluation_results_degree.append(df_evaluation_result)

        # Combine all results
        df_all_results = pd.concat(evaluation_results_degree, ignore_index=True)
        
        return df_all_results


    def plot_degree_resolution_sensitivity_auc(self, df_aucs, value_type='count'):
        import matplotlib.pyplot as plt
        import numpy as np
        import matplotlib.cm as cm
        
        # 筛选指定的value_type和citation=False的数据
        df_plot = df_aucs[(df_aucs['citation'] == False) & (df_aucs['value'] == value_type)].copy()
        df_plot_true = df_aucs[(df_aucs['citation'] == True) & (df_aucs['value'] == value_type)].copy()
        
        if df_plot.empty:
            print(f"No data for citation=False and value={value_type}")
            return
        
        unique_degrees = ['4', '5', '6', '7', '8', '9', '10', '11', '12', 'log']
        unique_resolutions = np.sort(df_plot['resolution'].unique())
        
        # 获取 Citation 的结果
        citation_auc = None
        if not df_plot_true.empty:
            citation_auc = df_plot_true['auc'].iloc[0]

        # 设置颜色映射
        colors = cm.get_cmap('tab10', 10)
        
        # 第一张图：横轴degree，每条线是不同resolution
        plt.figure(figsize=(5, 4.5))
        
        # 绘制不同resolution的AUC曲线
        for idx, resolution in enumerate(unique_resolutions):
            group = df_plot[df_plot['resolution'] == resolution].copy()
            if group.empty:
                continue
            
            # 按照 unique_degrees 的顺序排序
            group['degree_order'] = group['degree'].astype(str).map(
                lambda x: unique_degrees.index(x) if x in unique_degrees else 999
            )
            group = group.sort_values('degree_order')
            
            color = colors(idx)
            plt.plot(group['degree'].astype(str), group['auc'], 
                    marker='o', color=color, linestyle='-',
                    label=f'AUC (Res.={resolution:.1f})')
        
        # Citation AUC 水平线
        if citation_auc is not None:
            plt.axhline(y=citation_auc, color='black', 
                    linestyle='-', linewidth=2,
                    label='AUC (Citation)')

        plt.xlabel('Average Degree', fontsize=16)
        plt.ylabel('AUC', fontsize=16)
        # plt.title(f'AUC Sensitivity Analysis ({value_type.title()})', fontsize=14)
        
        # 手动设置 x 轴刻度顺序
        plt.xticks(unique_degrees, fontsize=16)
        
        plt.legend(prop={'size': 9}, loc='upper left', ncol=2, bbox_to_anchor=(0.06, 0.96), framealpha=0.5)
        plt.yticks(fontsize=16)
        plt.grid(True)
        plt.tight_layout()
        
        file_path = op.join(self.path_manager.base_figures_dir, f'sensitivity_degree_resolution_auc_{value_type}.pdf')
        self.path_manager.save_pdf_file(plt, abs_file_path=file_path, override=True, pad_inches=0.02)
        plt.show()
        
        # 第二张图：横轴resolution，每条线是不同degree
        plt.figure(figsize=(5, 4.5))
        
        # 绘制不同degree的AUC曲线
        for idx, degree in enumerate(unique_degrees):
            group = df_plot[df_plot['degree'] == degree].sort_values('resolution')
            if group.empty:
                continue
            
            color = colors(idx)
            plt.plot(group['resolution'], group['auc'], 
                    marker='o', color=color, linestyle='-',
                    label=f'AUC (Deg.={degree})')
        
        # Citation AUC 水平线
        if citation_auc is not None:
            plt.axhline(y=citation_auc, color='black', 
                    linestyle='-', linewidth=2,
                    label='AUC (Citation)')

        plt.xlabel('Louvain Resolution', fontsize=16)
        plt.ylabel('AUC', fontsize=16)
        # plt.title(f'AUC Sensitivity Analysis ({value_type.title()})', fontsize=14)
        plt.legend(prop={'size': 9}, loc='upper left', ncol=2, bbox_to_anchor=(0.06, 0.96), framealpha=0.5)
        
        # 设置x轴刻度，间隔显示
        plt.xticks([0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0], fontsize=16)
        plt.yticks(fontsize=16)
        plt.grid(True)
        plt.tight_layout()

        
        file_path = op.join(self.path_manager.base_figures_dir, f'sensitivity_resolution_degree_auc_{value_type}.pdf')
        self.path_manager.save_pdf_file(plt, abs_file_path=file_path, override=True, pad_inches=0.02)
        plt.show()



    @cache_results(
        cache_dir=path_manager.base_data_dir, 
        filename_pattern="df_aucs_degree_resolution_sensitivity_evaluation.csv",
        version=1
    )
    def calculate_auc_by_groups(self, df_all_results = None, skip_cache = False):
        import pandas as pd
        import numpy as np
        from sklearn.metrics import auc
        if df_all_results == None:
            df_all_results = self.kpm_degree_resolution_sensitivity_evaluation()
            df_all_results['degree'] = df_all_results['degree'].astype(str)
        def calculate_auc_for_group(group):
            """为单个组合计算AUC"""
            if len(group) < 2:
                return np.nan
            
            # 按threshold排序
            group = group.sort_values('threshold')
            
            # 计算总的正样本和负样本数量
            total_positives = group['tp'].max()
            total_negatives = group['fp'].max()
            if total_positives <= 0 or total_negatives <= 0:
                return np.nan
            
            # 计算TPR和FPR
            tpr = group['tp'] / total_positives
            fpr = group['fp'] / total_negatives
            
            # 使用sklearn的auc函数计算
            try:
                auc_score = auc(fpr, tpr)
                return auc_score
            except:
                return np.nan
        
        # 按条件分组并计算AUC
        auc_results = []
        
        # 对于citation=False的情况，按value, degree, resolution分组
        df_false = df_all_results[df_all_results['citation'] == False]
        if not df_false.empty:
            grouped = df_false.groupby(['value', 'degree', 'resolution'])
            
            for (value, degree, resolution), group in grouped:
                auc_score = calculate_auc_for_group(group)
                auc_results.append({
                    'value': value,
                    'degree': degree,
                    'resolution': resolution,
                    'citation': False,
                    'auc': auc_score
                })
        
        # 对于citation=True的情况，按value分组
        df_true = df_all_results[df_all_results['citation'] == True]
        if not df_true.empty:
            grouped = df_true.groupby(['value'])
            
            for (value), group in grouped:
                auc_score = calculate_auc_for_group(group)
                auc_results.append({
                    'value': value[0],
                    'degree': np.nan,
                    'resolution': np.nan,
                    'citation': True,
                    'auc': auc_score
                })
        
        # 转换为DataFrame
        df_auc = pd.DataFrame(auc_results)
        
        return df_auc


if __name__ == "__main__":
    
    kpn_parms = KPNParams(
        disciplines=[],
        row_normalize=False,
        threshold_mode='value_count_limit',
        quantile_threshold=0.65,
        cvm_threshold=0.95,
        cfm_threshold=0.2,
        matrix_filter=True,
        matrix_imshow=False,
        graph_show=True,
        layout="graphviz",
        save_kpn=True,
        old_matrix_version=False
    )
    # Initialize generator
    generator = KPMEvaluator(kpn_parms = kpn_parms, sub_discipline = True)
    # generator.cvm_cfm_sensitivity_analysis()
    # generator = KPMEvaluator(kpn_parms = kpn_parms, sub_discipline = True)
    generator.learning_recommendations()
    # df_results_degree = generator.calculate_precision_recall(type= 'degree')
    # generator.load_existing_kpm_dict_period()
    # generator.load_existing_kpm_dict_wcr()
    df_results_degree = generator.calculate_precision_recall(type= 'degree', skip_cache = True)
    df_results_resolution = generator.calculate_precision_recall(type= 'resolution', skip_cache = True)
    # df_results_wcr = generator.calculate_precision_recall(type= 'wcr')
    # df_results_period = generator.calculate_precision_recall(type= 'period')
    # generator = KPMEvaluator(kpn_parms = kpn_parms, sub_discipline = False)
    # df_results_degree_resolution = generator.calculate_precision_recall(type= 'all_degree_resolution')
    # df_all_results = generator.kpm_degree_resolution_sensitivity_evaluation(skip_cache = True)
    # df_aucs = generator.calculate_auc_by_groups(skip_cache = False)
    print(5)

# nohup python ./network/kpm_evaluation_metrics_cross.py >> kpm_evaluation_metrics_cross.log 2>&1 &
# ps aux | grep kpm_evaluation_metrics_cross.py
# pkill -f kpm_evaluation_metrics_cross.py
