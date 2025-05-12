import os, sys
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.append(project_dir)

import os
import os.path as op
import numpy as np
import pandas as pd
import tqdm, pickle
import logging
from utils.config import PathManager, calculate_runtime
from utils.database import DatabaseManager
from utils.concept import Concept
from network.kpn import KPN
from utils.params import KPNParams
import matplotlib.pyplot as plt


# Configure root logger to output to stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

class KPMEvaluator:
    """
    Class for constructing Knowledge Precedence Matrix (KPM) from research trajectories
    """
    def __init__(self, sub_concepts = False):
        """
        Initialize the KPN constructor
        
        Args:
            path_manager: Manager for handling file paths
            db_manager: Manager for database operations
        """
        self.KPN = KPN()
        self.path_manager = PathManager()
        self.db_manager = DatabaseManager()
        self.concepts_table = Concept.discipline_category_classification_llm(with_abbreviation = True)
        self.target_concept_table = pd.read_csv(op.join(project_dir, './llm_annotation/df_selected_top_concepts.csv'))
        self.discipline_to_ids = self.target_concept_table.groupby('llm_annotation')['id'].apply(list).to_dict()
        self.discipline_to_names = self.target_concept_table.groupby('llm_annotation')['display_name'].apply(list).to_dict()
        self.target_concept_ids = pd.read_csv(op.join(project_dir, './llm_annotation/df_selected_top_concepts.csv')).id.drop_duplicates().tolist()
        

        self.output_dir_communities = op.join(self.path_manager.external_file_dir, 'CCNs', 'CCNs_valid_authors')
        self.kpm_path = None
        self.evaluation_path = None
        self.processed_ids_file = None
        self.kpm_matrices = {}
        self.period_kpm_matrices = {}
        self.processed_ids = []

        self.sub_concepts = sub_concepts
        if self.sub_concepts:
            self.llm_pair_id_dicts, self.llm_pair_dicts, self.positive_gt_pairs = self.load_llm_annotation_results()
        else:
            self.llm_pair_id_dicts, self.llm_pair_dicts, self.positive_gt_pairs = self.load_llm_annotation_results_cross()
            
        self.kpm_path = self.path_manager.ensure_folder_exists(op.join(self.path_manager.ccns_dir, 'kpms_sub'))
        self.evaluation_path = self.path_manager.ensure_folder_exists(op.join(self.path_manager.ccns_dir, 'evaluation_sub'))
    
        self.cross_kpm_path = self.path_manager.ensure_folder_exists(op.join(self.path_manager.ccns_dir, 'kpms'))
        self.cross_evaluation_path = self.path_manager.ensure_folder_exists(op.join(self.path_manager.ccns_dir, 'evaluation'))
        
        # Define default WCR values list
        self.wcr_list = [1, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6]
        # Results save path
        self.results_path = self.path_manager.ensure_folder_exists(op.join(self.path_manager.ccns_dir, 'analysis'))
        if sub_concepts:
            self.parquet_path = op.join(self.results_path, 'all_authors_precision_sub_concepts.parquet')
        
        else: self.parquet_path = op.join(self.results_path, 'all_authors_precision.parquet')
        
        self.logger = logging.getLogger(__name__)
        
        self.logger.setLevel(logging.INFO)
        
    def load_llm_annotation_results_cross(self):
        
        llm_pair_dict_path = op.join(self.path_manager.base_file_dir, 'llm_pair_dict.pkl')
        llm_pair_id_dict_path = op.join(self.path_manager.base_file_dir, 'llm_id_pair_dict.pkl')
        
        with open(llm_pair_dict_path, 'rb') as f:
            llm_pair_dicts = pickle.load(f)
        with open(llm_pair_id_dict_path, 'rb') as f:
            llm_pair_id_dicts = pickle.load(f)

        positive_gt_pairs = [
            k for k, v in llm_pair_id_dicts.items() if v > 0
        ]
        return llm_pair_id_dicts, llm_pair_dicts, positive_gt_pairs
        
        
    def load_llm_annotation_results(self):
        
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
        positive_gt_pairs = {}
        for discipline, llm_pair_id_dict in llm_pair_id_dicts.items():
            positive_gt_pairs[discipline] = {pair for pair, vote in llm_pair_id_dict.items() if vote > 0}
        return llm_pair_id_dicts, llm_pair_dicts, positive_gt_pairs

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

    def cross_kpm_evaluation_wcr_recall(self, kpn_parms, debugging = True):
        evaluation_results = []
        evaluation_result_path = op.join(self.path_manager.base_data_dir, f'cross_wcr_results_percent_with_cvm_{kpn_parms.cvm_threshold}_cfm_{kpn_parms.cfm_threshold}.csv')
        
        if not op.exists(evaluation_result_path) or debugging:

            for wcr in self.wcr_list:
                wcr_str = f"{wcr:.1f}" if wcr == 1 else str(wcr)
                df_kpm_path = op.join(self.cross_kpm_path, f'''df_kpm_wcr_{wcr_str}.csv''')
                df_kpm_original = pd.read_csv(df_kpm_path, index_col=0)
                
                df_kpm = pd.DataFrame(self.KPN.get_max_symmetric_elements(df_kpm_original.values), index=df_kpm_original.index, columns=df_kpm_original.columns)
                
                df_edges = self.KPN.kpn_edges(df_kpm, kpn_parms)

                df_edges['ground_truth'] = df_edges[['source', 'target']].apply(tuple, axis=1).map(self.llm_pair_id_dicts)

                precesion = (df_edges.ground_truth>0).sum() / len(df_edges)

                recall = (df_edges.ground_truth>0).sum() / len(self.positive_gt_pairs)

                evaluation_results.append([wcr, precesion, recall])

            df_results = pd.DataFrame(evaluation_results, columns=['wcr', 'precesion','recall'])
            df_pivot = (
                df_results
                .melt(id_vars=['wcr'], value_vars=['precesion', 'recall'], var_name='metric', value_name='value')
                .pivot(index=['metric'], columns='wcr', values='value')
                .reset_index()
            )
            def to_percent(x):
                if isinstance(x, (int, float)):
                    return '{:.2f}%'.format(x * 100)
                else:
                    return x
            def percent_colname(col):
                try:
                    val = float(col)
                    if 0 < val <= 1:
                        return '{}%'.format(int(round(val * 100)))
                except (ValueError, TypeError):
                    pass
                return col

            df_pivot.rename(columns={col: percent_colname(col) for col in df_pivot.columns}, inplace=True)
            wcr_cols = df_pivot.columns.difference(['metric'])
            df_pivot[wcr_cols] = df_pivot[wcr_cols].apply(lambda col: col.map(to_percent))
            
            self.path_manager.save_csv_file(df_pivot, abs_file_path=evaluation_result_path)
        else: 
            df_pivot = pd.read_csv(evaluation_result_path)

        return df_pivot, df_evaluation_results

    def cross_kpm_evaluation_period_recall(self, kpn_parms, debugging = True):
        evaluation_results = []
        evaluation_result_path = op.join(self.path_manager.base_data_dir, f'cross_period_results_percent_with_cvm_{kpn_parms.cvm_threshold}_cfm_{kpn_parms.cfm_threshold}.csv')
        
        if not op.exists(evaluation_result_path) or debugging:

            for period in ['early', 'middle', 'later']:
                df_kpm_path = op.join(self.cross_kpm_path, f'''df_kpm_period_{period}.csv''')
                df_kpm_original = pd.read_csv(df_kpm_path, index_col=0)
                
                df_kpm = pd.DataFrame(self.KPN.get_max_symmetric_elements(df_kpm_original.values), index=df_kpm_original.index, columns=df_kpm_original.columns)
                
                df_edges = self.KPN.kpn_edges(df_kpm, kpn_parms, optimize=True)

                df_edges['ground_truth'] = df_edges[['source', 'target']].apply(tuple, axis=1).map(self.llm_pair_id_dicts)

                precesion = (df_edges.ground_truth>0).sum() / len(df_edges)

                recall = (df_edges.ground_truth>0).sum() / len(self.positive_gt_pairs)

                evaluation_results.append([period, precesion, recall])

            df_results = pd.DataFrame(evaluation_results, columns=['period', 'precesion','recall'])
            df_pivot = (
                df_results
                .melt(id_vars=['period'], value_vars=['precesion', 'recall'], var_name='metric', value_name='value')
                .pivot(index=['metric'], columns='period', values='value')
                .reset_index()
            )
            def to_percent(x):
                if isinstance(x, (int, float)):
                    return '{:.2f}%'.format(x * 100)
                else:
                    return x
            def percent_colname(col):
                try:
                    val = float(col)
                    if 0 < val <= 1:
                        return '{}%'.format(int(round(val * 100)))
                except (ValueError, TypeError):
                    pass
                return col

            df_pivot.rename(columns={col: percent_colname(col) for col in df_pivot.columns}, inplace=True)
            period_cols = df_pivot.columns.difference(['metric'])
            df_pivot[period_cols] = df_pivot[period_cols].apply(lambda col: col.map(to_percent))
            
            self.path_manager.save_csv_file(df_pivot, abs_file_path=evaluation_result_path)
        else: 
            df_pivot = pd.read_csv(evaluation_result_path)

        return df_pivot, df_evaluation_results


    def cross_kpm_evaluation_wcr(self, kpn_parms, debugging = True):
        evaluation_results = []
        evaluation_results_roc = []
        evaluation_result_path = op.join(self.path_manager.base_data_dir, f'cross_wcr_results_percent.csv')
        roc_results_path = op.join(self.path_manager.base_data_dir, 'cross_wcr_evaluation_results.csv')
        
        if not op.exists(roc_results_path) or not op.exists(evaluation_result_path) or debugging:
            for wcr in self.wcr_list:
                wcr_str = f"{wcr:.1f}" if wcr == 1 else str(wcr)
                df_kpm_path = op.join(self.cross_kpm_path, f'''df_kpm_wcr_{wcr_str}.csv''')
                df_kpm_original = pd.read_csv(df_kpm_path, index_col=0)
                
                df_kpm = pd.DataFrame(self.KPN.get_max_symmetric_elements(df_kpm_original.values), index=df_kpm_original.index, columns=df_kpm_original.columns)

                df_edges_roc = self.KPN.kpn_edges(df_kpm, kpn_parms, optimize=False)
                df_edges_roc['ground_truth'] = df_edges_roc[['source', 'target']].apply(tuple, axis=1).map(self.llm_pair_id_dicts)
                df_edges_roc['true_label'] = df_edges_roc['ground_truth'].isin([1, 2])
                df_evaluation_result = self.calculate_performance_metrics(df_edges_roc)
                df_evaluation_result['wcr'] = wcr
                evaluation_results_roc.append(df_evaluation_result)

                df_edges = self.KPN.kpn_edges(df_kpm, kpn_parms)
                df_edges['ground_truth'] = df_edges[['source', 'target']].apply(tuple, axis=1).map(self.llm_pair_id_dicts)
                precesion = (df_edges.ground_truth>0).sum() / len(df_edges)
                recall = (df_edges.ground_truth>0).sum() / len(self.positive_gt_pairs)
                evaluation_results.append([wcr, precesion, recall])

            df_evaluation_results = pd.concat(evaluation_results_roc)
            df_results = pd.DataFrame(evaluation_results, columns=['wcr', 'precesion','recall'])
            df_pivot = (
                df_results
                .melt(id_vars=['wcr'], value_vars=['precesion', 'recall'], var_name='metric', value_name='value')
                .pivot(index=['metric'], columns='wcr', values='value')
                .reset_index()
            )
            def to_percent(x):
                if isinstance(x, (int, float)):
                    return '{:.2f}%'.format(x * 100)
                else:
                    return x
            def percent_colname(col):
                try:
                    val = float(col)
                    if 0 < val <= 1:
                        return '{}%'.format(int(round(val * 100)))
                except (ValueError, TypeError):
                    pass
                return col

            df_pivot.rename(columns={col: percent_colname(col) for col in df_pivot.columns}, inplace=True)
            wcr_cols = df_pivot.columns.difference(['metric'])
            df_pivot[wcr_cols] = df_pivot[wcr_cols].apply(lambda col: col.map(to_percent))
            
            self.path_manager.save_csv_file(df_pivot, abs_file_path=evaluation_result_path)
            self.path_manager.save_csv_file(df_evaluation_results, abs_file_path=roc_results_path)
        else: 
            df_pivot = pd.read_csv(evaluation_result_path)
            df_evaluation_results = pd.read_csv(roc_results_path)

        return df_pivot, df_evaluation_results

    def cross_kpm_evaluation_period(self, kpn_parms):
        evaluation_results = []
        evaluation_results_roc = []
        evaluation_result_path = op.join(self.path_manager.base_data_dir, 'cross_period_results_percent.csv')
        roc_results_path = op.join(self.path_manager.base_data_dir, 'cross_period_evaluation_results.csv')
        
        if not op.exists(roc_results_path) or not op.exists(evaluation_result_path):
            for period in ['early', 'middle', 'later']:
                df_kpm_path = op.join(self.cross_kpm_path, f'''df_kpm_period_{period}.csv''')
                df_kpm_original = pd.read_csv(df_kpm_path, index_col=0)
                
                df_kpm = pd.DataFrame(self.KPN.get_max_symmetric_elements(df_kpm_original.values), index=df_kpm_original.index, columns=df_kpm_original.columns)

                df_edges_roc = self.KPN.kpn_edges(df_kpm, kpn_parms, optimize=False)
                df_edges_roc['ground_truth'] = df_edges_roc[['source', 'target']].apply(tuple, axis=1).map(self.llm_pair_id_dicts)
                df_edges_roc['true_label'] = df_edges_roc['ground_truth'].isin([1, 2])
                df_evaluation_result = self.calculate_performance_metrics(df_edges_roc)
                df_evaluation_result['period'] = period
                evaluation_results_roc.append(df_evaluation_result)

                df_edges = self.KPN.kpn_edges(df_kpm, kpn_parms)
                df_edges['ground_truth'] = df_edges[['source', 'target']].apply(tuple, axis=1).map(self.llm_pair_id_dicts)
                precesion = (df_edges.ground_truth>0).sum() / len(df_edges)
                recall = (df_edges.ground_truth>0).sum() / len(self.positive_gt_pairs)
                evaluation_results.append([period, precesion, recall])

            df_evaluation_results = pd.concat(evaluation_results_roc)
            df_results = pd.DataFrame(evaluation_results, columns=['period', 'precesion','recall'])
            df_pivot = (
                df_results
                .melt(id_vars=['period'], value_vars=['precesion', 'recall'], var_name='metric', value_name='value')
                .pivot(index=['metric'], columns='period', values='value')
                .reset_index()
            )
            def to_percent(x):
                if isinstance(x, (int, float)):
                    return '{:.2f}%'.format(x * 100)
                else:
                    return x
            def percent_colname(col):
                try:
                    val = float(col)
                    if 0 < val <= 1:
                        return '{}%'.format(int(round(val * 100)))
                except (ValueError, TypeError):
                    pass
                return col

            df_pivot.rename(columns={col: percent_colname(col) for col in df_pivot.columns}, inplace=True)
            period_cols = df_pivot.columns.difference(['metric'])
            df_pivot[period_cols] = df_pivot[period_cols].apply(lambda col: col.map(to_percent))
            
            self.path_manager.save_csv_file(df_pivot, abs_file_path=evaluation_result_path)
            self.path_manager.save_csv_file(df_evaluation_results, abs_file_path=roc_results_path)
        else: 
            df_pivot = pd.read_csv(evaluation_result_path)
            df_evaluation_results = pd.read_csv(roc_results_path)

        return df_pivot, df_evaluation_results


    def sub_kpm_evaluation_wcr(self, kpn_parms, debugging = False):
        evaluation_results = []
        evaluation_results_roc = []
        evaluation_result_path = op.join(self.path_manager.base_data_dir, 'sub_wcr_results_percent.csv')
        roc_results_path = op.join(self.path_manager.base_data_dir, 'sub_wcr_evaluation_results.csv')
        
        if not op.exists(roc_results_path) or not op.exists(evaluation_result_path) or debugging:
            for wcr in self.wcr_list:
                wcr_str = f"{wcr:.1f}" if wcr == 1 else str(wcr)
                df_kpm_path = op.join(self.kpm_path, f'''df_kpm_wcr_{wcr_str}.csv''')
                df_kpm_original = pd.read_csv(df_kpm_path, index_col=0)
                
                df_kpm = pd.DataFrame(self.KPN.get_max_symmetric_elements(df_kpm_original.values), index=df_kpm_original.index, columns=df_kpm_original.columns)

                for discipline in ['Computer science', 'Engineering', 'Mathematics', 'Physics']:

                    df_sub_kpm = df_kpm.loc[df_kpm.index.isin(self.discipline_to_ids[discipline]), df_kpm.index.isin(self.discipline_to_ids[discipline])]
                    df_edges_roc = self.KPN.kpn_edges(df_sub_kpm, kpn_parms, optimize=False)
                    df_edges_roc['ground_truth'] = df_edges_roc[['source', 'target']].apply(tuple, axis=1).map(self.llm_pair_id_dicts[discipline])
                    df_edges_roc['true_label'] = df_edges_roc['ground_truth'].isin([1, 2])
                    df_evaluation_result = self.calculate_performance_metrics(df_edges_roc)
                    df_evaluation_result['discipline'] = discipline
                    df_evaluation_result['wcr'] = wcr
                    evaluation_results_roc.append(df_evaluation_result)

                    df_edges = self.KPN.kpn_edges(df_sub_kpm, kpn_parms)
                    df_edges['ground_truth'] = df_edges[['source', 'target']].apply(tuple, axis=1).map(self.llm_pair_id_dicts[discipline])
                    precesion = (df_edges.ground_truth>0).sum() / len(df_edges)
                    recall = (df_edges.ground_truth>0).sum() / len(self.positive_gt_pairs[discipline])
                    evaluation_results.append([wcr, discipline, precesion, recall])

            df_evaluation_results = pd.concat(evaluation_results_roc)
            df_results = pd.DataFrame(evaluation_results, columns=['wcr', 'discipline', 'precesion','recall'])
            df_pivot = (
                df_results
                .melt(id_vars=['wcr', 'discipline'], value_vars=['precesion', 'recall'], var_name='metric', value_name='value')
                .pivot(index=['discipline', 'metric'], columns='wcr', values='value')
                .reset_index()
            )
            def to_percent(x):
                if isinstance(x, (int, float)):
                    return '{:.2f}%'.format(x * 100)
                else:
                    return x
            def percent_colname(col):
                try:
                    val = float(col)
                    if 0 < val <= 1:
                        return '{}%'.format(int(round(val * 100)))
                except (ValueError, TypeError):
                    pass
                return col

            df_pivot.rename(columns={col: percent_colname(col) for col in df_pivot.columns}, inplace=True)
            wcr_cols = df_pivot.columns.difference(['discipline', 'metric'])
            df_pivot[wcr_cols] = df_pivot[wcr_cols].apply(lambda col: col.map(to_percent))
            
            self.path_manager.save_csv_file(df_pivot, abs_file_path=evaluation_result_path)
            self.path_manager.save_csv_file(df_evaluation_results, abs_file_path=roc_results_path)
        else: 
            df_pivot = pd.read_csv(evaluation_result_path)
            df_evaluation_results = pd.read_csv(roc_results_path)

        return df_pivot, df_evaluation_results

    def sub_kpm_evaluation_period(self, kpn_parms, debugging = False):
        evaluation_results = []
        evaluation_results_roc = []
        evaluation_result_path = op.join(self.path_manager.base_data_dir, 'sub_period_results_percent.csv')
        roc_results_path = op.join(self.path_manager.base_data_dir, 'sub_period_evaluation_results.csv')
        if not op.exists(roc_results_path) or not op.exists(evaluation_result_path) or debugging:
            for period in ['early', 'middle', 'later']:
                df_kpm_path = op.join(self.kpm_path, f'''df_kpm_period_{period}.csv''')
                df_kpm_original = pd.read_csv(df_kpm_path, index_col=0)
                
                df_kpm = pd.DataFrame(self.KPN.get_max_symmetric_elements(df_kpm_original.values), index=df_kpm_original.index, columns=df_kpm_original.columns)

                for discipline in ['Computer science', 'Engineering', 'Mathematics', 'Physics']:

                    df_sub_kpm = df_kpm.loc[df_kpm.index.isin(self.discipline_to_ids[discipline]), df_kpm.index.isin(self.discipline_to_ids[discipline])]
                    df_edges_roc = self.KPN.kpn_edges(df_sub_kpm, kpn_parms, optimize=False)
                    df_edges_roc['ground_truth'] = df_edges_roc[['source', 'target']].apply(tuple, axis=1).map(self.llm_pair_id_dicts[discipline])
                    df_edges_roc['true_label'] = df_edges_roc['ground_truth'].isin([1, 2])
                    df_evaluation_result = self.calculate_performance_metrics(df_edges_roc)
                    df_evaluation_result['discipline'] = discipline
                    df_evaluation_result['period'] = period
                    evaluation_results_roc.append(df_evaluation_result)

                    df_edges = self.KPN.kpn_edges(df_sub_kpm, kpn_parms)
                    df_edges['ground_truth'] = df_edges[['source', 'target']].apply(tuple, axis=1).map(self.llm_pair_id_dicts[discipline])
                    precesion = (df_edges.ground_truth>0).sum() / len(df_edges)
                    recall = (df_edges.ground_truth>0).sum() / len(self.positive_gt_pairs[discipline])
                    evaluation_results.append([period, discipline, precesion, recall])
            
            df_evaluation_results = pd.concat(evaluation_results_roc)
            df_results = pd.DataFrame(evaluation_results, columns=['period', 'discipline', 'precesion','recall'])
            df_pivot = (
                df_results
                .melt(id_vars=['period', 'discipline'], value_vars=['precesion', 'recall'], var_name='metric', value_name='value')
                .pivot(index=['discipline', 'metric'], columns='period', values='value')
                .reset_index()
            )
            def to_percent(x):
                if isinstance(x, (int, float)):
                    return '{:.2f}%'.format(x * 100)
                else:
                    return x
            def percent_colname(col):
                try:
                    val = float(col)
                    if 0 < val <= 1:
                        return '{}%'.format(int(round(val * 100)))
                except (ValueError, TypeError):
                    pass
                return col

            df_pivot.rename(columns={col: percent_colname(col) for col in df_pivot.columns}, inplace=True)
            period_cols = df_pivot.columns.difference(['discipline', 'metric'])
            df_pivot[period_cols] = df_pivot[period_cols].apply(lambda col: col.map(to_percent))
            
            self.path_manager.save_csv_file(df_pivot, abs_file_path=evaluation_result_path)
            self.path_manager.save_csv_file(df_evaluation_results, abs_file_path=roc_results_path)
        else: 
            df_pivot = pd.read_csv(evaluation_result_path)
            df_evaluation_results = pd.read_csv(roc_results_path)
        # {dis: len(value) for dis, value in self.positive_gt_pairs.items()}
        # {dis: len(value)/(250*249/2) for dis, value in self.positive_gt_pairs.items()}
        return df_pivot, df_evaluation_results
    
    def plot_roc_curves_by_discipline(self, df=None, type='wcr', show_auc=True, legend_loc='lower right', 
                                    custom_colors=None, value_type='weight', fig_size = (16, 4), save_fig = False, font_size = 12):
        if df is None:
            kpn_parms = KPNParams(
                disciplines=[],
                row_normalize=False,
                threshold_mode='value_count_limit',  # direct
                quantile_threshold=0.65,
                cvm_threshold=0.9,
                cfm_threshold=0.15,
                matrix_filter=True,
                matrix_imshow=False,
                graph_show=True,
                layout="graphviz",
                save_kpn=True,
                old_matrix_version=False
            )
            if type == 'wcr':
                df_pivot, df = self.sub_kpm_evaluation_wcr(kpn_parms)
            elif type == 'period':
                df_pivot, df = self.sub_kpm_evaluation_period(kpn_parms)
            else:
                raise ValueError('undefined type')

        # Validate value_type
        if value_type not in df['value'].unique():
            raise ValueError(f"Invalid value_type: {value_type}. Available types: {df['value'].unique()}")

        # Filter data to only include the specified value type
        df = df[df['value'] == value_type]

        # Get unique disciplines and group values
        disciplines = sorted(df['discipline'].unique())
        group_values = sorted(df[type].unique())
        
        # Colors
        if custom_colors is None:
            # colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
            colors = [
                '#1e88e5',  # 亮蓝色
                '#ff0d57',  # 亮红色
                '#13b755',  # 亮绿色
                '#7e22ce',  # 紫色
                '#ff9500',  # 橙色
                '#00bcd4',  # 青色
                '#f06292',  # 粉色
                '#ffc107',  # 金黄色
                '#607d8b'   # 蓝灰色
            ]
        else:
            colors = custom_colors
        
        # Create figure with subplots (1 row, n columns)
        fig, axes = plt.subplots(1, len(disciplines), figsize=fig_size)
        
        # Handle case with only one discipline
        if len(disciplines) == 1:
            axes = [axes]
        
        # Loop through each discipline
        for i, discipline in enumerate(disciplines):
            ax = axes[i]
            ax.set_title(discipline, fontsize=font_size)
            ax.set_xlabel('False Positive Rate', fontsize=font_size)
            if i == 0:
                ax.set_ylabel('True Positive Rate', fontsize=font_size)
            ax.grid(True, alpha=0.3)
            
            # Set tick font size
            ax.tick_params(axis='both', which='major', labelsize=font_size)

            # Plot diagonal reference line
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
            
            # Loop through each group value (period or wcr)
            for j, group_val in enumerate(group_values):
                # Filter data for this discipline and group value
                mask = (df['discipline'] == discipline) & (df[type] == group_val)
                df_curve = df[mask].copy()
                
                if len(df_curve) == 0:
                    continue
                
                # Normalize tp and fp to get tpr and fpr
                max_tp = df_curve['tp'].max()
                max_fp = df_curve['fp'].max()
                
                if max_tp == 0 or max_fp == 0:
                    continue
                
                df_curve['tpr'] = df_curve['tp'] / max_tp
                df_curve['fpr'] = df_curve['fp'] / max_fp
                
                # Sort by threshold to properly draw the curve
                df_curve = df_curve.sort_values('threshold', ascending=False)
                
                # Add (0,0) and (1,1) points
                fpr = np.append(np.append(0, df_curve['fpr'].values), 1)
                tpr = np.append(np.append(0, df_curve['tpr'].values), 1)
                
                # Calculate AUC if requested
                if show_auc:
                    # Remove duplicate FPR values to ensure monotonicity
                    points = np.array([(x, y) for x, y in zip(fpr, tpr)])
                    # Sort by x values
                    points = points[np.argsort(points[:, 0])]
                    # Remove duplicates by keeping the max y value for each x
                    unique_x = np.unique(points[:, 0])
                    unique_points = np.array([(x, np.max(points[points[:, 0] == x, 1])) for x in unique_x])
                    
                    # Calculate AUC with clean data
                    from sklearn.metrics import auc
                    roc_auc = auc(unique_points[:, 0], unique_points[:, 1])
                    label = f"{group_val} (AUC={roc_auc:.3f})"
                else:
                    label = f"{group_val}"
                
                # Plot ROC curve
                ax.plot(fpr, tpr, color=colors[j % len(colors)], label=label)
            
            # Add legend
            ax.legend(loc=legend_loc, fontsize='small')
            ax.set_xlim([0, 1.05])
            ax.set_ylim([0, 1.05])
        
        plt.tight_layout()

        if save_fig:
            save_path = os.path.join(self.results_path, f'roc_curve_{type}.pdf')
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        

        plt.show()

    def plot_pr_curves_by_discipline(self, df=None, type='wcr', show_auc=True, legend_loc='lower left', 
                                custom_colors=None, value_type='weight', fig_size=(16, 4), save_fig=False, font_size=12):
        if df is None:
            kpn_parms = KPNParams(
                disciplines=[],
                row_normalize=False,
                threshold_mode='value_count_limit',  # direct
                quantile_threshold=0.65,
                cvm_threshold=0.9,
                cfm_threshold=0.15,
                matrix_filter=True,
                matrix_imshow=False,
                graph_show=True,
                layout="graphviz",
                save_kpn=True,
                old_matrix_version=False
            )
            if type == 'wcr':
                df_pivot, df = self.sub_kpm_evaluation_wcr(kpn_parms)
            elif type == 'period':
                df_pivot, df = self.sub_kpm_evaluation_period(kpn_parms)
            else:
                raise ValueError('undefined type')

        # Validate value_type
        if value_type not in df['value'].unique():
            raise ValueError(f"Invalid value_type: {value_type}. Available types: {df['value'].unique()}")

        # Filter data to only include the specified value type
        df = df[df['value'] == value_type]

        # Get unique disciplines and group values
        disciplines = sorted(df['discipline'].unique())
        group_values = sorted(df[type].unique())
        
        # Colors
        if custom_colors is None:
            colors = [
                '#7fc97f',  # 浅绿色
                '#beaed4',  # 淡紫色
                '#fdc086',  # 浅橙色
                '#ffff99',  # 黄色
                '#386cb0',  # 深蓝色
                '#f0027f',  # 洋红色
                '#bf5b17',  # 棕色
                '#666666',  # 灰色
                '#00cccc'   # 青绿色
            ]
        else:
            colors = custom_colors
        
        # Create figure with subplots (1 row, n columns)
        fig, axes = plt.subplots(1, len(disciplines), figsize=fig_size)
        
        # Handle case with only one discipline
        if len(disciplines) == 1:
            axes = [axes]
        
        # Loop through each discipline
        for i, discipline in enumerate(disciplines):
            ax = axes[i]
            ax.set_title(discipline, fontsize=font_size)
            ax.set_xlabel('Recall', fontsize=font_size)
            if i == 0:
                ax.set_ylabel('Precision', fontsize=font_size)
            ax.grid(True, alpha=0.3)
            
            # Set tick font size
            ax.tick_params(axis='both', which='major', labelsize=font_size)
            
            # Loop through each group value (period or wcr)
            for j, group_val in enumerate(group_values):
                # Filter data for this discipline and group value
                mask = (df['discipline'] == discipline) & (df[type] == group_val)
                df_curve = df[mask].copy()
                
                if len(df_curve) == 0:
                    continue
                
                # Calculate precision and recall
                df_curve['precision'] = df_curve['tp'] / (df_curve['tp'] + df_curve['fp'])
                df_curve['recall'] = df_curve['tp'] / (df_curve['tp'] + df_curve['fn'])
                
                # Handle NaN values (division by zero)
                df_curve['precision'] = df_curve['precision'].fillna(1.0)  # If tp+fp=0, precision=1
                
                # Sort by threshold to properly draw the curve
                df_curve = df_curve.sort_values('threshold', ascending=False)
                
                # Add (0,1) point (recall=0, precision=1) for the highest threshold
                precision = np.append(1.0, df_curve['precision'].values)
                recall = np.append(0.0, df_curve['recall'].values)
                
                # Calculate AUC (Average Precision) if requested
                if show_auc:
                    # Use sklearn's average_precision_score approximation
                    from sklearn.metrics import auc
                    # Ensure recall is sorted for AUC calculation
                    sorted_indices = np.argsort(recall)
                    sorted_recall = recall[sorted_indices]
                    sorted_precision = precision[sorted_indices]
                    
                    # Calculate AUC (area under PR curve)
                    pr_auc = auc(sorted_recall, sorted_precision)
                    label = f"{group_val} (AP={pr_auc:.3f})"
                else:
                    label = f"{group_val}"
                
                # Plot PR curve
                ax.plot(recall, precision, color=colors[j % len(colors)], label=label)
                
                # Add a baseline (random classifier)
                # For PR curves, the baseline is the ratio of positive samples
                # We can estimate this from the data
                if len(df_curve) > 0:
                    positive_ratio = df_curve['tp'].iloc[0] / (df_curve['tp'].iloc[0] + df_curve['fn'].iloc[0])
                    ax.axhline(y=positive_ratio, color='k', linestyle='--', alpha=0.3)
            
            # Add legend
            ax.legend(loc=legend_loc, fontsize='small')
            ax.set_xlim([0, 1.05])
            ax.set_ylim([0, 1.05])
        
        plt.tight_layout()

        if save_fig:
            save_path = os.path.join(self.results_path, f'pr_curve_{type}.pdf')
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()

    def plot_roc_curves_by_cross_discipline(self, df=None, type='wcr', show_auc=True, legend_loc='lower right', 
                                    custom_colors=None, value_type='weight', fig_size=(8, 6), save_fig=False, font_size=12):
        if df is None:
            kpn_parms = KPNParams(
                disciplines=[],
                row_normalize=False,
                threshold_mode='value_count_limit',  # direct
                quantile_threshold=0.65,
                cvm_threshold=0.9,
                cfm_threshold=0.15,
                matrix_filter=True,
                matrix_imshow=False,
                graph_show=True,
                layout="graphviz",
                save_kpn=True,
                old_matrix_version=False
            )
            if type == 'wcr':
                df_pivot, df = self.cross_kpm_evaluation_wcr(kpn_parms)
            elif type == 'period':
                df_pivot, df = self.cross_kpm_evaluation_period(kpn_parms)
            else:
                raise ValueError('undefined type')

        # Validate value_type
        if value_type not in df['value'].unique():
            raise ValueError(f"Invalid value_type: {value_type}. Available types: {df['value'].unique()}")

        # Filter data to only include the specified value type
        df = df[df['value'] == value_type]

        # Get unique group values
        group_values = sorted(df[type].unique())
        
        # Colors
        if custom_colors is None:
            colors = [
                '#7fc97f',  # 浅绿色
                '#beaed4',  # 淡紫色
                '#fdc086',  # 浅橙色
                '#ffff99',  # 黄色
                '#386cb0',  # 深蓝色
                '#f0027f',  # 洋红色
                '#bf5b17',  # 棕色
                '#666666',  # 灰色
                '#00cccc'   # 青绿色
            ]
        else:
            colors = custom_colors
        
        # Create a single figure
        fig, ax = plt.subplots(figsize=fig_size)
        
        # Set title and labels
        ax.set_title(f'ROC Curve by {type.upper()}', fontsize=font_size)
        ax.set_xlabel('False Positive Rate', fontsize=font_size)
        ax.set_ylabel('True Positive Rate', fontsize=font_size)
        ax.grid(True, alpha=0.3)
        
        # Set tick font size
        ax.tick_params(axis='both', which='major', labelsize=font_size)

        # Plot diagonal reference line
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        
        # Loop through each group value (period or wcr)
        for j, group_val in enumerate(group_values):
            # Filter data for this group value
            df_curve = df[df[type] == group_val].copy()
            
            if len(df_curve) == 0:
                continue
            
            df_curve = df_curve.sort_values('threshold', ascending=False)
            
            # Get total positives and negatives
            total_positives = df_curve['tp'].iloc[0] + df_curve['fn'].iloc[0]
            # Since we don't have TN directly, we'll estimate total negatives from FP at lowest threshold
            total_negatives = df_curve['fp'].max()
            
            if total_positives == 0 or total_negatives == 0:
                continue
            
            # Calculate TPR and FPR
            df_curve['tpr'] = df_curve['tp'] / total_positives
            df_curve['fpr'] = df_curve['fp'] / total_negatives
            
            # Add (0,0) and (1,1) points
            fpr = np.append(np.append(0, df_curve['fpr'].values), 1)
            tpr = np.append(np.append(0, df_curve['tpr'].values), 1)
            
            # Calculate AUC if requested
            if show_auc:
                # Remove duplicate FPR values to ensure monotonicity
                points = np.array([(x, y) for x, y in zip(fpr, tpr)])
                # Sort by x values
                points = points[np.argsort(points[:, 0])]
                # Remove duplicates by keeping the max y value for each x
                unique_x = np.unique(points[:, 0])
                unique_points = np.array([(x, np.max(points[points[:, 0] == x, 1])) for x in unique_x])
                
                # Calculate AUC with clean data
                from sklearn.metrics import auc
                roc_auc = auc(unique_points[:, 0], unique_points[:, 1])
                label = f"{group_val} (AUC={roc_auc:.3f})"
            else:
                label = f"{group_val}"
            
            # Plot ROC curve
            ax.plot(fpr, tpr, color=colors[j % len(colors)], label=label, linewidth=2)
        
        # Add legend
        ax.legend(loc=legend_loc, fontsize=10)
        ax.set_xlim([0, 1.05])
        ax.set_ylim([0, 1.05])
        
        plt.tight_layout()

        if save_fig:
            save_path = os.path.join(self.results_path, f'roc_curve_{type}_cross_discipline.pdf')
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()



# Usage example
if __name__ == "__main__":
    kpn_parms = KPNParams(
        disciplines=[],
        row_normalize=False,
        threshold_mode='value_count_limit', # direct
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

    # evaluator = KPMEvaluator(sub_concepts = False)
    # # evaluator.cross_kpm_evaluation_wcr_recall(kpn_parms)
    # evaluator.cross_kpm_evaluation_period_recall(kpn_parms) # 计算不同WCR下的Precision和Recall
    # df_pivot, df_evaluation_results = evaluator.cross_kpm_evaluation_wcr(kpn_parms)
    # df_pivot, df_evaluation_results = evaluator.cross_kpm_evaluation_period(kpn_parms)
    # evaluator.plot_roc_curves_by_cross_discipline(type='wcr', value_type='count', fig_size = (4, 4), save_fig = True, font_size = 16)


    evaluator = KPMEvaluator(sub_concepts = True)
    # evaluator.plot_roc_curves_by_discipline()
    df_pivot, df_evaluation_results = evaluator.sub_kpm_evaluation_wcr(kpn_parms, debugging=False)
    df_pivot, df_evaluation_results = evaluator.sub_kpm_evaluation_period(kpn_parms, debugging=False)

    print(5)