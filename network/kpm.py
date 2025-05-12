import os, sys
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.append(project_dir)

import os
import os.path as op
import numpy as np
import pandas as pd
import networkx as nx
import tqdm
import logging
import itertools
from functools import partial
import multiprocessing as mp
import csv
from utils.config import PathManager, calculate_runtime
from utils.database import DatabaseManager
from utils.params import TCPParams
from utils.concept import Concept
from network.apyd import APYDAnalyzer
from network.ccn import CCN

class KPM:
    """
    Class for constructing Knowledge Precedence Matrix (KPM) from research trajectories
    """
    def __init__(self, ):
        """
        Initialize the KPN constructor
        
        Args:f
            path_manager: Manager for handling file paths
            db_manager: Manager for database operations
        """
        self.path_manager = PathManager()
        self.db_manager = DatabaseManager()
        self.APYD = APYDAnalyzer()
        self.CCN = CCN()
        self.concepts_table = Concept.discipline_category_classification_llm(with_abbreviation = True)
        self.llm_pair_dict, self.positive_gt_pairs = self.load_llm_annotation_results()
        self.logger = logging.getLogger(__name__)
    
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
            # Build dictionary with forward and reverse relationships
            for _, row in llm_prereq_relations.iterrows():
                concept_a = row['concept_a']
                concept_b = row['concept_b']
                vote = row['majority_vote']
                
                # Add original direction (A,B)
                llm_pair_dict[(concept_a, concept_b)] = vote
                
                # Add reverse relationship (B,A) based on vote value
                if vote in [0, -2, 2]:  # Direction doesn't matter
                    llm_pair_dict[(concept_b, concept_a)] = vote
                elif vote in [1, -1]:  # A is prerequisite for B, reverse: B depends on A
                    llm_pair_dict[(concept_b, concept_a)] = -vote

            import pickle

            with open(llm_pair_dict_path, 'wb') as f:
                pickle.dump(llm_pair_dict, f)
        else:
            import pickle
            with open(llm_pair_dict_path, 'rb') as f:
                llm_pair_dict = pickle.load(f)
        positive_gt_pairs = {pair for pair, vote in llm_pair_dict.items() if vote > 0}
        return llm_pair_dict, positive_gt_pairs

    def valid_time_diffs_index_pairs(self, apyds):
        """
        Find valid index pairs based on average publication year differences (APYD)
        
        Args:
            mean_pub_years: List of mean publication years
            
        Returns:
            List of valid index pairs
        """
        lower_upper_dict = self.APYD.get_APYD_quantile_bounds(middle_high_frequency_ratio=0.3)
        lower, higher = lower_upper_dict.values()
        if len(apyds) == 2:
            return [[0, 1]]
        valid_pairs = []
        for r in range(len(apyds)):
            for s in range(r+1, len(apyds)):
                apyd = apyds[s] - apyds[r]
                if lower <= apyd <= higher:  # Based on our APYD range
                    valid_pairs.append([r, s])
        return valid_pairs
    
    def select_community_concepts(self, params, concept_statistics):
        # Select representative concepts based on select_mode
        if params.select_mode == 'collective':
            if concept_statistics.top_n_collective_wcr[0] < 1:
                min_index = np.argwhere(
                    (concept_statistics.top_n_collective_wcr >= params.wcr).values
                ).flatten()[0]
            else: 
                min_index = np.argwhere(
                    (concept_statistics.wcr >= params.wcr).values
                ).flatten()[-1]
            representative_concepts = concept_statistics.loc[:min_index, 'display_name'].tolist()
            
        elif params.select_mode == 'respective': 
            representative_concepts = concept_statistics.loc[
                concept_statistics.wcr >= params.wcr
            ].display_name.tolist()
            
        elif params.select_mode == 'cumulative': 
            candidate_indexes = np.argwhere(
                (concept_statistics.work_count_cum_ratio <= params.top_ratio).values
            ).flatten()
            if candidate_indexes.shape[0] > 0:
                max_index = candidate_indexes[-1]
                representative_concepts = concept_statistics.loc[:max_index, 'display_name'].tolist()
            else: 
                representative_concepts = []
                
        elif params.select_mode == 'all': 
            representative_concepts = concept_statistics.display_name.to_list()
            
        else: 
            raise ValueError('Undefined select mode')
        
        return representative_concepts

    def calculate_community_concept_statistics(self, community_concepts):

        # concept_statistics = community_concepts.groupby('display_name').apply(
        # lambda x: pd.Series({
        #     'work_count': x['work_id'].drop_duplicates().shape[0],
        #     'pub_years': sorted(set(x['pub_year']))
        #     })
        # ).reset_index()
        # concept_statistics = concept_statistics.sort_values('work_count', ascending=False).reset_index(drop=True)

        concept_statistics = community_concepts.groupby('display_name').apply(lambda x: x['work_id'].drop_duplicates().shape[0]).sort_values(ascending=False).reset_index()
        concept_statistics.columns = ['display_name','work_count']
        
        concept_statistics['wcr'] = (
            concept_statistics.work_count / community_concepts.work_id.drop_duplicates().shape[0]
        )
        concept_statistics['work_count_cum_ratio'] = (
            concept_statistics.work_count.cumsum() / concept_statistics.work_count.sum()
        )

        # The cumulative count of unique work IDs associated with the top N concepts
        concept_statistics['top_n_collective_work_ids'] = concept_statistics.apply(
            lambda x: set(
                community_concepts[
                    community_concepts.display_name.isin(
                        concept_statistics.loc[:x.name, 'display_name']
                    )
                ]['work_id'].tolist()
            ), 
            axis=1
        ).apply(len)
        
        concept_statistics['top_n_collective_wcr'] = (
            concept_statistics['top_n_collective_work_ids'] / concept_statistics.top_n_collective_work_ids.max()
        )

        return concept_statistics

    @calculate_runtime
    def get_CCN_temporal_community_pairs(self, params: TCPParams, df_author_communities, show_runtime=True):
        """
        Extract temporal community pairs and their representative concepts
        
        Args:
            params: Parameters for community concept pairs
            show_runtime: Whether to show runtime information
            
        Returns:
            Tuple of (author_representative_concepts, df_concept_pairs, 
                     df_concept_community_nodes_coverage_ratio, works_concepts_table)
        """
        if df_author_communities.shape[0] < params.min_community_size:
            return [], pd.DataFrame([], columns=['community_pair', 'concept_pair']), [], []
        
        communities_author = df_author_communities['community'].tolist()
        author_works = [work_id for community_ in communities_author for work_id in community_]
        
        valid_community_pair_indexes = self.valid_time_diffs_index_pairs(
            df_author_communities['mean_pub_year'].tolist()
        )
        # 再添加一个验证确保严格按照时间先后顺序，如果community paier (r, s) 各自的论文集合出现时间区间重叠的，去掉 r 中的时间重叠的 paper
        # workid2year = {k: v for d in df_author_communities.apply(lambda row: dict(zip(row['community'], row['pub_year'])), axis=1) for k, v in d.items()}
        works_concepts_table = self.db_manager.query_table(
            table_name='works_concepts', 
            columns=['work_id', 'concept_id', 'display_name', 'level'], 
            join_tables=['concepts'], 
            join_conditions=['works_concepts.concept_id = concepts.id'], 
            where_conditions=[f'''work_id IN ('{"','".join(author_works)}')'''], 
            show_runtime=False
        )
        # works_concepts_table['pub_year'] = works_concepts_table.work_id.map(workid2year)
        df_concept_community_nodes_coverage_ratio = pd.DataFrame(
            np.full((df_author_communities.shape[0], 10), None, dtype=object), 
            columns=[f'{i}_th' for i in range(1, 6)] + [f'top_{i}' for i in range(1, 6)]
        )
        
        author_representative_concepts = []
        
        for i, community_i in enumerate(communities_author):
            if 0 <= params.concept_level <= 5:
                concept_level_condition = (
                    (works_concepts_table.level <= params.concept_level) if params.less_than 
                    else (works_concepts_table.level == params.concept_level)
                )
            else: 
                concept_level_condition = pd.Series(np.ones(works_concepts_table.shape[0], dtype=bool))
                
            community_concepts = works_concepts_table.loc[
                (works_concepts_table.work_id.isin(community_i)) & concept_level_condition
            ]
            
            if community_concepts.shape[0] < 1:
                author_representative_concepts.append({})
                continue
                
            concept_statistics = self.calculate_community_concept_statistics(community_concepts)
            
            fill_length = min(concept_statistics.shape[0], 5)
            df_concept_community_nodes_coverage_ratio.iloc[i, :fill_length] = (
                concept_statistics.wcr[:fill_length]
            )
            df_concept_community_nodes_coverage_ratio.iloc[i, 5:5+fill_length] = (
                concept_statistics.top_n_collective_wcr[:fill_length]
            )
            df_concept_community_nodes_coverage_ratio['author_id'] = df_author_communities['author_id'].tolist()

            author_representative_concepts.append(self.select_community_concepts(params, concept_statistics))
            
        author_representative_concepts = np.array(author_representative_concepts, dtype=object)
        
        all_pairs = author_representative_concepts[valid_community_pair_indexes]
        concept_pairs_author = list(map(lambda row: list(itertools.product(*row)), all_pairs))
        concept_pairs_author = list(itertools.chain(*concept_pairs_author))

        def product_len(index_pair):
            return len(author_representative_concepts[index_pair[0]]) * len(author_representative_concepts[index_pair[1]])
        
        df_concept_pairs = pd.DataFrame({
            'community_pair': [
                tuple(pair_) for pair_ in valid_community_pair_indexes 
                for _ in range(product_len(pair_))
            ],
            'community_pair_1_based':[
                tuple((pair_[0] + 1, pair_[1] + 1)) for pair_ in valid_community_pair_indexes 
                for _ in range(product_len(pair_))
            ] # index pair_ + 1 to make it 1 based instead of 0 based
        })
        
        df_concept_pairs['concept_pair'] = concept_pairs_author

        return (
            author_representative_concepts.tolist(), 
            df_concept_pairs, 
            df_concept_community_nodes_coverage_ratio, 
            works_concepts_table
        )
    
    @calculate_runtime
    def get_CCN_temporal_community_pairs_multi_wcr(self, params: TCPParams, df_author_communities, show_runtime=True):
        """
        Extract temporal community pairs and their representative concepts
        
        Args:
            params: Parameters for community concept pairs
            show_runtime: Whether to show runtime information
            
        Returns:
            Tuple of (author_representative_concepts, df_concept_pairs, 
                     df_concept_community_nodes_coverage_ratio, works_concepts_table)
        """
        if df_author_communities.shape[0] < params.min_community_size:
            return [], pd.DataFrame([], columns=['community_pair', 'concept_pair']), [], []
        
        communities_author = df_author_communities['community'].tolist()
        author_works = [work_id for community_ in communities_author for work_id in community_]
        
        valid_community_pair_indexes = self.valid_time_diffs_index_pairs(
            df_author_communities['mean_pub_year'].tolist()
        )
        works_concepts_table = self.db_manager.query_table(
            table_name='works_concepts', 
            columns=['work_id', 'concept_id', 'display_name', 'level'], 
            join_tables=['concepts'], 
            join_conditions=['works_concepts.concept_id = concepts.id'], 
            where_conditions=[f'''work_id IN ('{"','".join(author_works)}')'''], 
            show_runtime=False
        )
        df_concept_community_nodes_coverage_ratio = pd.DataFrame(
            np.full((df_author_communities.shape[0], 10), None, dtype=object), 
            columns=[f'{i}_th' for i in range(1, 6)] + [f'top_{i}' for i in range(1, 6)]
        )
        
        author_representative_concepts_wcrs = []
        
        for i, community_i in enumerate(communities_author):
            if 0 <= params.concept_level <= 5:
                concept_level_condition = (
                    (works_concepts_table.level <= params.concept_level) if params.less_than 
                    else (works_concepts_table.level == params.concept_level)
                )
            else: 
                concept_level_condition = pd.Series(np.ones(works_concepts_table.shape[0], dtype=bool))
                
            community_concepts = works_concepts_table.loc[
                (works_concepts_table.work_id.isin(community_i)) & concept_level_condition
            ]
            
            if community_concepts.shape[0] < 1:
                author_representative_concepts_wcrs.append({})
                continue
                
            concept_statistics = self.calculate_community_concept_statistics(community_concepts)
            
            # 每个社团有多个参数下的代表性概念构成一个词典
            community_concepts_params = {}
            for params.wcr in params.wcr_ranges:
                community_concepts_params[params.wcr] = self.select_community_concepts(params, concept_statistics)

            author_representative_concepts_wcrs.append(community_concepts_params)
        
        df_concept_pairs_wcrs = []
        for wcr in params.wcr_ranges:
            author_representative_concepts = []
            for community_concepts_wcrs in author_representative_concepts_wcrs:
                author_representative_concepts.append(community_concepts_wcrs[wcr])
        
            author_representative_concepts = np.array(author_representative_concepts, dtype=object)
        
            all_pairs = author_representative_concepts[valid_community_pair_indexes]
            concept_pairs_author = list(map(lambda row: list(itertools.product(*row)), all_pairs))
            concept_pairs_author = list(itertools.chain(*concept_pairs_author))

            def product_len(index_pair):
                return len(author_representative_concepts[index_pair[0]]) * len(author_representative_concepts[index_pair[1]])
            
            df_concept_pairs = pd.DataFrame({
                'wcr': wcr,
                'community_pair': [
                    tuple(pair_) for pair_ in valid_community_pair_indexes 
                    for _ in range(product_len(pair_))
                ],
                'community_pair_1_based':[
                    tuple((pair_[0] + 1, pair_[1] + 1)) for pair_ in valid_community_pair_indexes 
                    for _ in range(product_len(pair_))
                ] # index pair_ + 1 to make it 1 based instead of 0 based
            })
                
            df_concept_pairs['concept_pair'] = concept_pairs_author
            df_concept_pairs_wcrs.append(df_concept_pairs)

        df_concept_pairs_wcrs = pd.concat(df_concept_pairs_wcrs)
        
        return df_concept_pairs_wcrs

    def sort_community_by_year(self, row):
        # Pair years with work_id and sort by year
        sorted_pairs = sorted(zip(row['pub_year'], row['community']))
        sorted_years, sorted_works = zip(*sorted_pairs)
        return pd.Series([list(sorted_works), list(sorted_years)])

    def evaluate_concept_pairs(self, df_all_concept_pairs, author_id, wcrs):
        """
        Evaluate predicted concept pairs against LLM annotation.

        Args:
            df_concept_pairs: DataFrame with a 'concept_pair' column (tuple of (concept_a, concept_b)).
            undirected: If True, treat (A, B) and (B, A) as the same pair.

        Returns:
            dict with TP, FP, TN, FN, Precision, Recall, F1.
        """
        df_all_concept_pairs['ground_truth'] = df_all_concept_pairs['concept_pair'].map(self.llm_pair_dict)
        result_wcrs = {}
        for wcr in wcrs:
            df_concept_pairs = df_all_concept_pairs.loc[df_all_concept_pairs.wcr == wcr]
            nan_rows = df_concept_pairs.loc[df_concept_pairs['ground_truth'].isna()]
            SLC  = nan_rows['concept_pair'].apply(lambda pair: pair[0] == pair[1]).sum() # self loop count 
            non_nan_rows = df_concept_pairs.loc[~df_concept_pairs['ground_truth'].isna()]
            tp = (non_nan_rows.ground_truth>0).sum()
            fp = (non_nan_rows.ground_truth<=0).sum()
            
            # Count false negatives (pairs that are positive in ground truth but not predicted)
            predicted_pairs = set(non_nan_rows['concept_pair'])
            
            fn = len(self.positive_gt_pairs - predicted_pairs)

            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            result_wcr = {
                'author_id': author_id,
                'wcr': wcr,
                'SLC': SLC,
                'TP': tp,
                'FP': fp,
                'FN': fn,
                'TotalPairs': df_concept_pairs.shape[0],
                'Precision': precision,
                'Recall': recall,
                'F1': f1
            }
            if author_id not in result_wcrs:
                result_wcrs[author_id] = {}
            result_wcrs[author_id][wcr] = result_wcr
        
        return result_wcrs
    
    def process_author_chunk(self, author_chunk, params):
        """Process a batch of authors"""
        evaluate_results = []
        local_concept_pairs = []
        
        for author_id, df_author_communities in author_chunk.groupby('author_id'):
            if df_author_communities.shape[0] < 2:
                continue
                
            # Copy parameters to avoid conflicts between processes
            df_author_communities[['community', 'pub_year']] = df_author_communities.apply(self.sort_community_by_year, axis=1)
            
            # _, df_concept_pairs, _, _ = self.get_CCN_temporal_community_pairs(params, df_author_communities, show_runtime=False)
            _, df_concept_pairs, _, _ = self.get_CCN_temporal_community_pairs_multi_wcr(params, df_author_communities, show_runtime=False)
            
            # Collect concept pairs
            if df_concept_pairs.shape[0] > 0:
                # Evaluate results
                result = self.evaluate_concept_pairs(df_concept_pairs, author_id)
                evaluate_results.append(result)
                local_concept_pairs.extend(df_concept_pairs.concept_pair.tolist())
        
        return local_concept_pairs, evaluate_results

    def parallel_process_authors(self, df_all_mean_pub_date_infor, params, n_processes=4):
        """Process author data in parallel using multiprocessing"""
        # Set number of processes, default to CPU core count

        if n_processes is None:
            n_processes = mp.cpu_count()

        author_ids = df_all_mean_pub_date_infor['author_id'].unique()
        author_chunks = np.array_split(author_ids, n_processes)
        # Split data into n_processes chunks
        df_chunks = [df_all_mean_pub_date_infor[df_all_mean_pub_date_infor['author_id'].isin(chunk)].copy() for chunk in author_chunks]
        
        # Create process pool
        pool = mp.Pool(processes=n_processes)
        
        # Prepare partially applied function
        process_func = partial(self.process_author_chunk, params=params)
        
        # Map function to data chunks using process pool
        print(f"Processing with {n_processes} processes...")
        all_evaluate_results = []
        all_concept_pairs = []
        # Use imap to process results with progress tracking
        for concept_pairs, evaluate_results in tqdm.tqdm(pool.imap(process_func, df_chunks), total=len(df_chunks)):
            all_concept_pairs.extend(concept_pairs)
            all_evaluate_results.extend(evaluate_results)
        
        # Close process pool
        pool.close()
        pool.join()
        
        if all_evaluate_results:
            # Prepare output file
            evaluation_path = op.join(self.path_manager.ccns_dir, 'author_evaluation_results.csv')
            fieldnames = ['author_id', 'SLC', 'TP', 'FP', 'FN', 'TotalPairs', 'Precision', 'Recall', 'F1']
            with open(evaluation_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_evaluate_results)
        
        print(f"Total processed authors: {len(all_evaluate_results)}")
        
        return all_concept_pairs

    def process_authors(self, df_all_mean_pub_date_infor, params):
        concept_pairs_all_authors = []
        
        with open(evaluation_path, 'w', newline='') as f:
            fieldnames = ['author_id', 'TP', 'FP', 'FN', 'Precision', 'Recall', 'F1']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
        results_buffer = {}
        buffer_size = 1000000
        processed_count = 0
        chunk_index = 0

        for author_id, df_author_communities in tqdm.tqdm(df_all_mean_pub_date_infor.groupby('author_id')):
            # It's important to pass the relevant parameters, or the result will be different
            if df_author_communities.shape[0] < 2: 
                continue
            df_author_communities[['community', 'pub_year']] = df_author_communities.apply(self.sort_community_by_year, axis=1)
            # _, df_concept_pairs, _, _ = self.get_CCN_temporal_community_pairs(params, df_author_communities, show_runtime=False)
            df_concept_pairs = self.get_CCN_temporal_community_pairs_multi_wcr(params, df_author_communities, show_runtime=False)
            
            results_buffer.update(self.evaluate_concept_pairs(df_concept_pairs, author_id, params.wcr_ranges))

            if len(results_buffer) >= buffer_size:
                evaluation_path = op.join(self.path_manager.ccns_dir, 'author_evaluation_results_{chunk_index}.csv')
                with open(evaluation_path, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writerows(results_buffer)
                
                processed_count += len(results_buffer)
                print(f"Processed {processed_count} authors")
                results_buffer = []

            if df_concept_pairs.shape[0] > 0:
                concept_pairs_all_authors.extend(df_concept_pairs.concept_pair.tolist())

        if results_buffer:
            with open(evaluation_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerows(results_buffer)
            
            processed_count += len(results_buffer)
            print(f"Processed final buffer with total {processed_count} authors")
        
        return concept_pairs_all_authors

    def generate_kpm(self, params, df_kpm_path, override):
        self.logger.info(f'Generating file {df_kpm_path}')
        
        df_all_mean_pub_date_infor = pd.DataFrame(
            self.CCN.load_author_community_info(test = True), 
            columns=['author_id', 'community', 'pub_year', 'mean_pub_date', 'mean_pub_year']
        )
        df_all_mean_pub_date_infor = df_all_mean_pub_date_infor[:1000]
        
        if params.parallel:
            concept_pairs_all_authors = self.parallel_process_authors(df_all_mean_pub_date_infor, params, n_processes = 7)
        else:
            concept_pairs_all_authors = self.process_authors(df_all_mean_pub_date_infor, params)
        
        # Filter concepts based on level
        concepts_required = self.concepts_table.loc[
            (self.concepts_table.level <= params.concept_level) if params.less_than 
            else (self.concepts_table.level == params.concept_level)
        ][['discipline_category_refined', 'llm_annotation', 'confidence', 'level', 'display_name']]
        
        concepts_required = concepts_required.sort_values(by=['discipline_category_refined', 'display_name'])
        concepts_index = concepts_required.display_name.drop_duplicates().tolist()
        
        # Create the concept flow matrix
        df_kpm = pd.DataFrame(0, index=concepts_index, columns=concepts_index)
        
        # Count concept pair occurrences
        value_counts = pd.DataFrame(concept_pairs_all_authors, columns=['source', 'target']).value_counts().reset_index()
        value_counts.columns = ['source', 'target', 'count_value']
        
        # Fill the matrix
        for _, row in value_counts.iterrows():
            df_kpm.loc[row['source'], row['target']] += row['count_value']
            
        # Save the matrix
        self.path_manager.save_csv_file(
            variable=df_kpm, 
            abs_file_path=df_kpm_path, 
            index=True, 
            override=override
        )
        self.logger.info(f'{df_kpm_path} cached')

        return df_kpm
    
    def get_kpm(
        self, 
        params: TCPParams, 
        override=False, 
        old_matrix_version=False, 
        params_str_old=False, 
        debugging = False
    ):
        """
        Generate the temporal concept flow matrix from co-citation network communities
        
        Knowledge Precedence Matrix (KPM)

        Args:
            params: Parameters for community concept pairs
            override: Whether to override existing files
            old_matrix_version: Whether to use old matrix version
            params_str_old: Whether to use old parameter string format
            
        Returns:
            Tuple of (df_kpm, params_str)
        """
        # Determine file name based on parameters
        if params.select_mode == 'respective' or params.select_mode == 'collective':
            file_name_end = f'{params.select_mode}_cover_{params.wcr}_works'
        elif params.select_mode == 'cumulative':
            file_name_end = f'{params.select_mode}_top_{params.top_ratio}_works'
        elif params.select_mode == 'all':
            file_name_end = 'cover_all_community_concepts'
        else: 
            raise ValueError('Undefined select mode')
            
        params_str = f"level_{'lt' if params.less_than else 'eq'}_{params.concept_level}_{file_name_end}"
        params_str = f"{params_str}{'_old' if params_str_old else ''}"
        params_str = 'old' if old_matrix_version else params_str
        
        df_kpm_path = op.join(self.path_manager.concepts_dir, f"kpm_{params_str}.csv")
        
        if not op.exists(df_kpm_path) or override or debugging:
            df_kpm = self.generate_kpm(params, df_kpm_path, override)
        else: 
            df_kpm = pd.read_csv(df_kpm_path, index_col=0)
            
        return df_kpm, params_str

if __name__ == "__main__":
    kpm = KPM()

    # parms_ = TCPParams(concept_level = 1, less_than = True, select_mode = 'collective', wcr = 0.9)

    parms_ = TCPParams(concept_level = 1, less_than = True, select_mode = 'collective', wcr = 0.9, parallel = False, debugging = True)

    df_kpm, params_str = kpm.get_kpm(parms_, debugging = parms_.debugging)

    print(5)

