import os, sys
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.append(project_dir)
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(line_buffering=True)

import networkx as nx
import pandas as pd
import numpy as np
import logging
import os.path as op

from utils.database import DatabaseManager
from utils.config import PathManager, calculate_runtime
from utils.params import TCPParams
from utils.concept import Concept
from network.apyd import APYDAnalyzer
from sklearn.metrics.pairwise import cosine_similarity
from community import community_louvain
import itertools
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from utils.smart_cache import cache_results
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from collections import Counter

class IntegratedKPMGenerator:
    """
    Integrated class for generating Knowledge Precedence Matrix (KPM) with sensitivity analysis
    """
    path_manager = PathManager()
    
    def __init__(self, debugging = False):
        self.logger = logging.getLogger(__name__)
        self.db_manager = DatabaseManager()
        self.APYD = APYDAnalyzer()
        self.debugging = debugging
        self.sciconnav_embedding, self.valid_concepts = self.load_sciconnav_embedding()
        self.concepts_table = Concept.discipline_category_classification_llm(with_abbreviation=True)
        self.llm_pair_dict, self.positive_gt_pairs = self.load_llm_annotation_results()
        self.target_concepts = pd.read_csv(op.join('./llm_cross_annotation/df_selected_cross_concepts.csv'))
        self.target_concepts = self.target_concepts.loc[self.target_concepts.display_name.isin(self.valid_concepts)]
        self.target_concept_ids = self.target_concepts.id.tolist()
        self.target_concept_names = self.target_concepts.display_name.tolist()
        self.concept_id_to_name = self.target_concepts.set_index('id')['display_name'].to_dict()

    def load_sciconnav_embedding(self):
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

            import pickle
            with open(llm_pair_dict_path, 'wb') as f:
                pickle.dump(llm_pair_dict, f)
        else:
            import pickle
            with open(llm_pair_dict_path, 'rb') as f:
                llm_pair_dict = pickle.load(f)
                
        positive_gt_pairs = {pair for pair, vote in llm_pair_dict.items() if vote > 0}
        return llm_pair_dict, positive_gt_pairs

    def query_author_data(self, author_ids, chunk_size=5000):
        """Query author works and concepts data from database"""
        df_authors_works = self.db_manager.query_table(
            table_name='works_authorships',
            columns=[
                'works_authorships.author_id',
                'works_authorships.work_id',
                'works.publication_date',
                'works_referenced_works.referenced_work_id'
            ],
            columns_trim_enter=['works_referenced_works.referenced_work_id'], 
            join_tables=['works', 'works_referenced_works'],
            join_conditions=[
                'works_authorships.work_id = works.id',
                'works_authorships.work_id = works_referenced_works.work_id'
            ],
            where_conditions=[f'''works_authorships.author_id in ('{"','".join(author_ids)}')'''],
            batch_read=False,
            show_runtime=False
        ).sort_values(['author_id', 'publication_date']).reset_index(drop=True)
        
        all_work_ids = df_authors_works.work_id.unique()
        
        all_results = []
        for i in range(0, len(all_work_ids), chunk_size):
            chunk_ids = all_work_ids[i:i+chunk_size]
            
            df_chunk = self.db_manager.query_table(
                table_name='works_concepts',
                columns=['work_id', 'concept_id'],
                where_conditions=[f'''work_id in ('{"','".join(chunk_ids)}')''', f'''concept_id in ('{"','".join(self.target_concept_ids)}')'''],
                batch_read=False,
                show_runtime=False
            )
            all_results.append(df_chunk)
            
        df_works_concepts = pd.concat(all_results, ignore_index=True)
        df_works_concepts['display_name'] = df_works_concepts['concept_id'].map(self.concept_id_to_name)
        
        return df_authors_works, df_works_concepts

    def query_author_data_with_manager(self, author_ids, db_manager, chunk_size=5000):
        """Query author works and concepts data with specific database manager"""
        try:
            df_authors_works = db_manager.query_table(
                table_name='works_authorships',
                columns=[
                    'works_authorships.author_id',
                    'works_authorships.work_id', 
                    'works.publication_date',
                    'works_referenced_works.referenced_work_id'
                ],
                columns_trim_enter=['works_referenced_works.referenced_work_id'],
                join_tables=['works', 'works_referenced_works'],
                join_conditions=[
                    'works_authorships.work_id = works.id',
                    'works_authorships.work_id = works_referenced_works.work_id'
                ],
                where_conditions=[f'''works_authorships.author_id in ('{"','".join(author_ids)}')'''],
                batch_read=False,
                show_runtime=False
            ).sort_values(['author_id', 'publication_date']).reset_index(drop=True)
            
            all_work_ids = df_authors_works.work_id.unique()
            
            if len(all_work_ids) == 0:
                self.logger.info(f"No works found for {len(author_ids)} authors")
                return df_authors_works, pd.DataFrame()
            
            # 分批查询works_concepts
            all_results = []
            for i in range(0, len(all_work_ids), chunk_size):
                chunk_ids = all_work_ids[i:i+chunk_size]
                
                df_chunk = db_manager.query_table(
                    table_name='works_concepts',
                    columns=['work_id', 'concept_id'],
                    where_conditions=[
                        f'''work_id in ('{"','".join(chunk_ids)}')''',
                        f'''concept_id in ('{"','".join(self.target_concept_ids)}')'''
                    ],
                    batch_read=False,
                    show_runtime=False
                )
                all_results.append(df_chunk)
                
            df_works_concepts = pd.concat(all_results, ignore_index=True)
            df_works_concepts['display_name'] = df_works_concepts['concept_id'].map(self.concept_id_to_name)
            
            return df_authors_works, df_works_concepts
            
        except Exception as e:
            self.logger.info(f"Database query error: {str(e)}")
            return pd.DataFrame(), pd.DataFrame()

    def semantic_similarity_matrix(self, df_work_concepts, work_ids):
        """Calculate semantic similarity matrix between works"""
        work_concept_groups = df_work_concepts.groupby('work_id')['display_name'].apply(list).reindex(work_ids)
        work_concept_groups = work_concept_groups.apply(lambda x: x if isinstance(x, list) else [])

        work_vectors = []
        for work_id, concepts in work_concept_groups.items():
            if len(concepts) > 0: 
                vecs = np.stack([self.sciconnav_embedding.wv[c] for c in concepts])
                mean_vec = vecs.mean(axis=0)
            else: 
                mean_vec = np.zeros(self.sciconnav_embedding.vector_size)
            work_vectors.append(mean_vec)
            
        work_vectors = np.vstack(work_vectors)
        return cosine_similarity(work_vectors)

    def co_citation_count_matrix(self, df_author_works):
        """Calculate co-citation count matrix"""
        valid_references = (
            df_author_works[['work_id', 'referenced_work_id']]
            .groupby('work_id', sort=False)['referenced_work_id']
            .apply(set)
            .apply(lambda s: set() if s == {None} else s)
            .reset_index(name='reference_ids')
        )
        
        co_citation_count_matrix = np.array([
            valid_references['reference_ids'].apply(lambda x: len(work_refs_i.intersection(x))).tolist() 
            for work_refs_i in valid_references['reference_ids']
        ])
        
        return co_citation_count_matrix, valid_references

    def get_similarity_threshold(self, triu_values, n_works, fixed_ave_degree=None):
        """Get similarity threshold based on quantile"""
        try:
            if fixed_ave_degree is None:
                quantile = 1 - np.log(10*n_works) / (n_works - 1)
            else:
                max_possible_degree = n_works - 1
                if fixed_ave_degree >= max_possible_degree:
                    return None
                quantile = 1 - fixed_ave_degree / (n_works - 1)
            
            threshold = np.quantile(triu_values, quantile)
            return threshold
            
        except Exception as e:
            self.logger.info(f"  Error: {e}")
            raise e

    def build_co_citation_network_sensitivity(self, author_id, df_author_works, df_work_concepts, 
                                            min_papers=10, min_edges=2, min_community_size=2, 
                                            base_semantic_threshold=0.1, fixed_degrees=None, resolutions=None):
        """Build co-citation network with sensitivity analysis for multiple parameters"""
        work_ids = df_author_works.work_id.unique()
        
        if len(work_ids) < min_papers:
            return None
            
        # Calculate matrices (only once)
        work_sim_matrix = self.semantic_similarity_matrix(df_work_concepts, work_ids)
        work_sim_matrix = np.maximum(work_sim_matrix, 0) # keep positive similarity
        triu_values = work_sim_matrix[np.triu_indices_from(work_sim_matrix, k=1)]
        co_citation_count_matrix, valid_references = self.co_citation_count_matrix(df_author_works)
        
        # Extract work information (only once)
        df_work_id_infos = df_author_works[['work_id', 'publication_date']].drop_duplicates()
        df_work_id_infos['publication_date'] = pd.to_datetime(df_work_id_infos.publication_date, errors='coerce')
        df_work_id_infos['publication_year'] = df_work_id_infos['publication_date'].dt.year
        
        def create_graph_from_threshold(threshold):
            """Helper function to create graph from similarity threshold"""
            weight_matrix = np.where(
                ((co_citation_count_matrix > 0) & (work_sim_matrix > base_semantic_threshold)) | 
                (work_sim_matrix >= threshold),
                work_sim_matrix,
                0
            )
            # 这个阈值threshold只用于筛选相似性矩阵
            # 原来的共引关系频数矩阵没有变
            
            x, y = np.nonzero(np.triu(weight_matrix, k=1))
            if len(x) < min_edges:
                return None
                
            weighted_edges = np.column_stack((
                valid_references.iloc[x]['work_id'].values, 
                valid_references.iloc[y]['work_id'].values, 
                weight_matrix[x, y]
            ))
            
            author_ccn = nx.Graph()
            author_ccn.add_nodes_from(work_ids)
            author_ccn.add_weighted_edges_from(weighted_edges)
            
            return author_ccn
        
        def process_communities(partition):
            """Helper function to process communities and add to results"""
            df_work_id_infos_copy = df_work_id_infos.copy()
            df_work_id_infos_copy['community_order'] = df_work_id_infos_copy.work_id.map(partition)
            
            df_author_communities = df_work_id_infos_copy.groupby('community_order').apply(
                lambda x: pd.Series({
                    'community': x['work_id'].tolist(), 
                    'pub_year': x['publication_year'].tolist(), 
                    'mean_pub_year': x['publication_year'].mean()
                }) if len(x) >= min_community_size else None,
                include_groups=False 
            ).dropna().reset_index()

            if len(df_author_communities) >= 2:
                df_author_communities['author_id'] = author_id
                df_author_communities = df_author_communities.sort_values('mean_pub_year').reset_index(drop=True)
                return df_author_communities
            return None
                
        results = {}
        
        # Sensitivity analysis for different fixed degrees (with resolution=1.0)
        for degree in (fixed_degrees or [None]):
            degree_str = "dynamic" if degree is None else str(degree)
            threshold = self.get_similarity_threshold(triu_values, len(work_ids), degree)
            if threshold is None:
                continue
            author_ccn = create_graph_from_threshold(threshold)
            if author_ccn is None:
                continue
            for resolution in (resolutions or [1.0]):
                param_key = f"degree_{degree_str}_res_{resolution}"
                partition = community_louvain.best_partition(author_ccn, resolution=resolution, random_state=42)
                community_result = process_communities(partition)
                if community_result is not None:
                    results[param_key] = community_result
        
        return results if results else None

    def valid_time_diffs_index_pairs(self, mean_pub_years):
        """Find valid index pairs based on publication year differences"""
        lower_upper_dict = self.APYD.get_APYD_quantile_bounds(middle_high_frequency_ratio=0.3)
        lower, higher = lower_upper_dict.values()
        
        if len(mean_pub_years) == 2:
            return [[0, 1]]
            
        valid_pairs = []
        for r in range(len(mean_pub_years)):
            for s in range(r+1, len(mean_pub_years)):
                apyd = mean_pub_years[s] - mean_pub_years[r]
                if lower <= apyd <= higher:
                    valid_pairs.append([r, s])
        return valid_pairs

    def calculate_community_concept_statistics(self, community_concepts):
        concept_statistics = community_concepts.groupby('display_name').apply(lambda x: x['work_id'].drop_duplicates().shape[0]).sort_values(ascending=False).reset_index()
        concept_statistics.columns = ['display_name','work_count']
        
        concept_statistics['wcr'] = (
            concept_statistics.work_count / community_concepts.work_id.drop_duplicates().shape[0]
        )
        
        top_n_collective_work_ids = concept_statistics.apply(
            lambda x: set(
                community_concepts[
                    community_concepts.display_name.isin(
                        concept_statistics.loc[:x.name, 'display_name']
                    )
                ]['work_id'].tolist()
            ), 
            axis=1
        ).apply(len)
        
        concept_statistics['top_n_collective_wcr'] = top_n_collective_work_ids / top_n_collective_work_ids.max()

        return concept_statistics

    def select_community_concepts(self, params, concept_statistics):
        """Select representative concepts based on parameters"""
        if concept_statistics.empty:
            return []
            
        # if params.select_mode == 'collective':
        #     if concept_statistics.top_n_collective_wcr[0] < 1:
        #         min_index_1 = np.argwhere(
        #             (concept_statistics.top_n_collective_wcr >= params.wcr).values
        #         ).flatten()[0]
        #     else: 
        #         min_index_1 = concept_statistics.loc[concept_statistics.wcr == 1].index.max()
            
        #     wcr_indices = np.argwhere(concept_statistics.wcr >= params.wcr).flatten()
        #     if len(wcr_indices) > 0:
        #         min_index_2 = wcr_indices[-1]
        #     else:
        #         min_index_2 = 0
        #     min_index = max(min_index_1, min_index_2)
        #     representative_concepts = concept_statistics.loc[:min_index, 'display_name'].tolist()

        if params.select_mode == 'collective': # original
            if concept_statistics.top_n_collective_wcr[0] < 1:
                min_index = np.argwhere(
                    (concept_statistics.top_n_collective_wcr >= params.wcr).values
                ).flatten()[0]
            else: 
                min_index = concept_statistics.loc[concept_statistics.wcr == 1].index.max()
            representative_concepts = concept_statistics.loc[:min_index, 'display_name'].tolist()
            
        elif params.select_mode == 'respective': 
            representative_concepts = concept_statistics.loc[
                concept_statistics.wcr >= params.wcr
            ].display_name.tolist()
        
        elif params.select_mode == 'all': 
            representative_concepts = concept_statistics.display_name.to_list()
            
        else: 
            raise ValueError('Undefined select mode')
        
        return representative_concepts

    def extract_temporal_community_pairs(self, params, df_author_communities, works_concepts_table):
        """Extract temporal community pairs and their representative concepts"""
        if df_author_communities.shape[0] < 2:
            return []
        
        communities_author = df_author_communities['community'].tolist()
        valid_community_pair_indexes = self.valid_time_diffs_index_pairs(
            df_author_communities['mean_pub_year'].tolist()
        )
        
        author_representative_concepts = []
        
        for i, community_i in enumerate(communities_author):
            community_concepts = works_concepts_table.loc[works_concepts_table.work_id.isin(community_i)]
            
            if community_concepts.shape[0] < 1:
                author_representative_concepts.append([])
                continue
                
            concept_statistics = self.calculate_community_concept_statistics(community_concepts)
            author_representative_concepts.append(
                self.select_community_concepts(params, concept_statistics)
            )
            
        author_representative_concepts = np.array(author_representative_concepts, dtype=object)
        
        # Generate concept pairs
        all_pairs = author_representative_concepts[valid_community_pair_indexes]
        concept_pairs_author = list(map(lambda row: list(itertools.product(*row)), all_pairs))
        concept_pairs_author = list(itertools.chain(*concept_pairs_author))

        return concept_pairs_author
    
    @cache_results(
        cache_dir=path_manager.ccns_dir, 
        filename_pattern="Total_authors_min_pubs_{min_papers}_v{version}.parquet",
        version=1
    )
    def get_author_batches(self, min_papers=10, max_papers=None):
        """Get batches of authors for processing"""
        where_conditions = [f"total_pubs >= '{min_papers}'"]
        if max_papers is not None:
            where_conditions.append(f"total_pubs < '{max_papers}'")
            
        df_author_ids = self.db_manager.query_table(
            table_name='author_yearlyfeature_field_geq10pubs',
            columns=['author_id'],
            where_conditions=where_conditions
        )
        
        return df_author_ids

    def get_kpm_filename(self, fixed_ave_degree=None, resolution=1.0):
        """Generate filename for KPM based on parameters"""
        if fixed_ave_degree is not None:
            return f"df_kpm_ave_degree_{fixed_ave_degree}_res_{resolution}.csv"
        else:
            return f"df_kpm_dynamic_degree_res_{resolution}.csv"

    def get_progress_filename(self, fixed_ave_degree=None, resolution=1.0):
        """Generate filename for progress tracking"""
        if fixed_ave_degree is not None:
            return f"progress_ave_degree_{fixed_ave_degree}_res_{resolution}.json"
        else:
            return f"progress_dynamic_degree_res_{resolution}.json"

    def load_existing_kpm_dict(self, param_combinations):
        """Load existing KPM matrices for all parameter combinations"""
        kpm_dict = {}
        for param_key in param_combinations:
            fixed_degree, resolution = self.parse_param_key(param_key)
            kpm_filename = self.get_kpm_filename(fixed_degree, resolution)
            kpm_path = op.join(self.path_manager.kpm_dir_all_deg_res_2, kpm_filename)
            
            if op.exists(kpm_path):
                kpm_dict[param_key] = pd.read_csv(kpm_path, index_col=0)
            else:
                kpm_dict[param_key] = pd.DataFrame(0, index=self.target_concept_names, columns=self.target_concept_names)
        
        return kpm_dict

    def load_progress_dict(self, param_combinations):
        """Load progress tracking information for all parameter combinations"""
        progress_dict = {}
        for param_key in param_combinations:
            fixed_degree, resolution = self.parse_param_key(param_key)
            progress_filename = self.get_progress_filename(fixed_degree, resolution)
            progress_path = op.join(self.path_manager.kpm_dir_all_deg_res_2, progress_filename)
            
            if op.exists(progress_path):
                with open(progress_path, 'r') as f:
                    progress_dict[param_key] = json.load(f)
            else:
                progress_dict[param_key] = {"completed_batches": []}
        
        return progress_dict

    def save_progress_dict(self, progress_dict):
        """Save progress tracking information for all parameter combinations"""
        for param_key, progress in progress_dict.items():
            fixed_degree, resolution = self.parse_param_key(param_key)
            progress_filename = self.get_progress_filename(fixed_degree, resolution)
            progress_path = op.join(self.path_manager.kpm_dir_all_deg_res_2, progress_filename)
            
            with open(progress_path, 'w') as f:
                json.dump(progress, f)

    def save_kpm_dict(self, kpm_dict):
        """Save KPM matrices for all parameter combinations"""
        saved_paths = {}
        for param_key, df_kpm in kpm_dict.items():
            fixed_degree, resolution = self.parse_param_key(param_key)
            kpm_filename = self.get_kpm_filename(fixed_degree, resolution)
            kpm_path = op.join(self.path_manager.kpm_dir_all_deg_res_2, kpm_filename)
            
            df_kpm.to_csv(kpm_path)
            saved_paths[param_key] = kpm_path
        
        return saved_paths

    def parse_param_key(self, param_key):
        """Parse parameter key to extract fixed_degree and resolution"""
        # param_key format: "degree_8_res_1.0"
        parts = param_key.split('_')
        # Handle dynamic degree case
        if parts[1] == 'dynamic':
            fixed_degree = None
        else:
            fixed_degree = int(parts[1])
        resolution = float(parts[3])
        return fixed_degree, resolution

    def generate_param_combinations(self, fixed_degrees, resolutions):
        """Generate parameter combinations for sensitivity analysis"""
        combinations = []
        
        # Fixed degrees with resolution=1.0
        for degree in fixed_degrees:
            if degree is None:
                combinations.append(f"degree_dynamic_res_1.0")
            else:
                combinations.append(f"degree_{degree}_res_1.0")

        # Different resolutions with fixed_degree=8
        for resolution in resolutions:
            combinations.append(f"degree_8_res_{resolution}")
        
        return combinations

    def generate_all_param_combinations(self, fixed_degrees, resolutions):
        """Generate parameter combinations for sensitivity analysis"""
        combinations = []
        
        # Fixed degrees with resolution=1.0
        combinations = []
        for degree in fixed_degrees:
            degree_str = "dynamic" if degree is None else str(degree)
            for resolution in resolutions:
                combinations.append(f"degree_{degree_str}_res_{resolution}")
        return combinations

    def process_batch_sensitivity(self, batch_data):
        """Process a single batch of authors with sensitivity analysis"""
        batch_idx, author_batch, params, fixed_degrees, resolutions = batch_data
        local_db_manager = DatabaseManager(
            pool_size=2,
            max_overflow=1,
            pool_timeout=60,
            pool_recycle=3600,
            pool_pre_ping=True
        )
        if self.debugging:
            author_batch = author_batch[:50]
        progress_interval = 200
        total_authors = len(author_batch)
        try:
            # Query data for current batch
            df_authors_works, df_works_concepts = self.query_author_data_with_manager(author_batch, local_db_manager)
            # df_authors_works, df_works_concepts = self.query_author_data(author_batch)
            
            # Initialize batch results for all parameter combinations
            param_combinations = self.generate_all_param_combinations(fixed_degrees, resolutions)
            batch_results = {param_key: [] for param_key in param_combinations}
            processed_authors = 0
            
            # Process each author in the batch
            for idx, author_id in enumerate(author_batch):
                try:
                    if (idx + 1) % progress_interval == 0:
                        self.logger.info(f"Batch {batch_idx}: Processed {idx + 1}/{total_authors} authors ({(idx + 1)/total_authors*100:.1f}%)")
                    
                    # Filter data for current author
                    df_author_works = df_authors_works[df_authors_works['author_id'] == author_id]
                    df_author_work_concepts = df_works_concepts[
                        df_works_concepts.work_id.isin(df_author_works.work_id.unique())
                    ]
                    
                    # Build co-citation network with sensitivity analysis
                    author_communities_dict = self.build_co_citation_network_sensitivity(
                        author_id, df_author_works, df_author_work_concepts,
                        fixed_degrees=fixed_degrees, resolutions=resolutions
                    )
                    
                    if author_communities_dict is None:
                        continue
                    
                    # Extract concept pairs for each parameter combination
                    for param_key, df_author_communities in author_communities_dict.items():
                        concept_pairs = self.extract_temporal_community_pairs(
                            params, df_author_communities, df_author_work_concepts
                        )
                        batch_results[param_key].extend(concept_pairs)
                    
                    processed_authors += 1
                    
                except Exception as e:
                    self.logger.info(f"Error processing author {author_id}: {str(e)}")
                    continue
            
            local_db_manager.close()
            batch_results_counted = {}
            total_pairs_count = {}
            
            self.logger.info(f"Batch {batch_idx}: Start counting frequency of pairs")
            for param_key, batch_concept_pairs in batch_results.items():
                if batch_concept_pairs:
                    
                    pair_counts = Counter(batch_concept_pairs)
                    batch_results_counted[param_key] = dict(pair_counts)
                    total_pairs_count[param_key] = len(batch_concept_pairs)
                else:
                    batch_results_counted[param_key] = {}
                    total_pairs_count[param_key] = 0

            self.logger.info(f"Batch {batch_idx}: Completed {processed_authors}/{total_authors} authors successfully")
            
            return batch_idx, batch_results_counted, total_pairs_count, processed_authors
            
        except Exception as e:
            self.logger.error(f"Error processing batch {batch_idx}: {str(e)}")
            param_combinations = self.generate_all_param_combinations(fixed_degrees, resolutions)
            empty_results_counted = {param_key: {} for param_key in param_combinations}
            empty_pairs_count = {param_key: 0 for param_key in param_combinations}
            return batch_idx, empty_results_counted, empty_pairs_count, 0
        
        finally:
            local_db_manager.close()

    @calculate_runtime
    def generate_kpm_parallel_sensitivity(self, params: TCPParams, fixed_degrees=None, resolutions=None,
                                        max_authors=None, n_processes=64):
        """Generate KPM using parallel processing with sensitivity analysis"""
        if fixed_degrees is None:
            fixed_degrees = [4, 5, 6, 7, 8, 9, 10, 11, 12, None] # None for dynameic degree
        if resolutions is None:
            resolutions = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        self.logger.info(f"Starting parallel KPM generation with sensitivity analysis...")
        self.logger.info(f"Fixed degrees: {fixed_degrees}")
        self.logger.info(f"Resolutions: {resolutions}")
        
        # Generate parameter combinations
        param_combinations = self.generate_all_param_combinations(fixed_degrees, resolutions)
        self.logger.info(f"Parameter combinations: {param_combinations}")
        
        # Load existing progress and KPM for all combinations
        progress_dict = self.load_progress_dict(param_combinations)
        kpm_dict = self.load_existing_kpm_dict(param_combinations)
        
        # Find common completed batches across all parameter combinations
        all_completed_batches = [set(progress["completed_batches"]) for progress in progress_dict.values()]
        completed_batches = set.intersection(*all_completed_batches) if all_completed_batches else set()
        
        # Get author batches
        author_ids = self.get_author_batches(
            min_papers=params.min_papers if hasattr(params, 'min_papers') else 10,
            max_papers=params.max_papers if hasattr(params, 'max_papers') else None
        )['author_id'].tolist()
        
        author_batches = [author_ids[i:i+params.batch_size] for i in range(0, len(author_ids), params.batch_size)]
        
        # Filter out completed batches
        remaining_batches = [(i, batch) for i, batch in enumerate(author_batches) if i not in completed_batches]
        
        if max_authors:
            total_authors = min(max_authors, sum(len(batch) for _, batch in remaining_batches))
            self.logger.info(f"Processing {total_authors} authors in {len(remaining_batches)} remaining batches")
        else:
            total_authors = sum(len(batch) for _, batch in remaining_batches)
            self.logger.info(f"Processing {total_authors} authors in {len(remaining_batches)} remaining batches")
        
        if not remaining_batches:
            self.logger.info("All batches already completed!")
            return kpm_dict
        
        # Prepare batch data for parallel processing
        batch_data_list = [
            (batch_idx, author_batch, params, fixed_degrees, resolutions)
            for batch_idx, author_batch in remaining_batches
        ]
        
        # Process batches in parallel using ProcessPoolExecutor
        total_new_pairs_dict = {param_key: 0 for param_key in param_combinations}
        
        if self.debugging:
            # Test with first batch only
            test_batch_data = batch_data_list[0]
            self.logger.info(f"Testing batch with {len(test_batch_data[1])} authors")
            
            # Process the test batch
            batch_idx, batch_results_counted, total_pairs_count, processed_authors = self.process_batch_sensitivity(test_batch_data)

            # Update KPM matrices for all parameter combinations
            for param_key, pair_counts  in batch_results_counted.items():
                if pair_counts:
                    # Count concept pair occurrences
                    for (source, target), count in pair_counts.items():
                        if source in kpm_dict[param_key].index and target in kpm_dict[param_key].columns:
                            kpm_dict[param_key].loc[source, target] += count

                    total_new_pairs_dict[param_key] += total_pairs_count[param_key]
            
            # Mark batch as completed for all parameter combinations
            for param_key in param_combinations:
                progress_dict[param_key]["completed_batches"].append(batch_idx)
            
            # Save progress and KPM after each batch
            self.save_progress_dict(progress_dict)
            self.save_kpm_dict(kpm_dict)
    
            return kpm_dict

        else:

            with ProcessPoolExecutor(max_workers=n_processes) as executor:
                # Submit all tasks
                future_to_batch = {
                    executor.submit(self.process_batch_sensitivity, batch_data): batch_data[0] 
                    for batch_data in batch_data_list
                }
                batch_counter = 0
                # Process completed tasks as they finish
                for future in as_completed(future_to_batch):
                    batch_idx = future_to_batch[future]
                    try:
                        batch_idx, batch_results_counted, total_pairs_count, processed_authors = future.result()
                        
                        # Update KPM matrices for all parameter combinations
                        for param_key, pair_counts  in batch_results_counted.items():
                            if pair_counts :
                                for (source, target), count in pair_counts.items():
                                    if source in kpm_dict[param_key].index and target in kpm_dict[param_key].columns:
                                        kpm_dict[param_key].loc[source, target] += count

                                total_new_pairs_dict[param_key] += total_pairs_count[param_key]
                        
                        # Mark batch as completed for all parameter combinations
                        for param_key in param_combinations:
                            progress_dict[param_key]["completed_batches"].append(batch_idx)
                        batch_counter += 1
                        # Save progress and KPM after each batch
                        self.save_progress_dict(progress_dict)
                        self.save_kpm_dict(kpm_dict)
                        
                        self.logger.info(f"Completed {batch_counter}-th batch {batch_idx}, processed {processed_authors} authors")
                        for param_key, count in total_pairs_count.items():
                            if count > 0:
                                self.logger.info(f"  {param_key}: {count} new pairs")
                        
                    except Exception as exc:
                        self.logger.info(f"Batch {batch_idx} generated an exception: {exc}")
                        continue
        self.logger.info(f"Parallel processing completed.")
        for param_key, total_pairs in total_new_pairs_dict.items():
            fixed_degree, resolution = self.parse_param_key(param_key)
            self.logger.info(f"Parameter {param_key}\t degree {fixed_degree}\t resolution {resolution}")
            self.logger.info(f"  Total new pairs: {total_pairs}")
            self.logger.info(f"  Final KPM matrix shape: {kpm_dict[param_key].shape}")
            self.logger.info(f"  Non-zero entries: {(kpm_dict[param_key] > 0).sum().sum()}")
        
        return kpm_dict

    @calculate_runtime
    def generate_kpm_parallel(self, params: TCPParams, fixed_ave_degree=None, resolution=1.0, 
                            max_authors=None, n_processes=64):
        """Generate KPM using parallel processing with ProcessPoolExecutor (single parameter set)"""
        self.logger.info(f"Starting parallel KPM generation (ave_degree={fixed_ave_degree}, resolution={resolution})...")
        
        # Load existing progress and KPM
        progress = self.load_progress(fixed_ave_degree, resolution)
        df_kpm = self.load_existing_kpm(fixed_ave_degree, resolution)
        completed_batches = set(progress["completed_batches"])
        
        # Get author batches
        author_ids = self.get_author_batches(
            min_papers=params.min_papers if hasattr(params, 'min_papers') else 10,
            max_papers=params.max_papers if hasattr(params, 'max_papers') else None
        )['author_id'].tolist()
        
        author_batches = [author_ids[i:i+params.batch_size] for i in range(0, len(author_ids), params.batch_size)]
        
        # Filter out completed batches
        remaining_batches = [(i, batch) for i, batch in enumerate(author_batches) if i not in completed_batches]
        
        if max_authors:
            total_authors = min(max_authors, sum(len(batch) for _, batch in remaining_batches))
            self.logger.info(f"Processing {total_authors} authors in {len(remaining_batches)} remaining batches")
        else:
            total_authors = sum(len(batch) for _, batch in remaining_batches)
            self.logger.info(f"Processing {total_authors} authors in {len(remaining_batches)} remaining batches")
        
        if not remaining_batches:
            self.logger.info("All batches already completed!")
            return df_kpm
        
        # Prepare batch data for parallel processing
        batch_data_list = [
            (batch_idx, author_batch, params, fixed_ave_degree, resolution)
            for batch_idx, author_batch in remaining_batches
        ]
        
        # Process batches in parallel using ProcessPoolExecutor
        total_new_pairs = 0
        
        with ProcessPoolExecutor(max_workers=n_processes) as executor:
            # Submit all tasks
            future_to_batch = {
                executor.submit(self.process_batch, batch_data): batch_data[0] 
                for batch_data in batch_data_list
            }
            
            # Process completed tasks as they finish
            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                try:
                    batch_idx, batch_concept_pairs, processed_authors = future.result()
                    
                    if batch_concept_pairs:
                        # Count concept pair occurrences
                        value_counts = pd.DataFrame(batch_concept_pairs, columns=['source', 'target']).value_counts().reset_index()
                        value_counts.columns = ['source', 'target', 'count_value']
                        
                        # Update the matrix
                        for _, row in value_counts.iterrows():
                            if row['source'] in df_kpm.index and row['target'] in df_kpm.columns:
                                df_kpm.loc[row['source'], row['target']] += row['count_value']
                        
                        total_new_pairs += len(batch_concept_pairs)
                    
                    # Mark batch as completed
                    completed_batches.add(batch_idx)
                    
                    # Save progress and KPM after each batch
                    progress["completed_batches"] = list(completed_batches)
                    self.save_progress(progress, fixed_ave_degree, resolution)
                    self.save_kpm(df_kpm, fixed_ave_degree, resolution)
                    
                    self.logger.info(f"Completed batch {batch_idx}, processed {processed_authors} authors, found {len(batch_concept_pairs)} pairs")
                    
                except Exception as exc:
                    self.logger.info(f"Batch {batch_idx} generated an exception: {exc}")
        
        self.logger.info(f"Parallel processing completed. Total new pairs: {total_new_pairs}")
        self.logger.info(f"Final KPM matrix shape: {df_kpm.shape}")
        self.logger.info(f"Non-zero entries: {(df_kpm > 0).sum().sum()}")
        
        return df_kpm

    def process_batch(self, batch_data):
        """Process a single batch of authors (single parameter set)"""
        batch_idx, author_batch, params, fixed_ave_degree, resolution = batch_data
        
        try:
            # Query data for current batch
            df_authors_works, df_works_concepts = self.query_author_data(author_batch)
            
            batch_concept_pairs = []
            processed_authors = 0
            
            # Process each author in the batch
            for author_id in author_batch:
                try:
                    # Filter data for current author
                    df_author_works = df_authors_works[df_authors_works['author_id'] == author_id]
                    df_author_work_concepts = df_works_concepts[
                        df_works_concepts.work_id.isin(df_author_works.work_id.unique())
                    ]
                    
                    # Build co-citation network and detect communities
                    df_author_communities = self.build_co_citation_network(
                        author_id, df_author_works, df_author_work_concepts,
                        fixed_ave_degree=fixed_ave_degree, resolution=resolution
                    )
                    
                    if df_author_communities is None:
                        continue
                        
                    # Extract concept pairs from temporal communities
                    concept_pairs = self.extract_temporal_community_pairs(
                        params, df_author_communities, df_author_work_concepts
                    )
                    
                    batch_concept_pairs.extend(concept_pairs)
                    processed_authors += 1
                    
                except Exception as e:
                    self.logger.info(f"Error processing author {author_id}: {str(e)}")
                    continue
            
            return batch_idx, batch_concept_pairs, processed_authors
            
        except Exception as e:
            self.logger.info(f"Error processing batch {batch_idx}: {str(e)}")
            return batch_idx, [], 0

    def build_co_citation_network(self, author_id, df_author_works, df_work_concepts, 
                                min_papers=10, min_edges=2, min_community_size=2, 
                                base_semantic_threshold=0.1, fixed_ave_degree=None, resolution=1.0):
        """Build co-citation network and detect communities for a single author (single parameter set)"""
        work_ids = df_author_works.work_id.unique()
        
        if len(work_ids) < min_papers:
            return None
            
        # Calculate matrices
        work_sim_matrix = self.semantic_similarity_matrix(df_work_concepts, work_ids)
        work_sim_matrix = np.maximum(work_sim_matrix, 0) # keep positive similarity
        triu_values = work_sim_matrix[np.triu_indices_from(work_sim_matrix, k=1)]
        co_citation_count_matrix, valid_references = self.co_citation_count_matrix(df_author_works)
        
        # Determine threshold
        threshold = self.get_similarity_threshold(triu_values, len(work_ids), fixed_ave_degree)
        
        # Create weight matrix
        weight_matrix = np.where(
            ((co_citation_count_matrix > 0) & (work_sim_matrix > base_semantic_threshold)) | 
            (work_sim_matrix >= threshold),
            work_sim_matrix,
            0
        )
        
        # Create weighted edges
        x, y = np.nonzero(np.triu(weight_matrix, k=1))
        if len(x) < min_edges:
            return None
            
        weighted_edges = np.column_stack((
            valid_references.iloc[x]['work_id'].values, 
            valid_references.iloc[y]['work_id'].values, 
            weight_matrix[x, y]
        ))
        
        # Create graph and detect communities
        author_ccn = nx.Graph()
        author_ccn.add_nodes_from(work_ids)
        author_ccn.add_weighted_edges_from(weighted_edges)
        
        partition = community_louvain.best_partition(author_ccn, resolution=resolution, random_state=42)

        # Extract work information
        df_work_id_infos = df_author_works[['work_id', 'publication_date']].drop_duplicates()
        df_work_id_infos['community_order'] = df_work_id_infos.work_id.map(partition)
        
        df_work_id_infos['publication_date'] = pd.to_datetime(df_work_id_infos.publication_date, errors='coerce')
        df_work_id_infos['publication_year'] = df_work_id_infos['publication_date'].dt.year

        # Group works by community
        df_author_communities = df_work_id_infos.groupby('community_order').apply(
            lambda x: pd.Series({
                'community': x['work_id'].tolist(), 
                'pub_year': x['publication_year'].tolist(), 
                'mean_pub_year': x['publication_year'].mean()
            }) if len(x) >= min_community_size else None,
            include_groups=False 
        ).dropna().reset_index()

        if len(df_author_communities) < 2:
            return None
            
        df_author_communities['author_id'] = author_id
        df_author_communities = df_author_communities.sort_values('mean_pub_year').reset_index(drop=True)
        
        return df_author_communities

    def load_existing_kpm(self, fixed_ave_degree=None, resolution=1.0):
        """Load existing KPM matrix if it exists"""
        kpm_filename = self.get_kpm_filename(fixed_ave_degree, resolution)
        kpm_path = op.join(self.path_manager.kpm_dir_all_deg_res_2, kpm_filename)
        
        if op.exists(kpm_path):
            return pd.read_csv(kpm_path, index_col=0)
        else:
            return pd.DataFrame(0, index=self.target_concept_names, columns=self.target_concept_names)

    def load_progress(self, fixed_ave_degree=None, resolution=1.0):
        """Load progress tracking information"""
        progress_filename = self.get_progress_filename(fixed_ave_degree, resolution)
        progress_path = op.join(self.path_manager.kpm_dir_all_deg_res_2, progress_filename)
        
        if op.exists(progress_path):
            with open(progress_path, 'r') as f:
                return json.load(f)
        else:
            return {"completed_batches": []}

    def save_progress(self, progress, fixed_ave_degree=None, resolution=1.0):
        """Save progress tracking information"""
        progress_filename = self.get_progress_filename(fixed_ave_degree, resolution)
        progress_path = op.join(self.path_manager.kpm_dir_all_deg_res_2, progress_filename)
        
        with open(progress_path, 'w') as f:
            json.dump(progress, f)

    def save_kpm(self, df_kpm, fixed_ave_degree=None, resolution=1.0):
        """Save KPM matrix to file"""
        kpm_filename = self.get_kpm_filename(fixed_ave_degree, resolution)
        kpm_path = op.join(self.path_manager.kpm_dir_all_deg_res_2, kpm_filename)
        
        df_kpm.to_csv(kpm_path)
        return kpm_path


if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description='Generate Knowledge Precedence Matrix with Sensitivity Analysis')
    parser.add_argument('--concept_level', type=int, default=1, help='Concept level')
    parser.add_argument('--wcr', type=float, default=1, help='Work coverage ratio')
    parser.add_argument('--select_mode', type=str, default='collective', help='Selection mode')
    parser.add_argument('--max_authors', type=int, default=None, help='Maximum number of authors to process')
    parser.add_argument('--n_processes', type=int, default=64, help='Number of parallel processes')
    parser.add_argument('--fixed_ave_degree', type=int, default=8, help='Fixed average degree')
    parser.add_argument('--resolution', type=float, default=1.0, help='Louvain resolution parameter')
    parser.add_argument('--run_sensitivity', default=True, help='Run full sensitivity analysis')
    args = parser.parse_args()
    
    # Create parameters
    params = TCPParams(
        concept_level=args.concept_level,
        less_than=True,
        select_mode=args.select_mode,
        wcr=args.wcr,
        batch_size=2000
    )
    
    # Initialize generator
    generator = IntegratedKPMGenerator(debugging = True)
    
    if args.run_sensitivity:
        # Run sensitivity analysis
        fixed_degrees = [4, 5, 6, 7, 8, 9, 10, 11, 12, None] # None for dynameic degree
        resolutions = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        
        kpm_dict = generator.generate_kpm_parallel_sensitivity(
            params, 
            fixed_degrees=fixed_degrees,
            resolutions=resolutions,
            max_authors=args.max_authors, 
            n_processes=args.n_processes
        )
        
        logger.info("Sensitivity analysis completed.")
        for param_key, df_kpm in kpm_dict.items():
            logger.info(f"Parameter {param_key}: Matrix shape {df_kpm.shape}, Non-zero entries: {(df_kpm > 0).sum().sum()}")
        
        # Save final KPM matrices
        saved_paths = generator.save_kpm_dict(kpm_dict)
        logger.info("KPM matrices saved:")
        for param_key, path in saved_paths.items():
            logger.info(f"  {param_key}: {path}")
    else:
        # Run single configuration
        df_kpm = generator.generate_kpm_parallel(
            params, 
            fixed_ave_degree=args.fixed_ave_degree, 
            resolution=args.resolution,
            max_authors=args.max_authors, 
            n_processes=args.n_processes
        )
        
        logger.info(f"KPM generation completed. Matrix shape: {df_kpm.shape}")
        logger.info(f"Non-zero entries: {(df_kpm > 0).sum().sum()}")
        
        # Save final KPM
        kpm_path = generator.save_kpm(df_kpm, args.fixed_ave_degree, args.resolution)
        logger.info(f"KPM saved to: {kpm_path}")

# nohup python ./network/direct_kpm_ave_degree_resolution_all_sensitivity.py >> direct_kpm_ave_degree_resolution_all_sensitivity.log 2>&1 &
# nohup python ./network/direct_kpm_ave_degree_resolution_all_sensitivity.py >> direct_kpm_ave_degree_resolution_all_sensitivity_2.log 2>&1 &

# ps aux | grep direct_kpm_ave_degree_resolution_all_sensitivity.py
# pkill -f direct_kpm_ave_degree_resolution_all_sensitivity.py
