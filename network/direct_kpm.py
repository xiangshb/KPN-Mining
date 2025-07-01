import os, sys
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.append(project_dir)

import networkx as nx
import pandas as pd
import numpy as np
import logging
import os.path as op
from typing import List, Union
import tqdm
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

class IntegratedKPMGenerator:
    """
    Integrated class for generating Knowledge Precedence Matrix (KPM) directly from database
    """
    path_manager = PathManager()
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.db_manager = DatabaseManager()
        self.APYD = APYDAnalyzer()
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
            print(f"Successfully loaded Word2Vec model with dimension: {model.vector_size}")
            return model, valid_concepts
        except Exception as e:
            print(f"Error loading model: {str(e)}")
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

    def semantic_similarity_matrix(self, df_work_concepts, work_ids):
        """Calculate semantic similarity matrix between works"""
        work_concept_groups = df_work_concepts.groupby('work_id')['display_name'].apply(list).reindex(work_ids)
        work_concept_groups = work_concept_groups.apply(lambda x: x if isinstance(x, list) else [])

        work_vectors = []
        for word_id, concepts in work_concept_groups.items():
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

    def average_degree_quantile(self, n_works, fixed_ave_degree = None):
        if fixed_ave_degree is None:
            quantile = 1 - np.log(10*n_works) / (n_works - 1)
        else:
            quantile = 1 - fixed_ave_degree / (n_works - 1)
        return quantile

    def get_similarity_threshold(self, work_sim_matrix, n_works):
        """Get similarity threshold based on quantile"""
        triu_values = work_sim_matrix[np.triu_indices_from(work_sim_matrix, k=1)]
        quantile = self.average_degree_quantile(n_works)
        return np.quantile(triu_values, quantile)

    def build_co_citation_network(self, author_id, df_author_works, df_work_concepts, 
                                min_papers=10, min_edges=2, min_community_size=2, 
                                base_semantic_threshold=0.1):
        """Build co-citation network and detect communities for a single author"""
        work_ids = df_author_works.work_id.unique()
        
        if len(work_ids) < min_papers:
            return None
            
        # Calculate matrices
        work_sim_matrix = self.semantic_similarity_matrix(df_work_concepts, work_ids)
        co_citation_count_matrix, valid_references = self.co_citation_count_matrix(df_author_works)
        
        # Determine threshold
        threshold = self.get_similarity_threshold(work_sim_matrix, len(work_ids))
        
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
        
        partition = community_louvain.best_partition(author_ccn, random_state=42)

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
        # The cumulative count of unique work IDs associated with the top N concepts
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
        if params.select_mode == 'collective':
            if concept_statistics.top_n_collective_wcr[0] < 1:
                min_index = np.argwhere(
                    (concept_statistics.top_n_collective_wcr >= params.wcr).values
                ).flatten()[0]
            else: 
                min_index = concept_statistics.loc[concept_statistics.wcr == 1].index.max()
            representative_concepts = concept_statistics.loc[:min_index, 'display_name'].tolist()
            
        if params.select_mode == 'hybrid':
            if concept_statistics.top_n_collective_wcr[0] < 1:
                min_index_1 = np.argwhere(
                    (concept_statistics.top_n_collective_wcr >= params.wcr).values
                ).flatten()[0]
            else: 
                min_index_1 = concept_statistics.loc[concept_statistics.wcr == 1].index.max()
            
            try:
                min_index_2 = np.argwhere(concept_statistics.wcr >= params.wcr).flatten()[-1]
            except IndexError:
                min_index_2 = 0

            min_index = max(min_index_1, min_index_2)
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

    @calculate_runtime
    def generate_kmp_from_database(self, params: TCPParams, max_authors=None):
        """Generate KMP directly from database"""
        print("Starting KMP generation from database...")
        
        # Get author batches
        author_ids = self.get_author_batches(
            min_papers=params.min_papers if hasattr(params, 'min_papers') else 10,
            max_papers=params.max_papers if hasattr(params, 'max_papers') else None
        )['author_id'].tolist()
        author_batches = [author_ids[i:i+params.batch_size] for i in range(0, len(author_ids), params.batch_size)]

        if max_authors:
            total_authors = min(max_authors, sum(len(batch) for batch in author_batches))
            print(f"Processing {total_authors} authors (limited by max_authors)")
        else:
            total_authors = sum(len(batch) for batch in author_batches)
            print(f"Processing {total_authors} authors")
        
        all_concept_pairs = []
        processed_authors = 0
        
        for batch_idx, author_batch in enumerate(author_batches):
            if max_authors and processed_authors >= max_authors:
                break
                
            print(f"Processing batch {batch_idx + 1}/{len(author_batches)}")
            
            # Query data for current batch
            df_authors_works, df_works_concepts = self.query_author_data(author_batch)
            
            # Process each author in the batch
            for author_id in tqdm.tqdm(author_batch, desc=f"Batch {batch_idx + 1}"):
                if max_authors and processed_authors >= max_authors:
                    break
                    
                try:
                    # Filter data for current author
                    df_author_works = df_authors_works[df_authors_works['author_id'] == author_id]
                    df_author_work_concepts = df_works_concepts[
                        df_works_concepts.work_id.isin(df_author_works.work_id.unique())
                    ]
                    
                    # Build co-citation network and detect communities
                    df_author_communities = self.build_co_citation_network(
                        author_id, df_author_works, df_author_work_concepts
                    )
                    
                    if df_author_communities is None:
                        continue
                        
                    # Extract concept pairs from temporal communities
                    concept_pairs = self.extract_temporal_community_pairs(
                        params, df_author_communities, df_author_work_concepts
                    )
                    
                    all_concept_pairs.extend(concept_pairs)
                    processed_authors += 1
                    
                except Exception as e:
                    print(f"Error processing author {author_id}: {str(e)}")
                    continue
        
        print(f"Processed {processed_authors} authors, found {len(all_concept_pairs)} concept pairs")
        
        # Build KMP matrix
        df_kmp = pd.DataFrame(0, index=self.target_concept_names, columns=self.target_concept_names)
        
        # Count concept pair occurrences
        if all_concept_pairs:
            value_counts = pd.DataFrame(all_concept_pairs, columns=['source', 'target']).value_counts().reset_index()
            value_counts.columns = ['source', 'target', 'count_value']
            
            # Fill the matrix
            for _, row in value_counts.iterrows():
                if row['source'] in df_kmp.index and row['target'] in df_kmp.columns:
                    df_kmp.loc[row['source'], row['target']] += row['count_value']
        
        return df_kmp

    def save_kmp(self, df_kmp, params: TCPParams):
        """Save KMP matrix to file"""
        if params.select_mode == 'respective' or params.select_mode == 'collective':
            file_name_end = f'{params.select_mode}_cover_{params.wcr}_works'
        elif params.select_mode == 'all':
            file_name_end = 'cover_all_community_concepts'
        else: 
            raise ValueError('Undefined select mode')
            
        params_str = f"level_{'lt' if params.less_than else 'eq'}_{params.concept_level}_{file_name_end}"
        df_kmp_path = op.join(self.path_manager.concepts_dir, f"kmp_{params_str}.csv")
        
        self.path_manager.save_csv_file(
            variable=df_kmp, 
            abs_file_path=df_kmp_path, 
            index=True, 
            override=True
        )
        
        print(f"KMP saved to: {df_kmp_path}")
        return df_kmp_path

if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description='Generate Knowledge Precedence Matrix')
    parser.add_argument('--concept_level', type=int, default=1, help='Concept level')
    parser.add_argument('--wcr', type=float, default=1, help='Work coverage ratio')
    parser.add_argument('--select_mode', type=str, default='collective', help='Selection mode')
    parser.add_argument('--max_authors', type=int, default=None, help='Maximum number of authors to process')
    args = parser.parse_args()
    
    # Create parameters
    params = TCPParams(
        concept_level=args.concept_level,
        less_than=True,
        select_mode=args.select_mode,
        wcr=args.wcr
    )
    
    # Generate KMP
    generator = IntegratedKPMGenerator()
    df_kmp = generator.generate_kmp_from_database(params, max_authors=args.max_authors)
    
    # Save KMP
    kmp_path = generator.save_kmp(df_kmp, params)
    
    print(f"KMP generation completed. Matrix shape: {df_kmp.shape}")
    print(f"Non-zero entries: {(df_kmp > 0).sum().sum()}")
