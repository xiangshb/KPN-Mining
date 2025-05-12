
import os, sys
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.append(project_dir)

import networkx as nx
import pandas as pd
import numpy as np
import logging
import os.path as op
from typing import List, Optional, Union
from concurrent.futures import ProcessPoolExecutor
import tqdm
from utils.database import DatabaseManager
from utils.config import PathManager, calculate_runtime
from sklearn.metrics.pairwise import cosine_similarity

class CCN:
    """
    Class for constructing and analyzing co-citation networks (bibliographic coupling).
    """
    
    def __init__(self):
        """
        Initialize the co-citation network (CCN) class.
        
        Args:
            config: Configuration object containing parameters
        """
        self.logger = logging.getLogger(__name__)
        self.db_manager = DatabaseManager()
        self.path_manager = PathManager()
        self.concept_embedding, self.valid_concepts = self.load_concept_embedding()

    def load_concept_embedding(self):
        from gensim.models import Word2Vec
    
        # model_path = op.join(self.path_manager.ccns_dir, 'Word2Vec_dim_24_epoch_100.model')
        model_path = op.join(self.path_manager.navigation_embedding_dir, 'model', 'Word2Vec_dim_24_epoch_100.model')
        # model_path = op.join(self.path_manager.ccns_dir, 'embedding_model_dim_24.model')
        
        try:
            # Load the model
            model = Word2Vec.load(model_path)
            valid_concepts = set()
            nan_concepts = []
            for concept in model.wv.index_to_key:
                vec = model.wv[concept]
                if not np.isnan(vec).any():
                    valid_concepts.add(concept)
                else:
                    nan_concepts.append(concept)
            print(f"Successfully loaded Word2Vec model with dimension: {model.vector_size}")
            return model, valid_concepts
        except FileNotFoundError:
            print(f"Error: Model file not found: {model_path}")
            return None
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return None

    def semantic_work_similarity_matrix(self, df_work_concepts, work_ids):
        
        work_concept_groups = df_work_concepts.groupby('work_id')['display_name'].apply(list).reindex(work_ids)
        work_concept_groups = work_concept_groups.apply(lambda x: x if isinstance(x, list) else [])

        work_vectors = []
        for idx, concepts in work_concept_groups.items():
            if len(concepts) > 0: 
                vecs = np.stack([self.concept_embedding.wv[c] for c in concepts])
                mean_vec = vecs.mean(axis=0)
            else: mean_vec = np.zeros(self.concept_embedding.vector_size)

            work_vectors.append(mean_vec)
        work_vectors = np.vstack(work_vectors)
        work_sim_matrix = cosine_similarity(work_vectors)

        return work_sim_matrix

    def co_citation_count_matrix(self, df_author_works):
        # Group references by work
        valid_references = (
            df_author_works[['work_id', 'referenced_work_id']]
            .groupby('work_id', sort=False)['referenced_work_id']
            .apply(set)
            .apply(lambda s: set() if s == {None} else s)
            .reset_index(name='reference_ids')
        )
        
        # Create co-citation matrix based on reference overlap
        co_citation_count_matrix = np.array([
            valid_references['reference_ids'].apply(lambda x: len(work_refs_i.intersection(x))).tolist() 
            for work_refs_i in valid_references['reference_ids']
        ])
        
        return co_citation_count_matrix, valid_references

    def overlap_stats_by_quantile(self, co_citation_count_matrix, work_sim_matrix, quantile_start=0.6, quantile_end=0.95, quantile_step=0.05):
        assert co_citation_count_matrix.shape == work_sim_matrix.shape
        n = co_citation_count_matrix.shape[0]
        tri_idx = np.triu_indices(n, k=1)
        
        co_cit_upper = co_citation_count_matrix[tri_idx]
        work_sim_upper = work_sim_matrix[tri_idx]
        
        result = []
        quantiles = np.arange(quantile_start, quantile_end + 1e-8, quantile_step)

        co_cit_mask = co_cit_upper > 0
        for q in quantiles:
            qv = np.quantile(work_sim_upper, q)
            work_sim_mask = work_sim_upper >= qv
            overlap = co_cit_mask & work_sim_mask
            union = co_cit_mask | work_sim_mask

            co_citation_only = co_cit_mask & (~overlap)
            work_sim_only = work_sim_mask & (~overlap)

            result.append(['quantile', round(q, 2), np.sum(co_citation_only), np.sum(overlap), np.sum(work_sim_only), np.sum(union)])
        
        sim_thresholds = np.arange(quantile_start, quantile_end + 1e-8, quantile_step)
        for threshold in sim_thresholds:
            work_sim_mask = work_sim_upper >= threshold
            overlap = co_cit_mask & work_sim_mask
            union = co_cit_mask | work_sim_mask

            co_citation_only = co_cit_mask & (~overlap)
            work_sim_only = work_sim_mask & (~overlap)
            result.append(['threshold', round(threshold, 2), np.sum(co_citation_only), np.sum(overlap), np.sum(work_sim_only), np.sum(union)])

        df = pd.DataFrame(result, columns = ['mode', 'quantile', 'co_citation_only', 'overlap', 'work_sim_only', 'combine'])

        return df

    def ccn_data_query(self, author_ids, chunk_size = 5000):
        # Get author's works
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
        total_work_ids = len(all_work_ids)
        for i in range(0, total_work_ids, chunk_size):
            chunk_ids = all_work_ids[i:i+chunk_size]
            df_chunk = self.db_manager.query_table(
                table_name='works_concepts',
                columns=[
                    'works_concepts.work_id',
                    'concepts.display_name'
                ],
                join_tables=['concepts'],
                join_conditions=[
                    'works_concepts.concept_id = concepts.id'
                ],
                where_conditions=[f'''works_concepts.work_id in ('{"','".join(chunk_ids)}')'''],
                batch_read=False,
                show_runtime=False
            )
            all_results.append(df_chunk)
        df_works_concepts = pd.concat(all_results, ignore_index=True)
        df_works_concepts = df_works_concepts.loc[df_works_concepts.display_name.isin(self.valid_concepts)]

        return df_authors_works, df_works_concepts

    def get_work_sim_quantile(self, n):
        return 1 - np.log(10*n) / (n - 1)

    def build_co_citation_networks(self, author_ids: str, min_papers: int = 10, min_edges: int = 2, base_semantic_threshold = 0.1, threshold: float = 0.95, mode = 'quantile') -> Optional[nx.DiGraph]:
        """
        Build a co-citation network (bibliographic coupling) for an author.
        
        Args:
            author_id: Author ID
            min_papers: Minimum number of papers required
            min_edges: Minimum number of edges required
            
        Returns:
            Directed graph representing the co-citation network or None if criteria not met
        """
        # Get author's works
        df_authors_works, df_works_concepts = self.ccn_data_query(author_ids)

        # Initialize list to store CCNs
        CCNs = []
        
        # Process each author individually
        for author_id in author_ids:
            # Filter data for current author
            df_author_works = df_authors_works[df_authors_works['author_id'] == author_id]
            work_ids = df_author_works.work_id.unique() # sorted with publication time
            
            # Check if author has minimum number of papers
            if len(work_ids) < min_papers:
                continue
            
            df_work_concepts = df_works_concepts.loc[df_works_concepts.work_id.isin(work_ids)]
            
            # Calculate semantic similarity between works
            work_sim_matrix = self.semantic_work_similarity_matrix(df_work_concepts, work_ids)
            
            co_citation_count_matrix, valid_references = self.co_citation_count_matrix(df_author_works)
            
            n = work_ids.shape[0]
            dynamic_quantile = self.get_work_sim_quantile(n)

            # Determine threshold based on mode
            if mode == 'quantile':
                triu_values = work_sim_matrix[np.triu_indices_from(work_sim_matrix, k=1)]
                threshold_ = np.quantile(triu_values, dynamic_quantile)
            elif mode == 'threshold':
                threshold_ = threshold
            else:
                raise ValueError('Undefined mode')
            
            # Create weight matrix combining co-citation and semantic similarity
            weight_matrix = np.where(
                ((co_citation_count_matrix > 0) & (work_sim_matrix > base_semantic_threshold)) | (work_sim_matrix >= threshold_),
                work_sim_matrix,
                0
            ) # remove thoese with co-citation and less similar with each other
            
            # Create weighted edges
            x, y = np.nonzero(np.triu(weight_matrix, k=1))
            weighted_edges = np.transpose((
                valid_references.iloc[x]['work_id'].values, 
                valid_references.iloc[y]['work_id'].values, 
                weight_matrix[x, y]
            ))
            print(f'edges {weighted_edges.shape[0]} works {n} edges/works {weighted_edges.shape[0]/n}')
            # Check if network has minimum number of edges
            if weighted_edges.shape[0] < min_edges:
                continue
            
            # Create directed graph for current author
            author_ccn = nx.DiGraph()
            # Add author ID as graph attribute
            author_ccn.graph['author_id'] = author_id
            
            # Add nodes and edges
            author_ccn.add_nodes_from(work_ids)
            author_ccn.add_weighted_edges_from(weighted_edges)
            
            # Add number of references as node attributes
            valid_references['n_refs'] = valid_references.reference_ids.apply(len)
            nx.set_node_attributes(author_ccn, valid_references.set_index('work_id')['n_refs'].to_dict(), "n_refs")
            
            # Add publication dates as node attributes
            work_pubdate_dict = df_author_works.drop_duplicates('work_id').set_index('work_id')['publication_date'].to_dict()
            nx.set_node_attributes(author_ccn, work_pubdate_dict, "publication_date")
            
            # Append network to list
            CCNs.append(author_ccn)
        
        return CCNs

    def get_author_communities(self, author_ccn, community_method: str = 'louvain', min_community_size: int = 2, new_ccn = False):
        """
        Calculate communities within an author's co-citation network (CCN).
        
        Parameters:
        -----------
        author_ccn : networkx.Graph
            The co-citation network for an author
        community_method : str, default='louvain'
            Community detection algorithm to use ('louvain' or 'greedy')
        min_community_size : int, default=2
            Minimum number of nodes required to consider a group as a community
            
        Returns:
        --------
        tuple
            (Updated author_ccn graph with community attributes, DataFrame of community information)
        """
        from community import community_louvain
        from networkx.algorithms import community as nxcommunity

        # Extract publication dates from node attributes
        df_work_id_infos = pd.DataFrame.from_dict(nx.get_node_attributes(author_ccn, 'publication_date'), orient='index', columns=['publication_date'])
        df_work_id_infos['publication_date'] = pd.to_datetime(df_work_id_infos.publication_date, errors='coerce')
        df_work_id_infos = df_work_id_infos.sort_values('publication_date')
        df_work_id_infos['publication_year'] = df_work_id_infos['publication_date'].dt.year

        # Detect communities using specified method
        if community_method == 'louvain':
            partition = community_louvain.best_partition(author_ccn.to_undirected(), random_state=42)
        elif community_method == 'greedy':
            partition = {node: idx for idx, comm in enumerate(nxcommunity.greedy_modularity_communities(author_ccn)) 
                        for node in comm}
        else:
            raise ValueError(f"Community detection method '{community_method}' not implemented. Use 'louvain' or 'greedy'.")
        
        # Assign community labels to works
        df_work_id_infos['community_order'] = df_work_id_infos.index.map(partition)

        # Group works by community and filter by minimum size
        df_author_communities = df_work_id_infos.groupby('community_order').apply(
            lambda x: pd.Series({'community':x.index.tolist(), 'pub_year':x.publication_year.tolist(), 'mean_pub_date':x.publication_date.mean(), 'mean_pub_year':x.publication_year.mean()}) 
            if len(x) >= min_community_size else None
        ).dropna().reset_index()

        # Add author ID to communities dataframe and reorder columns
        df_author_communities['author_id'] = author_ccn.graph['author_id']
        columns = df_author_communities.columns.tolist()
        df_author_communities = df_author_communities[[columns[-1]] + columns[:-1]]

        # Sort communities by publication year 
        df_author_communities = df_author_communities.sort_values('mean_pub_year').reset_index(drop=True)
        df_author_communities['mean_pub_date'] = df_author_communities['mean_pub_date'].astype(str)

        if new_ccn:
            # Filter graph to include only nodes in valid communities
            df_work_id_infos['selected'] = df_work_id_infos.community_order.isin(df_author_communities.community_order)
            selected_nodes = df_work_id_infos.loc[df_work_id_infos.selected].index.tolist()
            author_ccn = author_ccn.subgraph(selected_nodes)

            # Update node attributes with new community assignments
            community_partition_dict = {}
            for i, community_i in df_author_communities[['community']].itertuples(index=True):
                for node_id in community_i:
                    community_partition_dict[node_id] = i
                    
            nx.set_node_attributes(author_ccn, community_partition_dict, "community_order")
        
            return author_ccn, df_author_communities.drop(columns=['community_order'])
        
        return None, df_author_communities.drop(columns=['community_order'])

    @calculate_runtime
    def generate_ccns(self, author_ids: Union[np.ndarray, List, None] = None, 
                      min_papers: int = 10, min_edges: int = 2, min_community_size: int = 2) -> List[nx.DiGraph]:
        """
        Generate co-citation networks for multiple authors.
        
        Args:
            author_ids: List of author IDs
            min_papers: Minimum number of papers required
            min_edges: Minimum number of edges required
            progress_bar: Whether to show progress bar
            
        Returns:
            List of co-citation networks
        """
        CCNs = []
        author_community_date_infos = []
        valid_author_ids = []
        pbar = tqdm.tqdm(total=len(author_ids))

        for i in range(0, len(author_ids), 1000):
            batch = author_ids[i:i+1000]
            ccns = self.build_co_citation_networks(batch[20], min_papers, min_edges, threshold = 0.95)
            for ccn in ccns:
                _, df_author_communities = self.get_author_communities(ccn, min_community_size=min_community_size)
                author_community_date_infos.append(df_author_communities.values)
                CCNs.extend(ccn)
                valid_author_ids.append(ccn.graph['author_id'])
            
            pbar.update(len(batch))

        pbar.close()

        return CCNs, author_community_date_infos, valid_author_ids

    def get_author_batches(self, min_papers: int = 10, max_papers: int = None, range_size: int = 20000, min_cites: int = 0, n_range_splits = 20, debugging = False, iter_i: int = -1):
        """
        Get batches of authors who have published at least min_papers papers.
        
        Args:
            min_papers: Minimum number of papers for author selection
            range_size: Size of each author batch
            min_cites: Minimum number of citations for author selection
            
        Returns:
            Tuple of (author_ids_batches, iterate_ranges)
        """
        # Create directory for storing CCNs if it doesn't exist
        CCNs_dir = self.path_manager.ensure_folder_exists(op.join(self.path_manager.external_file_dir, 'CCNs'))

        if max_papers is not None and max_papers > min_papers:
            pub_part = f"min_pubs_{min_papers}_max_pubs_{max_papers}"
        else:
            pub_part = f"min_pubs_{min_papers}"
        cites_part = f"_min_cites_{min_cites}" if min_cites > 0 else ""
        
        # File path for cached author IDs
        file_path_author_ids_threshold = op.join(CCNs_dir, f"Total_author_ids_{pub_part}{cites_part}.npy")
        
        # Get author IDs if not already cached
        if not op.exists(file_path_author_ids_threshold) or debugging:
            # Construct where conditions based on parameters
            where_conditions_ = [f"total_pubs >= '{min_papers}'"]
            if max_papers is not None and max_papers > min_papers:
                where_conditions_.append(f"total_pubs < '{max_papers}'")
            if min_cites > 0:
                where_conditions_.append(f"total_cits >= '{min_cites}'")
            
            # Query database for author IDs
            author_id_cites = self.db_manager.query_table(
                table_name='author_yearlyfeature_field_geq10pubs',
                columns=['author_id', 'total_pubs', 'total_cits'],
                where_conditions=where_conditions_
            )
            
            author_ids = author_id_cites.author_id.to_list()
            
            # Split authors into batches
            from more_itertools import chunked
            author_ids_ranges = np.array(list(chunked(author_ids, range_size)), dtype=object)
            
            # Cache the batches
            np.save(file_path_author_ids_threshold, author_ids_ranges)
        else:
            # Load cached author batches
            author_ids_ranges = np.load(file_path_author_ids_threshold, allow_pickle=True)
        
        # Split into iteration ranges (for parallel processing)
        iterate_ranges = np.array_split(range(len(author_ids_ranges)), n_range_splits)

        if iter_i >= 0:
            iterate_range = iterate_ranges[iter_i]
            selected_author_ids_ranges = [author_ids_ranges[i] for i in iterate_range]
            return selected_author_ids_ranges, iterate_range
        
        else: return author_ids_ranges, iterate_ranges
    
    @calculate_runtime
    def generate_ccn_iteration(self, iter_i: int = 0, min_papers: int = 10, max_papers = None, min_edges: int = 2, debugging = True, s_id: int = 0):
        """
        Generate co-citation networks for batches of authors.
        
        Args:
            iter_i: Index of the iteration range to process
            min_papers: Minimum number of papers required
            min_edges: Minimum number of edges required
            
        Returns:
            List of co-citation networks for the batch
        """
        # Get author batches
        path_cnns_dir = self.path_manager.ensure_folder_exists(op.join(self.path_manager.external_file_dir, 'CCNs', 'CCNs_sub'))
        path_community_pub_date_dir = self.path_manager.ensure_folder_exists(op.join(self.path_manager.external_file_dir, 'CCNs', 'CCNs_community_pub_date', 'iteration'))

        author_ids_ranges, iterate_range = self.get_author_batches(min_papers=min_papers, max_papers = max_papers, iter_i=iter_i)

        for i, k_th in enumerate(iterate_range):
            # Create directory for storing batch results
            file_path_cnns_k = op.join(path_cnns_dir, f'CCNs_sub_{k_th + s_id}.npy')
            file_path_community_date_k = op.join(path_community_pub_date_dir, f'CCNs_{k_th + s_id}_th_community_pub_date.npy')

            if not op.exists(file_path_cnns_k) or not op.exists(file_path_community_date_k) or debugging:
                self.logger.info(f'Generating all CCNs with iter: {iter_i} range: {k_th + s_id}-th')
                
                # Generate CCNs for this batch
                
                CCNs_k_th, community_date_infos = self.generate_ccns(
                    author_ids=author_ids_ranges[i], 
                    min_papers=min_papers,
                    min_edges=min_edges
                )

                # Save results
                np.save(file_path_cnns_k, np.array(CCNs_k_th, dtype=object))
                np.save(file_path_community_date_k, np.array(community_date_infos, dtype=object))
                
                self.logger.info(f'{file_path_cnns_k} cached')
                self.logger.info(f'{file_path_community_date_k} cached')
            else:
                # Load cached results
                CCNs_k_th = np.load(file_path_cnns_k, allow_pickle=True)
                
        return CCNs_k_th
    
    def find_checkpoint_ids(self, output_dir, output_dir_2, output_dir_3):
        import re
        # Check if directories exist
        if not op.exists(output_dir) or not op.exists(output_dir_2) or not op.exists(output_dir_3):
            return -1, []
        
        def extract_ids(directory, pattern_str):
            """Helper function to extract IDs from filenames matching a pattern"""
            pattern = re.compile(pattern_str)
            ids_set = set()
            
            for filename in os.listdir(directory):
                match = pattern.match(filename)
                if match:
                    ids_set.add(int(match.group(1)))
            
            return ids_set
        
        # Get IDs from all three file types
        ids_set1 = extract_ids(output_dir, r'CCNs_sub_(\d+)\.npy')
        ids_set2 = extract_ids(output_dir_2, r'CCNs_(\d+)_th_community_pub_date\.npy')
        ids_set3 = extract_ids(output_dir_3, r'CCNs_(\d+)_th_valid_authors\.npy')
        
        # Find IDs that have all three files
        complete_ids = ids_set1.intersection(ids_set2, ids_set3)
        
        if not complete_ids:
            return 0, []
        
        max_id = max(max(ids_set1), max(ids_set2), max(ids_set3))
        
        # Check for missing IDs from 0 to max_id
        expected_ids = set(range(max_id + 1))
        missing_ids = sorted(expected_ids - complete_ids)
        
        return max_id + 1, missing_ids
    
    def process_futures_results(self, futures, future_to_info, s_id=0):
        from concurrent.futures import as_completed
        with tqdm.tqdm(total=len(futures)) as pbar:
            for future in as_completed(futures):
                i, output_file, output_file_2, output_file_3 = future_to_info[future]
                id_updated = i + s_id
                pbar.set_description(f"Processing Chunks {id_updated}")
                try:
                    result = future.result()
                    np.save(output_file, np.array(result[0], dtype=object))
                    np.save(output_file_2, np.array(result[1], dtype=object))
                    np.save(output_file_3, np.array(result[2], dtype=object))
                    self.logger.info(
                        f"Chunk {id_updated} saved: "
                        f"{len(result[0])} networks, {len(result[1])} communities, {len(result[2])} authors.\n"
                        f"Networks file: {output_file}\n"
                        f"Communities file: {output_file_2}\n"
                        f"Author file: {output_file_3}"
                    )
                except Exception as e:
                    self.logger.error(f"Error processing chunk {id_updated}: {str(e)}")
                pbar.update(1)

    def get_author_csv(self, with_pubs = False):
        file_dir = self.path_manager.ensure_folder_exists(op.join(self.path_manager.external_file_dir, 'CCNs'))
        if with_pubs:
            return pd.read_csv(op.join(file_dir, 'all_author_pubs.csv'))
        else: return pd.read_csv(op.join(file_dir, 'all_author.csv'))

    @calculate_runtime
    def generate_ccn_parallel(self, min_papers: int = 10, max_papers: int = None, min_edges: int = 2, n_workers: int = 16, s_id: int = 0, run_check = False):
        """
        Generate co-citation networks in parallel for multiple authors.

        Args:
            min_papers: Minimum number of papers required per author.
            max_papers: Maximum number of papers allowed per author.
            min_edges: Minimum number of edges required in the network.
            n_workers: Number of parallel workers to use.
            s_id: Start index for output file naming.

        This method divides all authors into batches (each batch contains about 10,000 authors),
        and processes each batch in parallel using a process pool. For each batch, two files are saved:
        one for the co-citation networks and one for the community publication dates.
        """
        # Get author batches: each batch contains about 10,000 authors
        author_ids_ranges, _ = self.get_author_batches(min_papers=min_papers, max_papers=max_papers)
        
        # df_author_pubs = self.get_author_csv(with_pubs=False)

        # Output directories
        output_dir = self.path_manager.ensure_folder_exists(
            op.join(self.path_manager.external_file_dir, 'CCNs', 'CCNs_sub_3'))
        output_dir_2 = self.path_manager.ensure_folder_exists(
            op.join(self.path_manager.external_file_dir, 'CCNs', 'CCNs_community_pub_date_3'))
        output_dir_3 = self.path_manager.ensure_folder_exists(
            op.join(self.path_manager.external_file_dir, 'CCNs', 'CCNs_valid_authors'))
        
        author_ids_ranges = author_ids_ranges[s_id:]
        self.logger.info(f"Total {len(author_ids_ranges)} batches, {n_workers} workers")
        
        # self.generate_ccns(author_ids_ranges[0], min_papers = 10)
        
        if run_check:
            s_id, missing_ids = self.find_checkpoint_ids(output_dir, output_dir_2, output_dir_3)
            if len(missing_ids) > 0:
                author_ids_ranges_missing = author_ids_ranges[missing_ids]
                with ProcessPoolExecutor(max_workers=n_workers) as executor:
                    futures = []
                    future_to_info = {}
                    for i, chunk in enumerate(author_ids_ranges_missing):
                        id_updated = missing_ids[i]
                        output_file = op.join(output_dir, f'CCNs_sub_{id_updated}.npy')
                        output_file_2 = op.join(output_dir_2, f'CCNs_{id_updated}_th_community_pub_date.npy')
                        output_file_3 = op.join(output_dir_3, f'CCNs_{id_updated}_th_valid_authors.npy')
                        if op.exists(output_file):
                            self.logger.info(f"Chunk {id_updated} already processed, skipping")
                            continue
                        future = executor.submit(self.generate_ccns, chunk, min_papers, min_edges)
                        futures.append(future)
                        future_to_info[future] = (id_updated, output_file, output_file_2, output_file_3)

                    self.process_futures_results(futures, future_to_info, 0)

        # self.generate_ccns(author_ids_ranges[0], min_papers = 10)
        # Parallel processing
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = []
            future_to_info = {}
            for i, chunk in enumerate(author_ids_ranges):
                id_updated = i + s_id
                output_file = op.join(output_dir, f'CCNs_sub_{id_updated}.npy')
                output_file_2 = op.join(output_dir_2, f'CCNs_{id_updated}_th_community_pub_date.npy')
                output_file_3 = op.join(output_dir_3, f'CCNs_{id_updated}_th_valid_authors.npy')
                if op.exists(output_file):
                    self.logger.info(f"Chunk {id_updated} already processed, skipping")
                    continue
                future = executor.submit(self.generate_ccns, chunk, min_papers, min_edges)
                futures.append(future)
                future_to_info[future] = (i, output_file, output_file_2, output_file_3)

            self.process_futures_results(futures, future_to_info, s_id)

        self.logger.info("Parallel processing completed")
        
    def load_all_co_citation_networks(self, parallel = True, return_paths = False):
        """
        Load all previously generated citation networks.
        
        Returns:
            Dictionary mapping author IDs to their citation networks
        """
        CCNs_dir = op.join(self.path_manager.external_file_dir, 'CCNs')
        all_networks = {}
        file_paths = []

        # Check main directory
        if not op.exists(CCNs_dir):
            self.logger.warning(f"Directory {CCNs_dir} does not exist")
            return all_networks
        subdir = 'CCNs_sub_parallel' if parallel else 'CCNs_sub'
        subdir_path = op.join(CCNs_dir, 'CCNs_sub_3')
        # Find all network files
        network_files = [f for f in os.listdir(subdir_path) if f.endswith('.npy')]
        
        for file_name in network_files:
            try:
                file_path = op.join(subdir_path, file_name)
                file_paths.append(file_path)

                if not return_paths:
                    networks = np.load(file_path, allow_pickle=True)
                    
                    # Add to overall dictionary
                    for network in networks:
                        if isinstance(network, nx.DiGraph):
                            author_id = network.graph.get('author_id')
                            if author_id:
                                all_networks[author_id] = network
                            
                    self.logger.info(f"Loaded {len(networks)} networks from {file_name}")
            except Exception as e:
                self.logger.error(f"Error loading {file_name}: {str(e)}")
                
        self.logger.info(f"Loaded a total of {len(all_networks)} citation networks")

        if return_paths:
            self.logger.info(f"Returning paths of {len(file_paths)} network files")
            return file_paths
    
        return all_networks
    
    def check_missing_files(self, file_path_dir, max_k=132):
        """
        Check for missing files in a directory
        
        Parameters:
        file_path_dir: Directory path
        max_k: Maximum value of k_th
        
        Returns:
        missing_files: List of k_th values for missing files
        existing_files: List of k_th values for existing files
        """
        missing_files = []
        existing_files = []
        
        # Check if the iteration subdirectory exists
        if not op.exists(file_path_dir):
            print(f"Warning: '{file_path_dir}' directory does not exist")
            return list(range(0, max_k)), []
        
        # Check each possible file
        for k in range(0, max_k):
            file_path = op.join(file_path_dir, f'CCNs_{k}_th_community_pub_date.npy')
            if not op.exists(file_path):
                missing_files.append(k)
            else:
                existing_files.append(k)
        
        return missing_files, existing_files

    @calculate_runtime
    def load_author_community_info(self, max_k = 20, debugging = True, test = False):
        """
        Calculate publication dates for communities
        
        Parameters:
        generating: Whether to generate new data
        iter_i: Current iteration index
        n_iters: Total number of iterations
        min_community_size: Minimum community size
        
        Returns:
        Publication date information for all communities
        """
        file_path_dir = self.path_manager.ensure_folder_exists(op.join(self.path_manager.external_file_dir, 'CCNs'))
        # missing_files, existing_files = self.check_missing_files(op.join(file_path_dir, 'iteration'), max_k=sum([len(sub_array) for sub_array in iterate_ranges]))

        # Load or aggregate existing publication date information
        if test:
            file_path_all_info = op.join(file_path_dir, 'all_CCNs_communities_pub_date_test.npy')
            max_k = 10
        else: 
            file_path_all_info = op.join(file_path_dir, 'all_CCNs_communities_pub_date.npy')
            max_k = 339

        if not op.exists(file_path_all_info) or debugging:
            all_mean_pub_date_info_list = []
            for k_th in tqdm.trange(max_k): 
                # file_path = op.join(file_path_dir, 'CCNs_sub_3', f'CCNs_sub_{k_th}.npy')
                file_path = op.join(file_path_dir, 'CCNs_community_pub_date_3', f'CCNs_{k_th}_th_community_pub_date.npy')
                all_mean_pub_date_info_list.extend(np.load(file_path, allow_pickle=True))
            all_mean_pub_date_info = np.concatenate(all_mean_pub_date_info_list, axis=0)
            np.save(file_path_all_info, all_mean_pub_date_info, allow_pickle=True)
        else: 
            all_mean_pub_date_info = np.load(file_path_all_info, allow_pickle=True)
        return all_mean_pub_date_info
    
if __name__ == '__main__':
    import argparse
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("citation_network.log"),
            logging.StreamHandler()
        ]
    )
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate Citation Networks')
    parser.add_argument('--min_papers', type=int, default=10, help='Minimum number of papers per author')
    parser.add_argument('--max_papers', type=int, default=None, help='Minimum number of papers per author')
    parser.add_argument('--min_cites', type=int, default=0, help='Minimum number of citations per author')
    parser.add_argument('--min_edges', type=int, default=2, help='Minimum number of edges in network')
    parser.add_argument('--iter_i', type=int, default=0, help='Iteration index for batch processing')
    # parser.add_argument('--parallel', action='store_true', help='Use parallel processing')
    parser.add_argument('--parallel', default=True, help='Use parallel processing')
    parser.add_argument('--n_workers', type=int, default=16, help='Number of parallel workers')
    parser.add_argument('--analyze', action='store_true', help='Analyze existing networks')
    parser.add_argument('--generate_CCN', type=bool, default=True, help='Generate CCN')
    parser.add_argument('--s_id', type=int, default=0, help='Generate CCN') # 从已有批次的id开始避免id重复
    args = parser.parse_args()
    
    # Create a simple config object
    # Create citation network object
    citation_network = CCN()

    if args.generate_CCN:
        if args.parallel:
            # Generate networks in parallel
            citation_network.generate_ccn_parallel(
                min_papers=args.min_papers,
                max_papers=args.max_papers,
                min_edges=args.min_edges,
                n_workers=args.n_workers,
                s_id=args.s_id
            )
        else:
            # Generate networks in batch mode
            citation_network.generate_ccn_iteration(
                iter_i=args.iter_i,
                min_papers=args.min_papers,
                max_papers=args.max_papers,
                min_edges=args.min_edges,
                s_id=args.s_id
            )
    else:
        # Load and combine all results
        # all_network_paths = citation_network.load_all_co_citation_networks(parallel = False, return_paths=True)
    
        # networks = np.load(all_network_paths[0], allow_pickle=True)
        
        community_infos = citation_network.load_author_community_info(max_k=20)
        
    print("Process completed successfully")
    
# nohup python ./network/ccn_optimized.py >> ccn_optimized_progress.log 2>&1 &
# ps aux | grep ccn_optimized.py
# pkill -f ccn_optimized.py
