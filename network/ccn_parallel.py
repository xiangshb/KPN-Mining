
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
from concurrent.futures import ProcessPoolExecutor, as_completed
import tqdm
from utils.database import DatabaseManager
from utils.config import PathManager, calculate_runtime
from sklearn.metrics.pairwise import cosine_similarity
from community import community_louvain
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

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
    
        model_path = op.join(self.path_manager.ccns_dir, 'Word2Vec_dim_24_epoch_100.model')
        # model_path = op.join(self.path_manager.navigation_embedding_dir, 'model', 'Word2Vec_dim_24_epoch_100.model')
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
    
    def build_co_citation_network_communities(self, author_ids: list[str], min_papers: int = 10, 
                                            min_edges: int = 2, min_community_size: int = 2, 
                                            base_semantic_threshold: float = 0.1, threshold: float = 0.95, 
                                            mode: str = 'quantile') -> tuple[pd.DataFrame, list[str]]:
        """
        Build co-citation networks (bibliographic coupling) for multiple authors and detect communities.
        
        Args:
            author_ids: List of author IDs to process
            min_papers: Minimum number of papers required for an author to be processed
            min_edges: Minimum number of edges required in the network
            min_community_size: Minimum size for a community to be included
            base_semantic_threshold: Base threshold for semantic similarity
            threshold: Similarity threshold when mode is 'threshold'
            mode: Method to determine threshold ('quantile' or 'threshold')
            
        Returns:
            Tuple containing:
                - DataFrame with community information for all valid authors
                - List of valid author IDs that met the criteria
        """
        # Get author's works
        df_authors_works, df_works_concepts = self.ccn_data_query(author_ids)

        all_author_communities = []
        valid_author_ids = []
        # Process each author individually
        for author_id in author_ids:
            try:
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
                
                # Determine threshold based on mode
                if mode == 'quantile':
                    triu_values = work_sim_matrix[np.triu_indices_from(work_sim_matrix, k=1)]
                    threshold_ = np.quantile(triu_values, self.get_work_sim_quantile(len(work_ids)))
                elif mode == 'threshold':
                    threshold_ = threshold
                else:
                    raise ValueError(f"Invalid mode: {mode}. Use 'quantile' or 'threshold'")
                
                # Create weight matrix combining co-citation and semantic similarity
                weight_matrix = np.where(
                    ((co_citation_count_matrix > 0) & (work_sim_matrix > base_semantic_threshold)) | (work_sim_matrix >= threshold_),
                    work_sim_matrix,
                    0
                ) # remove those with co-citation and less similar with each other
                
                # Create weighted edges
                x, y = np.nonzero(np.triu(weight_matrix, k=1))
                # Check if network has minimum number of edges
                if len(x) < min_edges:
                    continue
                    
                weighted_edges = np.column_stack((
                    valid_references.iloc[x]['work_id'].values, 
                    valid_references.iloc[y]['work_id'].values, 
                    weight_matrix[x, y]
                ))
                
                # Create graph for current author
                author_ccn = nx.Graph()
                author_ccn.add_nodes_from(work_ids)
                author_ccn.add_weighted_edges_from(weighted_edges)
                
                # Detect communities using Louvain algorithm
                partition = community_louvain.best_partition(author_ccn, random_state=42)

                # Extract work information
                df_work_id_infos = df_author_works[['work_id', 'publication_date']].drop_duplicates()
                df_work_id_infos['community_order'] = df_work_id_infos.work_id.map(partition)
                
                # Convert dates to datetime format
                df_work_id_infos['publication_date'] = pd.to_datetime(df_work_id_infos.publication_date, errors='coerce')
                df_work_id_infos['publication_year'] = df_work_id_infos['publication_date'].dt.year

                # Group works by community and calculate statistics
                df_author_communities = df_work_id_infos.groupby('community_order').apply(
                    lambda x: pd.Series({
                        'community': x['work_id'].tolist(), 
                        'pub_year': x['publication_year'].tolist(), 
                        'mean_pub_date': x['publication_date'].mean(), 
                        'mean_pub_year': x['publication_year'].mean()
                    }) if len(x) >= min_community_size else None,
                    include_groups=False 
                ).dropna().reset_index()

                # Check if we have enough communities
                if len(df_author_communities) < min_community_size: # at least two communities
                    continue
                selected_works = df_work_id_infos.loc[df_work_id_infos.community_order.isin(df_author_communities.community_order)]
                valid_author_ids.append([author_id, len(df_author_communities), selected_works.publication_year.min(), selected_works.publication_year.max()])

                # Add author ID to communities dataframe and reorder columns
                df_author_communities['author_id'] = author_id
                df_author_communities = df_author_communities[['author_id', 'community', 'pub_year', 'mean_pub_date', 'mean_pub_year']]

                # Sort communities by publication year 
                df_author_communities = df_author_communities.sort_values('mean_pub_year').reset_index(drop=True)
                df_author_communities['mean_pub_date'] = df_author_communities['mean_pub_date'].astype(str)

                all_author_communities.append(df_author_communities)
                
            except Exception as e:
                # Log the error but continue processing other authors
                print(f"Error processing author {author_id}: {str(e)}")
                continue

        # Handle case where no authors met the criteria
        if not all_author_communities:
            return pd.DataFrame(), []
            
        # Combine all author communities
        df_author_community_date_infos = pd.concat(all_author_communities, ignore_index=True)
        df_valid_author_ids = pd.DataFrame(valid_author_ids, columns=['author_id','n_communities','comm_start_year','comm_end_year'])

        return df_author_community_date_infos, df_valid_author_ids

    @calculate_runtime
    def generate_ccns(self, author_ids: Union[np.ndarray, List, None] = None, 
                    min_papers: int = 10, min_edges: int = 2, min_community_size: int = 2, batch_id: int = 0) -> tuple:
        """
        Generate co-citation networks for multiple authors.
        """
        all_communities = []
        all_valid_authors = []
        
        pbar = tqdm.tqdm(total=len(author_ids), desc=f"Sub batch {batch_id}")

        for i in range(0, len(author_ids), 1000):
            try:
                batch = author_ids[i:i+1000]
                df_batch_communities, df_valid_batch_authors = self.build_co_citation_network_communities(
                    batch, min_papers, min_edges, min_community_size, threshold=0.95
                )
                
                if not df_batch_communities.empty:
                    all_communities.append(df_batch_communities)
                    all_valid_authors.append(df_valid_batch_authors)
                    
                pbar.update(len(batch))
            except Exception as e:
                error_type = type(e).__name__
                self.logger.error(f"Error processing batch {i}: {error_type} - {str(e)}")
                pbar.update(len(batch))
                continue

        pbar.close()

        # Combine all results
        df_all_communities = pd.concat(all_communities, ignore_index=True) if all_communities else pd.DataFrame()
        df_all_valid_authors = pd.concat(all_valid_authors, ignore_index=True) if all_valid_authors else pd.DataFrame()
        
        return df_all_communities, df_all_valid_authors

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
    
    def find_checkpoint_ids(self, output_dir_communities, output_dir_authors, max_id):
        import re
        def extract_ids(directory, pattern_str):
            """Helper function to extract IDs from filenames matching a pattern"""
            pattern = re.compile(pattern_str)
            ids_set = set()
            
            try:
                for filename in os.listdir(directory):
                    match = pattern.match(filename)
                    if match:
                        ids_set.add(int(match.group(1)))
            except Exception as e:
                self.logger.error(f"Error extracting IDs from {directory}: {str(e)}")
                return set()
            
            return ids_set
        
        # Get IDs from both file types
        ids_set1 = extract_ids(output_dir_communities, r'CCNs_(\d+)_communities\.parquet')
        ids_set2 = extract_ids(output_dir_authors, r'CCNs_(\d+)_valid_authors\.csv')
        
        # Find IDs that have both files
        # complete_ids = ids_set1.intersection(ids_set2)
        complete_ids = set(ids_set1)
        
        if not ids_set1 and not ids_set2:
            self.logger.info("No existing files found in any directory")
            return []
        
        # Find the maximum ID to determine the range we need to check
        all_ids = ids_set1.union(ids_set2)
        if not all_ids:
            return []
        
        # Check for missing IDs from 0 to max_id
        expected_ids = set(range(max_id ))
        missing_ids = sorted(expected_ids - complete_ids)
        
        self.logger.info(f"Found {len(missing_ids)} missing batches out of {max_id + 1}")
        return missing_ids
    
    def process_futures_results(self, futures, future_to_info):
        """
        Process the results from parallel execution futures.
        
        Args:
            futures: List of futures to process
            future_to_info: Dictionary mapping futures to output information
        """
        for future in as_completed(futures):
            id_updated, output_file_communities, output_file_authors = future_to_info[future]
            try:
                df_communities, df_valid_authors = future.result()
                
                # Save communities dataframe to parquet
                if not df_communities.empty:
                    df_communities.to_parquet(output_file_communities)
                    self.logger.info(f"Saved communities for chunk {id_updated} with {len(df_communities)} rows")
                else:
                    self.logger.warning(f"No communities found for chunk {id_updated}")
                    
                # Save valid authors to CSV (changed from parquet to csv)
                if not df_valid_authors.empty:
                    self.path_manager.save_csv_file(df_valid_authors, abs_file_path=output_file_authors, index=False, override=True)
                    
                self.logger.info(f"Processed chunk {id_updated}, found {len(df_valid_authors) if not df_valid_authors.empty else 0} valid authors")
                
            except Exception as e:
                self.logger.error(f"Error processing chunk {id_updated}: {str(e)}")

    def get_author_csv(self, with_pubs = False):
        file_dir = self.path_manager.ensure_folder_exists(op.join(self.path_manager.external_file_dir, 'CCNs'))
        if with_pubs:
            return pd.read_csv(op.join(file_dir, 'all_author_pubs.csv'))
        else: return pd.read_csv(op.join(file_dir, 'all_author.csv'))

    @calculate_runtime
    def generate_ccn_parallel(self, min_papers: int = 10, max_papers: int = None, min_edges: int = 2, 
                            min_community_size: int = 2, n_workers: int = 16, s_id: int = 0, run_check = False):
        """
        Generate co-citation networks in parallel for multiple authors.

        Args:
            min_papers: Minimum number of papers required per author.
            max_papers: Maximum number of papers allowed per author.
            min_edges: Minimum number of edges required in the network.
            min_community_size: Minimum size for a community to be included.
            n_workers: Number of parallel workers to use.
            s_id: Start index for output file naming.
            run_check: Whether to check for missing batches and process only those.

        This method divides all authors into batches (each batch contains about 10,000 authors),
        and processes each batch in parallel using a process pool. For each batch, two files are saved:
        one for the community information and one for the valid author IDs.
        """
        # Get author batches: each batch contains about 10,000 authors
        author_ids_ranges, _ = self.get_author_batches(min_papers=min_papers, max_papers=max_papers)
        
        # Output directories
        output_dir_communities = self.path_manager.ensure_folder_exists(
            op.join(self.path_manager.external_file_dir, 'CCNs', 'CCNs_valid_authors'))
        output_dir_authors = self.path_manager.ensure_folder_exists(
            op.join(self.path_manager.external_file_dir, 'CCNs', 'CCNs_valid_authors'))
        
        author_ids_ranges = author_ids_ranges[s_id:]
        self.logger.info(f"Total {len(author_ids_ranges)} batches, {n_workers} workers")
        
        if run_check:
            missing_ids = self.find_checkpoint_ids(output_dir_communities, output_dir_authors, max_id = len(author_ids_ranges))
            self.logger.info(f"Total {len(missing_ids)} missing batches, {n_workers} workers")
            self.logger.info(f"Detailed missing ids {missing_ids}")
            if len(missing_ids) > 0:
                author_ids_ranges_missing = [author_ids_ranges[i] for i in missing_ids]
                with ProcessPoolExecutor(max_workers=n_workers) as executor:
                    futures = []
                    future_to_info = {}
                    for i, chunk in enumerate(author_ids_ranges_missing):
                        id_updated = missing_ids[i] + s_id
                        output_file_communities = op.join(output_dir_communities, f'CCNs_{id_updated}_communities.parquet')
                        output_file_authors = op.join(output_dir_authors, f'CCNs_{id_updated}_valid_authors.csv')
                        
                        future = executor.submit(
                            self.generate_ccns, chunk, min_papers, min_edges, min_community_size, batch_id=id_updated
                        )
                        futures.append(future)
                        future_to_info[future] = (id_updated, output_file_communities, output_file_authors)

                    self.process_futures_results(futures, future_to_info)
                return

        # Parallel processing for all batches
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = []
            future_to_info = {}
            for i, chunk in enumerate(author_ids_ranges):
                id_updated = i + s_id
                output_file_communities = op.join(output_dir_communities, f'CCNs_{id_updated}_communities.parquet')
                output_file_authors = op.join(output_dir_authors, f'CCNs_{id_updated}_valid_authors.csv')
                
                future = executor.submit(
                    self.generate_ccns, chunk, min_papers, min_edges, min_community_size, batch_id=id_updated
                )
                futures.append(future)
                future_to_info[future] = (id_updated, output_file_communities, output_file_authors)

            self.process_futures_results(futures, future_to_info)

        self.logger.info("Parallel processing completed")

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
    parser.add_argument('--min_edges', type=int, default=2, help='Minimum number of edges in network')
    parser.add_argument('--n_workers', type=int, default=20, help='Number of parallel workers')
    parser.add_argument('--s_id', type=int, default=0, help='Generate CCN') # 从已有批次的id开始避免id重复
    args = parser.parse_args()
    
    # Create a simple config object
    # Create citation network object
    citation_network = CCN()
    
    citation_network.generate_ccn_parallel(
        min_papers=args.min_papers,
        max_papers=args.max_papers,
        min_edges=args.min_edges,
        n_workers=args.n_workers,
        s_id=args.s_id,
        run_check=True
    )
        
    print("Process completed successfully")
    
# nohup python ./network/ccn_optimized_parallel.py >> ccn_optimized_parallel_progress.log 2>&1 &
# nohup python ./network/ccn_optimized_parallel.py >> ccn_optimized_parallel_progress.log 2>&1 &

# ps aux | grep ccn_optimized_parallel.py
# pkill -f ccn_optimized_parallel.py

