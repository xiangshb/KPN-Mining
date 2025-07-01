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
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
import threading
import time
from collections import defaultdict
import argparse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(line_buffering=True)

def process_author_batch(batch_data):
    """Worker function for processing author batch in subprocess"""
    batch_idx, author_batch, target_concept_ids, concept_id_to_name = batch_data
    
    try:
        logger.info(f"Processing batch {batch_idx} with {len(author_batch)} authors...")
        # Create local database manager for this process
        local_db_manager = DatabaseManager(
            pool_size=2,
            max_overflow=1,
            pool_timeout=60,
            pool_recycle=3600,
            pool_pre_ping=True
        )
        
        # Query data for current batch
        df_authors_works = local_db_manager.query_table(
            table_name='works_authorships',
            columns=[
                'works_authorships.work_id',
                'works_referenced_works.referenced_work_id',
            ],
            columns_trim_enter=['works_referenced_works.referenced_work_id'], 
            join_tables=['works_referenced_works'],
            join_conditions=[
                'works_authorships.work_id = works_referenced_works.work_id',
            ],
            where_conditions=[f'''works_authorships.author_id in ('{"','".join(author_batch)}')'''],
            batch_read=False,
            show_runtime=False
        ).dropna(subset=['referenced_work_id']).reset_index(drop=True)
        
        all_work_ids = list(set(df_authors_works.work_id.unique()).union(df_authors_works.referenced_work_id.unique()))
        
        # Process works in chunks
        all_results = []
        chunk_size = 5000
        for i in range(0, len(all_work_ids), chunk_size):
            chunk_ids = all_work_ids[i:i+chunk_size]
            
            df_chunk = local_db_manager.query_table(
                table_name='works_concepts',
                columns=['work_id', 'concept_id'],
                where_conditions=[f'''work_id in ('{"','".join(chunk_ids)}')''', f'''concept_id in ('{"','".join(target_concept_ids)}')'''],
                batch_read=False,
                show_runtime=False
            )
            all_results.append(df_chunk)
            
        if all_results: 
            df_works_concepts = pd.concat(all_results, ignore_index=True)
        else:
            df_works_concepts = pd.DataFrame(columns=['work_id', 'concept_id'])

        df_works_concepts['display_name'] = df_works_concepts['concept_id'].map(concept_id_to_name)
        
        # Process concept pairs
        work_concepts_dict = df_works_concepts.groupby('work_id')['display_name'].apply(set).to_dict()
        df_authors_works['work_concepts'] = df_authors_works['work_id'].map(work_concepts_dict)
        df_authors_works['referenced_work_concepts'] = df_authors_works['referenced_work_id'].map(work_concepts_dict)
        df_valid = df_authors_works.dropna(subset=['work_concepts', 'referenced_work_concepts'])
        
        concept_pairs = [
            (ref_concept, work_concept)
            for ref_concepts, work_concepts in zip(df_valid['referenced_work_concepts'], df_valid['work_concepts'])
            for ref_concept in ref_concepts
            for work_concept in work_concepts
        ]
        if concept_pairs:
            pair_counts = pd.Series(concept_pairs).value_counts().to_dict()
        else:
            pair_counts = {}
        # Count concept pair occurrences
        return {
            'batch_idx': batch_idx,
            'pair_counts': pair_counts,
            'total_pairs': len(concept_pairs),
            'status': 'success'
        }
        
    except Exception as e:
        logger.info(f"Batch {batch_idx} failed with error: {str(e)}")
        return {
            'batch_idx': batch_idx,
            'pair_counts': {},
            'total_pairs': 0,
            'status': 'error',
            'error': str(e)
        }
    finally:
        if local_db_manager is not None:
            try:
                local_db_manager.close()
            except:
                pass 

class ParallelKPMGenerator:
    """
    Parallel version of KPM generator with caching and progress tracking
    """
    path_manager = PathManager()
    
    def __init__(self, debugging = False):
        self.db_manager = DatabaseManager()
        self.logger = logging.getLogger(__name__)
        self.debugging = debugging
        self.sciconnav_embedding, self.valid_concepts = self.load_sciconnav_embedding()
        self.concepts_table = Concept.discipline_category_classification_llm(with_abbreviation=True)
        self.target_concepts = pd.read_csv(op.join('./llm_annotation/df_selected_top_concepts.csv'))
        self.target_concepts = self.target_concepts.loc[self.target_concepts.display_name.isin(self.valid_concepts)]
        self.target_concept_ids = self.target_concepts.id.tolist()
        self.target_concept_names = self.target_concepts.display_name.tolist()
        self.concept_id_to_name = self.target_concepts.set_index('id')['display_name'].to_dict()
        
        # Initialize cache paths
        
        self.progress_file = op.join(self.path_manager.sub_kpm_dir, 'progress_citation_based.json')
        self.kpm_file = op.join(self.path_manager.sub_kpm_dir, 'df_kpm_citation_based_parallel.csv')
        
        # Thread lock for safe file operations
        self.lock = threading.Lock()

    def __del__(self):
        """Cleanup database connections"""
        if hasattr(self, 'db_manager') and self.db_manager:
            try:
                self.db_manager.close()
            except:
                pass

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

    def load_progress(self):
        """Load existing progress and KPM matrix"""
        if op.exists(self.progress_file):
            with open(self.progress_file, 'rb') as f:
                progress = pickle.load(f)
        else:
            progress = {'completed_batches': set(), 'total_batches': 0}
        
        if op.exists(self.kpm_file):
            df_kpm = pd.read_csv(self.kpm_file, index_col=0)
            self.logger.info(f"Loaded existing KPM matrix with shape: {df_kpm.shape}")
        else:
            df_kpm = pd.DataFrame(0, index=self.target_concept_names, columns=self.target_concept_names)
            self.logger.info("Created new KPM matrix")
        
        return progress, df_kpm

    def save_progress(self, progress, df_kpm):
        """Save progress and KPM matrix"""
        with self.lock:
            # Save progress
            with open(self.progress_file, 'wb') as f:
                pickle.dump(progress, f)
            
            # Save KPM matrix
            df_kpm.to_csv(self.kpm_file)

    def update_kpm_with_results(self, df_kpm, results):
        """Update KPM matrix with batch results"""
        for pair, count in results['pair_counts'].items():
            source, target = pair
            if source in df_kpm.index and target in df_kpm.columns:
                df_kpm.loc[source, target] += count
        return df_kpm

    @calculate_runtime
    def generate_kpm_parallel(self, batch_size=2000, n_workers=64):
        """Generate KPM using parallel processing"""
        self.logger.info("Starting parallel KPM generation from database...")
        
        # Load progress and existing KPM
        progress, df_kpm = self.load_progress()
        
        # Get author batches
        author_ids = self.get_author_batches()['author_id'].tolist()
        author_batches = [author_ids[i:i+batch_size] for i in range(0, len(author_ids), batch_size)]
        
        total_batches = len(author_batches)
        progress['total_batches'] = total_batches
        
        # Filter out completed batches
        remaining_batches = [
            (idx, batch) for idx, batch in enumerate(author_batches)
            if idx not in progress['completed_batches']
        ]
        
        self.logger.info(f"Total batches: {total_batches}")
        self.logger.info(f"Completed batches: {len(progress['completed_batches'])}")
        self.logger.info(f"Remaining batches: {len(remaining_batches)}")
        
        if not remaining_batches:
            self.logger.info("All batches already completed!")
            return df_kpm
        
        # Prepare batch data for workers
        batch_data_list = [
            (batch_idx, author_batch, self.target_concept_ids, self.concept_id_to_name)
            for batch_idx, author_batch in remaining_batches
        ]
        
        if self.debugging:
            # Debug mode: process only the first batch sequentially
            self.logger.info("Running in debug mode - processing first batch only...")
            
            if batch_data_list:
                # Take the first batch for debugging
                first_batch_data = batch_data_list[0]
                batch_idx = first_batch_data[0]
                
                self.logger.info(f"Processing debug batch {batch_idx} with {len(first_batch_data[1])} authors...")
                
                # Process the batch
                results = process_author_batch(first_batch_data)
                
                self.logger.info(f"Debug batch results:")
                self.logger.info(f"  Status: {results['status']}")
                self.logger.info(f"  Total pairs: {results['total_pairs']}")
                self.logger.info(f"  Unique pair types: {len(results['pair_counts'])}")
                
                if results['status'] == 'success':
                    # Show some example pairs
                    if results['pair_counts']:
                        self.logger.info(f"  Sample pairs:")
                        sample_pairs = list(results['pair_counts'].items())[:5]
                        for pair, count in sample_pairs:
                            self.logger.info(f"    {pair[0]} -> {pair[1]}: {count}")
                    
                    # Update KPM matrix with debug results
                    df_kpm = self.update_kpm_with_results(df_kpm, results)
                    
                    # Update progress
                    progress['completed_batches'].add(batch_idx)
                    
                    # Save debug results
                    self.save_progress(progress, df_kpm)
                    
                    self.logger.info(f"Debug batch completed successfully!")
                    self.logger.info(f"KPM matrix updated - non-zero entries: {(df_kpm > 0).sum().sum()}")
                else:
                    self.logger.info(f"Debug batch failed with error: {results.get('error', 'Unknown error')}")
            else:
                self.logger.info("No batches available for debugging")
        else:
            # Process batches in parallel
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                # Submit all tasks
                future_to_batch = {
                    executor.submit(process_author_batch, batch_data): batch_data[0]
                    for batch_data in batch_data_list
                }
                
                # Process completed tasks
                with tqdm.tqdm(total=len(remaining_batches), desc="Processing batches") as pbar:
                    for future in as_completed(future_to_batch):
                        batch_idx = future_to_batch[future]
                        
                        try:
                            results = future.result()
                            
                            if results['status'] == 'success':
                                # Update KPM matrix
                                df_kpm = self.update_kpm_with_results(df_kpm, results)
                                
                                # Update progress
                                progress['completed_batches'].add(batch_idx)
                                
                                # Save progress every 10 batches
                                if len(progress['completed_batches']) % 100 == 0:
                                    self.save_progress(progress, df_kpm)
                                
                                pbar.set_postfix({
                                    'batch': batch_idx,
                                    'pairs': results['total_pairs'],
                                    'completed': len(progress['completed_batches'])
                                })
                            else:
                                self.logger.info(f"Batch {batch_idx} failed: {results['error']}")
                                
                        except Exception as e:
                            self.logger.info(f"Batch {batch_idx} encountered error: {str(e)}")
                        
                        pbar.update(1)
            
            # Final save
            self.save_progress(progress, df_kpm)
        
        self.logger.info(f"Parallel KPM generation completed!")
        self.logger.info(f"Matrix shape: {df_kpm.shape}")
        self.logger.info(f"Non-zero entries: {(df_kpm > 0).sum().sum()}")
        
        return df_kpm

    def save_final_kpm(self, df_kpm):
        """Save final KPM matrix"""
        final_kpm_path = op.join(self.path_manager.sub_kpm_dir, "df_kpm_citation_based_final.csv")
        
        self.path_manager.save_csv_file(
            variable=df_kpm, 
            abs_file_path=final_kpm_path, 
            index=True, 
            override=True
        )
        
        self.logger.info(f"Final KPM saved to: {final_kpm_path}")
        return final_kpm_path

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate KPM in parallel')
    parser.add_argument('--batch_size', type=int, default=2000, help='Author batch size')
    parser.add_argument('--n_workers', type=int, default=64, help='Number of parallel workers')
    args = parser.parse_args()
    
    # Generate KPM in parallel
    generator = ParallelKPMGenerator(debugging = False)
    df_kpm = generator.generate_kpm_parallel(
        batch_size=args.batch_size,
        n_workers=args.n_workers
    )
    
    # Save final KPM
    final_path = generator.save_final_kpm(df_kpm)
    
    logger.info(f"KPM generation completed. Final matrix saved to: {final_path}")


# nohup python ./network/citation_based_kpm_sub_paralell.py >> citation_based_kpm_sub_paralell.py.log 2>&1 &

# ps aux | grep citation_based_kpm_sub_paralell.py
# pkill -f citation_based_kpm_sub_paralell.py
