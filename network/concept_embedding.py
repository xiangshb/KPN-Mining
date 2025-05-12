import os, sys
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.append(project_dir)

import os
import time
import pickle
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from gensim.models import Word2Vec
from multiprocessing import Pool, cpu_count
from utils.database import DatabaseManager
from utils.config import PathManager
import os.path as op
from gensim.models.callbacks import CallbackAny2Vec

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("concept_embedding.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ConceptSequenceCreator:
    """Creates concept sequences for authors"""
    
    def __init__(self):
        """
        Initialize the creator
        
        Args:
            db_manager: Database manager instance
            cache_dir: Directory for caching results
        """
        self.db_manager = DatabaseManager()
        self.path_manager = PathManager()
        self.cache_dir = self.path_manager.concepts_sequences_dir
    
    def get_author_concept_sequence(self, author_ids, min_seq_len = 5):
        """
        Get a single author's concept sequence ordered by publication time
        
        Args:
            author_id: Author ID
            
        Returns:
            list: List of concept IDs ordered by publication time
        """
        try:
            # Query the author's works, publication years, and concepts
            df_query_result = self.db_manager.query_table(
                table_name='works_authorships',
                columns=[
                    'works_authorships.author_id',
                    'works.publication_year',
                    'works_authorships.work_id',
                    'concepts.display_name'
                ],
                join_tables=['works', 'works_concepts', 'concepts'],
                join_conditions=[
                    'works_authorships.work_id = works.id',
                    'works_authorships.work_id = works_concepts.work_id',
                    'works_concepts.concept_id = concepts.id'
                ],
                where_conditions=[
                    f'''works_authorships.author_id IN ('{"','".join(author_ids)}')'''
                ],
                batch_read=False,
                show_runtime=False
            )
            # Ensure publication_year is numeric
            df_query_result['publication_year'] = pd.to_numeric(df_query_result['publication_year'], errors='coerce')
            
            # Sort by year and work_id
            df_query_result = df_query_result.sort_values(by=['author_id', 'publication_year', 'work_id']).dropna(subset=['display_name'])
            
            # Extract the ordered concept list
            concept_sequence = df_query_result.groupby('author_id')['display_name'].agg(list)
            concept_sequence = concept_sequence[concept_sequence.apply(len) >= min_seq_len]

            return concept_sequence
            
        except Exception as e:
            logger.error(f"Error getting concept sequence for authors {len(author_ids)}: {str(e)}")
            return []
    
    def process_author_batch(self, author_ids, batch_idx, total_batches):
        """
        Process a batch of authors and save their concept sequences
        
        Args:
            author_ids: List of author IDs to process
            batch_idx: Current batch index
            total_batches: Total number of batches
            
        Returns:
            str: Path to the saved batch file
        """
        start_time = time.time()
        logger.info(f"Processing batch {batch_idx}/{total_batches} with {len(author_ids)} sub batchs")

        parquet_path = os.path.join(self.cache_dir, f"all_author_concepts_{batch_idx}.parquet")

        author_sequences = []
        for batch_author_ids in tqdm(author_ids, desc=f"Batch {batch_idx}/{total_batches}"):
            concept_sequence = self.get_author_concept_sequence(batch_author_ids)
            if len(concept_sequence) > 0:  # Only add non-empty sequences
                author_sequences.append(concept_sequence)
            
        df_all_sequences = pd.concat(author_sequences).reset_index()
        df_all_sequences.columns = ['author_id', 'concepts']

        # Save the batch
        df_all_sequences.to_parquet(parquet_path)

        elapsed = time.time() - start_time
        logger.info(f"Batch {batch_idx}/{total_batches} completed in {elapsed:.2f} seconds\n file saved at {parquet_path}")
        
        return parquet_path
    
    def create_all_sequences(self, author_id_batchs, batch_size=200, num_workers=8):
        """
        Create concept sequences for all given authors
        
        Args:
            author_ids: List of all author IDs to process
            batch_size: Number of authors to process in each batch
            num_workers: Number of parallel workers (default: CPU count - 1)
            
        Returns:
            list: Paths to all batch files
        """
        if num_workers is None:
            num_workers = max(1, cpu_count() - 1)
        
        # Split each batch of 2000 authors into batches
        batches = [author_id_batchs[i:i+batch_size] for i in range(0, len(author_id_batchs), batch_size)]

        total_batches = len(batches)
        logger.info(f"Processing {total_batches} batches with {num_workers} workers")

        # Process batches in parallel
        batch_files = []
        with Pool(processes=num_workers) as pool:
            tasks = [(batches[i], i+1, total_batches) for i in range(total_batches)]
            results = [pool.apply_async(self.process_author_batch, args=task) for task in tasks]
            
            for result in results:
                batch_file = result.get()
                batch_files.append(batch_file)
        
        return batch_files


class EpochLogger(CallbackAny2Vec):
    """Callback to log information about training."""

    def __init__(self):
        self.epoch = 0

    def on_epoch_begin(self, model):
        print(f"Epoch #{self.epoch} start")

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        print(f"Epoch #{self.epoch} end. Loss: {loss}")
        self.epoch += 1

class ConceptEmbeddingTrainer:
    """Trains Word2Vec model on concept sequences"""
    
    def __init__(self, model_dir='./models'):
        """
        Initialize the trainer
        
        Args:
            model_dir: Directory to save models
        """
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
    
    def load_sequences_from_batches(self, batch_files):
        """
        Load all concept sequences from batch files
        
        Args:
            batch_files: List of paths to batch files
            
        Returns:
            list: Combined list of all concept sequences
        """
        all_sequences = []
        for batch_file in tqdm(batch_files, desc="Loading batches"):
            df_concepts = pd.read_parquet(batch_file, engine='pyarrow')
            all_sequences.append(df_concepts['concepts'].apply(list))
            
        all_sequences = pd.concat(all_sequences, ignore_index=True)
        
        logger.info(f"Loaded {len(all_sequences)} concept sequences")

        return all_sequences.tolist()
    
    def train_word2vec(self, sequences, vector_size=24, window=10, min_count=1, negative = 15, workers=None, epochs=100):
        """
        Train Word2Vec model on concept sequences
        
        Args:
            sequences: List of concept sequences
            vector_size: Dimensionality of vectors
            window: Maximum distance between current and predicted word
            min_count: Minimum frequency of concepts
            workers: Number of parallel workers
            epochs: Number of training epochs
            
        Returns:
            Word2Vec: Trained model
        """
        if workers is None:
            workers = max(1, cpu_count() - 1)
            
        logger.info(f"Training Word2Vec model with {len(sequences)} sequences...")
        logger.info(f"Parameters: vector_size={vector_size}, window={window}, min_count={min_count}, workers={workers}, epochs={epochs}")
        
        start_time = time.time()

        model = Word2Vec(
            vector_size=vector_size,            # Dimensionality of the word vectors (embedding size)
            window=window,                      # Maximum distance between the current and predicted word within a sentence (context window size)
            min_count=min_count,                # Ignores all words with total frequency lower than this (includes all words if set to 1)
            sg=1,                               # Training algorithm: 1 = skip-gram; 0 = CBOW
            hs=0,                               # If 1, hierarchical softmax will be used for model training. If set to 0 (default), and negative is non-zero, negative sampling will be used.
            negative=negative,                  # Number of negative samples to use (if negative sampling is used)
            workers=workers,                    # Number of worker threads to train the model (faster training with multicore machines)
            min_alpha=0.0001,                   # Minimum learning rate (learning rate will linearly drop to min_alpha as training progresses)
            compute_loss=True,                  # If True, stores loss value which can be retrieved via model.get_latest_training_loss()
            callbacks=[EpochLogger()]           # Optional: List of callbacks to run at specific training events (e.g., logging after each epoch)
        )

        model.build_vocab(sequences, progress_per=100000)   # Build vocabulary from your data
        model.train(
            sequences,
            total_examples=model.corpus_count,
            epochs=epochs
        )

        elapsed = time.time() - start_time
        logger.info(f"Model training completed in {elapsed:.2f} seconds")
        
        # Save the model
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(self.model_dir, f"concept_w2v_{timestamp}.model")
        model.save(model_path)
        logger.info(f"Model saved to {model_path}")
        
        stats_df = self.get_concept_stats(sequences, model)

        return model, model_path
    
    def get_concept_stats(self, sequences, model):
        # 1. Get total frequency for each concept in the vocabulary
        
        concept_freq_path = os.path.join(self.model_dir, "concept_frequency_stats.csv")
        if not op.exists(concept_freq_path):
            concept_counts = {concept: model.wv.get_vecattr(concept, "count") for concept in model.wv.index_to_key}

            # 2. Count in how many sentences each concept appears
            sentence_counter = self.count_sentence_occurrences(sequences)

            # 3. Assemble the statistics into a DataFrame
            stats_df = pd.DataFrame({
                "concept": list(concept_counts.keys()),
                "total_count": [concept_counts[w] for w in concept_counts],
                "author_count": [sentence_counter[w] for w in concept_counts],
            })
            stats_df.to_csv(concept_freq_path, index=False)

            logger.info(f"Concept stats saved to {concept_freq_path}")

        else: stats_df = pd.read_csv(concept_freq_path)
        
        return stats_df
    
    def count_sentence_occurrences(self, sequences, n_threads = 8):
        from collections import Counter
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import numpy as np

        def worker(seqs):
            local_counter = Counter()
            for seq in seqs:
                local_counter.update(set(seq))
            return local_counter

        chunks = np.array_split(sequences, n_threads)

        counters = []
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = [executor.submit(worker, chunk) for chunk in chunks]
            for future in as_completed(futures):
                counters.append(future.result())

        # Merge all Counters
        total_counter = Counter()
        for c in counters:
            total_counter.update(c)

        return total_counter

class ConceptEmbeddingManager:
    """Main class to manage the entire process"""
    
    def __init__(self, db_config=None):
        """
        Initialize the manager
        
        Args:
            db_config: Database configuration
        """
        self.db_manager = DatabaseManager() if db_config is None else DatabaseManager(**db_config)
        self.sequence_creator = ConceptSequenceCreator()
        self.embedding_trainer = ConceptEmbeddingTrainer(model_dir = op.join(self.sequence_creator.path_manager.navigation_embedding_dir, 'model'))
        self.path_manager = PathManager()
    
    def create_concept_sequences(self, author_ids=None, batch_size=200, num_workers=8, chunk_size=2000):
        """
        Create concept sequences and return the list of batch file paths.
        """
        # Get author_ids if not provided
        if author_ids is None:
            file_path_author_ids = os.path.join(self.path_manager.external_file_dir, 'CCNs', "Total_author_ids_min_pubs_10_non_split.npy")
            if not os.path.exists(file_path_author_ids):
                logger.info("Querying author IDs from database...")
                author_ids = self.db_manager.query_table(
                    table_name='author_yearlyfeature_field_geq10pubs',
                    columns=['author_id', 'total_pubs'],
                    where_conditions=['total_pubs >= 10']
                )
                author_ids = author_ids.author_id.values
                np.save(file_path_author_ids, author_ids, allow_pickle=True)
            else:
                author_ids = np.load(file_path_author_ids, allow_pickle=True)
            logger.info(f"Found {len(author_ids)} authors")

        # Split into batches
        author_id_batches = [author_ids[i:i+chunk_size] for i in range(0, len(author_ids), chunk_size)]

        # Create sequences
        logger.info("Creating concept sequences for authors...")
        batch_files = self.sequence_creator.create_all_sequences(
            author_id_batchs=author_id_batches,
            batch_size=batch_size,
            num_workers=num_workers
        )
        logger.info("All concept sequences for authors finished")

        # Return the list of batch file paths
        return batch_files
    
    def train_from_concept_sequences(self, vector_size=24, window=10, min_count=1, epochs=100, num_workers=8):
        """
        Read batch files from concepts_sequences_dir and train the model.
        """
        dir_path = self.path_manager.concepts_sequences_dir
        batch_files = []
        for i in range(1, 18):
            filename = f"all_author_concepts_{i}.parquet"
            abs_path = os.path.abspath(os.path.join(dir_path, filename))
            if os.path.exists(abs_path):
                batch_files.append(abs_path)
        logger.info(f"Found {len(batch_files)} batch files for training.")

        # Load sequences and train the model
        logger.info("Loading sequences and training Word2Vec model...")
        sequences = self.embedding_trainer.load_sequences_from_batches(batch_files)
        model, model_path = self.embedding_trainer.train_word2vec(
            sequences=sequences,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=num_workers,
            epochs=epochs
        )
        return model, model_path

    def load_model(self, model_path):
        """
        Load a saved Word2Vec model
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            Word2Vec: Loaded model
        """
        logger.info(f"Loading model from {model_path}")
        model = Word2Vec.load(model_path)
        logger.info(f"Model loaded with {len(model.wv)} concepts")
        return model

if __name__ == "__main__":
    # Example usage
    manager = ConceptEmbeddingManager()
    
    # Option 1: Create concept sequences
    # batch_files = manager.create_concept_sequences()

    # Option 2: Train Word2Vec model
    model, path = manager.train_from_concept_sequences()
    
    # Option 3: Load existing model
    # model = manager.load_model("./models/concept_w2v_20250418_120000.model")
    
    # Example: Get vector for a concept
    # concept_id = "C123456"  # Replace with actual concept ID
    # if concept_id in model.wv:
    #     vector = model.wv[concept_id]
    #     print(f"Vector for {concept_id}: {vector[:5]}...")  # Print first 5 elements
    # else:
    #     print(f"Concept {concept_id} not in vocabulary")

# nohup python ./network/concept_embedding.py >> embedding_progress.log 2>&1 &
# ps aux | grep concept_embedding.py
# pkill -f concept_embedding.py
