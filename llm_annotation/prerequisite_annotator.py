import os
import sys
import time
import logging
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from typing import List, Tuple, Optional, Dict
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor, as_completed

# Setup project path
script_dir = Path(__file__).resolve().parent
project_dir = script_dir.parent
sys.path.append(str(project_dir))

logger = logging.getLogger(__name__)

class PrerequisiteAnnotator:
    """
    A class to annotate prerequisite relationships between concept pairs
    using LLM API calls with parallel processing across disciplines.
    """
    
    # Valid relation codes
    VALID_RELATIONS = ["-2", "-1", "0", "1", "2"]
    
    # Relation descriptions
    RELATION_DESCRIPTIONS = {
        -2: "related but no prerequisite (A — B)",
        -1: "B is prerequisite for A (B <- A)",
        0: "independent (no relationship)",
        1: "A is prerequisite for B (A -> B)",
        2: "mutual prerequisites (A <-> B)"
    }
    
    # Discipline to key mapping (2 keys per discipline)
    DISCIPLINE_KEY_MAPPING = {
        0: [1, 2],  # First discipline uses keys 1 and 2
        1: [3, 4],  # Second discipline uses keys 3 and 4
        2: [5, 6],  # Third discipline uses keys 5 and 6
        3: [7, 8],  # Fourth discipline uses keys 7 and 8
    }
    
    def __init__(self, model_name: str, batch_size: int = 20, url_api_type = False):
        """
        Initialize the annotator with model and processing settings.
        
        Args:
            model_name: Name of the LLM model to use
            batch_size: Number of annotations to process before saving
        """
        # Format model name for file naming
        self.model_name = model_name
        self.model_file_name = model_name.split('/')[1] if '/' in model_name else model_name
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # API keys configuration
        if url_api_type == 0:
            self.url = 'rul' # fill the LLM API address
            self.api_keys = {} # fill you keys

        elif url_api_type == 1: 
            if 'deepseek' in self.model_name: # specifically for DeepSeek
                pass
            if 'grok' in self.model_name: 
                pass
            if 'gemini' in self.model_name: 
                pass
        elif url_api_type == 2: # GPT4.1 
            pass
        elif url_api_type == 3: # Claude 
            pass
        else: raise ValueError('Undefined url api type')
            
        self.batch_size = batch_size
        
        # Setup paths
        self.output_dir = Path(self.current_dir) / 'llm_results'
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Single output file for all results
        self.output_file = self.output_dir / f'{self.model_file_name}-prerequisite-annotation-all.csv'
        
        # Load concept data
        self._load_concept_data()
        
        # Initialize OpenAI clients for each API key
        self.clients = {}
        for key_id, api_key in self.api_keys.items():
            self.clients[key_id] = OpenAI(
                base_url=self.url, 
                api_key=api_key
            )
        
        # Initialize results storage
        self.results = []
        self.processed_pairs = {}  # Dictionary to track processed pairs by discipline
        
        # Load any existing results
        self.load_existing_results()
    
    def _load_concept_data(self):
        """Load and prepare concept data from the concepts table."""
        concepts_path = Path(self.current_dir) / 'df_selected_top_concepts.csv'
        logger.info(f"Loading concepts from {concepts_path}")
        
        self.concepts_table = pd.read_csv(concepts_path)
        
        # Extract disciplines (level 0 concepts)
        self.disciplines = (self.concepts_table
                           .loc[self.concepts_table.level == 0]
                           .display_name
                           .drop_duplicates()
                           .sort_values()
                           .tolist())
        
        logger.info(f"Loaded {len(self.disciplines)} disciplines")
        
        # Extract all concepts
        self.all_concepts = (self.concepts_table
                            .loc[self.concepts_table.level >= 0]
                            .display_name
                            .drop_duplicates()
                            .tolist())
        
        logger.info(f"Loaded {len(self.all_concepts)} total concepts")
        
        # Verify we have exactly 4 disciplines
        if len(self.disciplines) != 4:
            logger.warning(f"Expected 4 disciplines, but found {len(self.disciplines)}")
    
    def load_existing_results(self):
        """Load previously annotated results if they exist."""
        if self.output_file.exists():
            try:
                df = pd.read_csv(self.output_file)
                self.results = df.to_dict(orient="records")
                
                # Extract processed pair indices by discipline
                self.processed_pairs = {}
                for discipline in self.disciplines:
                    discipline_df = df[df['discipline'] == discipline]
                    if 'pair_index' in discipline_df.columns:
                        self.processed_pairs[discipline] = set(
                            discipline_df['pair_index'].dropna().astype(int).tolist()
                        )
                    else:
                        self.processed_pairs[discipline] = set()
                
                logger.info(f"Loaded {len(self.results)} existing annotations from {self.output_file}")
                for discipline, pairs in self.processed_pairs.items():
                    logger.info(f"Found {len(pairs)} processed pairs for {discipline}")
            except Exception as e:
                logger.error(f"Error loading existing results: {e}")
                self.results = []
                self.processed_pairs = {discipline: set() for discipline in self.disciplines}
        else:
            logger.info(f"No existing results found at {self.output_file}")
            self.results = []
            self.processed_pairs = {discipline: set() for discipline in self.disciplines}
    
    def save_results(self, max_retries=5, retry_delay=1):
        """
        Save all results to CSV file with retry mechanism.
        
        Args:
            max_retries: Maximum number of retry attempts
            retry_delay: Initial delay between retries (seconds), will increase exponentially
        """
        if not self.results:
            logger.info("No results to save")
            return
        
        # Try to save results with retry mechanism
        for attempt in range(max_retries):
            try:
                df = pd.DataFrame(self.results)
                df.to_csv(self.output_file, index=False)
                logger.info(f"Saved {len(self.results)} results to {self.output_file}")
                return
                
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Error saving results (attempt {attempt+1}/{max_retries}): {e}. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Failed to save results after {max_retries} attempts: {e}")
                    
                    # After all retry attempts fail, try to save to a backup file
                    try:
                        backup_file = self.output_file.parent / f"{self.output_file.stem}-backup-{int(time.time())}.csv"
                        df.to_csv(backup_file, index=False)
                        logger.info(f"Saved {len(self.results)} results to backup file {backup_file}")
                    except Exception as backup_error:
                        logger.error(f"Error saving to backup file: {backup_error}")

    def is_pair_processed(self, discipline: str, pair_index: int) -> bool:
        """
        Check if a concept pair has already been processed for a specific discipline.
        
        Args:
            discipline: The discipline name
            pair_index: Index of the concept pair
            
        Returns:
            True if the pair has been processed, False otherwise
        """
        return pair_index in self.processed_pairs.get(discipline, set())
    
    def generate_prompt(self, concept_a: str, concept_b: str) -> Tuple[str, str]:
        """
        Generate the system message and prompt for a concept pair.
        
        Args:
            concept_a: First concept
            concept_b: Second concept
            
        Returns:
            Tuple of (system_message, prompt)
        """
        system_message = (
            "You are a professional learning prerequisite analyst, specialized in "
            "analyzing educational dependencies between concepts. Please provide "
            "precise, objective judgments based on the given concept pairs."
        )
        
        prompt = f"""
Consider the following two concepts as defined in Wikipedia:

Concept A: '{concept_a}'  
Concept B: '{concept_b}'

When analyzing their learning prerequisite relationship, please consider the following aspects:

1. Definition  
   - Core definition and fundamental principles  
   - Key theoretical foundations and essential characteristics

2. Origin  
   - Historical development and evolution  
   - Primary academic disciplines and research communities  
   - Influential foundational literature

3. Methodology  
   - Common research methods and experimental approaches  
   - Standard tools, techniques, and terminology used

4. Usage and Applications  
   - Typical practical applications and use cases  
   - Impact on industry, technology, or society

5. Context and Relationships  
   - Related or derived concepts  
   - Interdisciplinary connections and overlaps  
   - How concepts interact or depend on each other in broader knowledge systems

Using your internal knowledge of Wikipedia and pedagogical principles, analyze the learning prerequisite relationship between these two concepts.

Please respond with EXACTLY ONE NUMBER from the options below:

0: A and B are independent (no relationship)  
1: A is a prerequisite for B (A -> B)  
-1: B is a prerequisite for A (B <- A)  
2: A and B are mutual prerequisites (A ↔ B)  
-2: A and B are related but have no prerequisite relationship (A — B)

Your response must be a single digit: -2, -1, 0, 1, or 2
"""
        return system_message, prompt
    
    def call_api(self, client_id: int, concept_a: str, concept_b: str) -> Tuple[Optional[int], str, str]:
        """
        Call the LLM API to get the prerequisite relationship using a specific client.
        
        Args:
            client_id: ID of the client to use
            concept_a: First concept
            concept_b: Second concept
            
        Returns:
            Tuple of (relation_code, raw_response)
        """
        system_message, prompt = self.generate_prompt(concept_a, concept_b)
        client = self.clients[client_id]
        
        # Add exponential backoff for API rate limits
        max_retries = 5
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                    max_tokens=10
                )
                
                result_text = response.choices[0].message.content.strip()
                
                # Parse and validate the result
                relation = None
                status = None
                if result_text in self.VALID_RELATIONS:
                    relation = int(result_text)
                    status = 'correct'
                else:
                    # Try to extract a valid number from the response
                    for val in self.VALID_RELATIONS:
                        if val in result_text:
                            relation = int(val)
                            status = 'extract'
                            break
                
                return relation, status, result_text
                
            except Exception as e:
                logger.warning(f"API call with client {client_id} failed (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    sleep_time = 2 * retry_delay
                    logger.info(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    logger.error(f"Max retries exceeded for {concept_a} and {concept_b} with client {client_id}")
                    return None, "error", f"ERROR: {str(e)}"
    
    def get_discipline_concepts(self, discipline: str) -> List[str]:
        """
        Get all concepts for a specific discipline.
        
        Args:
            discipline: The discipline name
            
        Returns:
            List of concept names
        """
        concepts = (self.concepts_table.loc[self.concepts_table.llm_annotation == discipline]
                   .sort_values('works_count', ascending=False)['display_name']
                   .drop_duplicates()
                   .tolist())
        
        return concepts
    
    def generate_concept_pairs(self, discipline: str) -> List[Tuple[int, str, str]]:
        """
        Generate all unique concept pairs for a discipline with pair indices.
        
        Args:
            discipline: The discipline name
            
        Returns:
            List of tuples (pair_index, concept_a, concept_b)
        """
        discipline_concepts = self.get_discipline_concepts(discipline)
        
        if not discipline_concepts:
            logger.warning(f"No concepts found for discipline: {discipline}")
            return []
        
        # Generate all unique pairs with indices using combinations
        pairs = []
        for pair_index, (concept_a, concept_b) in enumerate(combinations(discipline_concepts, 2)):
            pairs.append((pair_index, concept_a, concept_b))
        
        return pairs
    
    def process_pair(self, discipline: str, client_id: int, pair_data: Tuple[int, str, str]) -> Dict:
        """
        Process a single concept pair using the specified client.
        
        Args:
            discipline: The discipline name
            client_id: ID of the client to use
            pair_data: Tuple of (pair_index, concept_a, concept_b)
            
        Returns:
            Result dictionary
        """
        pair_index, concept_a, concept_b = pair_data
        
        # Call API and get result
        relation, status, response_text = self.call_api(client_id, concept_a, concept_b)
        response_text_clean = response_text.replace('\n', ' ').replace('\r', ' ')
        # Create result entry
        result_entry = {
            "discipline": discipline,
            "pair_index": pair_index,
            "concept_a": concept_a,
            "concept_b": concept_b,
            "relation": relation,
            "client_id": client_id,
            "status": status,
            "response_text": response_text_clean,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        return result_entry
    
    def process_discipline_batch(self, discipline: str, batch_pairs: List[Tuple[int, str, str]]):
        """
        Process a batch of pairs for a discipline using both assigned clients.
        
        Args:
            discipline: The discipline name
            batch_pairs: List of (pair_index, concept_a, concept_b) tuples
        """
        # Get discipline index
        discipline_index = self.disciplines.index(discipline)
        
        # Get assigned client IDs for this discipline
        client_ids = self.DISCIPLINE_KEY_MAPPING[discipline_index]
        # Simply split the batch into two parts
        split_point = len(batch_pairs) // len(client_ids)
        client_batches = {}
        
        for i, client_id in enumerate(client_ids):
            start_idx = i * split_point
            end_idx = (i + 1) * split_point if i < len(client_ids) - 1 else len(batch_pairs)
            client_batches[client_id] = batch_pairs[start_idx:end_idx]

        # test results
        # results = self.process_pair(discipline, client_ids[0], batch_pairs[0])

        # Process pairs in parallel using both clients
        batch_results = []
        with ThreadPoolExecutor(max_workers=len(client_ids)) as executor:
            # Distribute pairs among clients
            futures = []
            for client_id, pairs in client_batches.items():
                for pair_data in pairs:
                    futures.append(executor.submit(self.process_pair, discipline, client_id, pair_data))
            
            # Collect results
            for future in as_completed(futures):
                try:
                    result = future.result()
                    batch_results.append(result)
                    # Mark as processed
                    self.processed_pairs[discipline].add(result["pair_index"])
                except Exception as e:
                    logger.error(f"Error processing pair: {e}")
        
        # Add batch results to overall results
        self.results.extend(batch_results)
    
    def process_discipline(self, discipline: str):
        """
        Process all concept pairs for a specific discipline.
        
        Args:
            discipline: The discipline to process
        """
        # Generate all concept pairs with indices
        all_pairs = self.generate_concept_pairs(discipline)
        
        if not all_pairs:
            logger.warning(f"No concept pairs to process for discipline: {discipline}")
            return
        
        total_pairs = len(all_pairs)
        logger.info(f"Generated {total_pairs} concept pairs for discipline: {discipline}")
        
        # Filter out already processed pairs
        unprocessed_pairs = [
            (idx, a, b) for idx, a, b in all_pairs 
            if not self.is_pair_processed(discipline, idx)
        ]
        
        logger.info(f"Found {len(unprocessed_pairs)} unprocessed pairs for {discipline} out of {total_pairs} total pairs")
        
        # self.process_discipline_batch(discipline, unprocessed_pairs[:100])

        # Process in batches
        for i in tqdm(range(0, len(unprocessed_pairs), self.batch_size), desc=f"Processing {discipline}"):
            batch_pairs = unprocessed_pairs[i:i+self.batch_size]
            
            # Process batch using both clients assigned to this discipline
            self.process_discipline_batch(discipline, batch_pairs)
            
            # Save after each batch
            self.save_results()
            logger.info(f"Processed {min(i+self.batch_size, len(unprocessed_pairs))}/{len(unprocessed_pairs)} pairs for {discipline}")
    
    def run_annotation(self):
        """
        Run the annotation process for all disciplines in parallel.
        """
        logger.info(f"Starting annotation process for all disciplines")
        
        # test discipline
        # self.process_discipline(discipline='Mathematics')

        # Process each discipline in parallel
        with ThreadPoolExecutor(max_workers=len(self.disciplines)) as executor:
            # Submit each discipline for processing
            futures = {
                executor.submit(self.process_discipline, discipline): discipline
                for discipline in self.disciplines
            }
            
            # Wait for all to complete
            for future in as_completed(futures):
                discipline = futures[future]
                try:
                    future.result()
                    logger.info(f"Completed annotation for discipline: {discipline}")
                except Exception as e:
                    logger.error(f"Error processing discipline {discipline}: {e}")
        
        # Final save
        self.save_results()
        logger.info(f"Annotation complete for all disciplines")

