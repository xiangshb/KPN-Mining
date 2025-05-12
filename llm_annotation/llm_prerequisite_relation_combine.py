import pandas as pd
import numpy as np
import os, sys
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.append(project_dir)

class PrerequisiteAnnotationProcessor:
    def __init__(self, script_dir):
        self.script_dir = script_dir
        self.project_dir = os.path.dirname(script_dir)
        sys.path.append(self.project_dir)
        
        # Define model name mapping
        self.model_names = {
            'claude-3-7-sonnet-20250219': 'claude-3-7-sonnet-20250219',
            'deepseek-chat': 'deepseek-chat',
            'gemini-2.5-pro-exp-03-25': 'gemini-2.5-pro-exp-03-25',
            'gpt-4.1': 'gpt-4.1',
            'grok-3-beta': 'grok-3-beta'
        }
        self.selected_concept_table = pd.read_csv(os.path.join(script_dir, 'df_selected_cross_concepts.csv'))
        # Define file paths
        self.subfolder = "llm_results"
        self.annotation_files = [
            os.path.join(self.script_dir, self.subfolder, 'claude-3-7-sonnet-20250219-prerequisite-annotation-cross.csv'),
            os.path.join(self.script_dir, self.subfolder, 'deepseek-chat-prerequisite-annotation-cross.csv'),
            os.path.join(self.script_dir, self.subfolder, 'gemini-2.5-pro-exp-03-25-prerequisite-annotation-cross.csv'),
            os.path.join(self.script_dir, self.subfolder, 'gpt-4.1-prerequisite-annotation-cross.csv'),
            os.path.join(self.script_dir, self.subfolder, 'grok-3-beta-prerequisite-annotation-cross.csv')
        ]
        self.output_file_path = os.path.join(self.script_dir, self.subfolder, 'llm_prerequisite_relations_comparison_cross.csv')
        # Initialize dataframes
        self.all_dfs = []
        self.combined_df = None
        self.pivot_df = None
        self.available_columns = None
    
    def read_files_and_combine(self):
        """Read all annotation files and store in all_dfs list"""
        for file in self.annotation_files:
            # Extract model name from filename
            base_filename = os.path.basename(file)
            model_key = next((key for key in self.model_names.keys() if key in base_filename), 'unknown')
            model_name = self.model_names.get(model_key, 'unknown')
            
            try:
                df = pd.read_csv(file)
                # Add model name column and keep only needed columns
                df['model'] = model_name
                df = df[['pair_index', 'concept_a', 'concept_b', 'relation', 'model']]
                self.all_dfs.append(df)
                print(f"Successfully read {file}, containing {len(df)} records")
            except Exception as e:
                print(f"Error reading {file}: {e}")
        
        if not self.all_dfs:
            print("No data to combine")
            return False
            
        self.combined_df = pd.concat(self.all_dfs, ignore_index=True)

        id_dict = self.selected_concept_table.set_index('display_name')['id'].to_dict()
        self.combined_df['concept_a_id'] = self.combined_df['concept_a'].map(id_dict)
        self.combined_df['concept_b_id'] = self.combined_df['concept_b'].map(id_dict)

        print(f"Combined dataset has {len(self.combined_df)} records")
    
    def get_majority_vote(self, row):
        """Calculate majority vote for a single row"""
        model_columns = list(self.model_names.values())
        self.available_columns = [col for col in model_columns if col in self.pivot_df.columns]
        
        values, counts = np.unique(row[self.available_columns], return_counts=True)
        max_count = np.max(counts)
        max_count_values = sorted(values[counts == max_count])
        n_max_count = len(max_count_values) # 1/2/5

        if n_max_count == 1: 
            vote_value = max_count_values[0]
            confidence = max_count / len(self.available_columns)
        
        elif n_max_count == 2: 
            if tuple(max_count_values) in [(-2, -1), (-2, 1)]:
                # two related contributes 1
                vote_value = max_count_values[1]  # Adjust to positive/negative prerequisite
                confidence = (2 + 1) / 5
                
            elif tuple(max_count_values) in [(-1, 2), (1, 2)]:
                # two mutual contributes 2
                vote_value = max_count_values[0]
                confidence = (2 + 2) / 5
                
            elif tuple(max_count_values)  in [(-1, 1), (-2, 2)]:
                # such cases contributes 1
                vote_value = 2  # Adjust to mutual prerequisite
                confidence = (2 + 1) / 5
                
            else: # Others with 0 like [(-2, 0), (-1, 0), (0, 1), (0, 2)] are not prerequisites
                vote_value = 0
                confidence = 2 / 5
        else: # case of [-2, -1, 0, 1, 2]
            vote_value = 2
            confidence = 2 / 5
        
        return n_max_count, vote_value, confidence
    
    def create_pivot_table(self):
        """Create pivot table with each model's annotation as separate columns"""
        if not os.path.exists(self.output_file_path):
            if self.combined_df is None:
                print("No combined data available")
                return False
            self.combined_df['relation'] = self.combined_df['relation'].astype(int)
            self.pivot_df = self.combined_df.pivot_table(
                index=['pair_index', 'concept_a_id', 'concept_b_id', 'concept_a', 'concept_b'],
                columns='model',
                values='relation',
                aggfunc='first'
            ).reset_index()
            
            # Apply majority vote function
            self.pivot_df[['n_max_count', 'majority_vote', 'confidence']] = self.pivot_df.apply(
                lambda row: pd.Series(self.get_majority_vote(row)), axis=1
            )
            self.pivot_df[['n_max_count', 'majority_vote']] = self.pivot_df[['n_max_count', 'majority_vote']].astype(int)

            self.pivot_df.to_csv(self.output_file_path, index=False)
        else: self.pivot_df = pd.read_csv(self.output_file_path)
        original_file = pd.read_csv('/home/OpenAlex-Analysis/KPN-Mining/llm_cross_annotation/llm_results/llm_prerequisite_relations_comparison.csv')
        return self.pivot_df

# Usage example
if __name__ == "__main__":
    processor = PrerequisiteAnnotationProcessor(script_dir)
    # processor.read_files_and_combine()
    result_df = processor.create_pivot_table()
    
    if result_df is not None:
        print("Processing completed successfully")
        print(result_df.head())
