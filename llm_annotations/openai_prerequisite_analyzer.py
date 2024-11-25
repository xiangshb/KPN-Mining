import os
import pandas as pd
import numpy as np
from openai import OpenAI
from datetime import datetime
import os.path as op
from pathlib import Path
import time
import csv, httpx
from collections import deque
from itertools import combinations
from tqdm import tqdm
import argparse
import qianfan

class ConceptRelationAnalyzer:
    def __init__(self, part = -1, model_name = 'gpt-4o', folder_path='summary'):
        """
        Initialize the analyzer
        
        Parameters:
        api_key (str): OpenAI API key
        folder_path (str): Folder path containing CSV files
        """
        self.part = part
        self.model_name = model_name
        self.model_abbreviations ={
            'claude-3-5-sonnet-20240620': 'claude-3-5',
            'gpt-4o':'gpt-4o'
        }
        if self.model_name in ['claude-3-5-sonnet-20240620', 'gpt-4o']:
            self.api_keys = {
                0: 'your_api_key'}
            self.client = OpenAI(base_url='', api_key=self.api_keys[part],
                                 http_client=httpx.Client(base_url='', follow_redirects=True))
        elif self.model_name in ['qwen-plus', 'qwen-max']:
            self.api_keys = {
                0: 'your_api_key'}
            self.client = OpenAI(base_url='https://dashscope.aliyuncs.com/compatible-mode/v1', api_key=self.api_keys[part])
        elif self.model_name in ['ERNIE-3.5-8K-0701', 'ERNIE-3.5-128K']:
            # self.api_keys = {
            #     0: 'your_api_key'}
            # self.secret_keys = {
            #     0: 'your_api_key'}
            # os.environ["QIANFAN_ACCESS_KEY"] = self.api_keys[self.part]
            # os.environ["QIANFAN_SECRET_KEY"] = self.secret_keys[self.part]
            
            self.api_keys = {
                0: 'your_api_key'}
            self.secret_keys = {
                0: 'your_api_key'}
            
            os.environ["QIANFAN_AK"] = self.api_keys[self.part]
            os.environ["QIANFAN_SK"] = self.secret_keys[self.part]
            self.client = qianfan.ChatCompletion()

        self.n_parts = len(self.api_keys)
        self.folder_path = folder_path
        self.summaries = {}
        self.concepts = []
        self.load_short_summaries()
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.results_file = Path(f'''./results_v2/{self.model_name}-concept_prerequisite_results{'' if self.part < 0 else f'_part_{self.part}'}.csv''')
        self.relationship_matrix = Path('./results/relationship_matrix.csv')
        
        if not self.results_file.exists():
            with open(self.results_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['index_i', 'index_j', 'prerequisite', 'timestamp'])
        
    def load_summaries(self):
        """Load concept summaries from all CSV files"""
        files = [f for f in os.listdir(self.folder_path) if f.endswith('.csv')]
        for file in files:
            try:
                file_path = op.join(self.folder_path, file)
                df = pd.read_csv(file_path)
                concept_name = file[:-4]  # Remove .csv extension
                if 'summary' in df.columns and not df['summary'].empty:
                    self.summaries[concept_name] = df['summary'].values[0]
                    self.concepts.append(concept_name)
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")
        
        print(f"Loaded {len(self.concepts)} concepts")
    
    def load_short_summaries(self):
        """Load concept summaries from all CSV files"""
        df_short_summaries = pd.read_csv('./data/final_GPT4_short_within_35_summaries.csv')
        
        self.concepts = df_short_summaries['display_name'].tolist()
        self.summaries = dict(df_short_summaries.values)
        
        print(f"Loaded {len(self.concepts)} concepts")
    
    def create_optimized_prompt(self, concept_a, concept_b):
        prompt = f"""Evaluate the prerequisite relationship between concept '{concept_a}' and concept '{concept_b}'.

Prerequisite Scale:
-2: '{concept_b}' and '{concept_a}' are related but have no prerequisite
-1: '{concept_b}' is a prerequisite for '{concept_a}'
0: '{concept_b}' and '{concept_a}' are not related
1: '{concept_a}' is a prerequisite for '{concept_b}'
2: '{concept_b}' and '{concept_a}' are bidirectional prerequisites

Analysis Dimensions:
1. Scientific Foundation
- Essential characteristics and core definitions
- Applicable instances and examples
- Historical development context
- Evolution and advancement

2. Cross-Disciplinary Analysis
- Relationships across academic disciplines
- Integration with established theories
- Consistency across scientific fields

3. Logical Structure
- Theoretical dependencies
- Hierarchical relationships
- Complexity levels
- Essential prerequisites

4. Learning Perspective
- Typical learning sequences
- Cognitive acquisition patterns
- Pedagogical approaches

5. Empirical Support
- Learning progressions
- Educational research findings
- Real-world case studies

Output Instruction:
Based on the analysis framework above, consider all outlined aspects and any other relevant factors 
that may influence the prerequisite relationship between the concepts.
Evaluate based on the inherent relationships between concepts, not mere on concept co-occurrence or association.
Provide a single integer score (-2 to 2) as defined in the prerequisite scale.
"""
        return prompt

    def create_prompt(self,concept_a, concept_b):
        prompt = (
            f"Based on the Wikipedia definitions:\n"
            f"Concept A: {concept_a}\n"
            f"Definition: {self.summaries[concept_a]}\n\n"
            f"Concept B: {concept_b}\n"
            f"Definition: {self.summaries[concept_b]}\n\n"
            f"Based on pedagogical principles, analyze their learning prerequisite relationship.\n"
            f"Reply with EXACTLY ONE NUMBER from these options:\n\n"
            f"0: A and B are independent (no relationship)\n"
            f"1: A is a prerequisite for B (A->B)\n"
            f"-1: B is a prerequisite for A (B->A)\n"
            f"2: A and B are mutual prerequisites (A<->B)\n"
            f"-2: A and B are related but have no prerequisite relationship (A-B)"
            f"Your response should be only a single digit: -2, -1, 0, 1, or 2"
        )
        return prompt

    def get_relationship(self, concept_a, concept_b, max_retries=3):
        """
        Use GPT-4 to determine the relationship between two concepts
        
        Parameters:
        concept_a (str): First concept
        concept_b (str): Second concept
        max_retries (int): Maximum number of retry attempts
        
        Returns:
        int: Relationship type (1, -1, 2, 0)
        """
        system_message = (
        "You are a professional learning prerequisite analyst, specialized in analyzing educational dependencies between concepts."
        "Please provide precise, objective judgments based on the given concept definitions."
        )
        
        # prompt = self.create_prompt(concept_a, concept_b)
        prompt = self.create_optimized_prompt(concept_a, concept_b)
        
        """
        0: Independent (no relationship)
        Clear and correct
        Concepts have no connection at all
        Example: "Photosynthesis" and "Ancient Roman History"

        1: One-way prerequisite (A->B)
        Clear and correct
        A must be learned before B
        Example: "Basic Arithmetic" -> "Algebra"

        -1: One-way prerequisite (B->A)
        Clear and correct
        B must be learned before A
        Example: Mirror of category 1

        2: Mutual prerequisites (A<->B)
        Clear and correct
        Each concept requires basic understanding of the other
        Example: (Reading, Writing) in early language learning

        -2: Related without prerequisites (A-B)
        Clear and correct
        Concepts are connected but can be learned independently
        Example: (Piano Playing, Music Theory), (Painting, Sculpture)
        """

        for attempt in range(max_retries):
            try:
                if self.model_name in ['ERNIE-3.5-8K-0701', 'ERNIE-3.5-128K']:
                    response = self.client.do(
                        model=self.model_name,
                        messages=[
                            {"role": "user", "content": f'{system_message}\n\n{prompt}'},
                        ],
                        temperature=0.3,
                        stream=False,
                        top_p = 0.1,
                        max_output_tokens = 5,
                        penalty_score=1.0
                    )
                    answer = response["body"]["result"]
                    input_tokens = response["body"]["usage"]["prompt_tokens"]      # 输入tokens数量
                    output_tokens = response["body"]["usage"]["completion_tokens"] # 输出tokens数量
                else:
                    response = self.client.chat.completions.create(
                        model=  self.model_name,
                        messages=[
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.3,
                        max_tokens=5
                    )
                    answer = response.choices[0].message.content.strip()
                    input_tokens = response.usage.prompt_tokens # 获取输入tokens数量
                    output_tokens = response.usage.completion_tokens # 获取输出tokens数量
                self.total_input_tokens += input_tokens # 累计总输入tokens数量
                self.total_output_tokens += output_tokens # 累计总输出tokens数量
                # Print token usage
                # print(f"Current tokens: {input_tokens}  All tokens so far: {self.total_input_tokens}")
                
                
                if answer in ['-2', '-1', '0', '1', '2']:
                    return int(answer), input_tokens, output_tokens
                else:
                    print(f"Error answer type: {answer}")

            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2)  # Wait 5 seconds before retrying
                    
        return 3, 0, 0  # Return 3 if all attempts fail
    
    def analyze_all_relationships(self):
        """Analyze relationships between all concept pairs and return relationship matrix"""
        print(f'Analyzing all relationships with model: {self.model_name}')
        n = len(self.concepts)

        if op.exists(self.relationship_matrix): 
            M = pd.read_csv(self.relationship_matrix, index_col=0).values # 加载已有的矩阵结果
        else: M = np.zeros((n, n), dtype=int)

        with open(self.results_file, 'r', newline='', encoding='utf-8') as f:
            last_line = deque(csv.DictReader(f), maxlen=1)
            if not last_line:  # 如果文件为空
                start_i, start_j = 0, 0
            else:
                row = last_line[0]
                start_i, start_j = int(row['index_i']), int(row['index_j'])
                
        if self.part < 0:
            total_pairs = n * (n - 1) // 2
            current_pair = 0

            if start_j < n - 1:
                start_j += 1
            else: 
                start_i += 1
            
            # chech unrecorded relations
            invalid_pairs = np.argwhere(np.triu(M)==3)
            for invalid_index, (i, j) in enumerate(invalid_pairs):
                print(f"Processing pair ({i} {j}) {invalid_index+1}/{invalid_pairs.shape[0]}: {self.concepts[i]} - {self.concepts[j]}")
                M = self.process_relationship(i, j, M)
                self.save_relation_matrix(M)

            for i in range(start_i, n):
                j_start = start_j if i == start_i else i + 1 # j 只有在第一轮的时候是start_j
                for j in range(j_start, n):
                    current_pair += 1
                    print(f"Processing pair ({i} {j}) {current_pair}/{total_pairs}: {self.concepts[i]} - {self.concepts[j]}")
                    
                    M = self.process_relationship(i, j, M)
                    self.save_relation_matrix(M)
        else:
            all_pairs = list(combinations(range(n), 2))
            all_pairs = [(int(pair[0]), int(pair[1])) for pair in np.array_split(all_pairs, self.n_parts)[self.part]]  # self.part range [0,7]
            
            if (start_i, start_j) in all_pairs:
                pair_strat_indes = all_pairs.index((start_i, start_j))
                all_pairs = all_pairs[pair_strat_indes+1:]
            
            total_pairs = len(all_pairs)

            with tqdm(enumerate(all_pairs), total=total_pairs) as pbar:
                for index, (i, j) in pbar:
                    # 使用更简洁的描述
                    pbar.set_description(f"({i},{j}) {self.concepts[i]}-{self.concepts[j]}")
                    
                    relation, input_tokens, output_tokens = self.get_relationship(self.concepts[i], self.concepts[j])
                    
                    # 使用更简洁的后缀
                    pbar.set_postfix({'in': input_tokens, 'out': output_tokens, 'sum':input_tokens + output_tokens, 'all': self.total_input_tokens + self.total_output_tokens})
                    
                    self.save_results(i, j, relation)

        return M
    
    def combine_results(self, model_name, n_parts = 8, debug = False, invalid_value = 3):
        dfs = []
        df_prerequisites_path = Path(f'''./results/{model_name}-concept_prerequisite_results.csv''')
        if not op.exists(df_prerequisites_path) or debug:
            for part in range(n_parts):
                results_file = Path(f'''./results/{model_name}-concept_prerequisite_results{'' if part < 0 else f'_part_{part}'}.csv''')
                df_results_file = pd.read_csv(results_file)
                dfs.append(df_results_file)
            df_prerequisites = pd.concat(dfs)
            df_prerequisites.to_csv(df_prerequisites_path, index=False)
        else: df_prerequisites = pd.read_csv(df_prerequisites_path)

        while True:
            invalid_prerequisites = df_prerequisites.loc[df_prerequisites.prerequisite==invalid_value]
            if len(invalid_prerequisites) == 0:  # 如果没有无效值了，就退出循环
                df_prerequisites.to_csv(df_prerequisites_path, index=False)
                break
                
            all_pairs = invalid_prerequisites[['index_i','index_j']].values
            total_pairs = len(all_pairs)
            
            with tqdm(enumerate(all_pairs), total=total_pairs) as pbar:
                for index, (i, j) in pbar:
                    pbar.set_description(f"({i},{j}) {self.concepts[i]}-{self.concepts[j]}")
                    
                    relation, input_tokens, output_tokens = self.get_relationship(self.concepts[i], self.concepts[j])
                    
                    df_prerequisites.loc[(df_prerequisites['index_i'] == i) & (df_prerequisites['index_j'] == j), 
                                    ['prerequisite', 'timestamp']] = [relation, datetime.now().strftime('%Y-%m-%d %H:%M:%S')]

                    pbar.set_postfix({'in': input_tokens, 
                                    'out': output_tokens, 
                                    'sum': input_tokens + output_tokens, 
                                    'all': self.total_input_tokens + self.total_output_tokens})
                    if index % 100 == 0:
                        df_prerequisites.to_csv(df_prerequisites_path, index=False)

        return df_prerequisites

    def process_relationship(self, i, j, M):
        relation, _ = self.get_relationship(self.concepts[i], self.concepts[j])
        M[i, j] = relation
        M[j, i] = -relation if relation == 1 else relation
        self.save_results(i, j, relation)
        return M

    # def save_results(self, i, j, relation):
    #     # 保存checkpoint到CSV
    #     with open(self.results_file, 'a', newline='', encoding='utf-8') as f:
    #         writer = csv.writer(f)
    #         writer.writerow([i, j, relation, time.strftime('%Y-%m-%d %H:%M:%S')])

    def save_results(self, i, j, relation):
        max_retries = 3
        retry_delay = 1  # seconds
        
        for attempt in range(max_retries):
            try:
                # Ensure the result directory exists
                os.makedirs(os.path.dirname(self.results_file), exist_ok=True)
                
                # Try to write to file
                with open(self.results_file, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([i, j, relation, time.strftime('%Y-%m-%d %H:%M:%S')])
                return  # Return directly if successful
                
            except PermissionError as e:
                if attempt < max_retries - 1:  # If not the last attempt
                    print(f"\nWrite failed (attempt {attempt + 1}/{max_retries}): {e}")
                    print(f"Waiting {retry_delay} seconds before retry...")
                    time.sleep(retry_delay)
                else:
                    print(f"\nCannot write to file {self.results_file}, max retries reached")
                    raise  # Re-raise the exception
                    
            except Exception as e:
                print(f"\nError occurred during writing: {type(e).__name__}: {e}")
                raise


    def save_relation_matrix(self, M):
        # 保存前提矩阵到CSV
        pd.DataFrame(M, index=self.concepts, columns=self.concepts).to_csv(self.relationship_matrix) # save matrix

if __name__ == "__main__":
    # 设置命令行参数
    debug = True

    # model_name = "gpt-4o"
    # n_parts = 8

    # model_name = "claude-3-5-sonnet-20240620" # "gpt-4o"
    # n_parts = 8

    # model_name = 'ERNIE-3.5-8K-0701'
    # n_parts = 4

    model_name = "qwen-plus" # "gpt-4o"
    n_parts = 4
    
    parser = argparse.ArgumentParser(description='Concept Relation Analyzer')
    parser.add_argument('--part', type=int, choices=range(8), default=0, help='Part to process (0-7)')
    parser.add_argument('--model', type=str, default='claude-3-5-sonnet-20240620', help='model name')

    # 解析命令行参数
    args = parser.parse_args()
    
    # Create analyzer instance
    analyzer = ConceptRelationAnalyzer(part=args.part, model_name = model_name) # part range [0,7]
    
    # Analyze all relationships
    print("Starting concept relationship analysis...")
    relationship_matrix = analyzer.analyze_all_relationships()
    analyzer.combine_results(model_name = model_name, n_parts = n_parts, debug = debug)
    # Create DataFrame and display results
    # df = pd.DataFrame(relationship_matrix, index=analyzer.concepts, columns=analyzer.concepts)
    # # Save to CSV file
    # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # output_filename = f'final_{model_name}_relationship_matrix_{timestamp}.csv'
    # df.to_csv(output_filename)

    # print("\nRelationship Matrix:")
    # print(df)

# python openai_prerequisite_analyzer.py --part 0  # 将使用 part=1
