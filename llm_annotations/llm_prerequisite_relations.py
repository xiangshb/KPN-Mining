


import os.path as op
import pandas as pd
import numpy as np

def combine_multiple_sources(model_names, folder_path = "llm_prerequisite_relations", debug = False):
    df_prerequisites_path = op.join(folder_path, 'llm_prerequisite_relations_comparison.csv')
    if not op.exists(df_prerequisites_path) or debug:
        for i, model_name in enumerate(model_names):
            file_path = op.join(folder_path, f'{model_name}-concept_prerequisite_results.csv')
            df_model_results = pd.read_csv(file_path).sort_values(by=['index_i', 'index_j']).rename(columns={'prerequisite': model_name})
            if i == 0: df_prerequisites = df_model_results[['index_i', 'index_j', model_name]]
            else: df_prerequisites = pd.concat([df_prerequisites, df_model_results[model_name]], axis=1)
        # 应用多数投票函数
        df_prerequisites[['majority_vote', 'confidence']] = df_prerequisites.apply(get_majority_vote, axis=1).to_list()
        df_prerequisites['majority_vote'] = df_prerequisites['majority_vote'].astype(int)
        df_prerequisites.to_csv(df_prerequisites_path, index=False)
    else: df_prerequisites = pd.read_csv(df_prerequisites_path)
    return df_prerequisites

def get_majority_vote(row):
    model_columns = ['gpt-4o', 'claude-3-5-sonnet-20240620', 'ERNIE-3.5-8K-0701', 'qwen-plus', 'Spark4.0-Ultra']
    values, counts = np.unique(row[model_columns], return_counts=True) # values default sort in ascending order
    max_count = np.max(counts)
    max_count_values = sorted(values[counts == max_count])
    
    if len(max_count_values) > 1: 
        if tuple(max_count_values) in [(-2, -1), (-2, 1)]: # two model: positive/negative prerequisite, two model: related
            most_common_vote = max_count_values[1] # adjust the relation to be positive/negative prerequisite
            confidence = (max_count + 1) / len(model_columns)  # 3/5 = 0.6
        elif tuple(max_count_values) == (-1, 1): # two positive, two negative
            most_common_vote = 2 # adjust the relation to be mutual prerequisite
            confidence = (max_count + 2) / len(model_columns)  # 4/5 = 0.8
        else: 
            most_common_vote = 0 # others like [(-2, 0), (-1, 0), (0, 1)] are not prerequisites
            confidence = max_count / len(model_columns)
    else: 
        most_common_vote = max_count_values[0]
        confidence = max_count / len(model_columns)

    return most_common_vote, confidence

def get_prerequisite_matrix(df_prerequisites, folder_path, debug = False):
    # 感觉没必要写这个函数，直接读取带index，带关系的csv文件即可
    df_prerequisite_matrix_path = op.join(folder_path, 'llm_prerequisite_relation_matrix.csv')
    df_short_summaries = pd.read_csv('final_GPT4_short_within_35_summaries.csv')
    idx_to_concept = {i:concept for i, concept in enumerate(df_short_summaries['display_name'].tolist())}
    df_prerequisites['concept_i'] = df_prerequisites.index_i.map(idx_to_concept)
    df_prerequisites['concept_j'] = df_prerequisites.index_j.map(idx_to_concept)
    df_prerequisites = df_prerequisites[['concept_i','concept_j','majority_vote','confidence']]
    df_prerequisites_ground_truth = df_prerequisites.loc[df_prerequisites.majority_vote.isin([-1, 1, 2])]
    if not op.exists(df_prerequisite_matrix_path) or debug:
        
        pass
    pass


# 使用示例
if __name__ == "__main__":
    model_names = ['gpt-4o', 'claude-3-5-sonnet-20240620', 'ERNIE-3.5-8K-0701', 'qwen-plus', 'Spark4.0-Ultra']
    folder_path = "llm_prerequisite_relations_v2"
    # df_prerequisites = combine_multiple_sources(model_names, debug=True)
    df_prerequisites = combine_multiple_sources(model_names, folder_path = folder_path, debug=False)
    df_prerequisite_matrix = get_prerequisite_matrix(df_prerequisites, folder_path = folder_path)
    print(5)

