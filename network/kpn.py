import os, sys
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.append(project_dir)

import numpy as np
import pandas as pd
import networkx as nx
import os.path as op
from utils.params import TCPParams, KPNParams
from utils.visualization import Visualizer
from utils.config import PathManager
from network.kpm import KPM

class KPN:
    """
    Class for constructing Knowledge Precedence Networks (KPNs) from research trajectories
    """
    def __init__(self):
        self.matrix = None
        self.visualization = Visualizer()  # Visualization tool
        self.path_manager = PathManager()  # Path manager
        self.KPM = KPM()

    def get_max_symmetric_elements(self, matrix, symmetry_preference='equal_priority'):
        """
        Create a symmetric matrix by keeping the larger values between symmetric positions
        
        Args:
            matrix: Input matrix
            symmetry_preference: How to handle equal values in symmetric positions
                - 'lower_priority': When values are equal, prefer values from lower triangle (row > column)
                - 'upper_priority': When values are equal, prefer values from upper triangle (column > row)
                - 'equal_priority': Keep both elements when they are equal (default)
                
        Returns:
            Matrix with retained larger symmetrical elements
        """
        matrix_tril, matrix_triu = np.tril(matrix, k=-1), np.triu(matrix, k=1)
        
        if symmetry_preference == 'lower_priority':
            matrix_maximum_symmetric = matrix_tril*(matrix_tril >= matrix_triu.T) + matrix_triu*(matrix_tril.T < matrix_triu)
        elif symmetry_preference == 'upper_priority':
            matrix_maximum_symmetric = matrix_tril*(matrix_tril > matrix_triu.T) + matrix_triu*(matrix_tril.T <= matrix_triu)
        elif symmetry_preference == 'equal_priority':
            matrix_maximum_symmetric = matrix_tril*(matrix_tril >= matrix_triu.T) + matrix_triu*(matrix_tril.T <= matrix_triu)
        else:
            raise ValueError('Invalid symmetry preference. Use "lower_priority", "upper_priority", or "equal_priority"')
            
        return matrix_maximum_symmetric


    def matrix_normalization(self, matrix, row_normalize: bool = True, epsilon: float = 1e-10):
        """
        Normalize the matrix either by row or column
        
        Args:
            matrix: Input matrix
            row_normalize: If True, normalize by row (row sum = 1); otherwise, normalize by column (column sum = 1)
            epsilon: Small constant added to sums to avoid division by zero
            
        Returns:
            Normalized matrix
        """
        if row_normalize:  # Row normalization (row sum = 1)
            sum_matrix = np.expand_dims(np.sum(matrix, axis=1), axis=1) + epsilon
        else:  # Column normalization (column sum = 1)
            sum_matrix = np.sum(matrix, axis=0) + epsilon
            
        normalized_matrix = np.nan_to_num(matrix / sum_matrix, nan=0)
        return normalized_matrix

    def get_sub_kpm_of_disciplines(self, params, df_kpm_non_symmetric, disciplines:list = None, reverse = False, with_llm_annotation = False):
        """
        Get concepts for matrix flow based on specified disciplines
        
        Args:
            disciplines: List of disciplines to filter concepts by
                - If ['All disciplines'], returns all concepts
                - If two disciplines, returns concepts specific to their interaction
                - If more than two disciplines, returns concepts shared among them
                
        Returns:
            Tuple containing (row_concepts, column_concepts, row_concepts_abbreviation, column_concepts_abbreviation)
        """
        if (len(disciplines) == 1) and (disciplines[0] != ['All disciplines']):
            disciplines = disciplines * 2
        elif len(disciplines) == 2:
            disciplines = disciplines[::-1] if reverse else disciplines
        elif len(disciplines) > 2:
            disciplines = disciplines

        concept_condition = self.KPM.concepts_table.level <= params.concept_level if params.less_than else self.KPM.concepts_table.level < params.concept_level
        concepts_table = self.KPM.concepts_table.loc[concept_condition]

        # Handle different discipline filtering scenarios
        if with_llm_annotation:
            if disciplines is None or (len(disciplines) == 1 and disciplines[0] == 'All disciplines'):
                row_concepts_table = column_concepts_table = concepts_table.copy()
                
            elif len(disciplines) == 2:
                row_concepts_table = concepts_table.loc[concepts_table.llm_annotation == disciplines[0]].copy()
                column_concepts_table = concepts_table.loc[concepts_table.llm_annotation == disciplines[1]].copy()

            elif len(disciplines) > 2:
                row_concepts_table = column_concepts_table = concepts_table.loc[concepts_table.llm_annotation.isin(disciplines)].copy()
        else:
            if disciplines is None or (len(disciplines) == 1 and disciplines[0] == 'All disciplines'):
                row_concepts_table = column_concepts_table = concepts_table.copy()
            elif (len(disciplines) == 1) & (disciplines[0] != 'All disciplines'):
                row_concepts_table = concepts_table.loc[concepts_table.multiple_level_0_ancestors.apply(lambda ancestors: disciplines[0] in ancestors)]
                row_concepts_refined_condition = (row_concepts_table.level_0_ancestor_refined==disciplines[0]) | row_concepts_table.multiple_level_0_ancestors.apply(lambda ancestors: disciplines[1] in ancestors)
                row_concepts_table = column_concepts_table = row_concepts_table.loc[row_concepts_refined_condition]
            elif len(disciplines) == 2:
                row_concepts_table = concepts_table.loc[concepts_table.multiple_level_0_ancestors.apply(lambda ancestors: disciplines[0] in ancestors)]
                row_concepts_refined_condition = (row_concepts_table.level_0_ancestor_refined==disciplines[0]) | row_concepts_table.multiple_level_0_ancestors.apply(lambda ancestors: disciplines[1] in ancestors)
                row_concepts_table = row_concepts_table.loc[row_concepts_refined_condition]

                column_concepts_table = concepts_table.loc[concepts_table.multiple_level_0_ancestors.apply(lambda ancestors: disciplines[1] in ancestors)]
                column_concepts_refined_condition = (column_concepts_table.level_0_ancestor_refined==disciplines[1]) | column_concepts_table.multiple_level_0_ancestors.apply(lambda ancestors: disciplines[0] in ancestors)
                column_concepts_table = column_concepts_table.loc[column_concepts_refined_condition]
            elif len(disciplines) > 2:
                temp_table = concepts_table.loc[concepts_table.multiple_level_0_ancestors.apply(lambda ancestors: any([discipline in ancestors for discipline in disciplines]))]
                filt_condition = (temp_table.level_0_ancestor_refined != 'Interdiscipline') | ((temp_table.level_0_ancestor_refined == 'Interdiscipline') & (temp_table.multiple_level_0_ancestors.apply(lambda ancestors: sum([discipline in ancestors for discipline in disciplines])>1)))
                row_concepts_table = column_concepts_table = temp_table.loc[filt_condition]

        # Extract concept names
        row_concepts = row_concepts_table.display_name.tolist()
        column_concepts = column_concepts_table.display_name.tolist()
        
        df_sub_kpm = df_kpm_non_symmetric.loc[row_concepts, column_concepts]

        return df_sub_kpm

    def optimize_kpm(self, matrix, kpn_parms):
        """
        Optimize the knowledge precedence matrix
        
        Args:
            matrix: Original knowledge precedence matrix
            cvm_threshold: Threshold for Cumulative Value Mass (cvm)
            cfm_threshold: Threshold for Cumulative Frequency Mass (cfm)
            row_normalize: If True, normalize by row; otherwise, normalize by column
            threshold_mode: Mode for thresholding ('value_count_limit' or 'direct')
            
        Returns:
            Optimized knowledge precedence matrix
        """
        optimized_matrix = np.zeros(matrix.shape)
        normalized_matrix = self.matrix_normalization(matrix, kpn_parms.row_normalize)
        
        if kpn_parms.threshold_mode == 'value_count_limit':
            values, counts = np.unique(matrix, return_counts=True)
            df_matrix_value = pd.DataFrame({'matrix_value': values, 'count': counts}).sort_values('matrix_value', ascending=False).reset_index(drop=True)
            df_matrix_value['cvm'] = df_matrix_value['matrix_value'].cumsum() / values.sum()
            df_matrix_value['cfm'] = df_matrix_value['count'].cumsum() / counts.sum()
            
            thresh_condition = (df_matrix_value.cvm >= kpn_parms.cvm_threshold) & (df_matrix_value.cfm <= kpn_parms.cfm_threshold)
            df_thresh_values = df_matrix_value.loc[thresh_condition].reset_index(drop=True)
            
            print(f"{'row norm' if kpn_parms.row_normalize else 'column norm'} matrix shape {matrix.shape}")
            print(f"df_thresh_values.shape[0]={df_thresh_values.shape[0]}")
            
            if df_thresh_values.shape[0] <= 0:
                print(f'cvm = {kpn_parms.cvm_threshold} too large\nfilter only by cfm={kpn_parms.cfm_threshold}')
                df_thresh_values = df_matrix_value.loc[df_matrix_value.cfm <= kpn_parms.cfm_threshold].reset_index(drop=True)
                matrix_thresh_value = df_thresh_values.iloc[-1].matrix_value
            else:
                matrix_thresh_value = df_thresh_values.iloc[0].matrix_value
        elif kpn_parms.threshold_mode == 'direct':
            matrix_thresh_value = np.quantile(matrix, kpn_parms.quantile_threshold)  # Use quantile as threshold
        else:
            raise ValueError(f"Unknown threshold mode: {kpn_parms.threshold_mode}")
            
        optimized_matrix[matrix >= matrix_thresh_value] = normalized_matrix[matrix >= matrix_thresh_value]

        return optimized_matrix

    def kpn_edges(self, df_sub_kpm, kpn_parms, optimize =True, original = False):
        """
        Return the KPN edges based on the optimized the KPM
        """
        # Optimize the matrix
        if optimize:
            optimized_matrix = self.optimize_kpm(df_sub_kpm.values, kpn_parms)
        else:
            original_matrix = df_sub_kpm.values
            optimized_matrix = self.matrix_normalization(df_sub_kpm.values, kpn_parms.row_normalize)

        if kpn_parms.matrix_imshow:
            self.visualization.matrix_imshow_with_value(matrix=optimized_matrix, row_names = df_sub_kpm.index, column_names=df_sub_kpm.columns, fontsize = 11,
                                            colorbar_shrink = kpn_parms.colorbar_shrink, file_name = 'test', save_pdf = False)

        if optimize:
            row_idx, col_idx = np.where(optimized_matrix > 0)
            # Create Edge DataFrame
            df_edges = pd.DataFrame({'source': df_sub_kpm.index[row_idx], 'target': df_sub_kpm.columns[col_idx], 'weight': optimized_matrix[row_idx, col_idx]})
        else:
            row_idx, col_idx = np.where(optimized_matrix > 0)
            original_row_idx, original_col_idx = np.where(original_matrix > 0)

            # Create Edge DataFrame
            df_edges = pd.DataFrame({'source': df_sub_kpm.index[row_idx], 'target': df_sub_kpm.columns[col_idx], 'weight': optimized_matrix[row_idx, col_idx], 'max_count': original_matrix[original_row_idx, original_col_idx]})
        
        return df_edges

    def build_kpn(self, df_edges, disciplines, node_color_set = ['lime', 'blue','yellow','gray','green']):
        """
        Build a knowledge precedence network from given edges
        
        Args:
            df_edges: DataFrame with weighted edges (columns: source, target, weight)
            disciplines: List of disciplines to consider
            params: General parameters
            node_color_set: List of colors for nodes
            save_files: Whether to save files
            visualization: Whether to visualize the graph
                
        Returns:
            NetworkX DiGraph representing the knowledge precedence network and DataFrame of edges
        """
        # Create the directed graph
        G = nx.DiGraph()
        G.add_weighted_edges_from(df_edges.values)
        
        print(f'Graph nodes {len(G.nodes)}')
        print(f'Graph edges {len(G.edges)}')
        
        graph_concepts = list(set(df_edges.source).union(df_edges.target))

        # Filter concepts table to only include nodes in the graph
        concepts_table = self.KPM.concepts_table.loc[self.KPM.concepts_table.display_name.isin(graph_concepts)]
        
        # Set up discipline color dictionary
        if disciplines is None or len(disciplines) < 1 or ((len(disciplines) == 1) and disciplines[0] == 'All disciplines'):
            discipline_color_dict = {discipline: node_color_set[0] for discipline in concepts_table.level_0_ancestor_refined.tolist()}
        elif len(disciplines) < 6: discipline_color_dict = {disciplines[i]: node_color_set[i] for i in range(len(disciplines))}
        else: raise ValueError('Too many disciplines, not defined yet')

        discipline_color_dict.update({'Interdiscipline': 'red'})
        
        # Add node colors
        concepts_table.loc[:, 'node_color'] = concepts_table['level_0_ancestor_refined'].map(discipline_color_dict)

        node_color_dict = dict(concepts_table[['display_name', 'node_color']].values)
        nx.set_node_attributes(G, node_color_dict, 'node_color')
        
        # Add level attribute to nodes
        if 'level' in concepts_table.columns:
            level_dict = dict(concepts_table.loc[concepts_table.display_name.isin(graph_concepts)][['display_name', 'level']].values)
            nx.set_node_attributes(G, level_dict, 'level')  # for hierarchical layout subset key
        
        # Add discipline attribute to nodes
        if 'level_0_ancestor_refined' in concepts_table.columns:
            discipline_dict = dict(concepts_table.loc[concepts_table.display_name.isin(graph_concepts)][['display_name', 'level_0_ancestor_refined']].values)
            nx.set_node_attributes(G, discipline_dict, 'discipline')
        
        return G, concepts_table, discipline_color_dict

    def kpn_analysis(self, params, kpn_parms):
        """
        Process KPM to KPN using parameter objects
        
        Args:
            params: TCPParams object
            kpn_parms: KPNParams object
            concepts: List of concepts (row/column labels for the matrix)
            node_attributes: Dictionary of node attributes
            
        Returns:
            NetworkX DiGraph and DataFrame of edges
        """
        if len(kpn_parms.disciplines) == 0:
            kpn_parms.disciplines = ['All disciplines']

        df_kpm, params_str = self.KPM.get_kpm(params)

        # Ensure the matrix is non symmetric
        df_kpm_non_symmetric = pd.DataFrame(self.get_max_symmetric_elements(df_kpm.values), index=df_kpm.index, columns=df_kpm.columns)

        if len(kpn_parms.disciplines) == 2: 
            df_sub_kpm_pos = self.get_sub_kpm_of_disciplines(params, df_kpm_non_symmetric, disciplines = kpn_parms.disciplines, reverse=False)
            df_sub_kpm_neg = self.get_sub_kpm_of_disciplines(params, df_kpm_non_symmetric, disciplines = kpn_parms.disciplines, reverse=True)
            
            df_edges_pos = self.kpn_edges(df_sub_kpm_pos, kpn_parms)
            df_edges_neg = self.kpn_edges(df_sub_kpm_neg, kpn_parms)
            df_edges = pd.concat([df_edges_pos, df_edges_neg]).drop_duplicates().reset_index(drop=True)
            df_edges = df_edges.loc[df_edges[['source','target']].drop_duplicates().index].reset_index(drop=True)
            df_edges['weight'] = df_edges['weight'].astype(float)

        else:
            df_sub_kpm = self.get_sub_kpm_of_disciplines(params, df_kpm_non_symmetric, disciplines = kpn_parms.disciplines)
            df_edges = self.kpn_edges(df_sub_kpm, kpn_parms)
        
        # Build knowledge precedence network
        G_kpn, concepts_table, discipline_color_dict = self.build_kpn(df_edges, kpn_parms.disciplines)
        
        # Save files if requested and parameters are provided
        discipline_str = "_".join(kpn_parms.disciplines).replace(' ', '_')
        
        # Save edges and nodes as CSV
        if kpn_parms.save_kpn_nodes_csv:
            self.path_manager.save_csv_file(variable=df_edges, file_name=f'{discipline_str}_edges.csv', index=False, override=True)
            self.path_manager.save_csv_file(variable=concepts_table, file_name=f'{discipline_str}_nodes.csv', index=False, override=True)
        
        # Save graph as GEXF
        if kpn_parms.save_kpn:
            gexf_file_name = f"G_concept_flow_{discipline_str}_matrix_filter_value_cum_{kpn_parms.cvm_threshold}_count_cum_{kpn_parms.cfm_threshold}_{params_str}.gexf"
            gexf_file_path = op.join(self.path_manager.ensure_folder_exists(op.join(self.path_manager.concepts_dir, 'Networks')), gexf_file_name)
            self.path_manager.save_gexf_file(G=G_kpn, abs_file_path=gexf_file_path, override=True, update_version=True)
        
        # Visualize if requested
        if kpn_parms.graph_show:
            self.visualization.draw_concept_flow_network(
                G_kpn, 
                row_normalize=kpn_parms.row_normalize, 
                layout=kpn_parms.layout, 
                discipline_color_dict=discipline_color_dict if 'discipline_color_dict' in locals() else None, 
                file_name=f'kpn_{discipline_str}.pdf', 
                save_pdf=kpn_parms.save_pdf, 
                draw_legend=kpn_parms.draw_legend
            )
            
        return G_kpn, df_edges

    def kpn_evaluation_wcr(self, params, kpn_parms):
        self.KPM.initialize_kpm_matrices(params)
        evaluation_results = []
        for wcr in params.wcr_ranges:
            df_kpm = self.KPM.kpm_matrices[wcr]
            df_edges = self.get_prediction_edges(df_kpm, params, kpn_parms)
            df_edges['ground_truth'] = df_edges[['source', 'target']].apply(tuple, axis=1).map(self.llm_pair_dict)
            
            precesion = (df_edges.ground_truth>0).sum() / len(df_edges)
            recall = (df_edges.ground_truth>0).sum() / len(self.positive_gt_pairs)
            evaluation_results.append([wcr, precesion, recall])
        
        def format_row(row):
            if row.name == 'wcr':
                return row.apply(lambda x: '{}%'.format(int(round(x * 100))))
            else:
                return row.apply(lambda x: '{:.2f}%'.format(x * 100))

        df_results = pd.DataFrame(evaluation_results, columns=['wcr','precesion','recall'])
        df_percent = df_results.T.apply(format_row)
        df_percent.to_csv(op.join(self.path_manager.base_data_dir, 'results_percent.csv'))

    def kpn_evaluation_period(self, params, kpn_parms):
        self.KPM.initialize_kpm_matrices(params)
        evaluation_results = []
        for period in params.period_ranges:
            df_kpm = self.KPM.period_kpm_matrices[period]
            df_edges = self.get_prediction_edges(df_kpm, params, kpn_parms)
            df_edges['ground_truth'] = df_edges[['source', 'target']].apply(tuple, axis=1).map(self.llm_pair_dict)
            
            precesion = (df_edges.ground_truth>0).sum() / len(df_edges)
            recall = (df_edges.ground_truth>0).sum() / len(self.positive_gt_pairs)
            evaluation_results.append([period, precesion, recall])

        df_results = pd.DataFrame(evaluation_results, columns=['period','precesion','recall'])
        for col in ['precesion', 'recall']:
            df_results[col] = df_results[col].apply(lambda x: '{:.2f}%'.format(float(x) * 100))

        df_results.T.to_csv(op.join(self.path_manager.base_data_dir, 'period_results_percent.csv'))

        print(5)


if __name__ == "__main__":

    params = TCPParams(
        concept_level=1, 
        less_than=True, 
        select_mode='all'
    )
    
    kpn_parms = KPNParams(
        disciplines=['Mathematics', 'Computer science'],
        row_normalize=False,
        threshold_mode='value_count_limit', # direct
        quantile_threshold=0.65,
        cvm_threshold=0.9,
        cfm_threshold=0.15,
        matrix_filter=True,
        matrix_imshow=False,
        graph_show=True,
        layout="graphviz",
        save_kpn=True,
        old_matrix_version=False
    )
    
    kpn = KPN()
    
    G_all, df_edges_all = kpn.kpn_analysis(params=params, kpn_parms=kpn_parms)

    print(5)

