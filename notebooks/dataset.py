from database import DatabaseManager
from config import PathManager, calculate_runtime
from community import community_louvain
from networkx.algorithms import community as nxcommunity
from concept import Concept
from visualization import Visualizer
from itertools import combinations
import tqdm, itertools
import os.path as op
import networkx as nx
import numpy as np
import pandas as pd
from typing import Union, List
import logging, ast
from param_class import Params_community_concept_pairs, Params_community_visualization, Params_concept_flow_matrix

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Dataset():
    path_manager = PathManager()
    db_manager = DatabaseManager()
    visualization = Visualizer()
    
    @calculate_runtime
    def get_author_communities(self, CCN_author, community_method: str = 'louvain', min_community_size: int = 2):
        df_work_id_infos =  pd.DataFrame.from_dict(nx.get_node_attributes(CCN_author, 'publication_date'), orient='index', columns=['publication_date'])
        df_work_id_infos['publication_date'] = pd.to_datetime(df_work_id_infos.publication_date, errors='coerce')
        if community_method == 'louvain':
            partition = community_louvain.best_partition(CCN_author.to_undirected())
        elif community_method == 'greedy':
            partition = {}
            for i, community_i in enumerate(nxcommunity.greedy_modularity_communities(CCN_author)):
                for node in community_i: partition[node] = i
        else: raise ValueError('Community detection method not implemented')
        df_work_id_infos['community_order'] = df_work_id_infos.index.map(partition)
        df_author_communities = pd.DataFrame({'community': df_work_id_infos.groupby('community_order').apply(lambda x: x.index.tolist() if len(x) >= min_community_size else None).reset_index(drop=True).dropna()})
        df_work_id_infos['selected'] = df_work_id_infos.community_order.isin(df_author_communities.index)
        CCN_author = CCN_author.subgraph(df_work_id_infos.loc[df_work_id_infos.selected].index.tolist())

        for i, community_i in df_author_communities.itertuples(index=True):
            df_author_communities.loc[i,'mean_pub_date'] = df_work_id_infos.loc[community_i].publication_date.mean()
            df_author_communities.loc[i,'mean_pub_year'] = df_work_id_infos.loc[community_i].publication_date.dt.year.mean()
        df_author_communities['author_id'] = CCN_author.graph['author_id']
        
        columns = df_author_communities.columns.tolist()
        df_author_communities = df_author_communities.sort_values('mean_pub_year').reset_index(drop=True)[[columns[-1]] + columns[:-1]]
        
        community_partition_dict = {}
        for i, community_i in df_author_communities[['community']].itertuples(index=True):
            for node_id in community_i: community_partition_dict[node_id] = i
        nx.set_node_attributes(CCN_author, community_partition_dict, "community_order")
        
        return CCN_author, df_author_communities

    @calculate_runtime
    def get_CCN_temporal_community_pairs(self, params: Params_community_concept_pairs, show_runtime: bool = True):
        # get_CCN_temporal_community_pairs(self, df_author_communities, concept_level: int = 1, less_than: bool = True, select_mode: str = 'respective', work_coverage_ratio: float = 0.8, top_ratio: float= 0.3, show_runtime: bool = True):
        if params.df_author_communities.shape[0] < 2: return [], pd.DataFrame([], columns = ['community_pair','concept_pair']), [], []
        communities_author = params.df_author_communities['community'].tolist()
        author_works = [work_id for community_ in communities_author for work_id in community_]
        works_concepts_table = self.db_manager.query_table(table_name = 'works_concepts', columns = ['work_id', 'concept_id', 'display_name', 'level'], 
                                                        join_tables = ['concepts'], join_conditions = ['works_concepts.concept_id = concepts.id'], where_conditions = [f'''work_id IN ('{"','".join(author_works)}')'''], show_runtime=False)
        df_concept_community_nodes_coverage_ratio = pd.DataFrame(np.full((params.df_author_communities.shape[0], 10), None, dtype=object), columns = [f'{i}_th' for i in range(1, 6)] + [f'top_{i}' for i in range(1, 6)])
        author_representive_concepts = []
        for i, community_i in enumerate(communities_author):
            community_concepts = works_concepts_table.loc[(works_concepts_table.work_id.isin(community_i)) & ((works_concepts_table.level <= params.concept_level) if params.less_than else (works_concepts_table.level == params.concept_level))]
            if community_concepts.shape[0] < 1:
                author_representive_concepts.append([])
                continue
            concept_statistics = community_concepts.groupby('display_name').apply(lambda x: x['work_id'].drop_duplicates().shape[0]).sort_values(ascending=False).reset_index()
            concept_statistics.columns = ['display_name','work_count']
            concept_statistics['wrok_coverage_ratio'] = concept_statistics.work_count / community_concepts.work_id.drop_duplicates().shape[0] # might be wrong if 
            concept_statistics['wrok_count_cum_ratio'] = concept_statistics.work_count.cumsum() / concept_statistics.work_count.sum()

            concept_statistics['top_n_collective_work_ids'] = concept_statistics.apply(lambda x: set(community_concepts[community_concepts.display_name.isin(concept_statistics.loc[:x.name, 'display_name'])]['work_id'].tolist()), axis=1).apply(len)
            concept_statistics['top_n_collective_coverage_ratio'] = concept_statistics['top_n_collective_work_ids'] / concept_statistics.top_n_collective_work_ids.max()
            fill_length = min(concept_statistics.shape[0], 5)
            df_concept_community_nodes_coverage_ratio.iloc[i,:fill_length] = concept_statistics.wrok_coverage_ratio[:fill_length]
            df_concept_community_nodes_coverage_ratio.iloc[i,5:5+fill_length] = concept_statistics.top_n_collective_coverage_ratio[:fill_length]

            if params.select_mode == 'collective':
                min_index = np.argwhere((concept_statistics.top_n_collective_coverage_ratio >= params.work_coverage_ratio).values).flatten()[0]
                representive_concepts = concept_statistics.loc[:min_index,'display_name'].tolist()
            elif params.select_mode == 'respective': 
                representive_concepts = concept_statistics.loc[concept_statistics.wrok_coverage_ratio >= params.work_coverage_ratio].display_name.tolist()
            elif params.select_mode == 'cumulative': 
                candidate_indexes = np.argwhere((concept_statistics.wrok_count_cum_ratio <= params.top_ratio).values).flatten()
                if candidate_indexes.shape[0] > 0:
                    max_index = np.argwhere((concept_statistics.wrok_count_cum_ratio <= params.top_ratio).values).flatten()[-1]
                    representive_concepts = concept_statistics.loc[:max_index,'display_name'].tolist()
                else: representive_concepts = []
            elif params.select_mode == 'all': 
                representive_concepts = concept_statistics.display_name.to_list()
            else: raise ValueError('Undefined select mode')

            author_representive_concepts.append(representive_concepts)
        df_concept_community_nodes_coverage_ratio['author_id'] = params.df_author_communities['author_id'].tolist()
        author_representive_concepts = np.array(author_representive_concepts, dtype=object)
        valid_community_pair_indexes = self.valid_time_diffs_index_pairs(params.df_author_communities['mean_pub_year'].agg(list))
        all_pairs = author_representive_concepts[valid_community_pair_indexes]
        concept_pairs_author = list(map(lambda row: list(itertools.product(*row)), all_pairs))
        concept_pairs_author = list(itertools.chain(*concept_pairs_author))

        product_len = lambda index_pair: len(author_representive_concepts[index_pair[0]]) * len(author_representive_concepts[index_pair[1]])
        df_concept_pairs = pd.DataFrame({'community_pair': [tuple(pair_) for pair_ in valid_community_pair_indexes for _ in range(product_len(pair_))]})
        df_concept_pairs['community_pair_1_based'] = [tuple(pair_+1) for pair_ in valid_community_pair_indexes for _ in range(product_len(pair_))] # index pair_ + 1 to make it 1 based instead of 0 based
        df_concept_pairs['concept_pair'] = concept_pairs_author

        return author_representive_concepts.tolist(), df_concept_pairs, df_concept_community_nodes_coverage_ratio, works_concepts_table

    @calculate_runtime
    def CCN_community_pub_date(self, generating = True, iter_i: int = 0, n_iters = 20, min_community_size = 2):
        assert (0 <= iter_i) & (iter_i < n_iters), f"Unexpected iteration: {iter_i} should be < {n_iters}"
        file_path_CCNs_k_th_mean_pub_date_infor_dir = self.path_manager.ensure_folder_exists(op.join(self.path_manager.external_file_dir, 'CCNs', 'CCNs_community_pub_date'))
        _, iterate_ranges = self.CCN_authors()
        if generating:
            for k_th in iterate_ranges[iter_i]: 
                file_path_CCNs_k_th_mean_pub_date_infor = op.join(file_path_CCNs_k_th_mean_pub_date_infor_dir, f'CCNs_{k_th}_th_community_pub_date.npy')
                if not op.exists(file_path_CCNs_k_th_mean_pub_date_infor):
                    CCNs_k_th = np.load(op.join(self.path_manager.external_file_dir, 'CCNs', 'CCNs_sub', f'CCNs_sub_{k_th}.npy'), allow_pickle = True)
                    author_communities_k_th = []
                    for CCN_author in tqdm.tqdm(CCNs_k_th):
                        CCN_author, df_author_communities = self.get_author_communities(CCN_author, min_community_size = min_community_size)
                        author_communities_k_th.append(df_author_communities.values)
                    mean_pub_date_infor_k_th = np.concatenate(author_communities_k_th, axis=0)
                    np.save(file_path_CCNs_k_th_mean_pub_date_infor, mean_pub_date_infor_k_th)
        else:
            file_path_all_pub_date_infor = op.join(file_path_CCNs_k_th_mean_pub_date_infor_dir, 'all_CCNs_communities_pub_date.npy')
            if not op.exists(file_path_all_pub_date_infor):
                all_mean_pub_date_infor_list = []
                for iter_i in range(n_iters):
                    for k_th in iterate_ranges[iter_i]: 
                        file_path_CCNs_k_th_mean_pub_date_infor = op.join(file_path_CCNs_k_th_mean_pub_date_infor_dir, f'CCNs_{k_th}_th_community_pub_date.npy')
                        all_mean_pub_date_infor_list.append(np.load(file_path_CCNs_k_th_mean_pub_date_infor, allow_pickle=True))
                all_mean_pub_date_infor = np.concatenate(all_mean_pub_date_infor_list, axis=0)
                np.save(file_path_all_pub_date_infor, all_mean_pub_date_infor)
            else: all_mean_pub_date_infor = np.load(file_path_all_pub_date_infor, allow_pickle=True)
            return all_mean_pub_date_infor
    
    def CCN_community_mean_time_diffs(self, year_diff_mode: bool = True, show_distribution: bool = False):
        file_path_mean_year_diffs = op.join(self.path_manager.external_file_dir, 'CCNs', 'CCNs_community_pub_date', 'CCN_community_mean_year_diffs.npy')
        file_path_date_year_diffs = op.join(self.path_manager.external_file_dir, 'CCNs', 'CCNs_community_pub_date', 'CCN_community_mean_date_diffs.npy')
        if not op.exists(file_path_mean_year_diffs) or not op.exists(file_path_date_year_diffs):
            df_all_mean_pub_date_infor = pd.DataFrame(self.CCN_community_pub_date(generating=False)[:, [0, 2, 3]], columns=['author_id', 'mean_pub_date', 'mean_pub_year'])
            df_all_mean_pub_date_infor['mean_pub_date'] = pd.to_datetime(df_all_mean_pub_date_infor.mean_pub_date)

            grouped_df_year = df_all_mean_pub_date_infor.groupby('author_id')['mean_pub_year'].agg(list).reset_index()
            all_year_diffs = grouped_df_year['mean_pub_year'].apply(self.calculate_diffs_with_year).tolist()
            time_diffs_year = [item for sublist in all_year_diffs for item in sublist]
            self.path_manager.save_npy_file(variable=np.array(time_diffs_year), abs_file_path=file_path_mean_year_diffs, override=True)

            grouped_df_date = df_all_mean_pub_date_infor.groupby('author_id', group_keys=False).apply(lambda x: x.sort_values('mean_pub_date'))[['author_id','mean_pub_date']]
            grouped_df_date = grouped_df_date.groupby('author_id')['mean_pub_date'].agg(list).reset_index()
            all_date_diffs = grouped_df_date['mean_pub_date'].apply(self.calculate_diffs_with_date).tolist()
            time_diffs_date = [item for sublist in all_date_diffs for item in sublist]
            self.path_manager.save_npy_file(variable=np.array(time_diffs_date), abs_file_path=file_path_date_year_diffs, override=True)
        else: 
            target_file_path = file_path_mean_year_diffs if year_diff_mode else file_path_date_year_diffs
            time_diffs = np.load(target_file_path)
            if show_distribution:
                lower_upper_dict = self.time_diff_pdf_two_side_quantiles(middle_high_frequency_ratio = 0.3) # make sure relevant function results are complete
                self.visualization.time_diff_pdf_plot(data = time_diffs, lower_upper_dict = lower_upper_dict)
            return time_diffs

    def CCN_community_mean_time_diff_quantiles(self, year_diff_mode: bool = True):
        df_path_mean_year_diff_quantiles = op.join(self.path_manager.external_file_dir, 'CCNs', 'CCNs_community_pub_date', 'CCN_community_mean_year_diff_quantiles.csv')
        df_path_mean_date_diff_quantiles = op.join(self.path_manager.external_file_dir, 'CCNs', 'CCNs_community_pub_date', 'CCN_community_mean_date_diff_quantiles.csv')
        if not op.exists(df_path_mean_year_diff_quantiles) or not op.exists(df_path_mean_date_diff_quantiles):
            percentiles = [float('{:g}'.format(num_)) for num_ in np.linspace(0.001, 0.999, num=999)]

            logger.info(f'generating {df_path_mean_year_diff_quantiles}')
            time_diffs_year = self.CCN_community_mean_time_diffs(year_diff_mode=True)
            df_time_diff_quantiles_year = pd.DataFrame(zip(percentiles, np.quantile(time_diffs_year, percentiles)), columns=['percentile','quantile'])
            df_time_diff_quantiles_year.to_csv(df_path_mean_year_diff_quantiles, index=False)
            
            logger.info(f'generating {df_path_mean_date_diff_quantiles}')
            time_diffs_date = self.CCN_community_mean_time_diffs(year_diff_mode=False)
            df_time_diff_quantiles_date = pd.DataFrame(zip(percentiles, np.quantile(time_diffs_date, percentiles)), columns=['percentile','quantile'])
            df_time_diff_quantiles_date.to_csv(df_path_mean_date_diff_quantiles, index=False)

        else: df_time_diff_quantiles_year, df_time_diff_quantiles_date = pd.read_csv(df_path_mean_year_diff_quantiles), pd.read_csv(df_path_mean_date_diff_quantiles)
        return df_time_diff_quantiles_year if year_diff_mode else df_time_diff_quantiles_date

    def get_selected_author_CCN_community(self, i_th: int = 0, discipline: str = 'Mathematics', params: Params_community_concept_pairs = Params_community_concept_pairs(), 
                                          visual_params: Params_community_visualization = Params_community_visualization()):
        # community_method: louvain, greedy
        file_path_selected_author_id_fields = op.join(self.path_manager.external_file_dir, 'CCNs', f'selected_author_id_fields_min_community_{params.min_community_size}_{params.select_mode}_cover_{params.work_coverage_ratio}_works_concept_size_{params.min_concept_size}_{params.max_concept_size}.csv')
        if not op.exists(file_path_selected_author_id_fields):
            df_selected_author_data = self.get_desired_author_CCN_community(params) # 采用louvian社团检测, params.min_community_size = 8
            all_selected_author_ids = df_selected_author_data['author_id'].drop_duplicates().tolist()
            df_author_id_fields = self.db_manager.query_table(table_name = 'author_yearlyfeature_field_geq10pubs', columns=['author_id', 'field'], where_conditions=[f'''author_id IN ('{"','".join(all_selected_author_ids)}')'''])
            self.path_manager.save_csv_file(variable=df_author_id_fields, abs_file_path=file_path_selected_author_id_fields, index = False)
        else: df_author_id_fields = pd.read_csv(file_path_selected_author_id_fields)
        selected_author_ids = df_author_id_fields.loc[df_author_id_fields.field==discipline].reset_index(drop=True)['author_id'].tolist()
        for author_id in selected_author_ids[i_th:i_th + 1]:
            # df_author_communities_0 = df_selected_author_data.loc[df_selected_author_data.author_id==author_id]
            CCN_author = self.CCN_generation_of_author(author_id=author_id)
            if CCN_author is not None: 
                params.min_community_size = 2 # in order to ensure select the desired community
                visual_params.G, params.df_author_communities = self.get_author_communities(CCN_author, community_method = params.community_method, min_community_size = params.min_community_size)
                visual_params.representive_concepts, df_concept_pairs, _, works_concepts_table = self.get_CCN_temporal_community_pairs(params)

                # number of pubs by communities
                works_concepts_table['community_order'] = works_concepts_table.work_id.map(nx.get_node_attributes(visual_params.G, 'community_order'))
                works_concepts_table['publication_year'] = works_concepts_table.work_id.map(nx.get_node_attributes(visual_params.G, 'publication_year'))
                dn_pubs_communities = []
                for community_order, table_community in works_concepts_table.groupby('community_order'):
                    df_n_pubs_community = table_community['publication_year'].value_counts().reset_index()
                    df_n_pubs_community.columns = ['pub_year', 'n_pubs']
                    df_n_pubs_community['community_order'] = community_order
                    dn_pubs_communities.append(df_n_pubs_community)
                df_n_pubs_communities = pd.concat(dn_pubs_communities)
                representive_concepts_dict = dict([[i, concept_list] for i, concept_list in enumerate(visual_params.representive_concepts)])
                df_n_pubs_communities['representative_concepts'] = df_n_pubs_communities.community_order.map(representive_concepts_dict)
                if visual_params.show_n_pubs_by_community: self.visualization.bar_n_pubs_by_community(df_n_pubs_communities, labels = visual_params.representive_concepts)

                research_areas = list(set(sum(visual_params.representive_concepts, [])))
                visual_params.mean_pub_years = params.df_author_communities.mean_pub_year.round(3).tolist()
                visual_params.file_name = f'{author_id}_CCN_{params.community_method}_min_community_size_{params.min_community_size}'
                self.path_manager.save_csv_file(variable = df_concept_pairs, abs_file_path = op.join(self.path_manager.concepts_dir, f'{author_id}_concept_pairs.csv'), index=True, override = False)
                if visual_params.show_community: self.visualization.draw_community(visual_params)
                return df_concept_pairs
            else: print(f'CCN for author {author_id} is none')

    def get_desired_author_CCN_community(self, params: Params_community_concept_pairs = Params_community_concept_pairs()):
        file_dir = self.path_manager.ensure_folder_exists(op.join(self.path_manager.external_file_dir, 'CCNs', 'CCNs_community_desired_author'))
        file_path_desired_author_CCN_data = op.join(file_dir, f'desired_author_CCN_min_community_{params.min_community_size}_{params.select_mode}_cover_{params.work_coverage_ratio}_works_concept_size_{params.min_concept_size}_{params.max_concept_size}.npy')
        if not op.exists(file_path_desired_author_CCN_data):
            df_all_mean_pub_date_infor = pd.DataFrame(self.CCN_community_pub_date(generating=False), columns = ['author_id', 'community', 'mean_pub_date', 'mean_pub_year'])
            logging.info(f'initiate generating {file_path_desired_author_CCN_data}')
            author_community_data = []
            for _, params.df_author_communities in tqdm.tqdm(df_all_mean_pub_date_infor.groupby('author_id')):
                if params.df_author_communities.shape[0] < params.min_community_size: continue # ensure CCN of each author contains at least min_community_size = 6 communites
                author_representive_concepts, _, df_concept_community_nodes_coverage_ratio, _ = self.get_CCN_temporal_community_pairs(params, show_runtime=False)
                if any(((len(sublist) < params.min_concept_size) or (len(sublist) > params.max_concept_size)) for sublist in author_representive_concepts): continue # ensure each sublist contains at leatst one element
                params.df_author_communities['representative_concepts'] = author_representive_concepts
                author_community_data.append(params.df_author_communities[['author_id', 'mean_pub_year','representative_concepts']].values)
            all_author_community_data = np.concatenate(author_community_data)
            self.path_manager.save_npy_file(variable=all_author_community_data, abs_file_path=file_path_desired_author_CCN_data, override = True)
        else: all_author_community_data = np.load(file_path_desired_author_CCN_data, allow_pickle = True)
        df_all_author_community_data = pd.DataFrame(all_author_community_data, columns = ['author_id', 'mean_pub_year','representative_concepts'])
        return df_all_author_community_data
    
    def CCN_community_nodes_coverage_ratio_of_concepts(self, params: Params_community_concept_pairs = Params_community_concept_pairs(), top: int = 0, concate:bool = False, show_pdf:bool = False):
        file_dir = self.path_manager.ensure_folder_exists(op.join(self.path_manager.external_file_dir, 'CCNs', 'CCNs_communities'))
        file_path_coverage_ratio_data = op.join(file_dir, f"CCN_community_nodes_coverage_ratio{f'_top_{top}' if top>0 else ''}.npy")
        file_path_n_communities = op.join(file_dir, f"all_CCN_n_communities{f'_top_{top}' if top>0 else ''}.npy")
        if not op.exists(file_path_coverage_ratio_data) or not op.exists(file_path_n_communities):
            df_all_mean_pub_date_infor = pd.DataFrame(self.CCN_community_pub_date(generating=False), columns = ['author_id', 'community', 'mean_pub_date', 'mean_pub_year'])
            if top > 0: df_all_mean_pub_date_infor = df_all_mean_pub_date_infor[:top]
            logging.info(f'initiate generating {file_path_coverage_ratio_data}')
            n_communities, all_community_nodes_coverage_ratio = [], []
            for author_id, params.df_author_communities in tqdm.tqdm(df_all_mean_pub_date_infor.groupby('author_id')):
                n_communities.append([f'{author_id}', params.df_author_communities.shape[0]])
                if params.df_author_communities.shape[0] < params.min_community_size: continue # ensure CCN of each author contains at least min_community_size = 6 communites
                _, _, df_concept_community_nodes_coverage_ratio, _ = self.get_CCN_temporal_community_pairs(params, show_runtime=False)
                all_community_nodes_coverage_ratio.append(df_concept_community_nodes_coverage_ratio)
            df_all_community_nodes_coverage_ratio = pd.concat(all_community_nodes_coverage_ratio, ignore_index=True)
            self.path_manager.save_npy_file(variable=df_all_community_nodes_coverage_ratio.values, abs_file_path=file_path_coverage_ratio_data)
            self.path_manager.save_npy_file(variable=np.array(n_communities, dtype=object), abs_file_path=file_path_n_communities)
        else: file_path_coverage_ratio_data = np.load(file_path_coverage_ratio_data, allow_pickle = True)
        df_all_community_nodes_coverage_ratio = pd.DataFrame(file_path_coverage_ratio_data, columns = [f'{i}_th' for i in range(1, 6)] + [f'top_{i}' for i in range(1, 6)] + ['author_id'])
        file_path_probability_of_community_top_n_concepts_coverage = op.join(file_dir, f"probability_of_community_top_n_concepts_coverage{f'_top_{top}' if top>0 else ''}.csv")
        if not op.exists(file_path_probability_of_community_top_n_concepts_coverage):
            percentiles = [float('{:g}'.format(num_)) for num_ in np.linspace(0.4, 0.95, num=12)]
            df_probability_of_community_top_n_concepts_coverage = pd.DataFrame(np.full((5, len(percentiles) + 1), None, dtype=object), columns = ['coverage'] + percentiles)
            for i in range(5):
                coverage_ratios = df_all_community_nodes_coverage_ratio[[f'top_{i+1}']][df_all_community_nodes_coverage_ratio[f'top_{i+1}'].notna()].values.flatten()
                coverage_counts = np.array([(coverage_ratios >= p_).sum() for p_ in percentiles])
                df_probability_of_community_top_n_concepts_coverage.iloc[i] = [f'Top {i+1}'] + (coverage_counts / coverage_ratios.shape[0]).round(3).tolist()
            self.path_manager.save_csv_file(variable=df_probability_of_community_top_n_concepts_coverage, abs_file_path=file_path_probability_of_community_top_n_concepts_coverage, override=True)
        else: df_probability_of_community_top_n_concepts_coverage = pd.read_csv(file_path_probability_of_community_top_n_concepts_coverage)
        if concate:
            dfs = []
            for column in [f'top_{i}' for i in range(1, 6)]:
                df_column = pd.DataFrame({'Category': column.capitalize().replace('_',' '), 'coverage':df_all_community_nodes_coverage_ratio[column].dropna()})
                dfs.append(df_column)
            df_data = pd.concat(dfs)
            if show_pdf: 
                self.visualization.multiple_pdf_plot_concate(df_data)
            return df_all_community_nodes_coverage_ratio, df_data
        else: 
            if show_pdf: 
                df_top = df_all_community_nodes_coverage_ratio[[f'top_{i}' for i in range(1, 6)]]
                self.visualization.multiple_pdf_plot(df_top)
            return df_all_community_nodes_coverage_ratio
        
    def all_CCN_n_communities(self, top: int = 0, first_n: int = 12, show_n_community_bar: bool = False):
        # prerequisite function: self.CCN_community_nodes_coverage_ratio_of_concepts()
        file_dir = self.path_manager.ensure_folder_exists(op.join(self.path_manager.external_file_dir, 'CCNs', 'CCNs_communities'))
        file_path_n_communities = op.join(file_dir, f"all_CCN_n_communities{f'_top_{top}' if top>0 else ''}.npy")
        n_communities = np.load(file_path_n_communities, allow_pickle = True)
        unique_values, counts = np.unique(n_communities[:,1], return_counts=True)
        value_strs = [str(num_) for num_ in unique_values[:first_n]] + [f'{unique_values[first_n]}+']
        counts_ratio = np.append(counts[:first_n], counts[first_n:].sum()) / counts.sum()
        if show_n_community_bar:
            self.visualization.bar_n_communities(value_strs, counts_ratio)

    def all_CCN_communitiies_node_size(self, top: int = 0, first_n: int = 20, show_communitiies_node_size_bar: bool = False):
        file_dir = self.path_manager.ensure_folder_exists(op.join(self.path_manager.external_file_dir, 'CCNs', 'CCNs_communities'))
        file_path_communitiies_node_size = op.join(file_dir, f"all_CCN_communitiies_node_size{f'_top_{top}' if top>0 else ''}.npy")
        if not op.exists(file_path_communitiies_node_size):
            df_all_mean_pub_date_infor = pd.DataFrame(self.CCN_community_pub_date(generating=False)[:,:2], columns = ['author_id', 'community'])
            if top > 0: df_all_mean_pub_date_infor = df_all_mean_pub_date_infor[:top]
            logging.info(f'initiate generating {file_path_communitiies_node_size}')
            df_all_mean_pub_date_infor['community_size'] = df_all_mean_pub_date_infor['community'].apply(len)
            self.path_manager.save_npy_file(variable=df_all_mean_pub_date_infor[['author_id','community_size']].values, abs_file_path=file_path_communitiies_node_size, override=True)
        else: communitiies_node_size = np.load(file_path_communitiies_node_size, allow_pickle = True)
        if show_communitiies_node_size_bar:
            unique_values, counts = np.unique(communitiies_node_size[:,1], return_counts=True)
            value_strs = [str(num_) for num_ in unique_values[:first_n]] + [f'{unique_values[first_n]}+']
            counts_ratio = np.append(counts[:first_n], counts[first_n:].sum()) / counts.sum()
            self.visualization.bar_communities_node_size(value_strs, counts_ratio)
        return communitiies_node_size

    def CCN_community_temporal_concept_flow_matrix(self, params: Params_community_concept_pairs):
        # CCN_community_temporal_concept_flow_matrix(self, concept_level = 1, less_than: bool = True, select_mode: str = 'respective', work_coverage_ratio: float = 0.8, top_ratio: float = 0.3):
        # all_mean_pub_date_infor = self.CCN_community_pub_date(generating=False) # generating=False means to directly read from stored file
        # df_concept_flow_matrix_path = op.join(self.path_manager.concepts_dir, f"concept_flow_matrix_old.csv")
        if params.select_mode == 'respective' or params.select_mode == 'collective':
            file_name_end = f'{params.select_mode}_cover_{params.work_coverage_ratio}_works'
        elif params.select_mode == 'cumulative':
            file_name_end = f'{params.select_mode}_top_{params.top_ratio}_works'
        elif params.select_mode == 'all':
            file_name_end = 'cover_all_community_concepts'
        else: raise ValueError('Undefined select mode')
        params_str = f"level_{'lt' if params.less_than else 'eq'}_{params.concept_level}_{file_name_end}"
        df_concept_flow_matrix_path = op.join(self.path_manager.concepts_dir, f"concept_flow_matrix_{params_str}.csv")
        if not op.exists(df_concept_flow_matrix_path):
            logger.info(f'generating file {df_concept_flow_matrix_path}')
            df_all_mean_pub_date_infor = pd.DataFrame(self.CCN_community_pub_date(generating=False), columns = ['author_id', 'community', 'mean_pub_date', 'mean_pub_year'])
            concept_pairs_all_authors = []
            for _, params.df_author_communities in tqdm.tqdm(df_all_mean_pub_date_infor.groupby('author_id')):
                # it's important to pass the relevant parameters, or the result will be different
                if params.df_author_communities.shape[0] < 2: continue
                _, df_concept_pairs,_, _ = self.get_CCN_temporal_community_pairs(params, show_runtime=False)
                if df_concept_pairs.shape[0] > 0:
                    concept_pairs_all_authors.extend(df_concept_pairs.concept_pair.tolist())
            concepts_table = Concept.discipline_category_classification() # or directly read from database: self.db_manager.query_table(table_name = 'concepts', columns = ['id', 'display_name', 'level'])
            concepts_required = concepts_table.loc[(concepts_table.level <= params.concept_level) if params.less_than else (concepts_table.level == params.concept_level)][['discipline_category_refined', 'level', 'display_name']]
            concepts_required = concepts_required.sort_values(by=['discipline_category_refined', 'display_name'])
            concepts_index = concepts_required.display_name.drop_duplicates().tolist()
            df_concept_flow_matrix = pd.DataFrame(0, index = concepts_index, columns = concepts_index)
            value_counts = pd.DataFrame(concept_pairs_all_authors, columns=['source', 'target']).value_counts().reset_index()
            value_counts.columns = ['source', 'target', 'count_value']
            for _, row in value_counts.iterrows():
                df_concept_flow_matrix.loc[row['source'], row['target']] = row['count_value']
            df_concept_flow_matrix.to_csv(df_concept_flow_matrix_path, index = True)
            logging.info(f'{df_concept_flow_matrix_path} cached')
        else: df_concept_flow_matrix = pd.read_csv(df_concept_flow_matrix_path, index_col=0)
        return df_concept_flow_matrix, params_str
    
    def matrix_normalization(self, matrix, row_normalize: bool = True):
        if row_normalize: # 按行归一化, 即归一化后行和为0
            normalized_matrix = matrix / np.expand_dims(np.sum(matrix, axis=1), axis=1)
        else: normalized_matrix = matrix / np.sum(matrix, axis=0) # 按列归一化, 即归一化后列和为0
        return normalized_matrix
    
    def retain_greater_matrix_diagonal_elements(self, matrix):
        matrix_tril, matrix_triu = np.tril(matrix, k=-1), np.triu(matrix, k=1), 
        matrix_maximum_symmetric = matrix_tril*(matrix_tril > matrix_triu.T) + matrix_triu*(matrix_tril.T <= matrix_triu)
        return matrix_maximum_symmetric
    
    def retain_top_cum_ratio_biggest_elements(self, matrix, row_normalize: bool = True, matrix_top_ratio: float = 1):
        # 最大值元素累计占比达 matrix_top_ratio 最小元素个数
        if matrix_top_ratio <= 0 or matrix_top_ratio > 1: raise ValueError('Unexpected top_ratio')
        if 0 < matrix_top_ratio < 1: 
            matrix_temp = matrix.copy() if row_normalize else matrix.copy().T
            for i, row_i in enumerate(matrix_temp):
                sorted_indices = np.argsort(row_i)[::-1] # 值从大到小排序后，其值对应原来未排序的index
                cumulative_sum = np.nan_to_num(np.cumsum(row_i[sorted_indices]) / np.sum(row_i))
                indexes = np.argwhere(cumulative_sum >= matrix_top_ratio).flatten()
                index_min = indexes[0] if indexes.shape[0] > 0 else -1 # 找到第一个累计占比>=阈值的最大值元素
                matrix_temp[i, sorted_indices[index_min + 1:]] = 0 # 将 index_min 之后的元素置为 0
            return np.round(matrix_temp if row_normalize else matrix_temp.T, 6)
        else: return np.round(matrix, 6)

    def concept_flow_matrix_analysis(self, params: Params_community_concept_pairs = Params_community_concept_pairs(), 
                                     concept_flow_params: Params_concept_flow_matrix = Params_concept_flow_matrix(),):
        df_concept_flow_matrix, params_str = self.CCN_community_temporal_concept_flow_matrix(params)
        concepts_table = Concept.discipline_category_classification()
        concepts_table = concepts_table.loc[concepts_table.display_name.isin(df_concept_flow_matrix.index)][['display_name','all_level_0_ancestors', 'level']]
        if len(concept_flow_params.disciplines) < 1: 
            source_concepts = target_concepts = concepts_table.display_name.tolist()
        elif len(concept_flow_params.disciplines) == 1: 
            source_concepts = target_concepts = concepts_table.loc[concepts_table.all_level_0_ancestors.apply(lambda ancestors: concept_flow_params.disciplines[0] in ancestors)].display_name.tolist()
        elif len(concept_flow_params.disciplines) == 2:
            source_concepts = concepts_table.loc[concepts_table.all_level_0_ancestors.apply(lambda ancestors: concept_flow_params.disciplines[0] in ancestors)].display_name.tolist()
            target_concepts = concepts_table.loc[concepts_table.all_level_0_ancestors.apply(lambda ancestors: concept_flow_params.disciplines[1] in ancestors)].display_name.tolist()
        else: raise ValueError('Undefined for discipline more than 2')
        df_sub_concept_flow_matrix = df_concept_flow_matrix.loc[source_concepts, target_concepts]
        
         # set diagonal to be 0 to exclude cyclic edges
        row_names, column_names = df_sub_concept_flow_matrix.index.tolist(), df_sub_concept_flow_matrix.columns.tolist()
        common_elements = list(set(row_names).intersection(column_names))
        # directed edge
        common_elements_symmetric_matrix = df_sub_concept_flow_matrix.loc[common_elements, common_elements].values
        df_sub_concept_flow_matrix.loc[common_elements, common_elements] = self.retain_greater_matrix_diagonal_elements(common_elements_symmetric_matrix)
        
        # originally without self loop, but discipline may be developed from the discipline itself, hence allow self loop
        # for concept_ in common_elements: 
        #     df_sub_concept_flow_matrix.loc[concept_, concept_] = 0
        
        if concept_flow_params.matrix_imshow: self.visualization.matrix_imshow(matrix=df_sub_concept_flow_matrix.values, row_names = row_names, column_names=column_names)

        if concept_flow_params.matrix_filter: 
            filtered_matrix = self.get_threshold_matrix(matrix=df_sub_concept_flow_matrix.values, row_normalize = concept_flow_params.row_normalize, 
                                                    matrix_value_cum_sum_ratio = concept_flow_params.matrix_value_cum_sum_ratio, matrix_value_count_cum_sum_ratio = concept_flow_params.matrix_value_count_cum_sum_ratio)
        else: filtered_matrix = self.get_threshold_matrix_old(matrix = df_sub_concept_flow_matrix.values, row_normalize = concept_flow_params.row_normalize, matrix_top_ratio = concept_flow_params.matrix_top_ratio)

        if concept_flow_params.matrix_imshow: self.visualization.matrix_imshow(matrix=filtered_matrix, row_names = row_names, column_names=column_names)

        co_x, co_y = np.where(filtered_matrix > 0)
        row_names_filter, column_names_filter = df_sub_concept_flow_matrix.index[co_x].tolist(), df_sub_concept_flow_matrix.columns[co_y].tolist()
        edges = np.transpose((row_names_filter, column_names_filter, filtered_matrix[co_x, co_y]))
        print(edges.shape)
        G_concept_flow = nx.DiGraph()
        graph_concepts = list(set(df_sub_concept_flow_matrix.index[co_x]).union(df_sub_concept_flow_matrix.columns[co_y]))
        level_dict = dict(concepts_table.loc[concepts_table.display_name.isin(graph_concepts)][['display_name','level']].values)
        G_concept_flow.add_weighted_edges_from(edges)
        nx.set_node_attributes(G_concept_flow, level_dict, 'level') # for hierarchical layout subset key

        if concept_flow_params.save_G_concept_flow: 
            if concept_flow_params.matrix_filter:
                gexf_file_name =  f"G_concept_flow_{'_'.join(concept_flow_params.disciplines).replace(' ','_')}_{'row' if concept_flow_params.row_normalize else 'column'}_norm_matrix_filter_value_cum_{concept_flow_params.matrix_value_cum_sum_ratio}_count_cum_{concept_flow_params.matrix_value_count_cum_sum_ratio}_{params_str}.gexf"
            else: gexf_file_name = f"G_concept_flow_{'_'.join(concept_flow_params.disciplines).replace(' ','_')}_{'row' if concept_flow_params.row_normalize else 'column'}_norm_top_{concept_flow_params.matrix_top_ratio}_{params_str}.gexf"
            gexf_file_path = op.join(self.path_manager.ensure_folder_exists(op.join(self.path_manager.concepts_dir, 'Networks')), gexf_file_name)
            self.path_manager.save_gexf_file(G = G_concept_flow, abs_file_path=gexf_file_path, override=True)
        if concept_flow_params.show: self.visualization.draw_concept_flow_network(G_concept_flow, row_normalize = concept_flow_params.row_normalize, layout=concept_flow_params.layout)
        return G_concept_flow
    