
import os, sys
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.append(project_dir)
import os.path as op
import numpy as np
import pandas as pd
import networkx as nx
import tqdm
from itertools import combinations
import logging
from utils.config import PathManager, calculate_runtime
from utils.visualization import Visualizer
from network.ccn import CCN

logger = logging.getLogger(__name__)

class APYDAnalyzer:
    """
    Class for analyzing temporal differences between communities
    
    Features include:
    - Calculating average publication dates for communities
    - Computing Average Publication Year Differences (APYD) between communities
    - Analyzing distributions of APYDs
    - Visualizing APYDs
    """
    
    def __init__(self, visualization_manager = None):
        """
        Initialize the community temporal analyzer
        
        Parameters:
        path_manager: Manager for file paths and directory creation
        visualization_manager: Manager for visualization functions
        """
        self.path_manager = PathManager()
        self.CCN = CCN()
        self.visualization = Visualizer() if visualization_manager == None else visualization_manager
    
    def calculate_APYD(self, year_diff_mode=True, show_distribution=False, fig_size=(8, 6), save_pdf=False):
        """
        Calculate and analyze Average Publication Year Differences (APYD) between communities
        Calculate and analyze Average Publication Date Differences (APPD) between communities
        
        Parameters:
        year_diff_mode: Whether to use year differences (True) or date differences (False)
        show_distribution: Whether to show the distribution of APYDs
        fig_size: Size of the figure for visualization
        save_pdf: Whether to save the visualization as PDF
        
        Returns:
        Average Publication Year Differences between communities
        """
        file_path_year_diffs = op.join(self.path_manager.external_file_dir, 'CCNs', 'CCNs_community_pub_date', 'CCN_community_APYD.npy')
        file_path_date_diffs = op.join(self.path_manager.external_file_dir, 'CCNs', 'CCNs_community_pub_date', 'CCN_community_APPD.npy')
        
        if not op.exists(file_path_year_diffs) or not op.exists(file_path_date_diffs):
            # Calculate and save APYDs if they don't exist
            df_all_mean_pub_date_info = pd.DataFrame(
                self.CCN.load_author_community_info()[:, [0, 3, 4]], 
                columns=['author_id', 'mean_pub_date', 'mean_pub_year']
            )
            df_all_mean_pub_date_info['mean_pub_date'] = pd.to_datetime(df_all_mean_pub_date_info.mean_pub_date)

            # Calculate year differences
            grouped_df_year = df_all_mean_pub_date_info.groupby('author_id')['mean_pub_year'].agg(list).reset_index()
            all_year_diffs = grouped_df_year['mean_pub_year'].apply(self.calculate_year_differences).tolist()
            APYD_values = [item for sublist in all_year_diffs for item in sublist]
            self.path_manager.save_npy_file(variable=np.array(APYD_values), abs_file_path=file_path_year_diffs, override=True)

            # Calculate date differences
            grouped_df_date = df_all_mean_pub_date_info.groupby('author_id', group_keys=False).apply(
                lambda x: x.sort_values('mean_pub_date')
            )[['author_id','mean_pub_date']]
            grouped_df_date = grouped_df_date.groupby('author_id')['mean_pub_date'].agg(list).reset_index()
            all_date_diffs = grouped_df_date['mean_pub_date'].apply(self.calculate_date_differences).tolist()
            time_diffs_date = [item for sublist in all_date_diffs for item in sublist]
            self.path_manager.save_npy_file(variable=np.array(time_diffs_date), abs_file_path=file_path_date_diffs, override=True)
        else:
            # Load and return APYDs
            target_file_path = file_path_year_diffs if year_diff_mode else file_path_date_diffs
            APYD_values = np.load(target_file_path)
        
        if show_distribution:
            lower_upper_dict = self.get_APYD_quantile_bounds(middle_high_frequency_ratio=0.3)
            self.visualization.time_diff_pdf_plot(data = APYD_values, lower_upper_dict = lower_upper_dict)
        
        return APYD_values
    
    def calculate_year_differences(self, years):
        """Calculate differences between years in a list"""
        return [year2 - year1 for year1, year2 in combinations(years, 2)]
    
    def calculate_date_differences(self, dates):
        """Calculate differences between dates in a list (in years)"""
        return [(date2 - date1).total_seconds() / (3600 * 24 * 365.25) for date1, date2 in combinations(dates, 2)]
    
    def get_APYD_quantile_bounds(self, middle_high_frequency_ratio=0.3, year_diff_mode=True):
        """
        Get the quantile bounds for Average Publication Year Differences
        
        Parameters:
        middle_high_frequency_ratio: Ratio of the middle high frequency region
        year_diff_mode: Whether to use year differences
        
        Returns:
        Dictionary with lower and upper bounds
        """
        APYD_quantiles = dict(self.get_APYD_quantiles(year_diff_mode=year_diff_mode).values)
        percentile_lower = float('{:g}'.format((1-middle_high_frequency_ratio)/2))
        percentile_higher = float('{:g}'.format((1+middle_high_frequency_ratio)/2))
        lower_limit = APYD_quantiles[percentile_lower]
        upper_limit = APYD_quantiles[percentile_higher]
        return {percentile_lower: lower_limit, percentile_higher: upper_limit}
    
    def get_APYD_quantiles(self, year_diff_mode=True):
        """
        Get quantiles of Average Publication Year Differences
        
        Parameters:
        year_diff_mode: Whether to use year differences
        
        Returns:
        DataFrame with percentiles and corresponding quantiles
        """
        df_path_APYD_quantiles = op.join(self.path_manager.external_file_dir, 'CCNs', 'CCNs_community_pub_date', 'CCN_community_APYD_quantiles.csv')
        df_path_date_quantiles = op.join(self.path_manager.external_file_dir, 'CCNs', 'CCNs_community_pub_date', 'CCN_community_APPD_quantiles.csv')
        
        if not op.exists(df_path_APYD_quantiles) or not op.exists(df_path_date_quantiles):
            percentiles = [float('{:g}'.format(num_)) for num_ in np.linspace(0.001, 0.999, num=999)]

            # Calculate year difference quantiles
            logger.info(f'generating {df_path_APYD_quantiles}')
            APYD_values = self.calculate_APYD(year_diff_mode=True)
            df_APYD_quantiles = pd.DataFrame(
                zip(percentiles, np.quantile(APYD_values, percentiles)), 
                columns=['percentile', 'quantile']
            )
            df_APYD_quantiles.to_csv(df_path_APYD_quantiles, index=False)
            
            # Calculate date difference quantiles
            logger.info(f'generating {df_path_date_quantiles}')
            time_diffs_date = self.calculate_APYD(year_diff_mode=False)
            df_APPD_quantiles = pd.DataFrame(
                zip(percentiles, np.quantile(time_diffs_date, percentiles)), 
                columns=['percentile', 'quantile']
            )
            df_APPD_quantiles.to_csv(df_path_date_quantiles, index=False)
        else:
            df_APYD_quantiles = pd.read_csv(df_path_APYD_quantiles)
            df_APPD_quantiles = pd.read_csv(df_path_date_quantiles)
            
        return df_APYD_quantiles if year_diff_mode else df_APPD_quantiles

if __name__ == "__main__":
    # Example usage of the APYDAnalyzer class
    import argparse
    analyzer = APYDAnalyzer()
    parser = argparse.ArgumentParser(description='Process some integers.')

    # 添加参数
    parser.add_argument('--i', type=int, default=0, help='The iteration index')

    # 解析参数
    args = parser.parse_args()
    
    # nohup python ./network/apyd.py --i 2 >> apyd_progress.log 2>&1 &

    analyzer = APYDAnalyzer()
    analyzer.get_APYD_quantiles()

    # time_diffs = analyzer.calculate_APYD(year_diff_mode=True, show_distribution=True)
    # print(f"Found {len(time_diffs)} time differences between communities")
    # print(f"Average APYD: {np.mean(time_diffs):.2f} years")
