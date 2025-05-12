import os, sys
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.append(project_dir)


import os
import sys
import os.path as op
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from utils.config import PathManager, calculate_runtime
import matplotlib.ticker as ticker
import matplotlib.cm as cm
from utils.database import DatabaseManager


class PeriodEvaluator:
    def __init__(self, sub_concepts=False):
        self.path_manager = PathManager()
        self.db_manager = DatabaseManager()
        self.sub_concepts = sub_concepts
        if sub_concepts:
            self.kpm_path = self.path_manager.ensure_folder_exists(op.join(self.path_manager.ccns_dir, 'kpms_sub'))
            self.evaluation_path = self.path_manager.ensure_folder_exists(op.join(self.path_manager.ccns_dir, 'evaluation_sub'))
        else:
            self.kpm_path = self.path_manager.ensure_folder_exists(op.join(self.path_manager.ccns_dir, 'kpms'))
            self.evaluation_path = self.path_manager.ensure_folder_exists(op.join(self.path_manager.ccns_dir, 'evaluation'))
        self.evaluation_path_period = self.path_manager.ensure_folder_exists(op.join(self.path_manager.ccns_dir, 'evaluation_period'))
        
        # Define default WCR values list
        self.wcr_list = [1, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6]
        # Results save path
        self.results_path = self.path_manager.ensure_folder_exists(op.join(self.path_manager.ccns_dir, 'analysis'))
        if sub_concepts:
            self.parquet_path = op.join(self.results_path, 'all_authors_precision_sub_concepts.parquet')
        else: self.parquet_path = op.join(self.results_path, 'all_authors_precision.parquet')
        self.period_parquet_path = op.join(self.results_path, 'all_authors_period_precision.parquet')
        
    @calculate_runtime
    def collect_all_precision_data_period(self, batch_range=(0, 339), test = False, save_df=True):
        """
        Collect precision data from all batch files or load from existing Parquet file
        
        Args:
            batch_range: Range of batch IDs, default (0, 339)
            save_df: Whether to save DataFrame locally
            
        Returns:
            DataFrame containing all authors' precision data
        """
        if test:
            self.period_parquet_test_path =  op.join(self.results_path, 'all_authors_period_test_precision.parquet')
            if not op.exists(self.period_parquet_test_path):
                df = pd.read_parquet(self.period_parquet_path)
                df_test = df[:5000000]
                df_test.to_parquet(self.period_parquet_test_path, index=False)
            else: df_test = pd.read_parquet(self.period_parquet_test_path)
            return df_test
        # Check if the data already exists in Parquet format
        if op.exists(self.period_parquet_path):
            print(f"Loading existing data from {self.period_parquet_path}")
            df = pd.read_parquet(self.period_parquet_path)
            return df
        
        # If no existing data, collect from batch files
        all_data = []
        # Iterate through all batches in the specified range (inclusive of start, exclusive of end)
        for batch_id in tqdm(range(batch_range[0], batch_range[1]), desc="Processing batches"):
            file_path = op.join(self.evaluation_path_period, f"author_evaluation_results_{batch_id}.pkl")
            skiped_author_path = op.join(self.evaluation_path_period, f'skipped_authors_{batch_id}.pkl')
            # Check if file exists
            if not op.exists(file_path):
                print(f"Warning: File {file_path} does not exist. Skipping.")
                continue
                
            # Load batch data
            try:
                with open(file_path, 'rb') as f:
                    batch_data = pickle.load(f)
                
                for author_id, wcr_dict in batch_data.items():
                    rows = [
                        {
                            'author_id': author_id,
                            'wcr': wcr,
                            'period': period,
                            **{
                                metric: wcr_dict.get(wcr, {}).get(period, {}).get(metric, None)
                                for metric in ['Precision', 'SLC', 'TP', 'FP', 'TotalPairs']
                            }
                        }
                        for wcr in self.wcr_list
                        for period in ['early', 'middle', 'later']
                    ]
                    all_data.extend(rows)
                            
            except Exception as e:
                print(f"Error processing batch {batch_id}: {str(e)}")
        
        # Check if we collected any data
        if not all_data:
            print("Warning: No data was collected from any batch file.")
            return pd.DataFrame()  # Return empty DataFrame
        
        # Create DataFrame
        df = pd.DataFrame(all_data)
        
        # Save DataFrame in Parquet format
        if save_df:
            df.to_parquet(self.period_parquet_path, index=False)
            print(f"DataFrame saved to {self.period_parquet_path} in Parquet format")

        return df

    def plot_precision_by_period_connected(self, df=None, save_fig=True, test=False, fig_size=(10, 4)):
        """
        Create a grouped violin plot to compare precision distribution across early, middle, and later periods
        for different WCR values, with mean values connected by lines within each WCR group
        
        Args:
            df: DataFrame containing precision data, if None will use loaded data
            save_fig: Whether to save the figure
            test: Whether to use test data
            fig_size: Size of the figure as a tuple (width, height)
        """
        if df is None:
            if hasattr(self, 'df'):
                df = self.df
            else:
                df = self.collect_all_precision_data_period(test=test)
        
        # Filter out rows with no data points (TotalPairs = 0)
        filtered_df = df[df['TotalPairs'] > 0].copy()
        periods = ['early', 'middle', 'later']
        color_map = {
            'early': '#6ACC64',
            'middle': '#7570b3',
            'later': '#d95f02'
        }
        # Set figure style and size
        plt.figure(figsize=fig_size)
        sns.set_style("whitegrid")
        
        # Create the grouped violin plot with updated parameter
        ax = sns.violinplot(
            x='wcr', 
            y='Precision', 
            hue='period',
            data=filtered_df,
            palette=color_map,
            split=False,
            inner='quartile',
            cut=0,
            density_norm='width'  # Updated from scale='width'
        )
        
        # Calculate mean values for each wcr and period combination
        mean_df = filtered_df.groupby(['wcr', 'period'])['Precision'].mean().reset_index()
        
        # Get unique wcr values
        wcr_values = sorted(filtered_df['wcr'].unique())
        
        # Extract the positions of violin plots
        # For seaborn violinplot, we need to determine the actual x-coordinates
        # First, get the number of categories on x-axis (wcr values)
        num_wcr = len(wcr_values)
        
        # For each wcr value, connect mean points across periods
        for wcr_idx, wcr in enumerate(wcr_values):
            wcr_means = mean_df[mean_df['wcr'] == wcr]
            
            # For each period, find the x-coordinate in the plot
            x_coords = []
            y_coords = []
            
            # In seaborn violinplot with hue, the x positions are arranged as:
            # [hue1_cat1, hue2_cat1, hue3_cat1, hue1_cat2, hue2_cat2, ...]
            # So we need to calculate the position for each period within each wcr
            for period_idx, period in enumerate(periods):
                period_data = wcr_means[wcr_means['period'] == period]
                if not period_data.empty:
                    # Calculate x position: each wcr has 3 periods side by side
                    # The position is the wcr index plus an offset for the period
                    # The offset depends on how many periods are in each wcr group
                    x_pos = wcr_idx + (period_idx - 1) * (1/3)
                    x_coords.append(x_pos)
                    y_coords.append(period_data['Precision'].values[0])
            
            # Connect the means with a line if we have at least 2 points
            if len(x_coords) > 1:
                plt.plot(x_coords, y_coords, color='black', linestyle='-', linewidth=1, alpha=0.7)
                for i, (x, y) in enumerate(zip(x_coords, y_coords)):
                    plt.text(x, y + 0.03, f"{y:.2f}", ha='center', va='bottom', fontsize=9, color='black')
        # Improve the appearance
        plt.title('Precision Distribution by Period Across Different WCR Values', fontsize=14)
        plt.xlabel('Work Coverage Rate (WCR)', fontsize=14)
        plt.ylabel('Precision', fontsize=14)

        plt.xticks(rotation=0)
        
        plt.tick_params(axis='y', labelsize=12)
        plt.tick_params(axis='x', labelsize=12)

        plt.subplots_adjust(bottom=0.2)
        plt.ylim(-0.1, 1.04)
        
        # Add legend at the bottom in a horizontal layout
        legend_elements = [plt.Line2D([0], [0], color=color_map[periods[i]], lw=4, label=periods[i].capitalize()) for i in range(3)]
        # Add a legend element for the mean connection lines
        legend_elements.append(plt.Line2D([0], [0], color='black', linestyle='-', linewidth=1, label='Period Means'))
        
        # Place legend below the plot but above x-axis labels
        plt.legend(handles=legend_elements, 
                loc='upper center', 
                bbox_to_anchor=(0.5, 0.12),  # Position below the plot
                ncol=4,  # Arrange in one row with 4 columns
                frameon=False,
                fontsize=12)  # Remove legend frame for cleaner look
        
        plt.tight_layout()
        
        # Save the figure if requested
        if save_fig:
            save_path = os.path.join(self.results_path, 'precision_by_period_violin.pdf')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()


    def plot_precision_by_period(self, df=None, save_fig=True, test=False, fig_size=(10, 4)):
        """
        Create a grouped violin plot to compare precision distribution across early, middle, and later periods
        for different WCR values
        
        Args:
            df: DataFrame containing precision data, if None will use loaded data
            save_fig: Whether to save the figure
            test: Whether to use test data
            fig_size: Size of the figure as a tuple (width, height)
        """
        if df is None:
            if hasattr(self, 'df'):
                df = self.df
            else:
                df = self.collect_all_precision_data_period(test=test)
        
        # Filter out rows with no data points (TotalPairs = 0)
        filtered_df = df[df['TotalPairs'] > 0].copy()
        periods = ['early', 'middle', 'later']
        color_map = {
            'early': '#6ACC64',
            'middle': '#7570b3',
            'later': '#d95f02'
        }
        # Set figure style and size
        plt.figure(figsize=fig_size)
        sns.set_style("whitegrid")
        
        # Create the grouped violin plot with updated parameter
        ax = sns.violinplot(
            x='wcr', 
            y='Precision', 
            hue='period',
            data=filtered_df,
            palette=color_map,
            split=False,
            inner='quartile',
            cut=0,
            density_norm='width'  # Updated from scale='width'
        )
        
        # Improve the appearance
        plt.title('Precision Distribution by Period Across Different WCR Values', fontsize=16)
        plt.xlabel('Work Coverage Rate (WCR)', fontsize=14)
        plt.ylabel('Precision', fontsize=14)

        plt.xticks(rotation=0)
        
        plt.tick_params(axis='y', labelsize=12)
        plt.tick_params(axis='x', labelsize=12)

        plt.subplots_adjust(bottom=0.2)
        plt.ylim(-0.1, 1.04)
        # Add legend at the bottom in a horizontal layout
        legend_elements = [plt.Line2D([0], [0], color=color_map[periods[i]], lw=4, label=periods[i].capitalize()) for i in range(3)]
        # Place legend below the plot but above x-axis labels
        plt.legend(handles=legend_elements, 
                # title="Author Discipline", 
                loc='upper center', 
                bbox_to_anchor=(0.5, 0.12),  # Position below the plot
                ncol=4,  # Arrange in one row with 4 columns
                frameon=False,
                fontsize=12)  # Remove legend frame for cleaner look
        
        plt.tight_layout()
        
        # Save the figure if requested
        if save_fig:
            save_path = os.path.join(self.results_path, 'precision_by_period_violin.pdf')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()
    
    def get_diff_data(self, df = None, test = False):
        if df is None:
            df = self.collect_all_precision_data_period(test=test)
        
        # Filter out rows with no data points (TotalPairs = 0)
        filtered_df = df[(df['TP'] + df['FP']) > 0].copy()
        
        # Prepare data for plotting differences
        all_diff_data = {wcr: {period: [] for period in ['early', 'middle', 'later']} for wcr in self.wcr_list}

        # Get unique author_ids and WCR values
        grouped = filtered_df.groupby(['author_id', 'wcr', 'period'])
    
        precision_dict = {}
        for (author_id, wcr, period), group in grouped:
            if wcr in self.wcr_list:  # 只处理我们关心的wcr值
                if author_id not in precision_dict:
                    precision_dict[author_id] = {}
                if wcr not in precision_dict[author_id]:
                    precision_dict[author_id][wcr] = {}
                
                precision_dict[author_id][wcr][period] = group['Precision'].values[0]
        
        for author_id, author_data in precision_dict.items():
            for wcr, wcr_data in author_data.items():
                if 'early' in wcr_data:
                    all_diff_data[wcr]['early'].append(wcr_data['early'])
                    
                    if 'middle' in wcr_data:
                        all_diff_data[wcr]['middle'].append(wcr_data['middle'])
                        
                        # 如果有later数据，添加到later列表
                        if 'later' in wcr_data:
                            all_diff_data[wcr]['later'].append(wcr_data['later'])
        return all_diff_data
        

    def get_diff_data_efficient(self, df=None, test=False):
        df_data_path = op.join(self.results_path, 'all_author_precision_diff_data.csv')
        if not op.exists(df_data_path):
            if df is None:
                if hasattr(self, 'df'):
                    df = self.df
                else:
                    df = self.collect_all_precision_data_period(test=test)
            
            # 过滤有效数据点
            filtered_df = df[(df['TP'] + df['FP']) > 0].copy()
            
            # 使用pivot_table重组数据，使每个作者-wcr组合成为一行，每个period成为一列
            pivot_df = filtered_df.pivot_table(
                index=['author_id', 'wcr'],
                columns='period',
                values='Precision',
                aggfunc='first'  # 每个组合应该只有一个值
            ).reset_index()
            
            complete_data = pivot_df.dropna(subset=['early', 'middle', 'later'])
            
            complete_data.loc[:, 'early_minus_middle'] = complete_data['early'] - complete_data['middle']
            complete_data.loc[:, 'middle_minus_later'] = complete_data['middle'] - complete_data['later']
            
            result = []
            
            for _, row in complete_data.iterrows():
                author_id = row['author_id']
                wcr = row['wcr']
                
                result.append({
                    'wcr': wcr,
                    'comparison': 'Early',
                    'value': row['early'],
                    'author_id': author_id
                })
                result.append({
                    'wcr': wcr,
                    'comparison': 'Early-Middle',
                    'value': row['early_minus_middle'],
                    'author_id': author_id
                })
                result.append({
                    'wcr': wcr,
                    'comparison': 'Middle-Later',
                    'value': row['middle_minus_later'],
                    'author_id': author_id
                })
            
            diff_df = pd.DataFrame(result)
            diff_df.to_csv(df_data_path, index=False)

        else: diff_df = pd.read_csv(df_data_path)

        return diff_df

    def plot_precision_differences_by_period(self, df=None, save_fig=True, test=False, fig_size=(10, 4)):
        """
        Create a grouped violin plot to compare precision differences across periods:
        1. Early precision (raw values)
        2. Early minus Middle precision (difference)
        3. Middle minus Later precision (difference)
        
        Args:
            df: DataFrame containing precision data, if None will use loaded data
            save_fig: Whether to save the figure
            test: Whether to use test data
            fig_size: Size of the figure as a tuple (width, height)
        """

        # diff_df = self.get_diff_data(df, test=test)
        diff_df = self.get_diff_data_efficient(df, test=test)

        # Set up plot
        plt.figure(figsize=fig_size)
        sns.set_style("whitegrid")
        
        # Define colors
        # 科学期刊风格颜色
        color_map = {
            'Early': '#6ACC64',       # 冷蓝色
            'Early-Middle': '#7570b3', # 适中绿色
            'Middle-Later': '#d95f02'  # 柔和红色
        }
        # Create the violin plot
        ax = sns.violinplot(
            x='wcr',
            y='value',
            hue='comparison',
            data=diff_df,
            palette=color_map,
            split=False,
            inner='quartile',
            cut=0,
            density_norm='width'
        )
        
        # Add a horizontal line at y=0 to highlight positive/negative differences
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        
        # Improve appearance
        plt.title('Precision and Precision Differences Across Career Periods', fontsize=14)
        plt.xlabel('Work Coverage Rate (WCR)', fontsize=14)
        plt.ylabel('Precision / Precision Difference', fontsize=14)
        plt.tick_params(axis='both', labelsize=12)
        
        # Calculate and display mean values
        for i, wcr in enumerate(self.wcr_list):
            for j, comp in enumerate(['Early', 'Early-Middle', 'Middle-Later']):
                subset = diff_df[(diff_df['wcr'] == wcr) & (diff_df['comparison'] == comp)]
                if not subset.empty:
                    mean_val = subset['value'].mean()
                    plt.text(i + (j-1)*0.3, mean_val + 0.02, f'{mean_val:.3f}', 
                            ha='center', va='bottom', fontsize=8, color='black')
        
        # Set y-axis limits with some padding
        plt.ylim(-1.2, 1.05)
        
        # Create custom legend
        legend_elements = [
            plt.Line2D([0], [0], color=color_map['Early'], lw=4, label='Early Precision'),
            plt.Line2D([0], [0], color=color_map['Early-Middle'], lw=4, label='Early-Middle Difference'),
            plt.Line2D([0], [0], color=color_map['Middle-Later'], lw=4, label='Middle-Later Difference')
        ]
        
        # Place legend below the plot
        plt.legend(
            handles=legend_elements,
            loc='upper center',
            bbox_to_anchor=(0.5, 0.12),
            ncol=3,
            frameon=False,
            fontsize=12
        )
        
        plt.tight_layout()
        
        # Save the figure if requested
        if save_fig:
            save_path = os.path.join(self.results_path, 'precision_period_differences.pdf')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()

# Usage example
if __name__ == "__main__":
    evaluator = PeriodEvaluator()
    # result_df = evaluator.collect_all_precision_data_period()
    # evaluator.get_diff_data()
    # evaluator.get_diff_data_efficient()
    evaluator.plot_precision_differences_by_period(test=True)

# nohup python ./network/period_evaluation.py >> period_evaluation.log 2>&1 &
# ps aux | grep period_evaluation.py
# pkill -f period_evaluation.py
