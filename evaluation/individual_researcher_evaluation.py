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

class IndividualEvaluator:
    def __init__(self, sub_concepts=False):
        self.path_manager = PathManager()
        self.sub_concepts = sub_concepts
        if sub_concepts:
            self.kpm_path = self.path_manager.ensure_folder_exists(op.join(self.path_manager.ccns_dir, 'kpms_sub'))
            self.evaluation_path = self.path_manager.ensure_folder_exists(op.join(self.path_manager.ccns_dir, 'evaluation_sub'))
        else:
            self.kpm_path = self.path_manager.ensure_folder_exists(op.join(self.path_manager.ccns_dir, 'kpms'))
            self.evaluation_path = self.path_manager.ensure_folder_exists(op.join(self.path_manager.ccns_dir, 'evaluation'))
        
        # Define default WCR values list
        self.wcr_list = [1, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6]
        # Results save path
        self.results_path = self.path_manager.ensure_folder_exists(op.join(self.path_manager.ccns_dir, 'analysis'))
        if sub_concepts:
            self.parquet_path = op.join(self.results_path, 'all_authors_precision_sub_concepts.parquet')
        else: self.parquet_path = op.join(self.results_path, 'all_authors_precision.parquet')
        
    @calculate_runtime
    def collect_all_precision_data(self, batch_range=(0, 339), save_df=True):
        """
        Collect precision data from all batch files or load from existing Parquet file
        
        Args:
            batch_range: Range of batch IDs, default (0, 339)
            save_df: Whether to save DataFrame locally
            
        Returns:
            DataFrame containing all authors' precision data
        """
        
        # Check if the data already exists in Parquet format
        if op.exists(self.parquet_path):
            print(f"Loading existing data from {self.parquet_path}")
            df = pd.read_parquet(self.parquet_path)
            return df
        
        # If no existing data, collect from batch files
        all_data = []
        # Iterate through all batches in the specified range (inclusive of start, exclusive of end)
        for batch_id in tqdm(range(batch_range[0], batch_range[1]), desc="Processing batches"):
            file_path = op.join(self.evaluation_path, f"author_evaluation_results_{batch_id}.pkl")
            
            # Check if file exists
            if not op.exists(file_path):
                print(f"Warning: File {file_path} does not exist. Skipping.")
                continue
                
            # Load batch data
            try:
                with open(file_path, 'rb') as f:
                    batch_data = pickle.load(f)
                
                if self.sub_concepts:
                    for author_id, discipline_dict in batch_data.items():
                        for discipline in ['Computer science', 'Engineering', 'Mathematics', 'Physics']:
                            wcr_dict = discipline_dict[discipline]
                            row = {'author_id': author_id, 'discipline': discipline}
                            for wcr in self.wcr_list:
                                try:
                                    row[f'Precision_{wcr}'] = wcr_dict[wcr]['Precision']
                                    # Optional: collect other metrics
                                    row[f'SLC_{wcr}'] = wcr_dict[wcr]['SLC']
                                    row[f'TP_{wcr}'] = wcr_dict[wcr]['TP']
                                    row[f'FP_{wcr}'] = wcr_dict[wcr]['FP']
                                except (KeyError, TypeError):
                                    row[f'Precision_{wcr}'] = None
                                    row[f'SLC_{wcr}'] = None
                                    row[f'TP_{wcr}'] = None
                                    row[f'FP_{wcr}'] = None

                            all_data.append(row)
                    
                else:
                    # Extract precision data for each author
                    for author_id, wcr_dict in batch_data.items():
                        row = {'author_id': author_id}
                        for wcr in self.wcr_list:
                            try:
                                row[f'Precision_{wcr}'] = wcr_dict[wcr]['Precision']
                                # Optional: collect other metrics
                                row[f'SLC_{wcr}'] = wcr_dict[wcr]['SLC']
                                row[f'TP_{wcr}'] = wcr_dict[wcr]['TP']
                                row[f'FP_{wcr}'] = wcr_dict[wcr]['FP']
                            except (KeyError, TypeError):
                                row[f'Precision_{wcr}'] = None
                        all_data.append(row)
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
            df.to_parquet(self.parquet_path, index=False)
            print(f"DataFrame saved to {self.parquet_path} in Parquet format")

        return df
    
    def plot_precision_distributions(self, df=None, wcr_list=None, figsize=(15, 10), save_fig=True):
        # 如果没有提供 DataFrame，尝试从本地文件加载
        if df is None:
            if op.exists(self.parquet_path):
                df = pd.read_parquet(self.parquet_path)
            else:
                print("No DataFrame provided and no saved file found. Collecting data...")
                df = self.collect_all_precision_data()
        
        # 如果没有指定 WCR 列表，使用默认列表
        if wcr_list is None:
            wcr_list = self.wcr_list
        
        # 计算子图布局
        n_plots = len(wcr_list)
        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        # 创建图形和子图
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True, sharey=True)
        axes = axes.flatten() if n_plots > 1 else [axes]
        
        # 使用不同的颜色
        colors = plt.cm.tab10(np.linspace(0, 1, len(wcr_list)))
        
        # 为每个 WCR 值绘制分布
        for i, wcr in enumerate(wcr_list):
            col = f'Precision_{wcr}'
            if col in df.columns and i < len(axes):
                # 过滤掉 NaN 值
                valid_data = df[col].dropna()
                if len(valid_data) > 0:
                    axes[i].hist(valid_data, bins=100, density=True, alpha=0.7, 
                            color=colors[i], histtype='stepfilled', edgecolor='black')
                    axes[i].set_title(f'WCR = {wcr}', fontsize=12)
                    axes[i].grid(True, linestyle='--', alpha=0.7)
        
        # 隐藏未使用的子图
        for i in range(len(wcr_list), len(axes)):
            fig.delaxes(axes[i])
        
        # 设置共享标签
        # fig.text(0.5, 0.04, 'Precision', ha='center', fontsize=14)
        # fig.text(0.04, 0.5, 'Density', va='center', rotation='vertical', fontsize=14)
        fig.suptitle('Precision Distribution for Different WCR Values', fontsize=16)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        # 保存图形
        if save_fig:
            fig_path = op.join(self.results_path, 'precision_distributions_subplots.png')
            plt.savefig(fig_path, dpi=300)
            print(f"Figure saved to {fig_path}")
        
        # 显示图形
        plt.show()
    
    def get_color(self, i, n = 11):
        r_off, g_off, b_off = 1, 1, 1
        low, high = 0.0, 1.0
        span = high - low
        r = low + span * (((i + r_off) * 3) % n) / (n - 1)
        g = low + span * (((i + g_off) * 5) % n) / (n - 1)
        b = low + span * (((i + b_off) * 7) % n) / (n - 1)
        return (r, g, b)
    
    def plot_precision_cdfs(self, df=None, wcr_list=None, figsize=(7, 7), fontsize = 12, save_fig=True):
        """
        Plot cumulative distribution functions (CDFs) of precision for different WCR values
        """
        # If no DataFrame provided, try to load from local file
        if df is None:
            parquet_path = op.join(self.results_path, 'all_authors_precision.parquet')
            if op.exists(parquet_path):
                df = pd.read_parquet(parquet_path)
            else:
                print("No DataFrame provided and no saved file found. Collecting data...")
                df = self.collect_all_precision_data()
        
        # If no WCR list specified, use default list
        if wcr_list is None:
            wcr_list = self.wcr_list
        
        # Calculate subplot layout
        n_plots = len(wcr_list)
        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        # Create figure and subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True, sharey=True)
        axes = axes.flatten() if n_plots > 1 else [axes]
        
        # Use different colors
        # colors = plt.cm.tab10(np.linspace(0, 1, len(wcr_list)))
        # colors = [
        #     '#ff7f00',  # Orange from 'Oranges'
        #     '#73c2fb',  # Light blue from 'PuBuGn'
        #     '#91cf60',  # Lime green (RdYlGn)
        #     '#e7298a',  # Magenta (Set2)
        #     '#6a3d9a',  # Royal purple (Set1)
        #     '#33a02c',   # Forest green (Set1)
        #     '#4575b4',  # Medium blue
        #     '#fc8d59',  # Salmon/Light orange
        #     '#e6ab02'   # Gold/Yellow
        # ]
        colors = [self.get_color(i) for i in range(len(wcr_list))]
        # colors = [
        #     '#ff7f00',  # Orange from 'Oranges'
        #     '#73c2fb',  # Light blue from 'PuBuGn'
        #     '#91cf60',  # Lime green (RdYlGn)
        #     '#e7298a',  # Magenta (Set2)
        #     '#6a3d9a',  # Royal purple (Set1)
        #     '#33a02c',  # Forest green (Set1)
        #     '#5e3c99',  # Deep purple (PuOr)
        #     '#e6ab02',  # Gold (PiYG)
        #     '#66a61e',  # Olive green (Set2)
        # ]


        def format_number(num):
            """Format number, removing unnecessary zeros"""
            formatted = f"{num:.3f}"
            # If the value is close to zero
            if abs(num) < 0.0005:  # Consider rounding error
                return "0.0"
            # Remove trailing zeros
            while formatted.endswith('0') and '.' in formatted:
                formatted = formatted[:-1]
            # If no digits after decimal point, remove the decimal point
            if formatted.endswith('.'):
                formatted = formatted[:-1]
            return formatted

        # Plot CDF for each WCR value
        for i, wcr in enumerate(wcr_list):
            col = f'Precision_{wcr}'
            if col in df.columns and i < len(axes):
                # Filter out NaN values
                valid_data = df[col].dropna()
                if len(valid_data) > 0:
                    print(f'{col} effective authors {len(valid_data)}')
                    # Calculate and plot cumulative distribution
                    sorted_data = np.sort(valid_data)
                    # Calculate cumulative probabilities (0 to 1)
                    cumulative_prob = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
                    
                    axes[i].plot(sorted_data, cumulative_prob, 
                            color=colors[i], linewidth=2)
                    axes[i].set_title(f'WCR = {wcr}', fontsize=fontsize)
                    axes[i].grid(True, linestyle='--', alpha=0.7)
                    
                    # Add vertical lines for median and mean
                    median = np.median(valid_data)
                    mean = np.mean(valid_data)
                    axes[i].axvline(x=median, color=self.get_color(9), linestyle='--', # 'dodgerblue'
                                label=f'Median: {median:.3f}')
                    axes[i].axvline(x=mean, color=self.get_color(10), linestyle='-.',# 'red'
                                label=f'Mean: {mean:.3f}')
                    
                    # Add horizontal line at 0.5 probability
                    axes[i].axhline(y=0.5, color='gray', linestyle=':', alpha=0.7)
                    axes[i].text(0.3, 0.45, f"Median: {format_number(median)}", transform=axes[i].transAxes,
                                verticalalignment='top', color=self.get_color(9), fontsize=fontsize-1)
                    axes[i].text(0.3, 0.3, f"Mean: {format_number(mean)}", transform=axes[i].transAxes,
                                verticalalignment='top', color=self.get_color(10), fontsize=fontsize-1)
                    axes[i].text(0.3, 0.15, f"Std: {format_number(valid_data.std())}", transform=axes[i].transAxes,
                                verticalalignment='top', fontsize=fontsize-1)
                    # axes[i].text(0.3, 0.3, f"Median: {median:.3f}", transform=axes[i].transAxes,
                    #         verticalalignment='top', color='red', fontsize = fontsize-2)
                    # axes[i].text(0.3, 0.2, f"Mean: {mean:.3f}", transform=axes[i].transAxes,
                    #         verticalalignment='top', color='lime', fontsize = fontsize-2)
                    # axes[i].text(0.3, 0.1, f"Std: {valid_data.std():.3f}", transform=axes[i].transAxes,
                    #         verticalalignment='top', fontsize = fontsize-2)
                    # axes[i].text(0.6, 0.15, f"N: {len(valid_data)}", transform=axes[i].transAxes,
                    #         verticalalignment='top')
                    
                    # # Add key statistics as text
                    # stats_text = (f"Mean: {mean:.3f}\n"
                    #             f"Median: {median:.3f}\n"
                    #             f"Std: {valid_data.std():.3f}\n"
                    #             f"N: {len(valid_data)}")
                    # axes[i].text(0.7, 0.3, stats_text, transform=axes[i].transAxes,
                    #         verticalalignment='top', bbox=dict(boxstyle='round', 
                    #         facecolor='white', alpha=0.8))
                    
                    # Set axis labels
                    axes[i].set_xlabel('Precision', fontsize = fontsize)
                    axes[i].set_ylabel('CDF', fontsize = fontsize)
        
        # Hide unused subplots
        for i in range(len(wcr_list), len(axes)):
            fig.delaxes(axes[i])
        
        # Set overall title
        # fig.suptitle('Precision Cumulative Distribution for Different WCR Values', fontsize=16)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        # Save figure
        if save_fig:
            fig_path = op.join(self.results_path, 'precision_cdf_subplots.pdf')
            plt.savefig(fig_path, bbox_inches='tight')
            print(f"Figure saved to {fig_path}")
        
        # Display figure
        plt.show()

    def plot_precision_boxplots(self, df=None, wcr_list=None, figsize=(15, 8), save_fig=True):
        """
        Compare precision distributions across different WCR values using boxplots
        
        Args:
            df: DataFrame containing precision data, if None will reload
            wcr_list: List of WCR values to plot, if None will use default list
            figsize: Figure size
            save_fig: Whether to save the figure
            
        Returns:
            None, displays the figure directly
        """
        # If no DataFrame provided, try to load from local file
        if df is None:
            csv_path = op.join(self.results_path, 'all_authors_precision.csv')
            if op.exists(csv_path):
                df = pd.read_csv(csv_path)
            else:
                print("No DataFrame provided and no saved file found. Collecting data...")
                df = self.collect_all_precision_data()
        
        # If no WCR list specified, use default list
        if wcr_list is None:
            wcr_list = self.wcr_list
        
        # Prepare boxplot data
        plot_data = []
        for wcr in wcr_list:
            col = f'Precision_{wcr}'
            if col in df.columns:
                valid_data = df[col].dropna()
                for val in valid_data:
                    plot_data.append({'WCR': wcr, 'Precision': val})
        
        plot_df = pd.DataFrame(plot_data)
        
        # Create figure
        plt.figure(figsize=figsize)
        sns.boxplot(x='WCR', y='Precision', data=plot_df, palette='viridis')
        
        plt.xlabel('WCR Value', fontsize=14)
        plt.ylabel('Precision', fontsize=14)
        plt.title('Precision Distribution Comparison Across Different WCR Values', fontsize=16)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save figure
        if save_fig:
            fig_path = op.join(self.results_path, 'precision_boxplots.png')
            plt.savefig(fig_path, dpi=300)
            print(f"Figure saved to {fig_path}")
        
        # Display figure
        plt.show()
    
    def generate_summary_statistics(self, df=None):
        """
        Generate summary statistics for precision across different WCR values
        
        Args:
            df: DataFrame containing precision data, if None will reload
            
        Returns:
            DataFrame containing summary statistics
        """
        # If no DataFrame provided, try to load from local file
        if df is None:
            csv_path = op.join(self.results_path, 'all_authors_precision.csv')
            if op.exists(csv_path):
                df = pd.read_csv(csv_path)
            else:
                print("No DataFrame provided and no saved file found. Collecting data...")
                df = self.collect_all_precision_data()
        
        # Create summary statistics
        stats = []
        for wcr in self.wcr_list:
            col = f'Precision_{wcr}'
            if col in df.columns:
                valid_data = df[col].dropna()
                stats.append({
                    'WCR': wcr,
                    'Count': len(valid_data),
                    'Mean': valid_data.mean(),
                    'Median': valid_data.median(),
                    'Std': valid_data.std(),
                    'Min': valid_data.min(),
                    '25%': valid_data.quantile(0.25),
                    '75%': valid_data.quantile(0.75),
                    'Max': valid_data.max()
                })
        
        stats_df = pd.DataFrame(stats)
        
        # Print summary statistics
        print("Summary Statistics for Precision across different WCR values:")
        print(stats_df.to_string(index=False))
        
        # Save summary statistics
        output_path = op.join(self.results_path, 'precision_statistics.csv')
        stats_df.to_csv(output_path, index=False)
        print(f"Statistics saved to {output_path}")
        
        return stats_df
        
    def analyze_all(self, batch_range=(0, 338)):
        """
        Execute all analyses in one go
        
        Args:
            batch_range: Range of batch IDs, default (0, 338)
            
        Returns:
            None
        """
        print("Step 1: Collecting precision data from all batches...")
        df = self.collect_all_precision_data(batch_range=batch_range)
        
        print("\nStep 2: Generating summary statistics...")
        self.generate_summary_statistics(df)
    

    # def plot_precision_categories(self, percent_df, colors = None):
    #     df = percent_df.copy()
        
    #     # Map row labels to more descriptive English names
    #     row_mapping = {
    #         '=0': '[=0] Zero',
    #         '(0.0-0.2]': '(0.0-0.2] Minimal',
    #         '(0.2-0.4]': '(0.2-0.4] Low',
    #         '(0.4-0.6]': '(0.4-0.6] Moderate',
    #         '(0.6-0.8]': '(0.6-0.8] Good',
    #         '(0.8-1.0]': '(0.8-1.0] High'
    #     }
        
    #     df = df.rename(index=row_mapping)
        
    #     # Transpose DataFrame for plotting
    #     df_plot = df.T
        
    #     # Convert index to float for proper sorting
    #     df_plot.index = df_plot.index.astype(float)
    #     df_plot = df_plot.sort_index(ascending=False)
        
    #     # Set up the plot
    #     plt.figure(figsize=(5, 4))
    #     sns.set_style("whitegrid")
        
    #     # Define colors for different precision categories
    #     if colors is None:
    #         colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33']
        
    #     # Plot each precision category
    #     for i, precision_category in enumerate(df.index):
    #         x_values = df_plot.index.tolist()
    #         y_values = df_plot[precision_category].tolist()
    #         plt.plot(x_values, y_values, marker='o', linewidth=2.5, label=precision_category, color=colors[i])
        
    #     # Add labels and title
    #     plt.xlabel('Work Coverage Ratio (WCR)', fontsize=14)
    #     plt.ylabel('Percentage (%)', fontsize=14)
    #     plt.grid(True, linestyle='--', alpha=0.7)
    #     plt.legend(title='Precision Category', fontsize=10, bbox_to_anchor=(-0.01, 0.8), loc='upper left', title_fontsize=11)
        
    #     # Set x-axis ticks
    #     plt.xticks(df_plot.index)
    #     plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x:.2g}'))

    #     plt.tick_params(axis='x', labelsize=12)
    #     plt.tick_params(axis='y', labelsize=12)
    #     # Add data labels
    #     for category in df.index:
    #         for wcr in df_plot.index:
    #             value = df_plot.loc[wcr, category]
    #             plt.text(wcr, value, f'{value:.2f}'.rstrip('0').rstrip('.'), ha='center', va='bottom', fontsize=10)

    #     plt.tight_layout()
    #     plt.show()
    
    def plot_precision_categories(self, percent_df, colors=None, fig_size=(5.5, 5), save_fig=True):
        df = percent_df.copy()
        
        # Map row labels to more descriptive English names
        row_mapping = {
            '=0': '[=0] Zero',
            '(0.0-0.2]': '(0.0-0.2] Minimal',
            '(0.2-0.4]': '(0.2-0.4] Low',
            '(0.4-0.6]': '(0.4-0.6] Moderate',
            '(0.6-0.8]': '(0.6-0.8] Good',
            '(0.8-1.0]': '(0.8-1.0] High'
        }
        
        df = df.rename(index=row_mapping)
        
        # Transpose DataFrame for plotting
        df_plot = df.T
        
        # Convert index to float for proper sorting
        df_plot.index = df_plot.index.astype(float)
        df_plot = df_plot.sort_index(ascending=True)
        
        # Set up the plot
        plt.figure(figsize=fig_size)
        sns.set_style("whitegrid")
        
        # Define colors for different precision categories
        if colors is None:
            colors = ['Greens', 'Blues', 'PuBuGn', 'Oranges', 'YlGnBu', 'Purples']
            # ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33']
        
        # Define different marker styles for each precision category
        markers = ['v', '^', '^', 'D', 'D', 'v'] # o
        
        # Define different line styles (all dashed variations)
        # linestyles = ['--', '-.', ':', (0, (5, 1)), (0, (3, 5, 1, 5)), (0, (3, 1, 1, 1))]
        linestyles = [':'] * 6
        
        # Plot each precision category
        for i, precision_category in enumerate(df.index):
            x_values = df_plot.index.tolist()
            y_values = df_plot[precision_category].tolist()
            plt.plot(
                x_values, 
                y_values, 
                marker=markers[i % len(markers)],  # Use different marker for each line
                linewidth=1.8,  # Slightly reduced from 2.5
                linestyle=linestyles[i % len(linestyles)],  # Use different dashed line style
                label=precision_category, 
                color=colors[i]
            )
        
        # Add labels and title
        plt.xlabel('Work Coverage Rate (WCR)', fontsize=14)
        plt.ylabel('Percentage (%)', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(title='Precision Category', fontsize=12, bbox_to_anchor=(-0.01, 0.78), loc='upper left', title_fontsize=12)
        
        # Set x-axis ticks
        plt.xticks(df_plot.index)
        plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x:.2g}'))

        plt.tick_params(axis='x', labelsize=12)
        plt.tick_params(axis='y', labelsize=12)
        
        # Dictionary to store custom label display settings for each category
        # Initially all set to display all labels
        display_ticks = {
            '[=0] Zero': df_plot.index.tolist(),
            '(0.0-0.2] Minimal': [0.6, 0.65, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0],
            '(0.2-0.4] Low': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
            '(0.4-0.6] Moderate': [0.9, 0.95, 1.0],
            '(0.6-0.8] Good': df_plot.index.tolist(),
            '(0.8-1.0] High': [0.6, 0.65, 0.75, 0.85, 0.9, 0.95, 1.0]
        }
        
        # Add data labels for all points
        for category in df.index:
            for wcr in df_plot.index:
                if wcr in display_ticks[category]:  # Check if this tick should display a label
                    value = df_plot.loc[wcr, category]
                    plt.text(wcr, value, f'{value:.2f}'.rstrip('0').rstrip('.'), 
                            ha='center', va='bottom', fontsize=10)

        plt.tight_layout()

        if save_fig: 
            fig_path = op.join(self.results_path, 'precision_categories_trend_by_wcr.pdf')
            self.path_manager.save_pdf_file(passed_in_plt=plt, abs_file_path=fig_path, override=True, pad_inches=0.02)

        plt.show()



    def analyze_precision_by_wcr(self, df = None, colors = None):

        if df is None:
            df = self.collect_all_precision_data(batch_range=(0, 339))
        
        # Create bins with 0.2 intervals
        bins = np.arange(0, 1.1, 0.2)
        
        # Initialize result dictionaries
        all_results = {}
        all_percentages = {}
        
        for wcr in self.wcr_list:
            col_name = f'Precision_{wcr}'
            if col_name in df.columns:
                # Create a copy of the data for manipulation
                temp_data = df[col_name].copy()
                
                # Count exact zeros
                zero_count = (temp_data == 0).sum()
                
                # Create categories for non-zero values with 0.2 intervals
                non_zero_data = temp_data[temp_data > 0]
                labels = [f'({i:.1f}-{i+0.2:.1f}]' for i in np.arange(0, 1.0, 0.2)]
                binned = pd.cut(non_zero_data, bins=bins, labels=labels, right=True).value_counts().sort_index()
                
                # Combine zeros with binned data
                result = pd.concat([pd.Series({'=0': zero_count}), binned])
                result_percentage = result / len(df) * 100
                
                all_results[col_name] = result
                all_percentages[wcr] = result_percentage
        
        freq_df = pd.DataFrame(all_results)
        percent_df = pd.DataFrame(all_percentages)

        self.plot_precision_categories(percent_df, colors)

        # percent_df = percent_df.apply(lambda col: col.map(lambda x: f"{x:.2f}%"))
        return {
            'frequency': freq_df,
            'percentage': percent_df
        }

    def analyze_high_precision_authors(self, df = None, colors = None):

        df = self.collect_all_precision_data()
        df_high_precision = df.loc[df['Precision_1']>0.8, ['SLC_1', 'TP_1', 'FP_1']]
        bins = [0, 1, 2, 3, 5, 7, 10, 20, float('inf')]
        labels = ['1', '2', '3', '4-5', '6-7', '8-10', '11-20', '21+']

        # 分箱
        tp1_grouped = pd.cut(df_high_precision['TP_1'] + df_high_precision['FP_1'], bins=bins, labels=labels, right=True)

        # 统计频数和比率
        freq = tp1_grouped.value_counts(sort=False)
        ratio = tp1_grouped.value_counts(normalize=True, sort=False)

        # 合并输出
        result = pd.DataFrame({'count': freq, 'ratio': ratio})
        result = result.reset_index().rename(columns={'index': 'n_pairs'})
        def format_ratio(r):
            if r < 0.01: # 小于1%，用千分比显示
                return f"{r*1000:.2f}‰"
            else: # 否则用百分比显示
                return f"{r*100:.2f}%"
        result['ratio'] = result['ratio'].apply(format_ratio)

        result.to_csv(op.join(self.results_path, 'high_precision_n_pairs.csv'))

        return result

    def plot_precision_distribution_by_level(self, df_all_results, wcr_list, labels, fig_size=(12, 6), save_fig = True):
        """
        Plot ratio distributions for each precision_level in a 2x3 subplot grid.
        Each subplot: one precision_level; each line: one wcr.
        """
        precision_levels = ['zero', 'minimal', 'low', 'moderate', 'good', 'high']
        n_pairs_order = {label: idx+1 for idx, label in enumerate(labels)}
        fig, axes = plt.subplots(2, 3, figsize=fig_size, sharey=True)
        axes = axes.flatten()
        
        # Define colors for different WCR values
        # colors = [
        #     '#ff7f00',  # Orange from 'Oranges'
        #     '#73c2fb',  # Light blue from 'PuBuGn'
        #     '#91cf60',  # Lime green (RdYlGn)
        #     '#e7298a',  # Magenta (Set2)
        #     '#6a3d9a',  # Royal purple (Set1)
        #     '#33a02c',  # Forest green (Set1)
        #     '#5e3c99',  # Deep purple (PuOr)
        #     '#e6ab02',  # Gold (PiYG)
        #     '#66a61e',  # Olive green (Set2)
        # ]
        # colors = [
        #     '#ff7f00',  # Orange from 'Oranges'
        #     '#73c2fb',  # Light blue from 'PuBuGn'
        #     '#91cf60',  # Lime green (RdYlGn)
        #     '#e7298a',  # Magenta (Set2)
        #     '#6a3d9a',  # Royal purple (Set1)
        #     '#33a02c',   # Forest green (Set1)
        #     '#4575b4',  # Medium blue
        #     '#fc8d59',  # Salmon/Light orange
        #     '#e6ab02'   # Gold/Yellow
        # ]
        colors = [self.get_color(i) for i in range(len(wcr_list))]
        # Define different marker styles
        markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'X', 'P']
        
        # Define different line styles
        linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 1))]
        
        # Assign color, marker, and line style for each WCR
        wcr_styles = {}
        for i, wcr in enumerate(wcr_list):
            wcr_styles[wcr] = {
                'color': colors[i % len(colors)],
                'marker': markers[i % len(markers)],
                'linestyle': linestyles[i % len(linestyles)]
            }
        
        # Thinner line widths - reduced from original
        line_widths = {wcr: 0.8 + (i * 0.1) for i, wcr in enumerate(wcr_list)}
        
        for idx, precision in enumerate(precision_levels):
            ax = axes[idx]
            df_plot_precision = df_all_results[df_all_results['precision_level'] == precision]
            
            for wcr in wcr_list:
                df_plot = df_plot_precision[df_plot_precision['wcr'] == wcr].copy()
                df_plot['n_pairs_order'] = df_plot['n_pairs'].map(n_pairs_order)
                df_plot = df_plot.sort_values('n_pairs_order')
                
                ax.plot(
                    df_plot['n_pairs_order'].tolist(),
                    df_plot['ratio'].tolist(),
                    marker=wcr_styles[wcr]['marker'],
                    markersize=6,  # Reduced from 8 to 6
                    linewidth=line_widths[wcr],
                    linestyle=wcr_styles[wcr]['linestyle'],
                    label=f'{wcr}',
                    color=wcr_styles[wcr]['color']
                )
                
            ax.set_xticks(range(1, len(labels)+1))
            ax.set_xticklabels(labels, fontsize=12)
            ax.set_title(f"{precision.capitalize()} Precision", fontsize=15)
            ax.set_xlabel('Number of Pairs', fontsize=12)
            if idx % 3 == 0:
                ax.set_ylabel('Ratio', fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0%}'.format(x)))
            
            # Create legend with better spacing and visibility
            legend = ax.legend(
                title='WCR', 
                title_fontsize=12, 
                fontsize=10, 
                loc='best', 
                ncol=3, 
                columnspacing=0.8,  # Increased spacing between columns
                frameon=True,
                framealpha=0.8,  # Semi-transparent background
                edgecolor='gray'  # Border color for legend
            )
            
        plt.tight_layout()
        if save_fig: 
            fig_path = op.join(self.results_path, 'pair_pdf_by_precision_across_wcr.pdf')
            self.path_manager.save_pdf_file(passed_in_plt=plt, abs_file_path=fig_path, override=True, pad_inches=0.02)

        plt.show()


    def analyze_tcp_ratio_under_various_precisions(self, df=None):
        lst = [0, 0.2, 0.4, 0.6, 0.8, 1]
        pairs = list(zip(lst[:-1], lst[1:]))
        precision_levels = ['zero', 'minimal', 'low', 'moderate', 'good', 'high']
        precision_to_pair = dict(zip(precision_levels[1:], pairs))
        bins = [0, 1, 2, 3, 5, 7, 10, 20, float('inf')]
        labels = ['1', '2', '3', '4-5', '6-7', '8-10', '11-20', '21+']

        if df is None:
            df = self.collect_all_precision_data()
        all_results = []
        for wcr in self.wcr_list:
            col_name = f'Precision_{wcr}'
            slc_col = f'SLC_{wcr}'
            tp_col = f'TP_{wcr}'
            fp_col = f'FP_{wcr}'
            for precision_level in precision_levels:
                if precision_level == 'zero':
                    df_precision = df.loc[df[col_name]==0, [slc_col, tp_col, fp_col]]
                else:
                    lower_, higher_ = precision_to_pair[precision_level]
                    df_precision = df.loc[(df[col_name] <= higher_) & (df[col_name] > lower_), [slc_col, tp_col, fp_col]]
                tp1_grouped = pd.cut(df_precision[tp_col] + df_precision[fp_col], bins=bins, labels=labels, right=True)
                freq = tp1_grouped.value_counts(sort=False)
                ratio = tp1_grouped.value_counts(normalize=True, sort=False)
                result = pd.DataFrame({'count': freq, 'ratio': ratio})
                result = result.reset_index().rename(columns={'index': 'n_pairs'})
                result['precision_level'] = precision_level
                result['wcr'] = wcr
                all_results.append(result)
        df_all_results = pd.concat(all_results, ignore_index=True)
        self.plot_precision_distribution_by_level(df_all_results, self.wcr_list, labels)

    def analyze_precision_by_prediction_volume(self, df = None, bins=None, labels=None):
        # Set default values
        if bins is None:
            bins = [0, 1, 2, 3, 5, 7, 10, 20, float('inf')]
        if labels is None:
            labels = ['1', '2', '3', '4-5', '6-7', '8-10', '11-20', '21+']
        if df is None:
            df = self.collect_all_precision_data()
        
        # Calculate the sum of TP+FP
        df['Total_Predictions'] = df['TP_1'] + df['FP_1']
        df = df[df['Total_Predictions'] > 0].copy()
        # Group by specified intervals
        df['Prediction_Group'] = pd.cut(df['Total_Predictions'], bins=bins, labels=labels, right=False)
        
        # Calculate statistics for each group - fix observed warning
        group_stats = df.groupby('Prediction_Group', observed=True)['Precision_1'].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).reset_index()
        
        # Calculate correlation coefficient
        correlation = df['Total_Predictions'].corr(df['Precision_1'])
        print(f"Correlation between TP+FP and Precision: {correlation:.4f}")
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Draw boxplot - fix palette warning
        ax = sns.boxplot(x='Prediction_Group', y='Precision_1', hue='Prediction_Group', 
                        data=df, legend=False)
        
        # Add mean points - fix observed warning
        means = df.groupby('Prediction_Group', observed=True)['Precision_1'].mean().values
        x_positions = np.arange(len(means))
        plt.plot(x_positions, means, 'ro-', linewidth=2, markersize=8)
        
        # Set chart title and labels
        plt.title('Precision by Prediction Volume (TP+FP)', fontsize=16)
        plt.xlabel('Total Predictions (TP+FP)', fontsize=14)
        plt.ylabel('Precision', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add sample size labels
        for i, count in enumerate(group_stats['count']):
            plt.annotate(f'n={count}', 
                        (i, -0.05),  # Display at the bottom
                        ha='center', va='top',
                        fontsize=10)
        
        # Add correlation information
        plt.figtext(0.01, 0.01, f"Correlation: {correlation:.4f}", fontsize=12)
        
        plt.tight_layout()
        plt.show()

# Usage example
if __name__ == "__main__":
    evaluator = IndividualEvaluator(sub_concepts = False)
    # evaluator.collect_all_precision_data()
    # Execute all analyses in one go
    # evaluator.analyze_all()
    # evaluator.analyze_precision_by_wcr()
    # evaluator.analyze_high_precision_authors()
    # evaluator.analyze_tcp_ratio_under_various_precisions()
    evaluator.analyze_precision_by_prediction_volume()
    # Or execute step by step
    # df = evaluator.collect_all_precision_data()
    # evaluator.plot_precision_distributions(df)
    # evaluator.plot_precision_boxplots(df)
    # evaluator.generate_summary_statistics(df)

# nohup python ./network/individual_researcher_evaluation.py >> individual_researcher_evaluation.log 2>&1 &

# ps aux | grep individual_researcher_evaluation.py
# pkill -f individual_researcher_evaluation.py
