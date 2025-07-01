import os, sys
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.append(project_dir)

import os
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging
from utils.config import PathManager
from joblib import Parallel, delayed
import os.path as op
from typing import Tuple, Optional, Dict, List, Union
from utils.smart_cache import cache_results

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DiversityAnalyzer:
    """Analyzes diversity metrics for author concept sequences"""
    path_manager = PathManager()
    def __init__(self, debugging):
        self.debugging = debugging
        self.cache_dir = self.path_manager.concepts_new_sequences_dir
        self.results_dir = os.path.join(self.cache_dir, 'diversity_analysis')
        os.makedirs(self.results_dir, exist_ok=True)
    
    def get_color(self, i, n=19):
        r_off, g_off, b_off = 1, 1, 1
        low, high = 0.0, 1.0
        span = high - low
        r = low + span * (((i + r_off) * 3) % n) / (n - 1)
        g = low + span * (((i + g_off) * 5) % n) / (n - 1)
        b = low + span * (((i + b_off) * 7) % n) / (n - 1)
        return (r, g, b)
    
    def analyze_author_diversity(self, author_id, concept_sequence, windows=[-1, 1, 3, 5]):
        """
        Calculate all diversity metrics for multiple windows in a single pass.
        
        Args:
            concept_sequence: List of concept lists (one per work)
            windows: List of window sizes; -1 means cumulative
        
        Returns:
            dict: {
                'unique_counts': {window: counts_list},
                'entropies': {window: entropy_list}, 
                'hhis': {window: hhi_list}
            }
        """
        
        # Initialize results structure
        results = {
            'author_id': author_id,
            'num_works': len(concept_sequence),
            'unique_counts': {},
            'entropies': {},
            'hhis': {}
        }
        
        # Prepare result lists for each window
        for w in windows:
            window_key = 'cumulative' if w == -1 else f'window_{w}'
            results['unique_counts'][window_key] = []
            results['entropies'][window_key] = []
            results['hhis'][window_key] = []
        
        # For cumulative calculation
        cumulative_concepts = []
        cumulative_set = set()
        
        # Single pass through the concept sequence
        for i, work_concepts in enumerate(concept_sequence):
            # Update cumulative data
            cumulative_concepts.extend(work_concepts)
            cumulative_set.update(work_concepts)
            
            # Calculate metrics for each window
            for w in windows:
                window_key = 'cumulative' if w == -1 else f'window_{w}'
                
                if w == -1:
                    # Cumulative calculations
                    window_concepts = cumulative_concepts
                    unique_count = len(cumulative_set)
                else:
                    # Sliding window calculations
                    window_start = max(0, i - w + 1)
                    window_concepts = sum(concept_sequence[window_start:i+1], [])
                    unique_count = len(set(window_concepts))
                
                # Store unique count
                results['unique_counts'][window_key].append(unique_count)
                
                # Calculate entropy and HHI
                if len(window_concepts) == 0:
                    results['entropies'][window_key].append(0)
                    results['hhis'][window_key].append(1.0)
                else:
                    concept_counts = Counter(window_concepts)
                    total_concepts = sum(concept_counts.values())
                    probabilities = [count / total_concepts for count in concept_counts.values()]
                    
                    # Shannon entropy
                    entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
                    results['entropies'][window_key].append(entropy)
                    
                    # HHI
                    hhi = sum(p**2 for p in probabilities)
                    results['hhis'][window_key].append(hhi)
        
        return results

    def load_concept_sequences(self, batch_files):
        """
        Load concept sequences from batch files
        
        Args:
            batch_files: List of paths to batch files
            
        Returns:
            pd.DataFrame: DataFrame with author_id and concept_sequences
        """
        all_sequences = []
        
        for batch_file in tqdm(batch_files, desc="Loading concept sequences"):
            try:
                df_batch = pd.read_parquet(batch_file)
                all_sequences.append(df_batch)
            except Exception as e:
                logger.error(f"Error loading batch file {batch_file}: {str(e)}")
                continue
        
        if all_sequences:
            return pd.concat(all_sequences, ignore_index=True)
        else:
            return pd.DataFrame(columns=['author_id', 'concept_sequences'])
    
    @cache_results(
        cache_dir=None,
        filename_pattern='diversity_results_debugging_v{version}.pkl',
        version=1,
        use_instance_cache_dir=True,
        use_exact_filename=True
    )
    def analyze_all_authors(self, windows=[1, 3, 5, -1], parallel=False, skip_cache = False, n_jobs=64):
        """
        Analyze diversity metrics for all authors
        
        Args:
            windows: List of window sizes; -1 means cumulative
            parallel: Whether to use parallel processing
            n_jobs: Number of parallel jobs
            
        Returns:
            list: List of diversity analysis results
        """
        # Load concept sequences
        if self.debugging:
            df_papers = pd.read_parquet(os.path.join(self.cache_dir, "all_author_work_concepts_1.parquet"))
            df_papers = df_papers[:500000]
        else:
            df_papers = pd.read_parquet(os.path.join(self.cache_dir, "all_author_work_concepts_merged.parquet"))
        
        n_authors = len(set(df_papers.author_id))
        logger.info(f"Analyzing diversity for {len(df_papers)} works of {n_authors} authors")
        
        # Convert publication_date to datetime if it's not already
        df_papers['publication_date'] = pd.to_datetime(df_papers['publication_date'])
        df_papers['concepts'] = df_papers['concepts'].apply(list)
        df_sequences = df_papers.groupby('author_id').agg(
            concept_sequences=('concepts', lambda x: list(x)),
            first_pub_date=('publication_date', 'min'),
            last_pub_date=('publication_date', 'max'),
            paper_count=('publication_date', 'count')
        ).reset_index()
        
        results = []

        if parallel:
            # Joblib parallel processing
            logger.info(f"Using {n_jobs} parallel jobs")
            results = Parallel(n_jobs=n_jobs, verbose=1)(
                delayed(self.analyze_author_diversity)(
                    row['author_id'], 
                    row['concept_sequences'], 
                    windows
                ) for _, row in df_sequences.iterrows()
            )
            
            # Filter out None results
            results = [r for r in results if r is not None]
        else:
            for _, row in tqdm(df_sequences.iterrows(), total=len(df_sequences), desc="Analyzing diversity"):
                author_result = self.analyze_author_diversity(
                    row['author_id'], 
                    row['concept_sequences'], 
                    windows
                )
                if author_result:
                    results.append(author_result)

        return results
    
    def plot_cumulative_unique_concepts(self, results=None, 
                                        figsize: Tuple[int, int] = (14, 8),
                                        font_size_dict: Optional[Dict[str, int]] = None,
                                        save_fig: bool = True,
                                        max_authors: int = 20):
        """
        Plot cumulative unique concept count trends
        """
        if font_size_dict is None:
            font_size_dict = {'title': 20, 'xlabel': 16, 'ylabel': 16, 'xtick': 16, 'ytick': 16, 'legend': 16, 'text': 14}
        
        # Filter authors with sufficient works
        filtered_results = [r for r in results if r['num_works'] >= 10]
        logger.info(f"Plotting cumulative unique concepts for {len(filtered_results)} authors with ≥10 works")
        
        if len(filtered_results) == 0:
            logger.warning("No authors with sufficient works for plotting")
            return
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Prepare colors
        colors = [self.get_color(i, 20) for i in range(20)]
        
        # Plot trends for first max_authors authors
        for idx, result in enumerate(filtered_results[:max_authors]):
            if 'cumulative' in result['unique_counts']:
                work_numbers = list(range(1, len(result['unique_counts']['cumulative']) + 1))
                ax.plot(
                    work_numbers, 
                    result['unique_counts']['cumulative'], 
                    alpha=0.8, 
                    color=colors[idx % len(colors)]
                )
        
        # Set labels and formatting
        ax.set_xlabel('Publication Order', fontsize=font_size_dict['xlabel'])
        ax.set_ylabel('Cumulative Unique Concepts', fontsize=font_size_dict['ylabel'])
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', labelsize=font_size_dict['xtick'])
        ax.tick_params(axis='y', labelsize=font_size_dict['ytick'])
        
        plt.tight_layout()
        
        if save_fig:
            fig_path = op.join(self.path_manager.base_figures_dir, 'cumulative_unique_concepts.pdf')
            self.path_manager.save_pdf_file(passed_in_plt=plt, abs_file_path=fig_path, override=True, pad_inches=0.02)
        plt.show()

    def plot_cumulative_unique_concepts_with_window(self, results=None, 
                                        window: Optional[int] = None,
                                        figsize: Tuple[int, int] = (14, 8),
                                        font_size_dict: Optional[Dict[str, int]] = None,
                                        save_fig: bool = True,
                                        max_authors: int = 20):
        """
        Plot cumulative unique concept count trends
        
        Args:
            results: List of diversity analysis results
            window: If provided (1/3/5), plot recent window unique concepts as dashed lines
            figsize: Figure size tuple
            font_size_dict: Dictionary of font sizes for different elements
            save_fig: Whether to save the figure
            max_authors: Maximum number of authors to plot
        """
        if font_size_dict is None:
            font_size_dict = {'title': 20, 'xlabel': 16, 'ylabel': 16, 'xtick': 16, 'ytick': 16, 'legend': 16, 'text': 14}
        
        # Filter authors with sufficient works
        filtered_results = [r for r in results if r['num_works'] >= 10]
        logger.info(f"Plotting cumulative unique concepts for {len(filtered_results)} authors with ≥10 works")
        
        if len(filtered_results) == 0:
            logger.warning("No authors with sufficient works for plotting")
            return
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Prepare colors
        colors = [self.get_color(i, 20) for i in range(20)]
        
        # Plot trends for first max_authors authors
        for idx, result in enumerate(filtered_results[:max_authors]):
            if 'cumulative' in result['unique_counts']:
                work_numbers = list(range(1, len(result['unique_counts']['cumulative']) + 1))
                
                # Plot cumulative unique concepts (solid line)
                ax.plot(
                    work_numbers, 
                    result['unique_counts']['cumulative'], 
                    alpha=0.8, 
                    color=colors[idx % len(colors)],
                    linestyle='-',
                    linewidth=2
                )
                
                # Plot window unique concepts if specified (dashed line)
                if window is not None and window in [1, 3, 5]:
                    window_key = f'window_{window}'
                    if window_key in result['unique_counts']:
                        # Skip the first 'window' points to avoid overlap with cumulative
                        window_data = result['unique_counts'][window_key]
                        if len(window_data) > window:
                            window_work_numbers = work_numbers[window:]  # Start from window+1
                            window_values = window_data[window:]
                            ax.plot(
                                window_work_numbers,
                                window_values,
                                alpha=0.6,
                                color=colors[idx % len(colors)],
                                linestyle='--',
                                linewidth=1.5
                            )
        
        # Set labels and formatting
        ax.set_xlabel('Publication Order', fontsize=font_size_dict['xlabel'])
        ax.set_ylabel('Unique Concepts', fontsize=font_size_dict['ylabel'])
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', labelsize=font_size_dict['xtick'])
        ax.tick_params(axis='y', labelsize=font_size_dict['ytick'])
        
        # Add legend if window is specified
        if window is not None and window in [1, 3, 5]:
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='black', linestyle='-', linewidth=2, label='Cumulative'),
                Line2D([0], [0], color='black', linestyle='--', linewidth=1.5, label=f'Recent {window}')
            ]
            ax.legend(handles=legend_elements, fontsize=font_size_dict['legend'])
        
        plt.tight_layout()
        
        if save_fig:
            window_suffix = f'_window{window}' if window is not None else ''
            fig_path = op.join(self.path_manager.base_figures_dir, f'cumulative_unique_concepts{window_suffix}.pdf')
            self.path_manager.save_pdf_file(passed_in_plt=plt, abs_file_path=fig_path, override=True, pad_inches=0.02)
        plt.show()

    def plot_unique_concepts_boxplot_comparison(self, results=None,
                                            figsize: Tuple[int, int] = (16, 10),
                                            font_size_dict: Optional[Dict[str, int]] = None,
                                            save_fig: bool = True,
                                            min_works: int = 30,
                                            max_publication_order: int = 30):
        """
        Plot boxplot comparison of unique concepts across publication orders for multiple authors
        
        Args:
            results: List of diversity analysis results
            figsize: Figure size tuple
            font_size_dict: Dictionary of font sizes for different elements
            save_fig: Whether to save the figure
            min_works: Minimum number of works required for an author to be included
            max_publication_order: Maximum publication order to plot
        """
        if font_size_dict is None:
            font_size_dict = {'title': 20, 'xlabel': 16, 'ylabel': 16, 'xtick': 14, 'ytick': 14, 'legend': 16}
        
        # Filter authors with sufficient works
        filtered_results = [r for r in results if r['num_works'] >= min_works]
        logger.info(f"Plotting boxplot for {len(filtered_results)} authors with ≥{min_works} works")
        
        if len(filtered_results) == 0:
            logger.warning("No authors with sufficient works for plotting")
            return
        
        # Prepare data structure for boxplot
        data_dict = {
            'cumulative': {},
            'window_1': {},
            'window_3': {},
            'window_5': {}
        }
        
        # Collect data for each publication order
        for pub_order in range(1, max_publication_order + 1):
            data_dict['cumulative'][pub_order] = []
            data_dict['window_1'][pub_order] = []
            data_dict['window_3'][pub_order] = []
            data_dict['window_5'][pub_order] = []
            
            for result in filtered_results:
                if len(result['unique_counts']['cumulative']) >= pub_order:
                    # Cumulative data (available from publication order 1)
                    data_dict['cumulative'][pub_order].append(result['unique_counts']['cumulative'][pub_order-1])
                    
                    # Window 1 data (available from publication order 2)
                    if pub_order >= 2 and 'window_1' in result['unique_counts']:
                        if len(result['unique_counts']['window_1']) >= pub_order:
                            data_dict['window_1'][pub_order].append(result['unique_counts']['window_1'][pub_order-1])
                    
                    # Window 3 data (available from publication order 4)
                    if pub_order >= 4 and 'window_3' in result['unique_counts']:
                        if len(result['unique_counts']['window_3']) >= pub_order:
                            data_dict['window_3'][pub_order].append(result['unique_counts']['window_3'][pub_order-1])
                    
                    # Window 5 data (available from publication order 6)
                    if pub_order >= 6 and 'window_5' in result['unique_counts']:
                        if len(result['unique_counts']['window_5']) >= pub_order:
                            data_dict['window_5'][pub_order].append(result['unique_counts']['window_5'][pub_order-1])
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Define colors for different metrics
        colors = {
            'cumulative': '#6a3d9a',
            'window_1': '#ff7f00', 
            'window_3': '#33a02c',
            'window_5': '#73c2fb'
        }
        
        # Define box width
        box_width = 0.15
        
        # For each publication order, determine which metrics have data and calculate positions
        for pub_order in range(1, max_publication_order + 1):
            # Determine which metrics have data for this publication order
            available_metrics = []
            if len(data_dict['cumulative'][pub_order]) > 0:
                available_metrics.append('cumulative')
            if len(data_dict['window_1'][pub_order]) > 0:
                available_metrics.append('window_1')
            if len(data_dict['window_3'][pub_order]) > 0:
                available_metrics.append('window_3')
            if len(data_dict['window_5'][pub_order]) > 0:
                available_metrics.append('window_5')
            
            if len(available_metrics) == 0:
                continue
            
            # Calculate positions to center the boxes around the publication order
            num_boxes = len(available_metrics)
            if num_boxes == 1:
                offsets = [0]
            elif num_boxes == 2:
                offsets = [-0.5 * box_width, 0.5 * box_width]
            elif num_boxes == 3:
                offsets = [-box_width, 0, box_width]
            else:  # num_boxes == 4
                offsets = [-1.5 * box_width, -0.5 * box_width, 0.5 * box_width, 1.5 * box_width]
            
            # Create boxplots for this publication order
            for i, metric in enumerate(available_metrics):
                position = pub_order + offsets[i]
                data = data_dict[metric][pub_order]
                
                bp = ax.boxplot([data], positions=[position], widths=box_width,
                            patch_artist=True, showfliers=False)
                
                # Color the box
                bp['boxes'][0].set_facecolor(colors[metric])
                bp['boxes'][0].set_alpha(0.7)
        
        # Set labels and formatting
        ax.set_xlabel('Publication Order', fontsize=font_size_dict['xlabel'])
        ax.set_ylabel('Unique Concepts', fontsize=font_size_dict['ylabel'])
        ax.set_xlim(0.5, max_publication_order + 0.5)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', labelsize=font_size_dict['xtick'])
        ax.tick_params(axis='y', labelsize=font_size_dict['ytick'])
        
        # Set x-axis ticks to show integer publication orders
        ax.set_xticks(range(1, max_publication_order + 1))
        ax.set_xticklabels([str(i) for i in range(1, max_publication_order + 1)])
        
        # Create legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=colors['cumulative'], alpha=0.7, label='Cumulative'),
            Patch(facecolor=colors['window_1'], alpha=0.7, label='Recent 1'),
            Patch(facecolor=colors['window_3'], alpha=0.7, label='Recent 3'),
            Patch(facecolor=colors['window_5'], alpha=0.7, label='Recent 5')
        ]
        ax.legend(handles=legend_elements, fontsize=font_size_dict['legend'], loc='upper left')
        
        plt.tight_layout()
        
        if save_fig:
            fig_path = op.join(self.path_manager.base_figures_dir, 'unique_concepts_boxplot_comparison.pdf')
            self.path_manager.save_pdf_file(passed_in_plt=plt, abs_file_path=fig_path, override=True, pad_inches=0.02)
        
        plt.show()

    def plot_diversity_metrics_boxplot_comparison(self, results=None,
                                                metric_type: str = 'unique_counts',
                                                figsize: Tuple[int, int] = (16, 10),
                                                font_size_dict: Optional[Dict[str, int]] = None,
                                                save_fig: bool = True,
                                                min_works: int = 30,
                                                max_publication_order: int = 30,
                                                show_legend = True,
                                                selected_orders: Optional[List[int]] = None):
        """
        Plot boxplot comparison of diversity metrics across publication orders for multiple authors
        
        Args:
            results: List of diversity analysis results
            metric_type: Type of metric to plot ('unique_counts', 'entropies', 'hhis')
            figsize: Figure size tuple
            font_size_dict: Dictionary of font sizes for different elements
            save_fig: Whether to save the figure
            min_works: Minimum number of works required for an author to be included
            max_publication_order: Maximum publication order to plot (ignored if selected_orders is provided)
            show_legend: Whether to show legend
            selected_orders: List of specific publication orders to plot (e.g., [6, 10, 15, 20, 25, 30])
        """
        if font_size_dict is None:
            font_size_dict = {'title': 20, 'xlabel': 16, 'ylabel': 16, 'xtick': 14, 'ytick': 14, 'legend': 16}
        
        # Validate metric_type
        valid_metrics = ['unique_counts', 'entropies', 'hhis']
        if metric_type not in valid_metrics:
            raise ValueError(f"metric_type must be one of {valid_metrics}, got {metric_type}")
        
        # Filter authors with sufficient works
        filtered_results = [r for r in results if r['num_works'] >= min_works]
        logger.info(f"Plotting boxplot for {len(filtered_results)} authors with ≥{min_works} works")
        
        if len(filtered_results) == 0:
            logger.warning("No authors with sufficient works for plotting")
            return
        
        # Determine which publication orders to plot
        if selected_orders is not None:
            plot_orders = sorted(selected_orders)
            logger.info(f"Plotting selected publication orders: {plot_orders}")
        else:
            plot_orders = list(range(1, max_publication_order + 1))
            logger.info(f"Plotting publication orders 1 to {max_publication_order}")
        
        # Prepare data structure for boxplot
        data_dict = {
            'window_1': {},
            'window_3': {},
            'window_5': {},
            'cumulative': {}
        }
        
        # Collect data for each publication order
        for pub_order in plot_orders:
            data_dict['window_1'][pub_order] = []
            data_dict['window_3'][pub_order] = []
            data_dict['window_5'][pub_order] = []
            data_dict['cumulative'][pub_order] = []
            
            for result in filtered_results:
                if len(result[metric_type]['cumulative']) >= pub_order:
                    # Cumulative data (available from publication order 1)
                    data_dict['cumulative'][pub_order].append(result[metric_type]['cumulative'][pub_order-1])
                    
                    # Window 1 data (available from publication order 2)
                    if pub_order >= 2 and 'window_1' in result[metric_type]:
                        if len(result[metric_type]['window_1']) >= pub_order:
                            data_dict['window_1'][pub_order].append(result[metric_type]['window_1'][pub_order-1])
                    
                    # Window 3 data (available from publication order 4)
                    if pub_order >= 4 and 'window_3' in result[metric_type]:
                        if len(result[metric_type]['window_3']) >= pub_order:
                            data_dict['window_3'][pub_order].append(result[metric_type]['window_3'][pub_order-1])
                    
                    # Window 5 data (available from publication order 6)
                    if pub_order >= 6 and 'window_5' in result[metric_type]:
                        if len(result[metric_type]['window_5']) >= pub_order:
                            data_dict['window_5'][pub_order].append(result[metric_type]['window_5'][pub_order-1])
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Define colors for different metrics (reordered)
        colors = {
            'window_1': '#ff7f00', 
            'window_3': '#33a02c',
            'window_5': '#73c2fb',
            'cumulative': '#6a3d9a'
        }
        
        # Define box width
        box_width = 0.15
        
        # For selected orders, we need to map them to x-axis positions
        # Use enumerate to get sequential positions
        x_positions = {}
        for i, pub_order in enumerate(plot_orders):
            x_positions[pub_order] = i + 1
        
        # Define the order of metrics to display
        metric_order = ['window_1', 'window_3', 'window_5', 'cumulative']
        
        # For each publication order, determine which metrics have data and calculate positions
        for pub_order in plot_orders:
            # Determine which metrics have data for this publication order (in the desired order)
            available_metrics = []
            for metric in metric_order:
                if len(data_dict[metric][pub_order]) > 0:
                    available_metrics.append(metric)
            
            if len(available_metrics) == 0:
                continue
            
            # Calculate positions to center the boxes around the x position
            num_boxes = len(available_metrics)
            if num_boxes == 1:
                offsets = [0]
            elif num_boxes == 2:
                offsets = [-0.5 * box_width, 0.5 * box_width]
            elif num_boxes == 3:
                offsets = [-box_width, 0, box_width]
            else:  # num_boxes == 4
                offsets = [-1.5 * box_width, -0.5 * box_width, 0.5 * box_width, 1.5 * box_width]
            
            # Create boxplots for this publication order
            for i, metric in enumerate(available_metrics):
                position = x_positions[pub_order] + offsets[i]
                data = data_dict[metric][pub_order]
                
                bp = ax.boxplot([data], positions=[position], widths=box_width,
                            patch_artist=True, showfliers=False)
                
                # Color the box
                bp['boxes'][0].set_facecolor(colors[metric])
                bp['boxes'][0].set_alpha(0.7)
        
        # Set labels and formatting based on metric type
        ylabel_dict = {
            'unique_counts': 'Unique Concepts',
            'entropies': 'Entropy',
            'hhis': 'HHI Index'
        }
        
        ax.set_xlabel('Publication Order', fontsize=font_size_dict['xlabel'])
        ax.set_ylabel(ylabel_dict[metric_type], fontsize=font_size_dict['ylabel'])
        ax.set_xlim(0.5, len(plot_orders) + 0.5)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', labelsize=font_size_dict['xtick'])
        ax.tick_params(axis='y', labelsize=font_size_dict['ytick'])
        
        # Set x-axis ticks to show the actual publication orders
        ax.set_xticks([x_positions[order] for order in plot_orders])
        ax.set_xticklabels([str(order) for order in plot_orders])
        
        # Create legend (reordered to match the new display order)
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=colors['window_1'], alpha=0.7, label='Recent 1'),
            Patch(facecolor=colors['window_3'], alpha=0.7, label='Recent 3'),
            Patch(facecolor=colors['window_5'], alpha=0.7, label='Recent 5'),
            Patch(facecolor=colors['cumulative'], alpha=0.7, label='Cumulative')
        ]
        if show_legend:
            ax.legend(handles=legend_elements, fontsize=font_size_dict['legend'], loc='upper left')
        
        plt.tight_layout()
        
        if save_fig:
            if selected_orders is not None:
                orders_str = '_'.join(map(str, selected_orders))
                filename = f'{metric_type}_boxplot_comparison.pdf'
            else:
                filename = f'{metric_type}_boxplot_comparison.pdf'
            fig_path = op.join(self.path_manager.base_figures_dir, filename)
            self.path_manager.save_pdf_file(passed_in_plt=plt, abs_file_path=fig_path, override=True, pad_inches=0.02)
        
        plt.show()

    def plot_cumulative_entropy(self, results=None,
                                figsize: Tuple[int, int] = (14, 8),
                                font_size_dict: Optional[Dict[str, int]] = None,
                                save_fig: bool = True,
                                max_authors: int = 20):
        """
        Plot cumulative Shannon entropy trends
        """
        if font_size_dict is None:
            font_size_dict = {'title': 20, 'xlabel': 16, 'ylabel': 16, 'xtick': 16, 'ytick': 16, 'legend': 16, 'text': 14}
        
        # Filter authors with sufficient works
        filtered_results = [r for r in results if r['num_works'] >= 10]
        logger.info(f"Plotting cumulative entropy for {len(filtered_results)} authors with ≥10 works")
        
        if len(filtered_results) == 0:
            logger.warning("No authors with sufficient works for plotting")
            return
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot trends for first max_authors authors
        for result in filtered_results[:max_authors]:
            if 'cumulative' in result['entropies']:
                work_numbers = list(range(1, len(result['entropies']['cumulative']) + 1))
                ax.plot(work_numbers, result['entropies']['cumulative'], alpha=0.8)
        
        # Set labels and formatting
        ax.set_xlabel('Publication Order', fontsize=font_size_dict['xlabel'])
        ax.set_ylabel('Cumulative Shannon Entropy', fontsize=font_size_dict['ylabel'])
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', labelsize=font_size_dict['xtick'])
        ax.tick_params(axis='y', labelsize=font_size_dict['ytick'])
        
        plt.tight_layout()
        
        if save_fig:
            fig_path = op.join(self.path_manager.base_figures_dir, 'cumulative_entropy.pdf')
            self.path_manager.save_pdf_file(passed_in_plt=plt, abs_file_path=fig_path, override=True, pad_inches=0.02)
        plt.show()

    def plot_entropy_comparison(self, results=None,
                                figsize: Tuple[int, int] = (14, 8),
                                font_size_dict: Optional[Dict[str, int]] = None,
                                save_fig: bool = True,
                                window_size: int = 5,
                                max_authors: int = 20):
        """
        Plot cumulative vs windowed entropy comparison
        """
        if font_size_dict is None:
            font_size_dict = {'title': 20, 'xlabel': 16, 'ylabel': 16, 'xtick': 16, 'ytick': 16, 'legend': 16, 'text': 14}
        
        # Filter authors with sufficient works
        filtered_results = [r for r in results if r['num_works'] >= window_size + 5]
        logger.info(f"Plotting entropy comparison for {len(filtered_results)} authors with ≥{window_size + 5} works")
        
        if len(filtered_results) == 0:
            logger.warning("No authors with sufficient works for plotting")
            return
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Color palette
        colors = [self.get_color(i, 20) for i in range(20)]
        
        # Plot trends for first max_authors authors
        for i, result in enumerate(filtered_results[:max_authors]):
            color = colors[i % len(colors)]
            
            # Plot cumulative entropy (solid line)
            if 'cumulative' in result['entropies']:
                work_numbers = list(range(1, len(result['entropies']['cumulative']) + 1))
                ax.plot(work_numbers, result['entropies']['cumulative'], 
                    alpha=0.8, linewidth=1.5, linestyle='-', 
                    color=color, label='Cumulative Entropy' if i == 0 else "")
            
            # Plot windowed entropy (dashed line)
            window_key = f'window_{window_size}'
            if window_key in result['entropies']:
                window_data = result['entropies'][window_key]
                # Skip the first 'window_size' points to avoid overlap with cumulative
                if len(window_data) > window_size:
                    work_numbers_window = list(range(window_size + 1, len(window_data) + 1))
                    window_values = window_data[window_size:]
                    ax.plot(work_numbers_window, window_values, 
                        alpha=0.6, linewidth=1.5, linestyle='--', 
                        color=color, label=f'Recent {window_size}-Work Entropy' if i == 0 else "")
        
        # Set labels and formatting
        ax.set_xlabel('Publication Order', fontsize=font_size_dict['xlabel'])
        ax.set_ylabel('Shannon Entropy', fontsize=font_size_dict['ylabel'])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=font_size_dict['legend'], loc='best', framealpha=0.9)
        ax.tick_params(axis='x', labelsize=font_size_dict['xtick'])
        ax.tick_params(axis='y', labelsize=font_size_dict['ytick'])
        
        plt.tight_layout()
        
        if save_fig:
            fig_path = op.join(self.path_manager.base_figures_dir, f'entropy_comparison_window_{window_size}.pdf')
            self.path_manager.save_pdf_file(passed_in_plt=plt, abs_file_path=fig_path, override=True, pad_inches=0.02)
        plt.show()


    def plot_cumulative_hhi(self, results=None,
                            figsize: Tuple[int, int] = (14, 8),
                            font_size_dict: Optional[Dict[str, int]] = None,
                            save_fig: bool = True,
                            max_authors: int = 20):
        """
        Plot cumulative HHI (concentration) trends
        """
        if font_size_dict is None:
            font_size_dict = {'title': 20, 'xlabel': 16, 'ylabel': 16, 'xtick': 16, 'ytick': 16, 'legend': 16, 'text': 14}
        
        # Filter authors with sufficient works
        filtered_results = [r for r in results if r['num_works'] >= 10]
        logger.info(f"Plotting cumulative HHI for {len(filtered_results)} authors with ≥10 works")
        
        if len(filtered_results) == 0:
            logger.warning("No authors with sufficient works for plotting")
            return
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot trends for first max_authors authors
        for result in filtered_results[:max_authors]:
            if 'cumulative' in result['hhis']:
                work_numbers = list(range(1, len(result['hhis']['cumulative']) + 1))
                ax.plot(work_numbers, result['hhis']['cumulative'], alpha=0.8)
        
        # Set labels and formatting
        ax.set_xlabel('Publication Order', fontsize=font_size_dict['xlabel'])
        ax.set_ylabel('HHI', fontsize=font_size_dict['ylabel'])
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', labelsize=font_size_dict['xtick'])
        ax.tick_params(axis='y', labelsize=font_size_dict['ytick'])
        
        plt.tight_layout()
        
        if save_fig:
            fig_path = op.join(self.path_manager.base_figures_dir, 'cumulative_hhi.pdf')
            self.path_manager.save_pdf_file(passed_in_plt=plt, abs_file_path=fig_path, override=True, pad_inches=0.02)
        plt.show()

    def plot_cumulative_hhi_with_window(self, results=None,
                            figsize: Tuple[int, int] = (14, 8),
                            font_size_dict: Optional[Dict[str, int]] = None,
                            save_fig: bool = True,
                            window_size: int = 5,
                            max_authors: int = 20):
        """
        Plot cumulative vs windowed HHI (concentration) comparison
        """
        if font_size_dict is None:
            font_size_dict = {'title': 20, 'xlabel': 16, 'ylabel': 16, 'xtick': 16, 'ytick': 16, 'legend': 16, 'text': 14}
        
        # Filter authors with sufficient works
        filtered_results = [r for r in results if r['num_works'] >= window_size + 5]
        logger.info(f"Plotting HHI comparison for {len(filtered_results)} authors with ≥{window_size + 5} works")
        
        if len(filtered_results) == 0:
            logger.warning("No authors with sufficient works for plotting")
            return
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Color palette
        colors = [self.get_color(i, 20) for i in range(20)]
        
        # Plot trends for first max_authors authors
        for i, result in enumerate(filtered_results[:max_authors]):
            color = colors[i % len(colors)]
            
            # Plot cumulative HHI (solid line)
            if 'cumulative' in result['hhis']:
                work_numbers = list(range(1, len(result['hhis']['cumulative']) + 1))
                ax.plot(work_numbers, result['hhis']['cumulative'], 
                    alpha=0.8, linewidth=1.5, linestyle='-', 
                    color=color, label='Cumulative HHI' if i == 0 else "")
            
            # Plot windowed HHI (dashed line)
            window_key = f'window_{window_size}'
            if window_key in result['hhis']:
                window_data = result['hhis'][window_key]
                # Skip the first 'window_size' points to avoid overlap with cumulative
                if len(window_data) > window_size:
                    work_numbers_window = list(range(window_size + 1, len(window_data) + 1))
                    window_values = window_data[window_size:]
                    ax.plot(work_numbers_window, window_values, 
                        alpha=0.6, linewidth=1.5, linestyle='--', 
                        color=color, label=f'Recent {window_size}-Work HHI' if i == 0 else "")
        
        # Set labels and formatting
        ax.set_xlabel('Publication Order', fontsize=font_size_dict['xlabel'])
        ax.set_ylabel('HHI (Herfindahl-Hirschman Index)', fontsize=font_size_dict['ylabel'])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=font_size_dict['legend'], loc='best', framealpha=0.9)
        ax.tick_params(axis='x', labelsize=font_size_dict['xtick'])
        ax.tick_params(axis='y', labelsize=font_size_dict['ytick'])
        
        plt.tight_layout()
        
        if save_fig:
            fig_path = op.join(self.path_manager.base_figures_dir, f'cumulative_hhi_window_{window_size}.pdf')
            self.path_manager.save_pdf_file(passed_in_plt=plt, abs_file_path=fig_path, override=True, pad_inches=0.02)
        plt.show()


    def plot_career_stage_diversity(self, results=None,
                                    figsize: Tuple[int, int] = (14, 8),
                                    font_size_dict: Optional[Dict[str, int]] = None,
                                    save_fig: bool = True):
        """
        Plot average diversity metrics by career stage
        """
        if font_size_dict is None:
            font_size_dict = {'title': 20, 'xlabel': 16, 'ylabel': 16, 'xtick': 16, 'ytick': 16, 'legend': 16, 'text': 14}
        
        # Filter authors with sufficient works
        filtered_results = [r for r in results if r['num_works'] >= 10]
        logger.info(f"Plotting career stage diversity for {len(filtered_results)} authors with ≥10 works")
        
        if len(filtered_results) == 0:
            logger.warning("No authors with sufficient works for plotting")
            return
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Calculate average metrics by career stage
        career_stages = []
        avg_entropies = []
        avg_hhis = []
        
        for stage in [5, 10, 15, 20, 25]:
            stage_entropies = []
            stage_hhis = []
            for result in filtered_results:
                if ('cumulative' in result['entropies'] and 
                    len(result['entropies']['cumulative']) >= stage):
                    stage_entropies.append(result['entropies']['cumulative'][stage-1])
                if ('cumulative' in result['hhis'] and 
                    len(result['hhis']['cumulative']) >= stage):
                    stage_hhis.append(result['hhis']['cumulative'][stage-1])
            
            if stage_entropies:
                career_stages.append(stage)
                avg_entropies.append(np.mean(stage_entropies))
                avg_hhis.append(np.mean(stage_hhis))
        
        # Create twin axis
        ax_twin = ax.twinx()
        
        # Plot lines
        line1 = ax.plot(career_stages, avg_entropies, 'b-o', label='Avg Entropy', linewidth=2, markersize=8)
        line2 = ax_twin.plot(career_stages, avg_hhis, 'r-s', label='Avg HHI', linewidth=2, markersize=8)
        
        # Set labels and formatting
        ax.set_xlabel('Publication Order', fontsize=font_size_dict['xlabel'])
        ax.set_ylabel('Average Shannon Entropy', color='b', fontsize=font_size_dict['ylabel'])
        ax_twin.set_ylabel('Average HHI', color='r', fontsize=font_size_dict['ylabel'])
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', labelsize=font_size_dict['xtick'])
        ax.tick_params(axis='y', labelsize=font_size_dict['ytick'], colors='b')
        ax_twin.tick_params(axis='y', labelsize=font_size_dict['ytick'], colors='r')
        
        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax_twin.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=font_size_dict['legend'])
        
        plt.tight_layout()
        
        if save_fig:
            fig_path = op.join(self.path_manager.base_figures_dir, 'career_stage_diversity.pdf')
            self.path_manager.save_pdf_file(passed_in_plt=plt, abs_file_path=fig_path, override=True, pad_inches=0.02)
        plt.show()

    def plot_career_stage_boxplots(self, results=None,
                                   figsize: Tuple[int, int] = (16, 8),
                                   font_size_dict: Optional[Dict[str, int]] = None,
                                   save_fig: bool = True,
                                   show_outliers: bool = False):
        """
        Plot diversity metrics by career stage with boxplots and mean lines
        """
        if font_size_dict is None:
            font_size_dict = {'title': 20, 'xlabel': 16, 'ylabel': 16, 'xtick': 16, 'ytick': 16, 'legend': 16, 'text': 14}
        
        # Filter authors with sufficient works
        filtered_results = [r for r in results if r['num_works'] >= 10]
        logger.info(f"Plotting career stage diversity for {len(filtered_results)} authors with ≥10 works")
        
        if len(filtered_results) == 0:
            logger.warning("No authors with sufficient works for plotting")
            return
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Define career stages
        career_stages = [5, 10, 15, 20, 25]
        
        # Collect data for each stage
        entropy_data = []
        hhi_data = []
        avg_entropies = []
        avg_hhis = []
        valid_stages = []
        
        for stage in career_stages:
            stage_entropies = []
            stage_hhis = []
            for result in filtered_results:
                if ('cumulative' in result['entropies'] and 
                    len(result['entropies']['cumulative']) >= stage):
                    stage_entropies.append(result['entropies']['cumulative'][stage-1])
                if ('cumulative' in result['hhis'] and 
                    len(result['hhis']['cumulative']) >= stage):
                    stage_hhis.append(result['hhis']['cumulative'][stage-1])
            
            if len(stage_entropies) >= 5:
                entropy_data.append(stage_entropies)
                hhi_data.append(stage_hhis)
                avg_entropies.append(np.mean(stage_entropies))
                avg_hhis.append(np.mean(stage_hhis))
                valid_stages.append(stage)
        
        if not valid_stages:
            logger.warning("No valid career stages with sufficient data")
            return
        
        # Create twin axis for HHI
        ax_twin = ax.twinx()
        
        # Adjust positions for side-by-side boxplots
        entropy_positions = [x - 0.3 for x in valid_stages]
        hhi_positions = [x + 0.3 for x in valid_stages]
        
        # Plot Shannon Entropy boxplots
        bp1 = ax.boxplot(
            entropy_data, positions=entropy_positions, widths=0.4, 
            patch_artist=True, showfliers=show_outliers,
            boxprops=dict(facecolor='seagreen', alpha=0.55, edgecolor='darkgreen'),
            medianprops=dict(color='darkgreen', linewidth=2),
            whiskerprops=dict(color='darkgreen', linewidth=1.5),
            capprops=dict(color='darkgreen', linewidth=1.5),
        )

        # Plot HHI boxplots
        bp2 = ax_twin.boxplot(
            hhi_data, positions=hhi_positions, widths=0.4, 
            patch_artist=True, showfliers=show_outliers,
            boxprops=dict(facecolor='khaki', alpha=0.85, edgecolor='goldenrod'),
            medianprops=dict(color='goldenrod', linewidth=2),
            whiskerprops=dict(color='goldenrod', linewidth=1.5),
            capprops=dict(color='goldenrod', linewidth=1.5),
        )

        # Plot mean lines
        line1 = ax.plot(
            entropy_positions, avg_entropies, 'o-', label='Mean Entropy', 
            color='mediumseagreen', linewidth=3, markersize=8, 
            markerfacecolor='mediumseagreen', markeredgecolor='darkgreen', zorder=8
        )
        line2 = ax_twin.plot(
            hhi_positions, avg_hhis, 'o-', label='Mean HHI', 
            color='goldenrod', linewidth=3, markersize=8, 
            markerfacecolor='khaki', markeredgecolor='goldenrod', zorder=8
        )

        # Customize axes
        ax.set_xlabel('Publication Order', fontsize=font_size_dict['xlabel'])
        ax.set_ylabel('Cumulative Shannon Entropy', color='mediumseagreen', fontsize=font_size_dict['ylabel'])
        ax_twin.set_ylabel('Herfindahl-Hirschman Index', color='goldenrod', fontsize=font_size_dict['ylabel'])
        ax.grid(True, alpha=0.3)

        # Set x-axis ticks to show only the main career stages
        ax.set_xticks(valid_stages)
        ax.set_xticklabels(valid_stages)

        # Set tick colors and sizes
        ax.tick_params(axis='x', labelsize=font_size_dict['xtick'])
        ax.tick_params(axis='y', labelsize=font_size_dict['ytick'], colors='mediumseagreen')
        ax_twin.tick_params(axis='y', labelsize=font_size_dict['ytick'], colors='goldenrod')

        plt.tight_layout()

        if save_fig:
            fig_path = op.join(self.path_manager.base_figures_dir, 'career_stage_diversity_combined.pdf')
            self.path_manager.save_pdf_file(passed_in_plt=plt, abs_file_path=fig_path, override=True, pad_inches=0.02)
        plt.show()

    def plot_windowed_comparison_boxplots(self, results=None,
                                          figsize: Tuple[int, int] = (16, 8),
                                          font_size_dict: Optional[Dict[str, int]] = None,
                                          save_fig: bool = True,
                                          show_outliers: bool = False,
                                          window_size: int = 5):
        """
        Plot diversity metrics by career stage with cumulative and windowed entropy comparison
        """
        if font_size_dict is None:
            font_size_dict = {'title': 20, 'xlabel': 16, 'ylabel': 16, 'xtick': 16, 'ytick': 16, 'legend': 16, 'text': 14}
        
        # Filter authors with sufficient works
        filtered_results = [r for r in results if r['num_works'] >= 10]
        logger.info(f"Plotting windowed comparison for {len(filtered_results)} authors with ≥10 works")
        
        if len(filtered_results) == 0:
            logger.warning("No authors with sufficient works for plotting")
            return
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Define career stages
        career_stages = [5, 10, 15, 20, 25]
        
        # Collect data for each stage
        cumulative_entropy_data = []
        windowed_entropy_data = []
        hhi_data = []
        avg_cumulative_entropies = []
        avg_windowed_entropies = []
        avg_hhis = []
        valid_stages = []
        
        window_key = f'window_{window_size}'
        
        for stage in career_stages:
            stage_cumulative_entropies = []
            stage_windowed_entropies = []
            stage_hhis = []
            
            for result in filtered_results:
                # Cumulative entropy
                if ('cumulative' in result['entropies'] and 
                    len(result['entropies']['cumulative']) >= stage):
                    stage_cumulative_entropies.append(result['entropies']['cumulative'][stage-1])
                
                # Windowed entropy
                if (window_key in result['entropies'] and 
                    len(result['entropies'][window_key]) >= stage):
                    stage_windowed_entropies.append(result['entropies'][window_key][stage-1])
                
                # HHI
                if ('cumulative' in result['hhis'] and 
                    len(result['hhis']['cumulative']) >= stage):
                    stage_hhis.append(result['hhis']['cumulative'][stage-1])
            
            if len(stage_cumulative_entropies) >= 5:
                cumulative_entropy_data.append(stage_cumulative_entropies)
                windowed_entropy_data.append(stage_windowed_entropies)
                hhi_data.append(stage_hhis)
                avg_cumulative_entropies.append(np.mean(stage_cumulative_entropies))
                avg_windowed_entropies.append(np.mean(stage_windowed_entropies))
                avg_hhis.append(np.mean(stage_hhis))
                valid_stages.append(stage)
        
        if not valid_stages:
            logger.warning("No valid career stages with sufficient data")
            return
        
        # Create twin axis for HHI
        ax_twin = ax.twinx()
        
        # Adjust positions for side-by-side boxplots
        cumulative_positions = [x - 0.3 for x in valid_stages]
        windowed_positions = valid_stages
        hhi_positions = [x + 0.3 for x in valid_stages]
        
        # Plot Cumulative Entropy boxplots
        bp1 = ax.boxplot(
            cumulative_entropy_data, positions=cumulative_positions, widths=0.23, 
            patch_artist=True, showfliers=show_outliers,
            boxprops=dict(facecolor='seagreen', alpha=0.55, edgecolor='darkgreen'),
            medianprops=dict(color='darkgreen', linewidth=2),
            whiskerprops=dict(color='darkgreen', linewidth=1.5),
            capprops=dict(color='darkgreen', linewidth=1.5),
        )

        # Plot Windowed Entropy boxplots
        bp3 = ax.boxplot(
            windowed_entropy_data, positions=windowed_positions, widths=0.23,
            patch_artist=True, showfliers=show_outliers,
            boxprops=dict(facecolor='mediumaquamarine', alpha=0.45, edgecolor='teal'),
            medianprops=dict(color='teal', linewidth=2),
            whiskerprops=dict(color='teal', linewidth=1.5),
            capprops=dict(color='teal', linewidth=1.5),
        )

        # Plot HHI boxplots
        bp2 = ax_twin.boxplot(
            hhi_data, positions=hhi_positions, widths=0.23, 
            patch_artist=True, showfliers=show_outliers,
            boxprops=dict(facecolor='khaki', alpha=0.85, edgecolor='goldenrod'),
            medianprops=dict(color='goldenrod', linewidth=2),
            whiskerprops=dict(color='goldenrod', linewidth=1.5),
            capprops=dict(color='goldenrod', linewidth=1.5),
        )

        # Plot mean lines
        line1 = ax.plot(
            cumulative_positions, avg_cumulative_entropies, 'o-', label='Mean Cumulative Entropy', 
            color='mediumseagreen', linewidth=3, markersize=8, 
            markerfacecolor='mediumseagreen', markeredgecolor='darkgreen', zorder=8
        )
        line3 = ax.plot(
            windowed_positions, avg_windowed_entropies, 's--', label=f'Mean Window-{window_size} Entropy',
            color='teal', linewidth=2, markersize=7, 
            markerfacecolor='mediumaquamarine', markeredgecolor='teal', zorder=8
        )
        line2 = ax_twin.plot(
            hhi_positions, avg_hhis, 'o-', label='Mean HHI', 
            color='goldenrod', linewidth=3, markersize=8, 
            markerfacecolor='khaki', markeredgecolor='goldenrod', zorder=8
        )

        # Customize axes
        ax.set_xlabel('Publication Order', fontsize=font_size_dict['xlabel'])
        ax.set_ylabel('Shannon Entropy', color='mediumseagreen', fontsize=font_size_dict['ylabel'])
        ax_twin.set_ylabel('Herfindahl-Hirschman Index', color='goldenrod', fontsize=font_size_dict['ylabel'])
        ax.grid(True, alpha=0.3)

        # Set x-axis ticks to show only the main career stages
        ax.set_xticks(valid_stages)
        ax.set_xticklabels(valid_stages)

        # Set tick colors and sizes
        ax.tick_params(axis='x', labelsize=font_size_dict['xtick'])
        ax.tick_params(axis='y', labelsize=font_size_dict['ytick'], colors='mediumseagreen')
        ax_twin.tick_params(axis='y', labelsize=font_size_dict['ytick'], colors='goldenrod')

        # Legends
        lines = line1 + line3
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left', fontsize=font_size_dict['legend'])
        ax_twin.legend([line2[0]], ['Mean HHI'], loc='upper right', fontsize=font_size_dict['legend'])

        plt.tight_layout()

        if save_fig:
            fig_path = op.join(self.path_manager.base_figures_dir, f'windowed_comparison_stage_{window_size}.pdf')
            self.path_manager.save_pdf_file(passed_in_plt=plt, abs_file_path=fig_path, override=True, pad_inches=0.02)
        plt.show()

    def plot_all_diversity_trends(self, results=None, save_plots=True):
        """
        Plot all diversity trends in a comprehensive figure
        """
        # Filter authors with sufficient works
        filtered_results = [r for r in results if r['num_works'] >= 10]
        logger.info(f"Plotting comprehensive trends for {len(filtered_results)} authors with ≥10 works")
        
        if len(filtered_results) == 0:
            logger.warning("No authors with sufficient works for plotting")
            return
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Research Diversity Evolution', fontsize=16)
        
        # Plot 1: Cumulative unique concept count
        ax1 = axes[0, 0]
        for result in filtered_results[:20]:  # Plot first 20 authors
            if 'cumulative' in result['unique_counts']:
                work_numbers = list(range(1, len(result['unique_counts']['cumulative']) + 1))
                ax1.plot(work_numbers, result['unique_counts']['cumulative'], alpha=0.3)
        ax1.set_xlabel('Publication Order')
        ax1.set_ylabel('Cumulative Unique Concepts')
        ax1.set_title('Cumulative Unique Concept Count')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Cumulative Shannon entropy
        ax2 = axes[0, 1]
        for result in filtered_results[:20]:
            if 'cumulative' in result['entropies']:
                work_numbers = list(range(1, len(result['entropies']['cumulative']) + 1))
                ax2.plot(work_numbers, result['entropies']['cumulative'], alpha=0.3)
        ax2.set_xlabel('Publication Order')
        ax2.set_ylabel('Shannon Entropy')
        ax2.set_title('Cumulative Shannon Entropy')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Cumulative HHI
        ax3 = axes[1, 0]
        for result in filtered_results[:20]:
            if 'cumulative' in result['hhis']:
                work_numbers = list(range(1, len(result['hhis']['cumulative']) + 1))
                ax3.plot(work_numbers, result['hhis']['cumulative'], alpha=0.3)
        ax3.set_xlabel('Publication Order')
        ax3.set_ylabel('HHI')
        ax3.set_title('Cumulative HHI (Concentration)')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Average metrics by career stage
        ax4 = axes[1, 1]
        career_stages = []
        avg_entropies = []
        avg_hhis = []
        
        for stage in [5, 10, 15, 20, 25]:
            stage_entropies = []
            stage_hhis = []
            for result in filtered_results:
                if ('cumulative' in result['entropies'] and 
                    len(result['entropies']['cumulative']) >= stage):
                    stage_entropies.append(result['entropies']['cumulative'][stage-1])
                if ('cumulative' in result['hhis'] and 
                    len(result['hhis']['cumulative']) >= stage):
                    stage_hhis.append(result['hhis']['cumulative'][stage-1])
            
            if stage_entropies:
                career_stages.append(stage)
                avg_entropies.append(np.mean(stage_entropies))
                avg_hhis.append(np.mean(stage_hhis))
        
        ax4_twin = ax4.twinx()
        line1 = ax4.plot(career_stages, avg_entropies, 'b-o', label='Avg Entropy')
        line2 = ax4_twin.plot(career_stages, avg_hhis, 'r-s', label='Avg HHI')
        
        ax4.set_xlabel('Career Stage (Number of Works)')
        ax4.set_ylabel('Average Shannon Entropy', color='b')
        ax4_twin.set_ylabel('Average HHI', color='r')
        ax4.set_title('Average Diversity by Career Stage')
        ax4.grid(True, alpha=0.3)
        
        # Combine legends
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4_twin.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.tight_layout()
        
        if save_plots:
            fig_path = os.path.join(self.results_dir, 'diversity_trends_comprehensive.png')
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            logger.info(f"Comprehensive diversity trends plot saved to {self.results_dir}")
        
        plt.show()
    
    def save_results(self, results, filename='diversity_results.parquet'):
        """
        Save diversity analysis results
        
        Args:
            results: List of diversity analysis results
            filename: Output filename
        """
        # Convert results to DataFrame
        df_results = pd.DataFrame(results)
        
        # Save to parquet
        output_path = os.path.join(self.results_dir, filename)
        df_results.to_parquet(output_path)
        
        logger.info(f"Diversity results saved to {output_path}")
        
        # Calculate summary statistics
        summary_stats = {}
        if results:
            summary_stats['total_authors'] = len(results)
            summary_stats['avg_num_works'] = np.mean([r['num_works'] for r in results])
            
            # Get final values for cumulative metrics
            final_unique_counts = []
            final_entropies = []
            final_hhis = []
            
            for r in results:
                if 'cumulative' in r['unique_counts'] and r['unique_counts']['cumulative']:
                    final_unique_counts.append(r['unique_counts']['cumulative'][-1])
                if 'cumulative' in r['entropies'] and r['entropies']['cumulative']:
                    final_entropies.append(r['entropies']['cumulative'][-1])
                if 'cumulative' in r['hhis'] and r['hhis']['cumulative']:
                    final_hhis.append(r['hhis']['cumulative'][-1])
            
            if final_unique_counts:
                summary_stats['avg_final_unique_concepts'] = np.mean(final_unique_counts)
            if final_entropies:
                summary_stats['avg_final_entropy'] = np.mean(final_entropies)
            if final_hhis:
                summary_stats['avg_final_hhi'] = np.mean(final_hhis)
        
        # Save summary statistics
        summary_path = os.path.join(self.results_dir, 'diversity_summary.txt')
        with open(summary_path, 'w') as f:
            for key, value in summary_stats.items():
                f.write(f"{key}: {value:.4f}\n")
        
        logger.info(f"Summary statistics saved to {summary_path}")
        
        return output_path


# Example usage
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = DiversityAnalyzer(debugging=True)
    
    # Analyze all authors with multiple windows
    windows = [1, 3, 5, -1]  # -1 for cumulative
    results = analyzer.analyze_all_authors(windows=windows, skip_cache = False)
    analyzer.plot_diversity_metrics_boxplot_comparison(results, metric_type = 'entropies', figsize=(12, 3), save_fig=False)
    # Save results
    analyzer.save_results(results)
    
    # Generate all plots
    analyzer.plot_cumulative_unique_concepts(results)
    analyzer.plot_cumulative_entropy(results)
    analyzer.plot_entropy_comparison(results, window_size=5)
    analyzer.plot_cumulative_hhi(results)
    analyzer.plot_career_stage_diversity(results)
    analyzer.plot_career_stage_boxplots(results)
    analyzer.plot_windowed_comparison_boxplots(results, window_size=5)
    analyzer.plot_all_diversity_trends(results)
