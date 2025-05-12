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


class SubFieldEvaluator:
    def __init__(self, sub_concepts=True):
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
            skiped_author_path = op.join(self.evaluation_path_period, f'skipped_authors_{batch_id}.pkl')
            # Check if file exists
            if not op.exists(file_path):
                print(f"Warning: File {file_path} does not exist. Skipping.")
                continue
                
            # Load batch data
            try:
                with open(file_path, 'rb') as f:
                    batch_data = pickle.load(f)
                
                with open(skiped_author_path, 'rb') as f:
                    skipped_authors = pickle.load(f)
                total_authors = len(batch_data) + len(skipped_authors)

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

    def _visualize_precision_heatmap(self, result_df, fig_size = (4.2,3.5), save_pdf = True):
        # Define discipline abbreviations
        discipline_abbrev = {
            'Computer science': 'C.S.', 
            'Engineering': 'Eng.', 
            'Mathematics': 'Math.',
            'Physics': 'Phys.',
        }
        disciplines = ['Computer science', 'Engineering', 'Mathematics', 'Physics']

        # Create abbreviated labels for both axes
        abbrev_disciplines = [discipline_abbrev.get(d, d[:4]) for d in disciplines]
        
        # Create a figure
        plt.figure(figsize=fig_size)
        
        # Create heatmap data
        heatmap_data = result_df.pivot(index='author_discipline', 
                                    columns='eval_discipline', 
                                    values='mean_precision')
        
        # Reorder to match disciplines order
        heatmap_data = heatmap_data.reindex(index=disciplines, columns=disciplines)
        
        # Create the heatmap with abbreviated labels cbar_kws={'label': 'Mean Precision'}
        sns.heatmap(heatmap_data, annot=True, cmap='BuPu', fmt='.3f', 
            linewidths=.5)
        
        # Set abbreviated labels
        plt.xticks(np.arange(len(abbrev_disciplines)) + 0.5, abbrev_disciplines, rotation=0, fontsize=12)
        plt.yticks(np.arange(len(abbrev_disciplines)) + 0.5, abbrev_disciplines, fontsize=12)
        
        # plt.title('Mean Precision by Author and Evaluation Discipline', fontsize=14)
        plt.xlabel('Evaluation Discipline', fontsize=14)
        plt.ylabel('Author Discipline', fontsize=14)
        
        plt.tight_layout()
        if save_pdf:
            fig_path = op.join(self.results_path, 'precision_heatmap_by_discipline.pdf')
            plt.savefig(fig_path, bbox_inches='tight')
            print(f'pdf file save at {fig_path}')
        plt.show()

    def _visualize_precision_violins(self, result_df, fig_size=(6, 3.5), save_pdf = True):
        # Define discipline abbreviations
        discipline_abbrev = {
            'Computer science': 'C.S.', 
            'Engineering': 'Eng.', 
            'Mathematics': 'Math.',
            'Physics': 'Phys.',
        }
        disciplines = ['Computer science', 'Engineering', 'Mathematics', 'Physics']

        # Create abbreviated labels
        abbrev_disciplines = [discipline_abbrev.get(d, d[:4]) for d in disciplines]
        
        # Create a figure with slightly more height to accommodate bottom legend
        plt.figure(figsize=fig_size)
        
        # Define color mapping, one color per author discipline
        colors = plt.cm.tab10(np.linspace(0, 1, len(disciplines)))
        
        # Prepare data for each evaluation discipline
        positions = []
        violin_data = []
        violin_colors = []
        
        # Calculate x-axis positions
        eval_positions = np.arange(1, len(disciplines) + 1) * 5  # Enough spacing between evaluation discipline groups
        
        for i, eval_disc in enumerate(disciplines):
            for j, author_disc in enumerate(disciplines):
                row = result_df[(result_df['author_discipline'] == author_disc) & 
                            (result_df['eval_discipline'] == eval_disc)]
                if not row.empty:
                    # Calculate violin position: evaluation position + author discipline offset
                    pos = eval_positions[i] + j - 1.5  # Offset to distribute violins evenly around evaluation position
                    positions.append(pos)
                    violin_data.append(row['precision_data'].iloc[0])
                    violin_colors.append(colors[j])  # Color based on author discipline
        
        # Draw violin plots
        for i, (pos, data, color) in enumerate(zip(positions, violin_data, violin_colors)):
            vp = plt.violinplot(data, positions=[pos], showmeans=True, widths=1.3)
            
            # Set violin plot colors
            for pc in vp['bodies']:
                pc.set_facecolor(color)
                pc.set_edgecolor(color)
                pc.set_alpha(0.7)
            vp['cbars'].set_color('black')
            vp['cbars'].set_linewidth(1)
            vp['cmeans'].set_color('black')
            vp['cmeans'].set_linewidth(1)

        # Set x-axis labels
        plt.xticks(eval_positions, abbrev_disciplines)
        plt.tick_params(axis='y', labelsize=12)
        plt.tick_params(axis='x', labelsize=12)
        
        # Add legend at the bottom in a horizontal layout
        legend_elements = [plt.Line2D([0], [0], color=colors[i], lw=4, label=abbrev_disciplines[i]) 
                        for i in range(len(disciplines))]
        
        # Adjust the bottom margin to make room for the legend
        plt.subplots_adjust(bottom=0.2)
        plt.ylim(-0.1, 1.04)

        # Place legend below the plot but above x-axis labels
        plt.legend(handles=legend_elements, 
                # title="Author Discipline", 
                loc='upper center', 
                bbox_to_anchor=(0.5, 0.128),  # Position below the plot
                ncol=4,  # Arrange in one row with 4 columns
                frameon=False,
                fontsize = 12)  # Remove legend frame for cleaner look
        
        # plt.title('Precision Distribution by Evaluation Discipline', fontsize=14)
        plt.xlabel('Evaluation Discipline', fontsize=14)
        plt.ylabel('Precision', fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        if save_pdf:
            fig_path = op.join(self.results_path, 'precision_distribution_violin_plots.pdf')
            plt.savefig(fig_path, bbox_inches='tight')
            print(f'pdf file save at {fig_path}')
        plt.show()

    def cross_sub_field_validation(self, df=None):
        if df is None:
            df = self.collect_all_precision_data()
        
        disciplines = ['Computer science', 'Engineering', 'Mathematics', 'Physics']
        author_ids = df['author_id'].drop_duplicates().tolist()
        
        # Load or query author field data
        author_field_path = op.join(self.results_path, 'author_field.csv')
        if not op.exists(author_field_path):
            author_field = self.db_manager.query_table(
                table_name='author_yearlyfeature_field_geq10pubs',
                columns=['author_id', 'field'],
                where_conditions=[f'''author_id in ('{"','".join(author_ids)}')''']
            )
            self.path_manager.save_csv_file(author_field, abs_file_path=author_field_path, index=False)
        else: 
            author_field = pd.read_csv(author_field_path)
        
        # Create field to authors mapping
        field2authors = author_field.groupby('field')['author_id'].apply(set).to_dict()
        
        # Get authors with positive precision in each discipline
        discipline_authors = {}
        for discipline in disciplines:
            df_discipline = df.loc[df.discipline == discipline]
            df_discipline_effective = df_discipline.loc[df_discipline['Precision_1'] > 0]
            discipline_authors[discipline] = set(df_discipline_effective['author_id'].drop_duplicates())
            
            # Ensure we only include authors that are actually in that field
            if discipline in field2authors:
                discipline_authors[discipline] = discipline_authors[discipline].intersection(field2authors[discipline])
        
        cross_validation_path = op.join(self.results_path, 'cross_validation_path.parquet')
        if not op.exists(cross_validation_path):
            # Create a dataframe for analysis
            result_rows = []
            
            # For each author discipline
            for author_discipline in disciplines:
                # Get authors from this discipline
                authors = discipline_authors[author_discipline]
                
                # For each evaluation discipline
                for eval_discipline in disciplines:
                    # Get precision data for these authors in this evaluation discipline
                    precision_data = df[(df['author_id'].isin(authors)) & 
                                    (df['discipline'] == eval_discipline)]['Precision_1']
                    
                    # Calculate metrics
                    mean_precision = precision_data.mean()
                    median_precision = precision_data.median()
                    std_precision = precision_data.std()
                    count = len(precision_data)
                    
                    # Store results
                    result_rows.append({
                        'author_discipline': author_discipline,
                        'eval_discipline': eval_discipline,
                        'mean_precision': mean_precision,
                        'median_precision': median_precision,
                        'std_precision': std_precision,
                        'count': count, # number of authors
                        'precision_data': precision_data.tolist()  # Store raw data for plotting
                    })
            
            # Convert to DataFrame
            result_df = pd.DataFrame(result_rows)
            result_df.to_parquet(cross_validation_path, index=False)
        else: result_df = pd.read_parquet(cross_validation_path)
        
        # Perform statistical tests
        # test_results_df = self._statistical_test_cross_discipline(result_df, disciplines)

        return result_df
        

    def _statistical_test_cross_discipline(self, result_df, disciplines):
        """Perform statistical tests to verify if authors perform better in their own discipline"""
        from scipy import stats
        
        # Create a DataFrame to store test results
        test_results = []
        
        for author_disc in disciplines:
            # Get precision for authors in their own discipline
            own_disc_row = result_df[(result_df['author_discipline'] == author_disc) & 
                                    (result_df['eval_discipline'] == author_disc)]
            
            if own_disc_row.empty:
                continue
                
            own_disc_precision = own_disc_row['precision_data'].iloc[0]
            
            # Compare with other disciplines
            for eval_disc in disciplines:
                if eval_disc == author_disc:
                    continue
                    
                other_disc_row = result_df[(result_df['author_discipline'] == author_disc) & 
                                        (result_df['eval_discipline'] == eval_disc)]
                
                if other_disc_row.empty:
                    continue
                    
                other_disc_precision = other_disc_row['precision_data'].iloc[0]
                
                # Perform paired t-test
                t_stat, p_value = stats.ttest_ind(own_disc_precision, other_disc_precision, 
                                                equal_var=False)
                
                test_results.append({
                    'author_discipline': author_disc,
                    'compared_discipline': eval_disc,
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'higher_in_own': t_stat > 0
                })
        
        # Convert to DataFrame and save
        test_results_df = pd.DataFrame(test_results)
        test_results_df.to_csv(op.join(self.results_path, 'cross_discipline_tests.csv'), index=False)
        
        return test_results_df


# Usage example
if __name__ == "__main__":
    evaluator = SubFieldEvaluator()
    result_df = evaluator.cross_sub_field_validation()
    # evaluator._visualize_precision_heatmap(result_df)
    evaluator._visualize_precision_violins(result_df)
    # df = evaluator.collect_all_precision_data()

    print(5)