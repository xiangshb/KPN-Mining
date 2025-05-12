import os.path as op
import pandas as pd
import numpy as np
import time, os
from functools import wraps
import matplotlib.pyplot as plt
import networkx as nx
from typing import Union
import logging

def format_time(seconds):
    if seconds < 60: return "{:.3f}s".format(seconds)
    days, seconds_remainder = divmod(seconds, 86400)
    hours, seconds_remainder = divmod(seconds_remainder, 3600)
    minutes, seconds_remainder = divmod(seconds_remainder, 60)
    result = ""
    if days > 0: result += "{:02d}:".format(int(days))
    if hours > 0 or result: result += "{:02d}:".format(int(hours))
    if minutes > 0 or result: result += "{:02d}:".format(int(minutes))
    result += "{:.3f}".format(seconds_remainder)
    final_output = "{:.3f}s ({})".format(seconds, result.strip())
    return final_output

def calculate_runtime(func):
    """Decorator to calculate and log function runtime."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        if kwargs.get('show_runtime', True):
            print(f"{func.__name__} completed in {format_time(end_time - start_time)}")
        return result
    return wrapper

# cmd_running = False # set True only when is_in_server = False
# filedir = '../files' if cmd_running else './files'

class PathManager:
    # Modify manually to run on multiple synchronous PCs
    active_device = 'server_1'
    external_file_dir_dict = {'server_1': '../../../data/files/prerequisite-learning-files', 
                              'nut_cloud_based_pc': 'D:\data\prerequisite-learning-files',
                              'server_2': '/data/prerequisite-learning-files'
                              }
    is_in_sustech_ds_public_server = (active_device == 'server_1')
    current_folder_dir = op.dirname(op.dirname(op.abspath(__file__)))
    current_working_dir = os.getcwd() # default current_folder_dir = current_working_dir
    
    def __init__(self): # exists via class instantiation
        self.base_file_dir = self.ensure_folder_exists(op.join(self.current_folder_dir, 'files'))
        self.base_data_dir = self.ensure_folder_exists(op.join(self.base_file_dir, 'data'))
        self.base_figures_dir = self.ensure_folder_exists(op.join(self.base_file_dir, 'figures'))
        self.base_edges_dir = self.ensure_folder_exists(op.join(self.base_file_dir, 'edges'))
        self.external_file_dir = self.ensure_folder_exists(self.external_file_dir_dict[self.active_device])
        self.external_data_dir = self.ensure_folder_exists(op.join(self.external_file_dir, 'data'))
        self.concepts_dir = self.ensure_folder_exists(op.join(self.external_file_dir, 'Concepts'))
        self.ccns_dir = self.ensure_folder_exists(op.join(self.external_file_dir, 'CCNs'))
        self.concepts_sequences_dir = self.ensure_folder_exists(op.join(self.concepts_dir, 'Sequences'))

        self.external_outer_file_dir = op.dirname(self.external_file_dir)
        self.navigation_embedding_dir = self.ensure_folder_exists(op.join(self.external_outer_file_dir, 'Embeddings'))

    @classmethod
    def ensure_folder_exists(cls, folder_path):
        if not op.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Created folder: {folder_path}")
        return folder_path
    
    def check_file_write_permission(self, abs_file_path: str = None, override: bool = False):
        if not op.exists(abs_file_path): 
            save_file = True
            file_status_infor = f"file: {abs_file_path} does not exists and saved"
        elif override: 
            save_file = True
            file_status_infor = f"file: {abs_file_path} exists and override"
        else: 
            save_file = False
            file_status_infor = f"file: {abs_file_path} exists and no override (did not save)"
        return save_file, file_status_infor
    
    def save_gexf_file(self, G: Union[nx.Graph, nx.DiGraph] = None, abs_file_path: str = None, override: bool = False, update_version:bool = False):
        save_file, file_status_infor = self.check_file_write_permission(abs_file_path, override)
        if save_file:
            nx.write_gexf(G, abs_file_path)
        print(file_status_infor)
        if update_version:
            import xml.etree.ElementTree as ET
            tree = ET.parse(abs_file_path) # Read the saved GEXF file as XML
            root = tree.getroot()
            root.attrib['version'] = '1.3'
            tree.write(abs_file_path)
            print('updated GEXF with version 1.3')
    
    def save_csv_file(self, variable, file_name: str = None, abs_file_path:str = None, index: bool = False, override: bool = False):
        abs_file_path = op.join(self.base_edges_dir, file_name) if abs_file_path == None else abs_file_path
        save_file, file_status_infor = self.check_file_write_permission(abs_file_path, override)
        if save_file: variable.to_csv(abs_file_path, index = index)
        print(file_status_infor)
    
    def save_npy_file(self, variable, abs_file_path, override: bool = False):
        save_file, file_status_infor = self.check_file_write_permission(abs_file_path, override)
        if save_file: np.save(abs_file_path, variable)
        print(file_status_infor)
    
    def save_png_file(self, passed_in_plt: plt = None, file_name: str = None,  override: bool = False, bbox_inches: str = 'tight', dpi: int = 600):
        if passed_in_plt is None: raise ValueError('The passed plt is None')
        abs_file_path = op.join(self.base_figures_dir, file_name)
        save_file, file_status_infor = self.check_file_write_permission(abs_file_path, override)
        if save_file: plt.savefig(abs_file_path, dpi = dpi, bbox_inches = bbox_inches)
        print(file_status_infor)
    
    def save_pdf_file(self, passed_in_plt: plt = None, file_name: str = None, abs_file_path = None, override: bool = False, bbox_inches: str = 'tight', pad_inches: float = 0): # pad_inches = 0.1
        if passed_in_plt is None: raise ValueError('The passed plt is None')
        if file_name is None: 
            if abs_file_path == None:
                raise ValueError('The file name is None')
        else:
            if abs_file_path == None:
                abs_file_path = op.join(self.base_figures_dir, file_name)
        save_file, file_status_infor = self.check_file_write_permission(abs_file_path, override)
        if save_file: 
            passed_in_plt.savefig(abs_file_path ,bbox_inches = bbox_inches, pad_inches = pad_inches)
        print(file_status_infor)
    
    def generate_filename(self,):
        pass

    @calculate_runtime
    def read_csv_file(self, abs_file_path, index_col = None):
        df_data = pd.read_csv(abs_file_path) if index_col is None else pd.read_csv(abs_file_path, index_col = index_col)
        return df_data

class DatabaseParams:
    # the following default parameters are randomly set for example
    def __init__(self, server_ip: str = '89.128.116.15', server_port: int = 22, database_name: str = 'your_database_name', # openalex2022
                 username: str = 'user_name', password: str = 'user_password', charset:str = 'utf8mb4'):
        self.server_ip = server_ip
        self.server_port = server_port
        self.database_name = database_name
        self.username = username
        self.password = password
        self.charset = charset

if __name__=='__main__':
    path_manager = PathManager()
    database_params = DatabaseParams(database_name = 'openalex2024')
    print(5)
