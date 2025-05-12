import os, sys
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.append(project_dir)

import pandas as pd
import numpy as np
from typing import Union, List
from tqdm import tqdm, trange
from urllib.parse import quote_plus
from utils.config import DatabaseParams, calculate_runtime
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool
from sqlalchemy import text

# Base class with common functionality
class BaseDBManager(DatabaseParams):
    def __init__(self, **kwargs):
        super(BaseDBManager, self).__init__(**kwargs)
    
    def build_query(self, table_name: str, 
                   columns: Union[List[str], None] = None, 
                   columns_trim_enter: Union[List[str], None] = None, 
                   join_tables: Union[List[str], None] = None, 
                   join_conditions: Union[List[str], None] = None, 
                   where_conditions: Union[List[str], None] = None) -> str:
        # Process column names
        column_clause = '*' if columns is None else ', '.join(columns)
        if columns_trim_enter is not None:
            for column_ in columns_trim_enter:
                alias = column_.split('.')[-1]
                column_clause = column_clause.replace(column_, f'''trim("\r" from {column_}) as {alias}''')
                
        # Process JOIN statements
        join_clause = ''
        if join_tables and join_conditions:
            assert len(join_tables) == len(join_conditions), "Join tables and conditions must match"
            for i, (table, condition) in enumerate(zip(join_tables, join_conditions)):
                join_clause += f"LEFT JOIN `{table}` ON {condition}"
                if i < len(join_tables) - 1:  
                    join_clause += ' '

        # Process WHERE conditions
        where_clause = ''
        if where_conditions:
            where_clause = 'WHERE ' + ' AND '.join(where_conditions)

        # Build the final SQL query statement
        query = f"SELECT {column_clause} FROM `{table_name}` {join_clause} {where_clause}"

        return query

class DatabaseManager(BaseDBManager):
    def __init__(self, **kwargs):
        super(DatabaseManager, self).__init__(**kwargs)
        self.engine = None
        
    def get_engine(self):
        """Initialize and return SQLAlchemy engine with connection pooling"""
        if self.engine is None:
            self.engine = create_engine(
                f"mysql+pymysql://{self.username}:{quote_plus(self.password)}@{self.server_ip}:{self.server_port}/{self.database_name}?charset={self.charset}",
                poolclass=QueuePool,
                pool_size=90,
                max_overflow=8,
                pool_timeout=30,
                pool_recycle=3600,
                pool_pre_ping=True,
            )
        return self.engine
    
    def close(self):
        """Close the database connection"""
        if self.engine is not None:
            self.engine.dispose()
            
    def get_total_rows(self, query: str) -> int:
        """Get total number of rows that would be returned by a query"""
        count_query = f"SELECT COUNT(*) FROM ( {query} ) AS subquery"
        with self.get_engine().connect() as connection:
            result = connection.execute(count_query).scalar()
            return result if result else 0
    
    def execute_query(self, query: str, batch_read: bool = False, batch_size: int = 5000) -> pd.DataFrame:
        try:
            if batch_read:
                df_batches = []
                with self.get_engine().connect() as connection:
                    result = connection.execute(text(query)).yield_per(batch_size)
                    columns = result.keys()
                    
                    for chunk in result.partitions(batch_size):
                        df_batch = pd.DataFrame(chunk, columns=columns)
                        df_batches.append(df_batch)
                        
                df = pd.concat(df_batches, ignore_index=True) if df_batches else pd.DataFrame()
            else:
                with self.get_engine().connect() as connection:
                    result = connection.execute(text(query))
                    df = pd.DataFrame(result.fetchall(), columns=result.keys())
                    
            return df.drop_duplicates().reset_index(drop=True)
        except Exception as e:
            print(f"Error executing query: {e}")
            raise

    def execute_query_parallel(self, query: str, batch_size: int = 50000, max_workers: int = 8) -> pd.DataFrame:
        from concurrent import futures
        try:
            total_rows = self.get_total_rows(query)
            batches = np.ceil(total_rows / batch_size).astype(int)
            
            if batches <= 1:
                # For small queries, just use regular execution
                return pd.read_sql(query, con=self.get_engine())
            
            # Prepare batch offsets
            offsets = [i * batch_size for i in range(batches)]
            
            # Define a worker function to fetch a single batch
            def fetch_batch(offset):
                batch_query = f"{query} LIMIT {batch_size} OFFSET {offset}"
                with self.get_engine().connect() as connection:
                    return pd.read_sql(batch_query, con=connection)
            
            # Execute batches in parallel
            df_batches = []
            with futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_offset = {executor.submit(fetch_batch, offset): i for i, offset in enumerate(offsets)}
                
                with tqdm(total=batches, desc='Reading batches in parallel', unit='batch') as pbar:
                    for future in futures.as_completed(future_to_offset):
                        try:
                            df_batch = future.result()
                            df_batches.append(df_batch)
                        except Exception as e:
                            batch_idx = future_to_offset[future]
                            print(f"Batch {batch_idx} generated an exception: {e}")
                        pbar.update(1)
            
            # Combine all batches
            df = pd.concat(df_batches, ignore_index=True)
            return df.drop_duplicates().reset_index(drop=True)
        except Exception as e:
            print(f"Error executing parallel query: {e}")
            raise

    @calculate_runtime
    def query_table(self, table_name: str, 
                   columns: Union[List[str], None] = None, 
                   columns_trim_enter: Union[List[str], None] = None, 
                   join_tables: Union[List[str], None] = None, 
                   join_conditions: Union[List[str], None] = None, 
                   where_conditions: Union[List[str], None] = None,
                   batch_read: bool = False,
                   batch_size: int = 5000,
                   parallel: bool = False,
                   show_runtime: bool = True) -> pd.DataFrame:
        
        query = self.build_query(
                    table_name=table_name,
                    columns=columns,
                    columns_trim_enter=columns_trim_enter,
                    join_tables=join_tables,
                    join_conditions=join_conditions,
                    where_conditions=where_conditions
                )
        
        if not parallel:
            return self.execute_query(query=query, batch_read=batch_read, batch_size=batch_size)
        else: return self.execute_query_parallel(query=query)
        
if __name__=='__main__':
    # db_manager = DatabaseManager()
    db_manager = DatabaseManager()
    # data = db_manager.query_table(parallel = True, table_name = 'concepts', columns=['id', 'level', 'display_name'], show_runtime=True) # driver= 'sqlalchemy' or pymysql
    # data = db_manager.query_table(table_name = 'concepts', columns=['id', 'level', 'display_name'], batch_read=True) # driver= 'sqlalchemy' or pymysql
    data = db_manager.query_table(table_name = 'concepts', columns=['id', 'level', 'display_name'], batch_read=False) # driver= 'sqlalchemy' or pymysql
    print('Testing')
