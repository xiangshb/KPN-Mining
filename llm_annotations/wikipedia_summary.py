import os, tiktoken
import time, itertools
import wikipedia
import pandas as pd
import requests
import re
from bs4 import BeautifulSoup
import wikipediaapi
from requests.exceptions import ReadTimeout, RequestException
from openai import OpenAI

class TokenCounter:
    def __init__(self, model="gpt-4-1106-preview"):
        self.encoding = tiktoken.encoding_for_model(model)
    
    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))
    
    def count_tokens_batch(self, texts: list[str]) -> list[int]:
        return [len(self.encoding.encode(text)) for text in texts]

class WikipediaSummarizer:
    def __init__(self, csv_file, summary_folder="summary", api_key=None):
        self.csv_file = csv_file
        self.summary_folder = summary_folder
        self.concepts_df = None
        self.summaries = {}
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is not set.")
        self.client = OpenAI(base_url='', api_key=self.api_key)
        self.total_input_tokens = 0

    def load_concepts(self):
        self.concepts_df = pd.read_csv(self.csv_file)
        self.concepts_df = self.concepts_df.loc[self.concepts_df.level <= 1][
            ['display_name', 'discipline_category_refined', 'all_level_0_ancestors']
        ].reset_index(drop=True)

    def get_wikipedia_content_url(self, concept, first_paragraph=False):
        url = f"https://en.wikipedia.org/wiki/{concept.replace(' ', '_')}"
        with requests.Session() as session:
            try:
                response = session.get(url)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, 'html.parser')
                content_paragraphs = soup.select("div.mw-parser-output p")
                if not content_paragraphs:
                    return None
                if first_paragraph:
                    concept_summary = content_paragraphs[0].get_text()
                else:
                    concept_summary = ' '.join([p.get_text() for p in content_paragraphs])
                concept_summary = re.sub(r'\[[0-9]+\]', '', concept_summary)
                if concept_summary:
                    return concept_summary.strip()
                else:
                    return None
            except requests.RequestException as e:
                print(f"Error fetching Wikipedia content for '{concept}': {e}")
                return None

    def get_wikipedia_summary_api(self, concept, retries=3, delay=2):
        wiki_wiki = wikipediaapi.Wikipedia(
            language='en', 
            user_agent="YourAppName/1.0 (contact@example.com)"
        )
        for attempt in range(retries):
            try:
                page = wiki_wiki.page(concept)
                if page.exists():
                    return page.summary
                else:
                    print(f"Page '{concept}' does not exist via API. Trying to fetch via URL.")
                    return self.get_wikipedia_content_url(concept)
            except ReadTimeout:
                print(f"Read timeout error on attempt {attempt + 1} for '{concept}'. Retrying in {delay} seconds...")
                time.sleep(delay)
            except RequestException as e:
                print(f"Request error on attempt {attempt + 1} for '{concept}': {e}. Retrying in {delay} seconds...")
                time.sleep(delay)
        return None

    def get_wikipedia_summary(self, concept):
        try:
            concept_summary = wikipedia.summary(concept)
            if concept_summary:
                return concept_summary
            else:
                return None
        except wikipedia.exceptions.DisambiguationError as e:
            print(f"Disambiguation error for '{concept}'. Trying the first suggested option '{e.options[0]}'.")
            return self.get_wikipedia_summary(e.options[0])
        except Exception as e:
            print(f"Error accessing Wikipedia for concept '{concept}': {e}")
            print("Trying access via API.")
            return self.get_wikipedia_summary_api(concept)

    def save_summary_to_file(self, concept_name, summary):
        if summary is not None:
            if not os.path.exists(self.summary_folder):
                os.makedirs(self.summary_folder)
            filename = os.path.join(self.summary_folder, f"{concept_name}.csv")
            df_summary = pd.DataFrame({'summary': [summary]})
            df_summary.to_csv(filename, index=False)

    def load_summary_from_file(self, concept_name):
        filename = os.path.join(self.summary_folder, f"{concept_name}.csv")
        if os.path.exists(filename):
            df_summary = pd.read_csv(filename)
            summary = df_summary['summary'].values[0]
            if pd.isna(summary):
                return None
            return summary
        return None

    def summarize_with_gpt4(self, concept, text, model_name = 'gpt-4o'):
        try:
            system_message = "You are a helpful assistant that summarizes texts."
            prompt = f"Summarize the description of concept '{concept}' as one clear definitional sentence within 35 words:\n\n{text}"
            response = self.client.chat.completions.create(
                # model="gpt-4", # gpt-4 davinci
                model = model_name, # gpt-4-1106-preview
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=100 # max outpu tokens
            )
            # Update token counts
            input_tokens = response.usage.prompt_tokens
            self.total_input_tokens += input_tokens
            # Print token usage
            print(f"Current tokens: {input_tokens}  All tokens so far: {self.total_input_tokens}")

            summary = response.choices[0].message.content.strip()
            return summary
        except Exception as e:
            print(f"Error summarizing with {model_name}: {e}")
            return None

    def summary_simplify(self, save_path):
        if self.concepts_df is not None:
            if 'wikipedia_summary' not in self.concepts_df.columns:
                print("No summaries found to simplify.")
                return
            # 初始化待处理的索引列表和简化摘要列
            pending_indices = list(self.concepts_df.index)
            self.concepts_df['simplified_summary'] = None
            while pending_indices:
                print(f"\nRemaining entries to process: {len(pending_indices)}")
                for index in pending_indices[::-1]:
                    row = self.concepts_df.iloc[index]
                    long_summary = row['wikipedia_summary']
                    
                    print(f"Processing summary for {index}-th '{row['display_name']}':")
                    short_summary = self.summarize_with_gpt4(row['display_name'], long_summary)
                    if short_summary is not None:
                        self.concepts_df.at[index, 'simplified_summary'] = short_summary
                        pending_indices.remove(index)
                        print(f"Successful")# 保存到CSV文件
                        self.concepts_df[['display_name','simplified_summary']].to_csv(save_path, index=False)
                    else:
                        print(f"Failed")
            print("\nAll entries have been successfully processed!")
        else:
            print("Concepts DataFrame is None. Please load concepts first.")

    def process_concepts(self):
        if self.concepts_df is None:
            self.load_concepts()
        # for index, row in self.concepts_df.iterrows():
        #     concept_name = row['display_name']
        #     print(f'Processing concept {index}: {concept_name}')
        #     summary = self.load_summary_from_file(concept_name)
        #     if summary is None:
        #         summary = self.get_wikipedia_summary(concept_name)
        #         if summary is None:
        #             print(f"No valid summary found for {concept_name}")
        #         else:
        #             self.save_summary_to_file(concept_name, summary)
        #     self.concepts_df.at[index, 'wikipedia_summary'] = summary
        save_path='final_GPT4_short_within_35_summaries.csv'
        self.concepts_df = pd.read_csv('final_GPT4_short_summaries.csv')
        self.concepts_df.columns = ['display_name', 'wikipedia_summary']
        self.summary_simplify(save_path)
        # 如果需要，可以将结果保存到文件
        self.concepts_df[['display_name','simplified_summary']].to_csv(save_path, index=False)
        print(self.concepts_df[['display_name', 'simplified_summary']])

    def combination_token_counts(self, n_fix_tokens = 137):
        df_short_summary = pd.read_csv('final_GPT4_short_summaries.csv')
        token_counter = TokenCounter()
        total_tokens = 0
        df_short_summary['simplified_summary'] = df_short_summary['simplified_summary'].apply(lambda x: x.replace('\n\n\n', '\n\n').replace('\n\n', '\n'.replace('\n', ' ')))
        df_short_summary['tokens'] = df_short_summary['simplified_summary'].apply(token_counter.count_tokens)
        total_tokens = sum(t1 + t2 + n_fix_tokens for t1, t2 in itertools.combinations(df_short_summary['tokens'], 2))

        return total_tokens

if __name__ == "__main__":
    # 创建 WikipediaSummarizer 实例并处理概念
    api_key = 'you_api_key' # pament 2
    summarizer = WikipediaSummarizer('All_concepts_with_ancestors.csv', api_key=api_key)
    summarizer.process_concepts()
    # total_tokens = summarizer.combination_token_counts()
