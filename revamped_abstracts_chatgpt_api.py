# -*- coding: utf-8 -*-


!pip install openai
import pandas as pd
import os
import openai
import time
import math
from google.colab import files

openai.api_key="xxxxx"
\!curl https://sdk.cloud.google.com | bash

!gcloud auth login

!gcloud config set project abstract-output

"""# load data"""

try:
    df_batch_10 = pd.read_csv('/content/df_batch_10_processed_8500.csv',error_bad_lines=False)

except pd.errors.ParserError as e:
    # Handle the parsing error by returning NaN rows
    print(f"Parser error: {e}")
    abstract_df = pd.DataFrame(columns=['abstract'])  # Empty DataFrame with 'abstract' column
    abstract_df['abstract'] = float('NaN')  # Fill the 'abstract' column with NaN values

"""# set up api"""

def commercialize_abstract(abstract):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Prompts: Please act as if you are an academic researcher,\\\
                and now you are editing the abstract of your paper to make it more commercial, let the readers\\\
                have the impression that the paper should have some commercial application, but do not add\\\
                any new information. Keep all the original details for the revamped text and remember that the\\\
                revamped text should be proper for academic journals:\n{abstract}"},
            ]
        )
        message = response['choices'][0]['message']['content']
        tokens = response['usage']['total_tokens']
        return message, tokens
    except openai.error.RateLimitError as e:
        print("Rate limit exceeded. Retrying in 60 seconds...")
        time.sleep(60)
        return commercialize_abstract(abstract)
    except openai.error.APIError as e:
        print("Bad Gateway error. Retrying in 60 seconds...")
        time.sleep(60)
        return commercialize_abstract(abstract)

"""# process batch"""

if 'Revamped_abstract' not in df_batch_10.columns:
    df_batch_10['Revamped_abstract'] = ''
if 'Tokens_used' not in df_batch_10.columns:
    df_batch_10['Tokens_used'] = 0

empty_indices = df_batch_10[df_batch_10['Tokens_used'] == 0].index
if len(empty_indices) == 0:
    print("No rows with empty 'Revamped_abstract' column found.")
else:
    empty_index = empty_indices[0]

    processed_count = empty_index  # Variable to keep track of processed rows

    # Iterate over the DataFrame starting from the empty index
    for index, row in df_batch_10.iloc[empty_index:].iterrows():
        abstract = row['abstract']
        revamped_abstract, tokens_used = commercialize_abstract(abstract)
        df_batch_10.at[index, 'Revamped_abstract'] = revamped_abstract
        df_batch_10.at[index, 'Tokens_used'] = tokens_used

        processed_count += 1

        # Check if the number of processed rows is a multiple of 100
        if processed_count % 500 == 0:
            # Export the DataFrame to a CSV file
            filename = f'df_batch_10_processed_{processed_count}.csv'
            df_batch_10.to_csv(filename, index=False)
            print(f"Processed {processed_count} rows. Exported to {filename}")

            !gsutil cp df_batch_10_processed_{processed_count}.csv gs://abstract-output-0613/abstract_output


final_filename = f'df_batch_10_processed_final.csv'
df_batch_10.to_csv(final_filename, index=False)
!gsutil cp df_batch_10_processed_final.csv gs://abstract-output-0613/abstract_output


print("All rows processed.")


            # Download the CSV file


print("All rows processed.")

"""# Merge back the batches"""

import pandas as pd

# List the filenames of the processed final CSV files
csv_files = [
    'df_batch_1_processed_final.csv',
    'df_batch_2_processed_final.csv',
    'df_batch_3_processed_final.csv',
    'df_batch_4_processed_final.csv',
    'df_batch_5_processed_final.csv',
    'df_batch_6_processed_final.csv',
    'df_batch_7_processed_final.csv',
    'df_batch_8_processed_final.csv',
    'df_batch_9_processed_final.csv',
    'df_batch_10_processed_final.csv'
]

# Create an empty list to store the DataFrames
dataframes = []

# Iterate over the CSV files and load them as DataFrames
for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    dataframes.append(df)

# Concatenate the DataFrames vertically
merged_df = pd.concat(dataframes, axis=0)

# Export the merged DataFrame to a single CSV file
merged_filename = 'merged_processed_final.csv'
merged_df.to_csv(merged_filename, index=False)

print(f"All processed final CSV files merged into {merged_filename}.")



!gsutil cp merged_processed_final.csv gs://abstract-output-0613/abstract_output

merged_df = pd.read_csv('merged_processed_final.csv')

# Drop duplicate rows based on the "doi" column
merged_df.drop_duplicates(subset='doi', keep='first', inplace=True)

# Export the updated DataFrame to a new CSV file
deduplicated_filename = 'merged_processed_final_deduplicated.csv'
merged_df.to_csv(deduplicated_filename, index=False)
