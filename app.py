from google.cloud import bigquery

client = bigquery.Client()

query = """
    SELECT subject AS subject, COUNT(*) AS num_duplicates
    FROM bigquery-public-data.github_repos.commits
    GROUP BY subject
    ORDER BY num_duplicates
    DESC LIMIT 10
"""
job_config = bigquery.job.QueryJobConfig(use_query_cache=False)
results = client.query(query, job_config=job_config)

for row in results:
    subject = row['subject']
    num_duplicates = row['num_duplicates']
    print(f'{subject:<20} | {num_duplicates:>9,}')

print('-'*60)
print(f'Created: {results.created}')
print(f'Ended:   {results.ended}')
print(f'Bytes:   {results.total_bytes_processed:,}')

# Loading data into BigQuery

# gcs_uri = 'gs://cloud-samples-data/bigquery/us-states/us-states.json'

# dataset = client.create_dataset('us_states_dataset')
# table = dataset.table('us_states_table')

# job_config = bigquery.job.LoadJobConfig()
# job_config.schema = [
#     bigquery.SchemaField('name', 'STRING'),
#     bigquery.SchemaField('post_abbr', 'STRING'),
# ]
# job_config.source_format = bigquery.SourceFormat.NEWLINE_DELIMITED_JSON

# load_job = client.load_table_from_uri(gcs_uri, table, job_config=job_config)

# print('JSON file loaded to BigQuery')
