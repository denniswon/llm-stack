from google.cloud import bigquery
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

client = bigquery.Client()

query = """
SELECT
  SUM(value/POWER(10,18)) AS sum_tx_ether,
  AVG(gas_price*(receipt_gas_used/POWER(10,18))) AS avg_tx_gas_cost,
  DATE(timestamp) AS tx_date
FROM
  `bigquery-public-data.crypto_ethereum.transactions` AS transactions,
  `bigquery-public-data.crypto_ethereum.blocks` AS blocks
WHERE TRUE
  AND transactions.block_number = blocks.number
  AND receipt_status = 1
  AND value > 0
GROUP BY tx_date
HAVING tx_date >= '2018-01-01' AND tx_date <= '2018-12-31'
ORDER BY tx_date
"""

job_config = bigquery.job.QueryJobConfig(use_query_cache=True)
query_job = client.query(query, job_config=job_config)

iterator = query_job.result(timeout=30)
rows = list(iterator)

# Transform the rows into a nice pandas dataframe
df = pd.DataFrame(data=[list(x.values())
                  for x in rows], columns=list(rows[0].keys()))

# Look at the first 10
df.head(10)

print(f'Created: {query_job.created}')
print(f'Ended:   {query_job.ended}')
print(f'Bytes:   {query_job.total_bytes_processed:,}')

plt.style.use('ggplot')
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

f, g = plt.subplots(figsize=(12, 9))
g = sns.lineplot(x="tx_date", y="avg_tx_gas_cost", data=df, palette="Blues_d")
plt.title("Average Ether transaction cost over time")
plt.show()

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
