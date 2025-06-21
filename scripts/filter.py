import pandas as pd


employment_df = pd.read_csv('employment.csv', low_memory=False)
productivity_df = pd.read_csv('productivity.csv', low_memory=False)


employment_df['SCALAR_FACTOR'] = employment_df['SCALAR_FACTOR'].replace({'thousands': 3})
productivity_df['SCALAR_FACTOR'] = productivity_df['SCALAR_FACTOR'].replace({'units': 0})


employment_df['SCALAR_FACTOR'] = pd.to_numeric(employment_df['SCALAR_FACTOR'], errors='coerce')
employment_df['VALUE'] = pd.to_numeric(employment_df['VALUE'], errors='coerce')

productivity_df['SCALAR_FACTOR'] = pd.to_numeric(productivity_df['SCALAR_FACTOR'], errors='coerce')
productivity_df['VALUE'] = pd.to_numeric(productivity_df['VALUE'], errors='coerce')


employment_df = employment_df.dropna(subset=['SCALAR_FACTOR', 'VALUE'])
productivity_df = productivity_df.dropna(subset=['SCALAR_FACTOR', 'VALUE'])


employment_df['REF_DATE'] = pd.to_datetime(employment_df['REF_DATE'], format='%Y-%m')
productivity_df['REF_DATE'] = pd.to_datetime(productivity_df['REF_DATE'], format='%Y-%m')


employment_df = employment_df[
    (employment_df['North American Industry Classification System (NAICS)'] == 'Finance, insurance, real estate, rental and leasing [52-53]') &
    (employment_df['GEO'] == 'Canada') &
    (employment_df['Statistics'] == 'Estimate') &
    (employment_df['Data type'] == 'Seasonally adjusted')
]


productivity_df = productivity_df[
    (productivity_df['North American Industry Classification System (NAICS)'] == 'Finance and insurance, and holding companies') &
    (productivity_df['GEO'] == 'Canada')
]


employment_df['Employment'] = employment_df['VALUE'] * (10 ** employment_df['SCALAR_FACTOR'])
productivity_df['Productivity'] = productivity_df['VALUE'] * (10 ** productivity_df['SCALAR_FACTOR'])


employment_df = employment_df[['REF_DATE', 'Employment']]
productivity_df = productivity_df[['REF_DATE', 'Productivity']]


data_merged = pd.merge(employment_df, productivity_df, on='REF_DATE', how='inner')


data_merged = data_merged[(data_merged['REF_DATE'] >= '1995-01-01') & (data_merged['REF_DATE'] <= '2024-12-31')]


data_merged.to_csv('filtered_finance_canada.csv', index=False)


print("Summary of the filtered dataset:")
print(data_merged.describe())
print("\nSample of the filtered dataset:")
print(data_merged.head())
