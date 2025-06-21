import pandas as pd


employment_df = pd.read_csv('employment.csv', low_memory=False)


employment_df['SCALAR_FACTOR'] = employment_df['SCALAR_FACTOR'].replace({'thousands': 3})


employment_df['SCALAR_FACTOR'] = pd.to_numeric(employment_df['SCALAR_FACTOR'], errors='coerce')
employment_df['VALUE'] = pd.to_numeric(employment_df['VALUE'], errors='coerce')


employment_df = employment_df.dropna(subset=['SCALAR_FACTOR', 'VALUE'])


employment_df['REF_DATE'] = pd.to_datetime(employment_df['REF_DATE'], format='%Y-%m', errors='coerce')


employment_df = employment_df[
    (employment_df['North American Industry Classification System (NAICS)'].isin([
        'Finance, insurance, real estate, rental and leasing [52-53]',
        'Manufacturing [31-33]'
    ])) &
    (employment_df['GEO'] == 'Canada') &
    (employment_df['Statistics'] == 'Estimate') &
    (employment_df['Data type'] == 'Seasonally adjusted')
]


employment_df['Employment'] = employment_df['VALUE'] * (10 ** employment_df['SCALAR_FACTOR'])


employment_df = employment_df[['REF_DATE', 'North American Industry Classification System (NAICS)', 'Employment']]


employment_df['Group'] = employment_df['North American Industry Classification System (NAICS)'].apply(
    lambda x: 1 if x == 'Finance, insurance, real estate, rental and leasing [52-53]' else 0
)


employment_df = employment_df[employment_df['REF_DATE'] >= '1990-01-01']


employment_df.to_csv('filtered_employment_finance_manufacturing.csv', index=False)


print("Summary of the filtered dataset:")
print(employment_df.describe())
print("\nSample of the filtered dataset:")
print(employment_df.head())
