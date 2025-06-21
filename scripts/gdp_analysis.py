import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from datetime import datetime
import numpy as np


df = pd.read_csv('gdp.csv', low_memory=False)

df['SCALAR_FACTOR'] = df['SCALAR_FACTOR'].replace({'millions': 6})
df['SCALAR_FACTOR'] = pd.to_numeric(df['SCALAR_FACTOR'], errors='coerce')
df['VALUE'] = pd.to_numeric(df['VALUE'].replace('..', None), errors='coerce')  # Handle '..' as NaN
df.dropna(subset=['SCALAR_FACTOR', 'VALUE'], inplace=True)


df['GDP'] = df['VALUE'] * (10 ** df['SCALAR_FACTOR'])


df_total = df[df['North American Industry Classification System (NAICS)'] == 'All industries [T001]'].copy()
df_finance = df[df['North American Industry Classification System (NAICS)'] == 'Finance and insurance [52]'].copy()
df_ict = df[df['North American Industry Classification System (NAICS)'] == 'Information and communication technology sector [T013]'].copy()


df_total['REF_DATE'] = pd.to_datetime(df_total['REF_DATE'])
df_finance['REF_DATE'] = pd.to_datetime(df_finance['REF_DATE'])
df_ict['REF_DATE'] = pd.to_datetime(df_ict['REF_DATE'])


merged_df = pd.merge(df_total[['REF_DATE', 'GDP']], df_finance[['REF_DATE', 'GDP']], on='REF_DATE', how='outer', suffixes=('_Total', '_Finance'))
merged_df = pd.merge(merged_df, df_ict[['REF_DATE', 'GDP']], on='REF_DATE', how='outer')
merged_df.rename(columns={'GDP': 'GDP_ICT'}, inplace=True)


merged_df.set_index('REF_DATE', inplace=True)
merged_df = merged_df.resample('M').mean()


ict_missing = merged_df.loc['1997-01-01':'2005-12-31', 'GDP_ICT'].isnull()
if ict_missing.any():
    post_2005_avg = merged_df.loc['2006-01-01':, 'GDP_ICT'].mean()
    merged_df.loc['1997-01-01':'2005-12-31', 'GDP_ICT'] = merged_df.loc['1997-01-01':'2005-12-31', 'GDP_ICT'].fillna(post_2005_avg)


print("Missing Data Summary:")
print(merged_df.isnull().sum())


X = merged_df[['GDP_Finance', 'GDP_ICT']].copy()
y = merged_df['GDP_Total'].copy()


X = X.replace([np.inf, -np.inf], np.nan).dropna()
y = y.loc[X.index]  
X = sm.add_constant(X) 


if X.isnull().any().any() or y.isnull().any():
    raise ValueError("Missing or invalid data found in regression variables.")


model = sm.OLS(y, X).fit()


with open('regression_results.txt', 'w') as f:
    f.write(model.summary().as_text())


plt.figure(figsize=(12, 6))
plt.plot(merged_df.index, merged_df['GDP_Total'], label='Total GDP', color='blue')


plt.axvspan(datetime(1997, 1, 1), datetime(2005, 12, 31), color='gray', alpha=0.3, label='Tech Bubble 1.0')
plt.axvspan(datetime(2006, 1, 1), datetime(2015, 12, 31), color='lightblue', alpha=0.3, label='Tech Bubble 2.0')
plt.axvspan(datetime(2016, 1, 1), datetime(2024, 12, 31), color='lightgreen', alpha=0.3, label='Tech Bubble 3.0')


plt.annotate('Dot-Com Bubble Burst', xy=(datetime(2001, 3, 1), 1.1e12), 
             xytext=(datetime(2001, 6, 1), 1.2e12),
             arrowprops=dict(facecolor='black', arrowstyle='->'))

plt.annotate('2008 Financial Crisis', xy=(datetime(2008, 9, 1), 1.2e12), 
             xytext=(datetime(2009, 1, 1), 1.3e12),
             arrowprops=dict(facecolor='black', arrowstyle='->'))

plt.annotate('2020 Pandemic Dip', xy=(datetime(2020, 4, 1), 1.4e12), 
             xytext=(datetime(2020, 7, 1), 1.5e12),
             arrowprops=dict(facecolor='black', arrowstyle='->'))


plt.title('Total GDP (1997-2024)')
plt.xlabel('Year')
plt.ylabel('GDP (in billions)')
plt.legend()
plt.grid()
plt.savefig('total_gdp.png')
plt.show()


merged_df['Finance_Percentage'] = (merged_df['GDP_Finance'] / merged_df['GDP_Total']) * 100
merged_df['ICT_Percentage'] = (merged_df['GDP_ICT'] / merged_df['GDP_Total']) * 100


merged_df = merged_df.dropna(subset=['Finance_Percentage', 'ICT_Percentage'])


fig, ax1 = plt.subplots(figsize=(14, 7))


ax1.plot(merged_df.index, merged_df['Finance_Percentage'], label='Finance Contribution (%)', color='green', linewidth=2)
ax1.set_ylabel('Finance Contribution (%)', color='green', fontsize=12)
ax1.tick_params(axis='y', labelcolor='green')


ax2 = ax1.twinx()
ax2.plot(merged_df.index, merged_df['ICT_Percentage'], label='ICT Contribution (%)', color='orange', linewidth=2)
ax2.set_ylabel('ICT Contribution (%)', color='orange', fontsize=12)
ax2.tick_params(axis='y', labelcolor='orange')


ax1.axvspan(datetime(1997, 1, 1), datetime(2005, 12, 31), color='gray', alpha=0.2, label='Tech Bubble 1.0')
ax1.axvspan(datetime(2006, 1, 1), datetime(2015, 12, 31), color='lightblue', alpha=0.2, label='Tech Bubble 2.0')
ax1.axvspan(datetime(2016, 1, 1), datetime(2024, 12, 31), color='lightgreen', alpha=0.2, label='Tech Bubble 3.0')


plt.title('Finance and ICT Contributions to GDP Over Time (1997-2024)', fontsize=14)
ax1.set_xlabel('Year', fontsize=12)


lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', fontsize=10)


plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('finance_ict_contributions_dual_y_axis.png')
plt.show()
