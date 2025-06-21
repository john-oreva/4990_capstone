import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt


data = pd.read_csv('filtered_employment_finance_manufacturing.csv', parse_dates=['REF_DATE'])


pre_period = (data['REF_DATE'] >= '2015-01-01') & (data['REF_DATE'] < '2020-01-01')  
post_period = (data['REF_DATE'] >= '2020-01-01') & (data['REF_DATE'] <= '2024-12-31')  


data['Post'] = 0
data.loc[post_period, 'Post'] = 1


data = data[pre_period | post_period]


data['Group_Post'] = data['Group'] * data['Post']


X = sm.add_constant(data[['Group', 'Post', 'Group_Post']]) 
y = data['Employment']

model = sm.OLS(y, X).fit()


with open('did_regression_bubble_3.txt', 'w') as f:
    f.write(model.summary().as_text())


print(model.summary())


did_summary = data.groupby(['Group', 'Post'])['Employment'].mean().reset_index()


manufacturing_pre = did_summary[(did_summary['Group'] == 0) & (did_summary['Post'] == 0)]['Employment'].values[0]
manufacturing_post = did_summary[(did_summary['Group'] == 0) & (did_summary['Post'] == 1)]['Employment'].values[0]
finance_pre = did_summary[(did_summary['Group'] == 1) & (did_summary['Post'] == 0)]['Employment'].values[0]
finance_post = did_summary[(did_summary['Group'] == 1) & (did_summary['Post'] == 1)]['Employment'].values[0]


finance_counterfactual_post = finance_pre + (manufacturing_post - manufacturing_pre)


plt.figure(figsize=(10, 6))


plt.plot([0, 1], [manufacturing_pre, manufacturing_post], marker='o', label='Manufacturing (Control)', color='blue')


plt.plot([0, 1], [finance_pre, finance_post], marker='o', label='Finance (Treatment)', color='orange')


plt.plot([0, 1], [finance_pre, finance_counterfactual_post], linestyle='--', color='orange', label='Finance Counterfactual')


plt.title('Difference-in-Differences (DID) Analysis: Employment Trends (2015-2024)', fontsize=14)
plt.xticks([0, 1], ['Pre-Treatment (2015–2019)', 'Post-Treatment (2020–2024)'])
plt.xlabel('Period')
plt.ylabel('Average Employment')
plt.legend()
plt.grid(True)


plt.tight_layout()
plt.savefig('did_bubble_3.png')
plt.show()
