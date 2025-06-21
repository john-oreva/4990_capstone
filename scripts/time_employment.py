import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('filtered_finance_canada.csv', parse_dates=['REF_DATE'])


data.set_index('REF_DATE', inplace=True)


data_agg = data.groupby(data.index).mean()


def get_nearest_date(dataframe, target_date):
    try:
        return dataframe.index.get_loc(pd.Timestamp(target_date), method='nearest')
    except KeyError:
        return None

events = [
    ("2000-03-01", "Dot-com Bubble Burst (2000)"),
    ("2008-09-01", "2008 Financial Crisis"),
    ("2020-04-01", "2020 Pandemic Dip"),
]


fig, ax1 = plt.subplots(figsize=(14, 7))


ax1.set_xlabel('Year')
ax1.set_ylabel('Employment', color='blue')
ax1.plot(data_agg.index, data_agg['Employment'], label='Employment', color='blue', alpha=0.7)
ax1.tick_params(axis='y', labelcolor='blue')


ax2 = ax1.twinx()
ax2.set_ylabel('Productivity', color='green')
ax2.plot(data_agg.index, data_agg['Productivity'], label='Productivity', color='green', alpha=0.7)
ax2.tick_params(axis='y', labelcolor='green')


ax1.axvspan('1995-01-01', '2005-12-31', color='orange', alpha=0.2, label='Tech Bubble 1.0 (1995–2005)')
ax1.axvspan('2008-01-01', '2015-12-31', color='purple', alpha=0.2, label='Tech Bubble 2.0 (2008–2015)')
ax1.axvspan('2020-01-01', data_agg.index.max(), color='red', alpha=0.2, label='Tech Bubble 3.0 (2020–present)')


for date, description in events:
    nearest_date_idx = get_nearest_date(data_agg, date)
    if nearest_date_idx is not None:
        actual_date = data_agg.index[nearest_date_idx]
        ax1.annotate(
            description,
            xy=(actual_date, data_agg['Employment'][actual_date]),
            xytext=(actual_date - pd.DateOffset(years=1), data_agg['Employment'].max() * 0.8),
            arrowprops=dict(facecolor='black', arrowstyle='->'),
            fontsize=10,
        )


fig.suptitle('Employment and Productivity Trends (1997-2024)')
fig.tight_layout()
ax1.legend(loc='upper left', fontsize=10)


plt.grid(True)
plt.savefig('employment_trends.png')
plt.show()


data_agg.to_csv('filtered_finance_canada_aggregated.csv')

#Print summary of the updated dataset
#print("Summary of the dataset (aggregated):")
#print(data_agg[['Employment', 'Productivity']].describe())
