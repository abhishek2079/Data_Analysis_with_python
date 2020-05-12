# Importing lib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
from itertools import combinations
from collections import Counter


# Resetting the values
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 400)

# Plot resizing

plt.figure(figsize=(15, 5), dpi=100)


# Merging 12 months of sales data
files=[file for file in os.listdir('D:\Courses\Python\Pandas-Data-Science-Tasks-master\Pandas-Data-Science-Tasks-master\SalesAnalysis\Sales_Data')]
all_months_data = pd.DataFrame()

for file in files:
    df = pd.read_csv('D:\Courses\Python\Pandas-Data-Science-Tasks-master\Pandas-Data-Science-Tasks-master\SalesAnalysis\Sales_Data/' + file)
    all_months_data = pd.concat([all_months_data, df])

# all_months_data.to_csv('D:\Courses\Python\Pandas-Data-Science-Tasks-master\Pandas-Data-Science-Tasks-master\SalesAnalysis\Sales_Data\All_data.csv', index=False)   #Saving the file

all_data = pd.read_csv('D:\Courses\Python\Pandas-Data-Science-Tasks-master\Pandas-Data-Science-Tasks-master\SalesAnalysis\Sales_Data\All_data.csv')
print(all_data.head())


# Cleaning

# print(all_data.isna())
all_data.replace('Order Date', np.nan, inplace=True)
all_data.dropna(axis='index', how='any', inplace=True)
# print(all_data.isna())

# for data in all_data['Order Date']:
# print(data)


# date and time parsing

# data = datetime.strptime(data, '%m/%d/%y %H:%M')
all_data['Order Date'] = pd.to_datetime(all_data['Order Date'], format='%m/%d/%y %H:%M')
print(all_data.head())


# Month Analysis

all_data['Month'] = all_data['Order Date'].dt.month_name()
print(all_data.head())

all_data['Quantity Ordered'] = all_data['Quantity Ordered'].astype(int)

all_data['Price Each'] = all_data['Price Each'].astype(float)

all_data['Total Sales'] = all_data['Quantity Ordered'] * all_data['Price Each']

print(all_data.head())

sales_data = pd.DataFrame(all_data.groupby(['Month']).sum())  # .sort_values('Total Sales', ascending=False)
sales_data.reset_index(inplace=True)
# sales_data.to_csv('D:\Courses\Python\Pandas-Data-Science-Tasks-master\Monthly_sales.csv')
print(sales_data.columns)
print(sales_data)
print(sales_data['Total Sales'])

sales_data=pd.read_csv('D:\Courses\Python\Pandas-Data-Science-Tasks-master\Monthly_sales.csv')

plt.plot(sales_data['Month'], sales_data['Quantity Ordered'], marker='.')
plt.xlabel('Month')
plt.ylabel('Number of Orders')
plt.xticks(sales_data['Month'])
# plt.xticks(['April','August','December','February','January','July','June','March','May','November','October','September'])
plt.title('Distribution of orders monthly', fontdict={'fontname': 'Gabriola', 'fontsize': 20})
plt.xticks(sales_data['Month'])
plt.show()

plt.plot(sales_data['Month'], sales_data['Total Sales'], marker='.')
plt.title('Distribution of sales monthly', fontdict={'fontname': 'Gabriola', 'fontsize': 20})
plt.xlabel('Month')
plt.ylabel('Sales in USD')
#plt.show()
plt.bar(sales_data['Month'], sales_data['Total Sales'], color='#aabbcc')
plt.xlabel('Month')
plt.ylabel('Total sales in USD $')
plt.title('Distribution of sales monthly', fontdict={'fontname': 'Gabriola', 'fontsize': 20})
plt.show()


print(all_data.head())


# City Analysis

results = '669 Spruce St, Los Angeles, CA 90001'
sp = results.split(',')
dumpjson = json.dumps(sp[1])
dumpjson = dumpjson.strip('"')
print(dumpjson)


city_data = []
for address in all_data['Purchase Address']:
    sp = address.split(',')
    city = json.dumps(sp[1])
    state = json.dumps(sp[2])
    st = state.split(' ')
    state = json.dumps(st[1])
    city = city.strip('"')
    state = state.strip('"')
    add = city + '(' + state + ')'
    city_data.append(add)

all_data['City'] = city_data
print(all_data['City'].value_counts())
print(all_data.groupby('City').sum())
city_sales = pd.DataFrame(all_data.groupby('City').sum())
city_sales.reset_index(inplace=True)
print(city_sales)

plt.bar(city_sales['City'], city_sales['Total Sales'], color='#aabbcc')
plt.xlabel('City')
plt.xticks(rotation='vertical')
plt.ylabel('Total Sales in USD')
plt.title('Distribution of sales city wise', fontdict={'fontname': 'Gabriola', 'fontsize': 20})
plt.show()


# Time Analysis

print(all_data)

all_data['Hour'] = all_data['Order Date'].dt.hour
all_data['Minute'] = all_data['Order Date'].dt.minute

# Histogram
bins = np.arange(25)
print(bins)
plt.hist(all_data['Hour'], bins=bins, color='#aabbcc')
plt.xticks(bins)
plt.xlabel('Time in Hour')
plt.ylabel('Frequency')
plt.title('Time Distribution of order placed', fontdict={'fontname': 'Gabriola', 'fontsize': 20})
plt.show()


hours = [hour for hour, some_random_shit in all_data.groupby(['Hour'])]
print(hours)
plt.plot(hours, all_data.groupby(['Hour']).count(), color='r')
print(all_data.groupby(['Hour']).count())
plt.xlabel('Time in Hour')
plt.ylabel('Frequency')
plt.xticks(hours)
plt.title('Time Distribution of order placed', fontdict={'fontname': 'Gabriola', 'fontsize': 20})
plt.grid()
plt.show()


Hour_df = pd.DataFrame(all_data.groupby(['Hour']).sum())
Hour_df.reset_index(inplace=True)
print(Hour_df)
plt.bar(Hour_df['Hour'], Hour_df['Quantity Ordered'], color='#aabbbb')
plt.xlabel('Time in Hour')
plt.ylabel('Frequency')
plt.xticks(Hour_df['Hour'])
plt.grid()
plt.title('Time Distribution of order placed', fontdict={'fontname': 'Gabriola', 'fontsize': 20})
plt.show()
all_data['count'] = 1

'''

NY = all_data.loc[all_data['City'] == ' New York City(NY)']
NY = NY.reset_index(drop=True)
hours = [hour for hour, some_random_shit in NY.groupby(['Hour'])]
plt.plot(hours, NY.groupby(['Hour']).count(), color='b')
plt.xlabel('Time in Hour')
plt.ylabel('Frequency')
plt.xticks(hours)
plt.title('Time Distribution of order placed in NY', fontdict={'fontname': 'Gabriola', 'fontsize': 20})
plt.grid()
plt.show()
'''

def City(City_name):
    Ct = all_data.loc[all_data['City'] == City_name]
    Ct = Ct.reset_index(drop=True)
    hours = [hour for hour, some_random_shit in Ct.groupby(['Hour'])]
    plt.plot(hours, Ct.groupby(['Hour']).count(), color='#aabbbb', label=City_name)
    plt.xlabel('Time in Hour')
    plt.ylabel('Frequency')
    plt.xticks(hours)
    plt.title('Time Distribution of order placed in' + City_name, fontdict={'fontname': 'Gabriola', 'fontsize': 20})
    plt.grid()
    plt.show()


City(' New York City(NY)')
City(' Atlanta(GA)')
City(' San Francisco(CA)')

# Products bought together

df = all_data[all_data['Order ID'].duplicated(keep=False)]


df['Grouped']= df.groupby('Order ID')['Product'].transform(lambda x: ','.join(x))
print(df)


df = df[['Order ID', 'Grouped']].drop_duplicates()
print(df['Grouped'].value_counts())

# sample
list = [[2, 3], [2, 3, 4]]
count = Counter()
print(count)
for sub in list:
    count.update(Counter(combinations(sub, 3)))

print(count)

count = Counter()

for row in df['Grouped']:
    row_list = row.split(',')
    #print(row_list)
    count.update(Counter(combinations(row_list, 2)))

for key, value in count.most_common(10):
    print(key, value)


product_count = all_data['Product'].value_counts()
product_count = product_count.reset_index()
product_count.rename(columns={'index':'Product', 'Product':'Count'}, inplace=True)
print(product_count)
plt.bar(product_count['Product'], product_count['Count'])
plt.xticks(rotation='vertical')
plt.show()


product_group = all_data.groupby(['Product','Price Each']).sum()
product_group.reset_index(inplace=True)
print(product_group)
plt.bar(product_group['Product'], product_group['Quantity Ordered'])
plt.xlabel('Product')
plt.ylabel('# Ordered')
plt.xticks(rotation='vertical')
plt.show()

price = all_data[['Product', 'Price Each']].drop_duplicates()
print(price)
plt.bar(product_group['Product'], product_group['Price Each'])
plt.xlabel('Product')
plt.ylabel('Cost of Product')
plt.xticks(rotation='vertical')
plt.show()

# subplot for comparing
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.bar(product_group['Product'], product_group['Quantity Ordered'], color='g')
ax2.plot(product_group['Product'], product_group['Price Each'])

ax1.set_xlabel('Product Name')
ax1.set_ylabel('# Ordered', color='g')
ax2.set_ylabel('Cost of Product', color='b')
ax1.set_xticklabels(product_group['Product'], rotation='vertical')
plt.show()



