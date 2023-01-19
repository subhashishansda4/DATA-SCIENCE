# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 03:16:31 2022

@author: Subhashis Hansda
"""
# importing libraries
import pandas as pd
import numpy as np



# DATA EXPLORATION

# -----------------------------------------------------------------------------
# GOOGLE ADS
# checking columns
print('')
print('Google_Ads')
google_ads = pd.read_csv('google_ads.csv')
print(google_ads.columns)
# unique values
campaign_gs = google_ads['Campaign'].unique()
ad_group_gs = google_ads['Ad group'].unique() #no use since we have both search keyword and keyword type
search_keyword_gs = google_ads['Search Keyword'].unique() #consists of rubbish and null values
currency_gs = google_ads['Currency'].unique() #no use since only one variable
country_gs = google_ads['Country'].unique() #no use since only one variable
keyword_type_gs = google_ads['Keyword type'].unique()
payment_gs = google_ads['Payment ($)'].unique()
week_gs = google_ads['Week'].unique() #total 14 weeks

# -----------------------------------------------------------------------------
# LISTING SITE
# checking columns
print('')
print('Listing_Site')
listing_site = pd.read_csv('listing_site.csv')
print(listing_site.columns)
# unique values
product_name_ls = listing_site['Product Name'].unique() #no use since only one variable
categories_ls = listing_site['Categories'].unique()
channel_ls = listing_site['Channel'].unique()
location_ls = listing_site['Location'].unique() #consists of duplicate values
date_report_ls = listing_site['Date of Report'].unique()
paid_ls = listing_site['Paid'].unique() #some have $ symbol; some dont

# Statistical Analysis
google_ads_stats = google_ads.describe()
listing_site_stats = listing_site.describe()
# -----------------------------------------------------------------------------



# DATA CLEANING

# -----------------------------------------------------------------------------
# GOOGLE ADS
# -----------------------------------------------------------------------------
# dataframe for null and rubbish values in search keyword
search_keyword_null = google_ads.loc[google_ads['Ad group'] == 'Pool_Reservation_Software_Open_Broad']

# removing rubbish and null values
clean_google_ads = pd.read_csv('google_ads.csv')
clean_google_ads.drop(clean_google_ads.index[clean_google_ads['Ad group'] == 'Pool_Reservation_Software_Open_Broad'], inplace=True)
clean_google_ads.drop(['Currency', 'Country', 'Ad group'], axis = 1, inplace = True)
clean_google_ads['Payment ($)'] = clean_google_ads['Payment ($)'].replace(np.nan, '0')
clean_google_ads['Payment Date'] = clean_google_ads['Payment Date'].replace(np.nan, '0')
# clean column ['Search Keyword']
clean_search_keyword_gs = clean_google_ads['Search Keyword'].unique()

# splitting ['Week'] column into day, month and year
clean_google_ads[['Day', 'Month', 'Year']] = clean_google_ads['Week'].str.split("-", expand = True)
clean_google_ads['Year'] = clean_google_ads['Year'].replace('21', '2021')

# removing ['Week'], ['Leads'] and ['Payment Date']
clean_google_ads.drop(['Week', 'Leads', 'Payment Date'], axis = 1, inplace = True)

# adding ['Week'] column by merging ['Day'] and ['Month']
clean_google_ads['Week'] = clean_google_ads[['Day', 'Month']].agg('-'.join, axis = 1)
clean_google_ads.drop(['Day'], axis = 1, inplace = True)

# removing '$' from ['Payment ($)']
clean_google_ads['Payment'] = clean_google_ads['Payment ($)'].str.replace('$', '')
clean_google_ads.drop(['Payment ($)'], axis = 1, inplace = True)

# adding ['Total'] column
clean_google_ads.rename(columns = {'Cost ($)':'Cost', 'Keyword type':'Keyword_Type'}, inplace = True)
clean_google_ads.rename(columns = {'Search Keyword':'Search_Keyword'}, inplace = True)
clean_google_ads['Payment'] = clean_google_ads['Payment'].str.replace(',', '')

print(clean_google_ads.Payment.dtype)
clean_google_ads['Payment'] = pd.to_numeric(clean_google_ads['Payment'])
print(clean_google_ads.Payment.dtype)
clean_google_ads['Total'] = clean_google_ads['Payment'] - clean_google_ads['Cost']

# checking unique values
clicks_gs = clean_google_ads['Clicks'].unique()
impressions_gs = clean_google_ads['Impressions'].unique()
cost_gs = clean_google_ads['Cost'].unique()
prospects_gs = clean_google_ads['Prospects'].unique()

# statistical analysis
clean_google_ads_stats = clean_google_ads.describe()
# dataframe to csv
clean_google_ads.to_csv('clean_google_ads.csv', index = False)


# -----------------------------------------------------------------------------
# LISTING SITE
# -----------------------------------------------------------------------------
clean_listing_site = pd.read_csv('listing_site.csv')
# removing duplicate values
# USA
clean_listing_site['Location'] = clean_listing_site['Location'].replace(['UNITED STATES', 'USA', 'US', 'United States'], 'USA')
# UK
clean_listing_site['Location'] = clean_listing_site['Location'].replace(['UNITED KINGDOM', 'UK', 'United Kingdom'], 'UK')
# Singapore
clean_listing_site['Location'] = clean_listing_site['Location'].replace(['SINGAPORE', 'Singapore'], 'Singapore')
# Australia
clean_listing_site['Location'] = clean_listing_site['Location'].replace(['AUSTRALIA', 'Australia'], 'Australia')
# Canada
clean_listing_site['Location'] = clean_listing_site['Location'].replace(['CANADA', 'Canada'], 'Canada')
clean_listing_site.drop(['Product Name'], axis = 1, inplace = True)
# replacing nan values with 0
clean_listing_site['Money Spent ($)'] = clean_listing_site['Money Spent ($)'].replace(np.nan, '0')
clean_listing_site['Clicks'] = clean_listing_site['Clicks'].replace(np.nan, '0')
clean_listing_site['Average Position'] = clean_listing_site['Average Position'].replace(np.nan, '0')
clean_listing_site['Prospects'] = clean_listing_site['Prospects'].replace(np.nan, '0')
clean_listing_site['Paid'] = clean_listing_site['Paid'].replace(np.nan, '0')
clean_listing_site['Paid Date'] = clean_listing_site['Paid Date'].replace(np.nan, '0') 
# clean column ['Location']
clean_location_ls = clean_listing_site['Location'].unique()

# checking unique values
money_spent_ls = clean_listing_site['Money Spent ($)'].unique()

# replacing '$' and ',' from ['Paid']
clean_listing_site['Paid'] = clean_listing_site['Paid'].str.replace('$', '')
clean_listing_site['Paid'] = clean_listing_site['Paid'].str.replace(',', '')

# adding ['Total'] column
clean_listing_site.rename(columns = {'Money Spent ($)':'Money_Spent'}, inplace = True)
clean_listing_site.rename(columns = {'Average Position':'Average_Position'}, inplace = True)
clean_listing_site['Paid'] = pd.to_numeric(clean_listing_site['Paid'])
clean_listing_site['Money_Spent'] = clean_listing_site['Money_Spent'].astype(float)
clean_listing_site['Total'] = clean_listing_site['Paid'] - clean_listing_site['Money_Spent']

# splitting ['Date of Report'] column into day, month and year
ls_1_date = clean_listing_site.loc[0:1692]
ls_1_date[['Day', 'Month', 'Year']] = ls_1_date['Date of Report'].str.split('-', expand = True)
ls_1_date['Year'] = ls_1_date['Year'].replace('20', '2020')
ls_1_date['Year'] = ls_1_date['Year'].replace('21', '2021')

ls_2_date = clean_listing_site.loc[1693:1784]
ls_2_date[['Month', 'Day', 'Year']] = ls_2_date['Date of Report'].str.split('/', expand = True)
ls_2_date['Month'] = ls_2_date['Month'].replace('03', 'Mar')

ls_3_date = clean_listing_site.loc[1785:1914]
ls_3_date[['Day', 'Month', 'Year']] = ls_3_date['Date of Report'].str.split('/', expand = True)
ls_3_date['Month'] = ls_3_date['Month'].replace('4', 'Apr')

ls_4_date = clean_listing_site.loc[1915:2090]
ls_4_date[['Month', 'Day', 'Year']] = ls_4_date['Date of Report'].str.split('-', expand = True)
ls_4_date['Month'] = ls_4_date['Month'].replace('04', 'Apr')

clean_listing_site = pd.concat([ls_1_date, ls_2_date, ls_3_date, ls_4_date])

# removing ['Date of Report'], ['Leads'] and ['Paid Date']
clean_listing_site.drop(['Date of Report', 'Leads', 'Paid Date'], axis = 1, inplace = True)

clean_listing_site['Prospects'] = pd.to_numeric(clean_listing_site['Prospects'])
clean_listing_site['Clicks'] = pd.to_numeric(clean_listing_site['Clicks'])
clean_listing_site['Average_Position'] = pd.to_numeric(clean_listing_site['Average_Position'])

# statistical analysis
clean_listing_site_stats = clean_listing_site.describe()
# dataframe to csv
clean_listing_site.to_csv('clean_listing_site.csv', index = False)
# -----------------------------------------------------------------------------

# Statistical Analysis
z1 = clean_google_ads.describe()
z2 = clean_listing_site.describe()
