## Data Cleaning
#### Google Ads
Columns:
* Week
    Shows the date in weekly format 03-Mar-21. No other different variations of the format
* Campaign
    Total 15 categorical variables with no unwanted or rubbish format
* Ad group
    Total 27 categorical variables with no duplicate or rubbish format. Not helpful since it is the combination of Keyword Type and Search Keyword
* Keyword type
    Total 3 categorical variables i.e. Broad, Phrase and Exact
* Search Keyword
    Total 21 categorical variables with 2 variables unwanted, =+pool +reservation software and Err:509
* Currency
    Only one variable type i.e. USD. Will delete later because of no use
* Clicks
    Number of clicks on the ads
* Impressions
    Count of people who view the ad
* Cost ($)
    Money spend on ads in format $500
* Leads
    Dont know of what use is this. Will delete the column later
* Prospects
    Count of people that are interested in the products
* Payment ($)
    Payment recieved from ads
* Payment Date
    Date of payment recieved
* Country
    Contains only one country name i.e. USD
\
Removing the below mentioned columns:
* Payment Date
* Currency
* Country
* Leads
* Ad group
\
Removing all the nan values for Clicks, Impressions, Cost, Payment, Prospects\
\
Splitting the Week column in week, month and year\
\
Removing unwanted data from Search Keyword and making the column clean\
\
Removing the $ sign from Payment and Cost columns\
\
Adding a Total column which is basically the profit = Payment - Cost\
\
**Stats**\
| Index | Clicks | Impressions | Cost   | Prospects | Payment | Total |
| :--- | :---- | :--------- | :--   | :------- | :----- | :--- |
|  mean | 0.931873  |   16.8905| 6.70859|0.0510949  |28.9051  |22.1965|
|  std  | 1.91534  |   47.105  | 14.3576|0.287661   |220.656  |217.363|
|  min  |   0    |           0 |0       |0          |0        |-92.09 |
|  max  |   12   |         357 |92.09   |3          |2388     |2368.36| 

#### Listing Site
Columns:
* Product Name
    Contains only one product Omnify
* Categories
    Includes total 19 columns with no rubbish data
* Date of Report
    This column includes different date formats e.g. 03-Mar-21, 4/13/2020, 31/03/2020, 15-5-2021
* Average Position
    Indicates the average position of the ad in the charts
* Clicks
    Number of clicks on the ads
* Leads
    Dont know of what use is this. Will delete this column later
* Money Spent ($)
    Money spend on ads in format $500
* Channel
    Includes only 3 categorical variables i.e. Capterra, GetApp, Software Advice
* Location
    Contains name of the countries and has duplicate values
* Prospects
    Count of people that are interested in the products
* Paid
    Payment recieved from ads
* Paid Date
    Date of payment recieved
\
Removing the below mentioned columns:
* Paid Date
* Leads
* Product Name
\
Removing all the nan values for Clicks, Average Position, Money Spent ($), Paid\
\
Splitting the Date of Report column into day, month and year with correct date format for all the dates\
\
Merging the duplicate values in the Location column\
\
Removing the $ sign from Paid and Money Spent ($) column\
\
Adding a Total column which is basically the profit = Paid - Money Spent ($)\
\
**Stats**\
| Index | Clicks | Average Position | Money Spent | Prospects | Paid | Total |
| :---: | :----: | :--------------: | :---------: | :-------: | :--: | :---: |
|  mean | 10.7263|   1.66284        | 12.0628     |0.0325203  |11.2195|-0.843257|
|  std  | 11.2196|   2.79844        | 39.1553     |0.182734   |154.16| 158.294 |
|  min  |   0    |     0            |     0       |0          |0     | -1460  |
|  max  |   74   |     80           |    1460     |2          |4000  |  3998 | 

## SQL Data Analysis
#### Google Ads
* Total number of Clicks\
`SELECT SUM(Clicks) FROM google_ads;`

* Total number of people interested in the products\
`SELECT SUM(Prospects) FROM google_ads;`

* Profit on EXACT keyword type\
`SELECT SUM(Profit) from google_ads WHERE Keyword_Type = 'Exact';`

* Distinct values of Campaign for month March\
`SELECT distinct Campaign FROM google_ads WHERE Month = 'Mar';`

* Total Profit in Feb month\
`SELECT SUM(Profit) FROM google_ads WHERE Month = 'Feb';`

#### Listing Site
* Highest Average Position in USA
`SELECT MAX(Average_Position) FROM listing_site WHERE Location = 'USA';`

* Total Clicks in Jan month
`SELECT SUM(Clicks) FROM listing_site WHERE Month = 'Jan';`

* Total money spend on GetApp channel
`SELECT SUM(Money_Spent) FROM listing_site WHERE Channel = 'GetApp';`

* Total payment recieved from Class Registration campaign in April month
`SELECT SUM(Paid) FROM listing_site WHERE Campaign = 'Class Registration' AND Month = 'Apr';`

* Total people interested in the products from Autralia in year 2021
`SELECT SUM(Prospects) FROM listing_site WHERE Location = 'Australia' AND Year = 2021;`

## Power BI Dashboard
![dash 1]() \
![dash 2]() \
![dash 3]() \
![dash 4]()