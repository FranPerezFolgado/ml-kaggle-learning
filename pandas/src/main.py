import pandas as pd

wine_data_path = './pandas/resources/winemag-data-130k-v2.csv'
ca_videos_data = './pandas/resources/CAvideos.csv'
gb_videos_data = './pandas/resources/GBvideos.csv'

'''
DataFrame:
A DataFrame is a table. Contains an array of individual entries.

Simple dataframe:
'''
simple_dataframe = pd.DataFrame({'Yes' : [50,21],'No' : [131,2]})
print(simple_dataframe)


'''
You can create dataFrames with other types of data too. 
The lit of row labels used in DataFrames are called Index 
'''

dataframe_with_index = pd.DataFrame({'Bob': ['I liked it.', 'It was awful.'], 
              'Sue': ['Pretty good.', 'Bland.']},
             index=['Product A', 'Product B'])

print(dataframe_with_index)


'''
Pandas Series:
A series is a sequence of data values. A list. 
A series is a single column of a DataFrame. 
'''

simple_series = pd.Series([1,2,3,4,5])
print(simple_series)

'''
You can assign row labels with an index parameter. 
A series doesn't have a column name, it only has one overall name
'''

named_series = pd.Series([30,40,50], index=['2015 Sales', '2016 Sales','2017 Sales'],
name='Product A')
print(named_series)

#Read files

wine_reviews = pd.read_csv(wine_data_path)

#We can use .shape to check how large the resulting DataFrame is:
print(wine_reviews.shape)
# So our new DataFrame has 130,000 records split across 14 different columns.
#  That's almost 2 million entries!

# We can examine the first 5 rows of the dataframe with .head()
print(wine_reviews.head())


'''
In the read_csv function we have over 30 optional parameters.
One of them is  index_col, to specify the index to use in case the CSV-file has a built-in index
'''
wine_reviews_with_index = pd.read_csv(wine_data_path, index_col=0)
print(wine_reviews_with_index.head())
print('-------------------------------------------------------')
# Accessing data

# Ways of selecting a specific Series out of a DataFrame
print(wine_reviews.country)
print('---------------')
print(wine_reviews['country'])
print(wine_reviews['country'][0])

print('---------------')

## Indexing in pandas
# Index-based selection
print(wine_reviews.iloc[0])
print(wine_reviews.loc[0])
# Both iloc and loc are row-first, column-second
print(wine_reviews.iloc[:, 1])
# We can select with range of values
print(wine_reviews.iloc[:5, 1])
# We can also pass the range as a list
print(wine_reviews.iloc[[0,1,4],1])
#The negative numbers can be used in selection to start counting forward from the end of the values
print(wine_reviews.iloc[-5:])
print('---------------')

# Label-based selection
# with .loc the second operator would be the index value, not its position
print(wine_reviews.loc[0,'country'])
# iloc is conceptually simpler than loc because we treat the dataset as a list of list (a matrix)
# but with loc will be usually easier because our dataset usually has meaningful indices
print(wine_reviews.loc[:, ['taster_name', 'taster_twitter_handle', 'points']])

'''
Choosing between loc and iloc
The two methods use slightly different indexing schemes.
While iloc is exclusive:
0:10 will select entries 0,...,9
loc is inclusive, so
0:10 will select 0,...,10
'''
print('---------------')

## Manipulating the index
# we can manipulate the index  to fit whatever we want
print(wine_reviews.set_index('title'))

print('---------------')

## Conditional Selection
# We can select data based on conditions
print(wine_reviews.country == 'Italy')
# This operation generated a Series of True/False boolean based on the country of each record.
# This can be used inside of loc  to select the relevant data 
print(wine_reviews.loc[wine_reviews.country == 'Italy'])
# We can add conditions with & (and condition)
print(wine_reviews.loc[(wine_reviews.country == 'Italy') & (wine_reviews.points >= 90)])
# We can add conditions with | (or condition) too
print(wine_reviews.loc[(wine_reviews.country == 'Italy') |(wine_reviews.points >= 90)])
# Pandas has built-in conditional selector
# the first one is  isin
print(wine_reviews.loc[wine_reviews.country.isin(['Italy','France'])])
# the second one are isnull and notnull (select values that are or are not empty (NaN))
print(wine_reviews.loc[wine_reviews.price.notnull()])

## Assigning data
# We can assign data to a dataframe
# either a constant value
wine_reviews['critic'] = 'everyone'
print(wine_reviews['critic'])
# or with an iterable of values
wine_reviews['index_backwards'] = range(len(wine_reviews), 0, -1)
print(wine_reviews['index_backwards'])
print('-------------------------------------------------------')

## Summary functions
# pandas has many simple summary functions which restructure the data is some useful way.
# the describe method generates a high-level summart of the attributes of the given column
print(wine_reviews.points.describe())
# that was an example with numeric data, here's one with string
print(wine_reviews.taster_name.describe())
# we can get some particular simple summary statistic about a column
# average
print(wine_reviews.points.mean())
# unique
print(wine_reviews.taster_name.unique())
# all unique values and how often they occur in the dataset
print(wine_reviews.taster_name.value_counts())

## Map 
# A map is a term, borrowed from mathematics, for a function that takes one set of values and "maps" them to another set of values.
#  In data science we often have a need for creating new representations from existing data,
#  or for transforming data from the format it is in now to the format that we want it to be in later.
#  Maps are what handle this work, making them extremely important for getting your work done!

# remean the score the wines received to 0
review_points_mean = wine_reviews.points.mean()
print(wine_reviews.points.map(lambda p: p - review_points_mean))

# The function you pass to map() should expect a single value from the Series (a point value, in the above example), 
# and return a transformed version of that value.
# map() returns a new Series where all the values have been transformed by your function.

# apply() is the equivalent method if we want to transform a whole DataFrame by calling a custom method on each row.
def remean_points(row):
    row.points = row.points - review_points_mean
    return row

print(wine_reviews.apply(remean_points, axis='columns'))

# If we had called reviews.apply() with axis='index', then instead of passing a function to transform each row, 
# we would need to give a function to transform each column.

# Note that map() and apply() return new, transformed Series and DataFrames, respectively. 
# They don't modify the original data they're called on. 

# Pandas provides many common mapping operations as built-ins. For example, here's a faster way of remeaning our points column:

review_points_mean = wine_reviews.points.mean()
print(wine_reviews.points - review_points_mean)

# we can also combine fields like this
print(wine_reviews.country + "-" + wine_reviews.region_1)

# Create a variable bargain_wine with the title of the 
# wine with the highest points-to-price ratio in the dataset.
bargain_idx = (wine_reviews.points / wine_reviews.price).idxmax()
print(wine_reviews.loc[bargain_idx, 'title'])

#There are only so many words you can use when describing a bottle of wine.
#  Is a wine more likely to be "tropical" or "fruity"? 
# Create a Series descriptor_counts counting how many times each of these two words appears in the description column 
# in the dataset. 
# (For simplicity, let's ignore the capitalized versions of these words.)
tropical = wine_reviews.description.map(lambda desc: 'tropical' in desc).sum()
fruity = wine_reviews.description.map(lambda desc: 'fruity' in desc).sum()
descriptor_counts = pd.Series([tropical, fruity], index=['tropical','fruity'])
print(descriptor_counts)


# We'd like to host these wine reviews on our website, but a rating system ranging from 80 to 100 points is too hard 
# to understand - we'd like to translate them into simple star ratings. 
# A score of 95 or higher counts as 3 stars, a score of at least 85 but less than 95 is 2 stars.
#  Any other score is 1 star.
# Also, the Canadian Vintners Association bought a lot of ads on the site,
#  so any wines from Canada should automatically get 3 stars, regardless of points.
# Create a series star_ratings with the number of stars corresponding to each review in the dataset.

def check_rating(row):
    if row.points >= 95 or row.country == 'Canada':
        return 3
    elif row.points >=85:
        return 2
    else:
        return 1
star_ratings = wine_reviews.apply(check_rating, axis='columns')

## Grouping and sorting

print(wine_reviews.groupby('points').points.count())
# groupby() created a group of reviews which allotted the same point values to the given wines. 
# Then, for each of these groups, we grabbed the points() column and counted how many times it appeared.
# value_counts() is just a shortcut to this groupby() operation.

# we can use any of the summary functions with this data
print(wine_reviews.groupby('points').price.min())
# You can think of each group we generate as being a slice of our DataFrame containing only data with values that match.
#  This DataFrame is accessible to us directly using the apply() method, 
# and we can then manipulate the data in any way we see fit.
#  For example, here's one way of selecting the name of the first wine reviewed from each winery in the dataset:

print(wine_reviews.groupby('winery').apply(lambda df: df.title.iloc[0]))

# We can adjust this even more grouping by more than one column.
# here's how we would pick out the best wine by country and province
print(wine_reviews.groupby(['country','province']).apply(lambda df: df.loc[df.points.idxmax()]))

# another amazing method is agg(), it lets you run a bunch of different functions on the dataframe simultaneously
#for example, generate a simple statistical summary of the dataset
print(wine_reviews.groupby('country').price.agg([len,min,max]))

## Multi-indexes
# We can group by multiple levels. for example:
countries_reviewed = wine_reviews.groupby(['country','province']).description.agg([len])
print(countries_reviewed)
# we can see the index type like this:
mi = countries_reviewed.index
print(type(mi))
# We can reset a multi-index back to a regular index
countries_reviewed = countries_reviewed.reset_index()
print(countries_reviewed)
## Sorting
# The data is automatically ordered by the index, but we can sort it by the field we want.
print(countries_reviewed.sort_values(by='len'))
# sort_values() defaults to an ascending sort, we can change it by:
print(countries_reviewed.sort_values(by='len', ascending=False))
# we can sort by index with the companion method, it has the same arguments and default order
print(countries_reviewed.sort_index())
# we can also sort by more than one column at a time:
print(countries_reviewed.sort_values(by=['country','len']))


### Data types and missing values
# the data type for a column in a DataFrame or a Series is known as dtype
# we can use dtpe property to get the type of a specific column
print(wine_reviews.price.dtype) 
# we can get the dtype of every column by:
print(wine_reviews.dtypes)
# the columns consisting entirely of string do not get their own type, they are instead given the object type
# we can convert a column wherever a conversion makes sense by using astype()
print(wine_reviews.points.astype('float64'))

# a DataFrame or Series index has its own dtype:
print(wine_reviews.index.dtype)

## Missing data
# missing data are given the value NaN (short for Not a Number), NaN values are always float64 dtype
# we can select NaN entries by using pd.isnull() (or pd.notnull()) 
print(wine_reviews[pd.isnull(wine_reviews.country)])
# We can replace missing values with fillna()
reviews_filled = wine_reviews.region_2.fillna('Unknown')
print(reviews_filled.head())
# Or we could fill each NaN with the first non-null value thet appears sometime after the given record th the database

# we also can replace data 
print(wine_reviews.taster_twitter_handle.replace('@kerinokeefe','@kerino'))
# the replace method is useful for replacing missing data which is given like 'Unknown', 'Undisclosed', 'Invalid'


### Renaming and Combining
## Renaming
# rename() can be used to change index and/or column names
print(wine_reviews.rename(columns={'points':'score'}))
print(wine_reviews.rename(index={0: 'firstEntry', 1:'secondEntry'}))
# Usually set_index() is more convinient than rename index
# both row and column index have their own name attr. You can rename these:
print(wine_reviews.rename_axis('wines',axis='rows').rename_axis('fields',axis='columns'))
## Combining
# methods for combining: concat(), join(), merge()
# concat() smush all elements together along an axis.
canadian_youtube = pd.read_csv(ca_videos_data)
british_youtube = pd.read_csv(gb_videos_data)

print(pd.concat([canadian_youtube, british_youtube]))

# join() combine different DF which have an index in common.
# For example pull down videos that happened to be trending on the same day in both Canada an UK
left = canadian_youtube.set_index(['title','trending_date'])
right = british_youtube.set_index(['title','trending_date'])

print(left.join(right, lsuffix='_CAN', rsuffix='_UK'))