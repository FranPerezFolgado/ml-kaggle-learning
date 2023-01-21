# ml-kaggle-learning
 
# Decision Tree
In this scenario when you want to predict the price of a house, this would be the simplest decision tree

![Simple Decision Tree Example](https://i.imgur.com/7tsb5b1.png)

The step of giving data to a model to determine outputs is called _fitting_** or _training_**.
The data used for this training is called _training data_**.

## Improving the decision tree

You can improve the decision tree by deepening it.

![Deeper tree](https://i.imgur.com/R3ywQsR.png)

The point at the bottom where we make a prediction is called a _leaf_**.

## Pandas

You can import the data using pandas.
```python
import pandas as pd
```

The most important part of Pandas library is DataFrame. A dataframe holds the data similar to a sheet in Excel or a table in SQL database.
We can explore the data inside a dataframe with:
```python
data_file_path = './data.csv'
data = pd.read_csv(data_file)
print(data.describe())
```

```
              Rooms         Price      Distance      Postcode      Bedroom2      Bathroom           Car       Landsize  BuildingArea    YearBuilt     Lattitude    Longtitude  Propertycount
count  13580.000000  1.358000e+04  13580.000000  13580.000000  13580.000000  13580.000000  13518.000000   13580.000000   7130.000000  8205.000000  13580.000000  13580.000000   13580.000000
mean       2.937997  1.075684e+06     10.137776   3105.301915      2.914728      1.534242      1.610075     558.416127    151.967650  1964.684217    -37.809203    144.995216    7454.417378
std        0.955748  6.393107e+05      5.868725     90.676964      0.965921      0.691712      0.962634    3990.669241    541.014538    37.273762      0.079260      0.103916    4378.581772
min        1.000000  8.500000e+04      0.000000   3000.000000      0.000000      0.000000      0.000000       0.000000      0.000000  1196.000000    -38.182550    144.431810     249.000000
25%        2.000000  6.500000e+05      6.100000   3044.000000      2.000000      1.000000      1.000000     177.000000     93.000000  1940.000000    -37.856822    144.929600    4380.000000
50%        3.000000  9.030000e+05      9.200000   3084.000000      3.000000      1.000000      2.000000     440.000000    126.000000  1970.000000    -37.802355    145.000100    6555.000000
75%        3.000000  1.330000e+06     13.000000   3148.000000      3.000000      2.000000      2.000000     651.000000    174.000000  1999.000000    -37.756400    145.058305   10331.000000
max       10.000000  9.000000e+06     48.100000   3977.000000     20.000000      8.000000     10.000000  433014.000000  44515.000000  2018.000000    -37.408530    145.526350   21650.000000
```

With `data.head()` you can see the top data from the DataFrame
```
   Rooms  Bathroom  Landsize  Lattitude  Longtitude
1      2       1.0     156.0   -37.8079    144.9934
2      3       2.0     134.0   -37.8093    144.9944
4      4       1.0     120.0   -37.8072    144.9941
6      3       2.0     245.0   -37.8024    144.9993
7      2       1.0     256.0   -37.8060    144.9954
```


## SCIKIT-SKLEARN

Using this package you can create models. While coding, this package is written as `sklearn`. 
Scikit-learn is easily the most popular library for modeling the types of data typically stored in DataFrames.

The steps to building a model are:

* **Define**: Declare the type of the model and some other parameters.
* **Fit**: Capture patterns from provided data. The heart of modeling.
* **Predict**: Just what it sounds like.
* **Evaluate**: Determine how accurate the model predictions are.

When you have the data stored in a DataFrame, you can declare your prediction target. By convention is called **y**.

```python
y = data.Price
```

The columns passed to our model for prediction are called **Features**. 
Now select the features you want to use to predict the **Price**. By convention this list is called **X**.

```python
features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']

X = data[features]
```

### Example of declaring a Model and predict.
```python
from sklearn.tree import DecisionTreeRegressor

# Define model. Specify a number for random_state to ensure same results each run
model = DecisionTreeRegressor(random_state=1)

# Fit model
model.fit(X, y)
print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(model.predict(X.head()))
```

The random_state value does reference to this:

> random_state : int, RandomState instance or None, optional (default=None)
> If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by np.random.


### Mean Absolute Error (MAE)

Mean Absolute Error is one of the many mertrics for summarizing model quality.  

The prediction error for each house is:
``` 
error = actual - predicted
``` 

For example if a house costs $150,000 and the model predicted $100,000 the error is $50,000.

With the MAE metric, we take the absolute value of each error, this converts each error to a positive number and take the average of those absolute errors. 

To calculate mae:

```python
from sklearn.metrics import mean_absolute_error

predicted_home_prices = model.predict(X)
mean_absolute_error(y, predicted_home_prices)
```

### Split data

The problem with that prediction is that we used the whole dataset to get it, so the model will appear accurate in the training data.
Since model's practical value come from making predictions on new data, we should measure performance on data that wasn't used to build the model. The most easy way to do this is excluding some data from the model-building process, and then use those to test model's accuracy on data it hasn't seen before.
This data is calles **validation data**.
