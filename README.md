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
```
import pandas as pd
```

The most important part of Pandas library is DataFrame. A dataframe holds the data similar to a sheet in Excel or a table in SQL database.
We can explore the data inside a dataframe with:
```
data_file_path='./data.csv'
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

