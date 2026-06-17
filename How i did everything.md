# House Price Prediction

## Task 1 -  Data Loading and Exploration
print(df.shape) - rows x columns 
print(df.columns) - columns in the dataset 

(545, 13)
543 houses and 13 pieces of informates for each house 
the first column "Price" is what we want to predict 
'price',
'area',
'bedrooms',
'bathrooms',
'stories',
'mainroad',
'guestroom',
'basement',
'hotwaterheating',
'airconditioning',
'parking',
'prefarea',
'furnishingstatus'

we ran df.info() -
<class 'pandas.core.frame.DataFrame'> we got this output meaning datafrfame is a pandas dataframe 
RangeIndex: 545 entries, 0 to 544 
Data columns (total 13 columns)

 #   Column            Non-Null Count  Dtype 
---  ------            --------------  ----- 
 0   price             545 non-null    int64 
 1   area              545 non-null    int64 
 2   bedrooms          545 non-null    int64 
 3   bathrooms         545 non-null    int64 
 4   stories           545 non-null    int64 
 5   mainroad          545 non-null    object
 6   guestroom         545 non-null    object
 7   basement          545 non-null    object
 8   hotwaterheating   545 non-null    object
 9   airconditioning   545 non-null    object
 10  parking           545 non-null    int64 
 11  prefarea          545 non-null    object
 12  furnishingstatus  545 non-null    object

0   price             545 non-null    int64 -- this means that all 545 rows contains a value for price 
if we see price 530 non-null that means 15 values are missing 
good news for us there are NO MISSING VALUES

this is confirmed by 
df.isnull().sum()

price               0
area                0
bedrooms            0
bathrooms           0
stories             0
mainroad            0
guestroom           0
basement            0
hotwaterheating     0
airconditioning     0
parking             0
prefarea            0
furnishingstatus    0

for all the columns there is no missing values 

dtypes: int64(6), object(7) 
6 numeric columns and 7 text Columns

memory usage: 55.5+ KB
small dataset 

df.describe()
Mean Price = 4,766,729
Min Price = 1,750,000
Max Price = 13,300,000

Cheapest house costs about 1.75 million
Most expensive house costs about 13.3 million
Average house costs about 4.77 million

Mean Area = 5150 sq ft
Min Area = 1650 sq ft
Max Area = 16200 sq ft

Min = 1
Max = 6
Average ≈ 3

Min = 1
Max = 4
Average ≈ 1.3

Min = 1
Max = 4
Average ≈ 1.8

Min = 0
Max = 3
Average ≈ 0.69

df.duplicated().sum() - output:0 
no duplicate rows 

for col in df.columns:
    print("\n", col)
    print(df[col].unique()) 

prints only unique values of the columns 


Before cleaning data, we need to know:
what values are in each of the columns 

unique() returns all distinct values present in a column.

## Task 2: Data Cleaning

df.isnull().sum() - no missing values 
df.isnull().sum() - no diplicate rows

the standard practice is to 
df = df.drop_duplicates() run this 

beacause if duplicates existed 
the model would learn biased information

now machine learning models cannot learn yes or no so we have to convert this into 1 or 2
so we will convert binary column's data from yes or no to 1 or 2 

binary_cols = [
    'mainroad',
    'guestroom',
    'basement',
    'hotwaterheating',
    'airconditioning',
    'prefarea'
]
first define all the binary columns 

for binary columns like these 
yes/no
true/false
male/female

use .map()

for col in binary_cols:
    df[col] = df[col].map({
        'yes': 1,
        'no': 0
    })

we used .map instead of if yes: 
Because .map() replaces every value in the column automatically
if yes: will iterate through all the values 

now df['furnishingstatus'].unique()

this does not give a binary output it has 3 categories 
so we have to somehow convert this into binary output 1 or 2 

df = pd.get_dummies(
    df,
    columns=['furnishingstatus'],
    drop_first=True
)

why drop_first = True? 
without it we would create 3 columns but 
But if two are known, the third is automatically known.

we split it into 

| furnishingstatus_semi-furnished | furnishingstatus_unfurnished |
| ------------------------------- | ---------------------------- |
| 0                               | 0                            |
| 1                               | 0                            |
| 0                               | 1                            |

semi = 0
unfurnished = 0
then that would mean 
furnsished = 1 

So we remove one column to avoid redundancy.
This is called the Dummy Variable Trap

the problem was we had a column "furnishingstatus" 
and the outputs we text based - furnished/unfurnished/semi-furnished 

ml model cant understand text 

in ml 1 or 0 means yes or no but if we introduce another number like 0,1,2 then it will think that the value with 2 has higher priority so this is a bad solution 
furnished = 2
semi-furnished = 1
unfurnished = 0

so instead of one column, we create separate columns.
| furnished | semi-furnished | unfurnished |
| --------- | -------------- | ----------- |
| 1         | 0              | 0           |
| 0         | 1              | 0           |
| 0         | 0              | 1           |

df["furnishingstatus"] 

[
'furnished',
'semi-furnished',
'unfurnished'
]

when we run 
pd.get_dummies(df, columns=['furnishingstatus'])

furnishingstatus_furnished
furnishingstatus_semi-furnished
furnishingstatus_unfurnished

drop_first=True
Pandas removes the first category: furnished 
because two columns are enough to tell us the full info and
if we kept all three columns 
then for every row - furnished + semi-furnished + unfurnished = 1

Linear Regression models dont like redundant data 


## Task 3: Model Building

its my first ever ML project so im learning side by side as well 
we cannot train the model on ALL 545 houses 

example - your teacher gave you 100 practice questions for a unit you memorised them all 
but did you actually learn anything? no you just memorised it 

ml has the same problem 
this is called OVERFITTING 

# Solution: Train-Test Split
we divide the dataset 

80% Training Data
20% Testing Data

436 houses → Training
109 houses → Testing

training set is use to teach the model, the model learns patterns 

test set - the model never sees this data later we will ask our model to predict the prices of houses from test set and see if its accurate 

X = df.drop('price', axis=1)

y = df['price']

print("X Shape:", X.shape)
print("y Shape:", y.shape)

x is all the other columns 
y is the target variable 

total 14 columns 

X Shape: (545, 13) 
y Shape: (545,)

# actually splitting the data 
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

Full Dataset (545 rows)
│
├── Training Set (80%)
│   ├── X_train (436, 13)
│   └── y_train (436,)
│
└── Testing Set (20%)
    ├── X_test (109, 13)
    └── y_test (109,)


# import the algorithm from sickit-learn
from sklearn.linear_model import LinearRegression

# create a model
model = LinearRegression()

# now we train the model 
model.fit(X_train, y_train)

print(model.coef_) - lets see what the model learned 
What are coefficients?
These are the weights assigned to each feature.
how much does these features influence the price 

every feature gets its own coefficient 

and we when are training the model the model creates an equation 

Price =
235 × Feature1
+
76778 × Feature2
+
1094444 × Feature3
+
...
-
413645 × Feature13
+
260032

positive coeff means 
Feature increases
↓
House price increases

negative coeff means 
Feature increases
↓
House price decreases



# Linear Regression Coefficients Interpretation

After training the Linear Regression model, each feature was assigned a coefficient. A coefficient indicates how much the predicted house price changes when that feature increases by one unit, assuming all other features remain constant.

| Feature           | Coefficient | Interpretation                                                                                                                             |
| ----------------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| Area              | 235.97      | Every additional square foot increases the predicted house price by approximately ₹236.                                                    |
| Bedrooms          | 76,779      | Each additional bedroom increases the predicted house price by approximately ₹76,779.                                                      |
| Bathrooms         | 1,094,445   | Each additional bathroom increases the predicted house price by approximately ₹10.94 lakh, making it one of the most influential features. |
| Stories           | 407,477     | Each additional floor/story increases the predicted house price by approximately ₹4.07 lakh.                                               |
| Main Road         | 367,920     | Houses located on a main road are predicted to cost approximately ₹3.68 lakh more than those that are not.                                 |
| Guest Room        | 231,610     | Having a guest room increases the predicted house price by approximately ₹2.32 lakh.                                                       |
| Basement          | 390,251     | Houses with a basement are predicted to cost approximately ₹3.90 lakh more.                                                                |
| Hot Water Heating | 684,650     | Houses with hot water heating are predicted to cost approximately ₹6.85 lakh more.                                                         |
| Air Conditioning  | 791,427     | Houses with air conditioning are predicted to cost approximately ₹7.91 lakh more.                                                          |
| Parking           | 224,842     | Each additional parking space increases the predicted house price by approximately ₹2.25 lakh.                                             |
| Preferred Area    | 629,891     | Houses located in a preferred area are predicted to cost approximately ₹6.30 lakh more.                                                    |
| Semi-Furnished    | -126,882    | Semi-furnished houses are predicted to cost approximately ₹1.27 lakh less than furnished houses.                                           |
| Unfurnished       | -413,645    | Unfurnished houses are predicted to cost approximately ₹4.14 lakh less than furnished houses.                                              |

# Intercept

The model learned an intercept value of **₹260,032.36**. The intercept acts as the baseline value from which the effects of all features are added or subtracted to generate the final house price prediction.

# Key Insights

* **Bathrooms** have the strongest positive impact on house price among all features.
* **Air Conditioning**, **Hot Water Heating**, and **Preferred Area** also significantly increase house prices.
* Houses that are **unfurnished** are generally predicted to be cheaper than furnished houses.
* Larger **area**, additional **stories**, and more **parking spaces** contribute positively to the house price.
* The model achieved an **R² Score of 0.653**, meaning it explains approximately **65.3% of the variation in house prices**.
* The model's predictions are reasonably accurate but still leave room for improvement through feature engineering, scaling, or more advanced regression techniques.

now lets rank which feature matters the most 

# Feature Importance Ranking

To better understand which features have the strongest influence on house prices, the absolute value of each coefficient was calculated and sorted in descending order.

| Rank | Feature           | Coefficient | Interpretation                                                                                                               |
| ---- | ----------------- | ----------- | ---------------------------------------------------------------------------------------------------------------------------- |
| 1    | Bathrooms         | 1,094,445   | Strongest positive influence on house price. Adding one bathroom increases the predicted price by approximately ₹10.94 lakh. |
| 2    | Air Conditioning  | 791,427     | Houses with air conditioning are predicted to cost approximately ₹7.91 lakh more.                                            |
| 3    | Hot Water Heating | 684,650     | Houses with hot water heating are predicted to cost approximately ₹6.85 lakh more.                                           |
| 4    | Preferred Area    | 629,891     | Houses located in a preferred area are predicted to cost approximately ₹6.30 lakh more.                                      |
| 5    | Unfurnished       | -413,645    | Unfurnished houses are predicted to cost approximately ₹4.14 lakh less than furnished houses.                                |
| 6    | Stories           | 407,477     | Each additional floor/story increases the predicted price by approximately ₹4.07 lakh.                                       |
| 7    | Basement          | 390,251     | Houses with a basement are predicted to cost approximately ₹3.90 lakh more.                                                  |
| 8    | Main Road         | 367,920     | Houses on a main road are predicted to cost approximately ₹3.68 lakh more.                                                   |
| 9    | Guest Room        | 231,610     | Houses with a guest room are predicted to cost approximately ₹2.32 lakh more.                                                |
| 10   | Parking           | 224,842     | Each additional parking space increases the predicted price by approximately ₹2.25 lakh.                                     |
| 11   | Semi-Furnished    | -126,882    | Semi-furnished houses are predicted to cost approximately ₹1.27 lakh less than furnished houses.                             |
| 12   | Bedrooms          | 76,779      | Each additional bedroom increases the predicted price by approximately ₹76,779.                                              |
| 13   | Area              | 235.97      | Every additional square foot increases the predicted price by approximately ₹236.                                            |

# Observations

* **Bathrooms** emerged as the most influential feature in the model.
* Premium amenities such as **air conditioning** and **hot water heating** significantly increase house value.
* **Location-related factors** such as being in a **preferred area** or on a **main road** have a strong positive impact on price.
* **Unfurnished houses** are considerably cheaper than furnished houses, indicating that furnishing status affects property valuation.
* Although **area** is generally expected to be important, its coefficient appears relatively small because it is measured in square feet and can vary by thousands of units. Therefore, coefficient magnitude alone should not be used as the sole measure of feature importance.
* Features such as **stories**, **basement**, **guest room**, and **parking** contribute positively to the overall house price and represent desirable property characteristics.

# Important Note

The ranking above is based on the absolute coefficient values of a Linear Regression model. Since features are measured on different scales (e.g., area in square feet versus bathrooms as a count), larger coefficients do not necessarily imply greater real-world importance. A more accurate comparison would require feature standardization before training the model.


## Task 4: Visualization

## Task 5: Insights and Summary



conda create -n internship 
conda activate internship 
mkdir HousePricePrediction_KartikSharma
cd HousePricePrediction_KartikSharma
mkdir charts 

downloaded the dataset from Kaggle 
extracted it and copied the house.csv file to HousePricePrediction_KartikSharma folder 

then opened jupyter lab by typing "jupyter lab" in anaconda prompt

created a python 3 analysis.ipynb file 

HousePricePrediction_KartikSharma
│
├── Housing.csv
├── analysis.ipynb
└── charts

this is the folder structure rn 

then downloaded the required packages 
pip install pandas numpy matplotlib seaborn scikit-learn


