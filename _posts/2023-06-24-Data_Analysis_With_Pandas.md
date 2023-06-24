---
title:  "Pandas: Data Analysis Essentials"
category: posts
date: 2023-06-08
excerpt: "This article outlines the essential Pandas methods for preliminary data analysis."
toc: true
toc_label: "Contents"
tags:
  - pandas
  - data analysis
---

A significant portion of working on real-world projects, around 70-80%, is actually spent on preparing and cleaning the data. This is where Pandas comes in handy: Pandas is a powerful tool that empowers analysts to efficiently manipulate, analyze, and visualize data. Here's a brief overview of how you can perform data analysis with Pandas.

## Introduction

- **Pandas**: Python library for data manipulation and analysis.
- **Installation**: Run `pip install pandas` in your Python environment.
- **DataFrames**: Two-dimensional labeled data structures in Pandas, used for storing and analyzing tabular data efficiently.
- **Series**: One-dimensional labeled data structure in Pandas, representing a single column or row of data, capable of holding various data types.


## Data
The [UCI Adult Income dataset](https://archive.ics.uci.edu/ml/datasets/adult)
is used for this article. It contains information about individuals from the 1994 United States Census, and the goal is to predict whether a person's income exceeds \$50,000 per year based on various attributes. It consists of 14 attributes, including features such as age, education level, work-class, marital status, occupation, race, sex, capital gain, capital loss, hours per week, native country, and the target variable, which indicates whether the income exceeds $50,000 or not.

## Setup

* import numpy and pandas
* setup precision for two decimal places
* read csv from URL into a dataframe



```python
import numpy as np
import pandas as pd

pd.set_option("display.precision", 2)

DATA_URL = "https://vmlverse.github.io/assets/datasets/adult.data.csv"
df = pd.read_csv(DATA_URL)
```

## Exploring the data

Once data loaded into a DataFrame, you can examine its contents and structure.


### 1. df.head(n)
Returns the first n rows of the DataFrame. (Default:5):


```python
df.head()
```





  <div id="df-342399f4-01de-43da-a06f-a85462a4bef0">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>workclass</th>
      <th>fnlwgt</th>
      <th>education</th>
      <th>education-num</th>
      <th>marital-status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>sex</th>
      <th>capital-gain</th>
      <th>capital-loss</th>
      <th>hours-per-week</th>
      <th>native-country</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>39</td>
      <td>State-gov</td>
      <td>77516</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Never-married</td>
      <td>Adm-clerical</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>2174</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>1</th>
      <td>50</td>
      <td>Self-emp-not-inc</td>
      <td>83311</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>13</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>2</th>
      <td>38</td>
      <td>Private</td>
      <td>215646</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Divorced</td>
      <td>Handlers-cleaners</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>3</th>
      <td>53</td>
      <td>Private</td>
      <td>234721</td>
      <td>11th</td>
      <td>7</td>
      <td>Married-civ-spouse</td>
      <td>Handlers-cleaners</td>
      <td>Husband</td>
      <td>Black</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>4</th>
      <td>28</td>
      <td>Private</td>
      <td>338409</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Prof-specialty</td>
      <td>Wife</td>
      <td>Black</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>Cuba</td>
      <td>&lt;=50K</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-342399f4-01de-43da-a06f-a85462a4bef0')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-342399f4-01de-43da-a06f-a85462a4bef0 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-342399f4-01de-43da-a06f-a85462a4bef0');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




### 2. df.tail(n)
Returns the last n rows of the DataFrame.(Default:5)


```python
df.tail()
```





  <div id="df-62613947-3468-4726-84fe-4ef57780f9c5">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>workclass</th>
      <th>fnlwgt</th>
      <th>education</th>
      <th>education-num</th>
      <th>marital-status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>sex</th>
      <th>capital-gain</th>
      <th>capital-loss</th>
      <th>hours-per-week</th>
      <th>native-country</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>32556</th>
      <td>27</td>
      <td>Private</td>
      <td>257302</td>
      <td>Assoc-acdm</td>
      <td>12</td>
      <td>Married-civ-spouse</td>
      <td>Tech-support</td>
      <td>Wife</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>38</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>32557</th>
      <td>40</td>
      <td>Private</td>
      <td>154374</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Married-civ-spouse</td>
      <td>Machine-op-inspct</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&gt;50K</td>
    </tr>
    <tr>
      <th>32558</th>
      <td>58</td>
      <td>Private</td>
      <td>151910</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Widowed</td>
      <td>Adm-clerical</td>
      <td>Unmarried</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>32559</th>
      <td>22</td>
      <td>Private</td>
      <td>201490</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Never-married</td>
      <td>Adm-clerical</td>
      <td>Own-child</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>20</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>32560</th>
      <td>52</td>
      <td>Self-emp-inc</td>
      <td>287927</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Wife</td>
      <td>White</td>
      <td>Female</td>
      <td>15024</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&gt;50K</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-62613947-3468-4726-84fe-4ef57780f9c5')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-62613947-3468-4726-84fe-4ef57780f9c5 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-62613947-3468-4726-84fe-4ef57780f9c5');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




### 3. df.shape
Returns the dimensions (rows, columns) of the DataFrame.


```python
df.shape
```




    (32561, 15)



### 4. df.info()
Provides information about the DataFrame, including the data - types of columns and missing values.


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 32561 entries, 0 to 32560
    Data columns (total 15 columns):
     #   Column          Non-Null Count  Dtype 
    ---  ------          --------------  ----- 
     0   age             32561 non-null  int64 
     1   workclass       32561 non-null  object
     2   fnlwgt          32561 non-null  int64 
     3   education       32561 non-null  object
     4   education-num   32561 non-null  int64 
     5   marital-status  32561 non-null  object
     6   occupation      32561 non-null  object
     7   relationship    32561 non-null  object
     8   race            32561 non-null  object
     9   sex             32561 non-null  object
     10  capital-gain    32561 non-null  int64 
     11  capital-loss    32561 non-null  int64 
     12  hours-per-week  32561 non-null  int64 
     13  native-country  32561 non-null  object
     14  salary          32561 non-null  object
    dtypes: int64(6), object(9)
    memory usage: 3.7+ MB


### 5. df.describe()
Generates descriptive statistics for numerical columns:
- descriptive statistics (count, mean, min, max, etc.)
- numerical columns (int64 and float64 types)
- In the below code, we can observe the metrics for numerical columns like age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week.


```python
df.describe()
```





  <div id="df-7a77c468-5b9d-4da9-a572-6c3c28218a36">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>fnlwgt</th>
      <th>education-num</th>
      <th>capital-gain</th>
      <th>capital-loss</th>
      <th>hours-per-week</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>32561.00</td>
      <td>3.26e+04</td>
      <td>32561.00</td>
      <td>32561.00</td>
      <td>32561.00</td>
      <td>32561.00</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>38.58</td>
      <td>1.90e+05</td>
      <td>10.08</td>
      <td>1077.65</td>
      <td>87.30</td>
      <td>40.44</td>
    </tr>
    <tr>
      <th>std</th>
      <td>13.64</td>
      <td>1.06e+05</td>
      <td>2.57</td>
      <td>7385.29</td>
      <td>402.96</td>
      <td>12.35</td>
    </tr>
    <tr>
      <th>min</th>
      <td>17.00</td>
      <td>1.23e+04</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>28.00</td>
      <td>1.18e+05</td>
      <td>9.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>40.00</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>37.00</td>
      <td>1.78e+05</td>
      <td>10.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>40.00</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>48.00</td>
      <td>2.37e+05</td>
      <td>12.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>45.00</td>
    </tr>
    <tr>
      <th>max</th>
      <td>90.00</td>
      <td>1.48e+06</td>
      <td>16.00</td>
      <td>99999.00</td>
      <td>4356.00</td>
      <td>99.00</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-7a77c468-5b9d-4da9-a572-6c3c28218a36')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-7a77c468-5b9d-4da9-a572-6c3c28218a36 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-7a77c468-5b9d-4da9-a572-6c3c28218a36');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




- df.describe(include=[]): to see statistics on non-numerical features:

One has to explicitly indicate data types of interest in the include parameter. <br/>
In the below code, we can observe the metrics for non-numerical columns like workclass, education, marital-status, occupation, relationship, race, sex, native-country & salary.<br/>
However, note that unlike the stats for numerical columns, we can see statss like count, number of unique values, top occuring value and frequently occuring value.


```python
df.describe(include=["object", "bool"])
```





  <div id="df-1463ebee-c30d-4a68-9065-c2651a7ff1e9">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>workclass</th>
      <th>education</th>
      <th>marital-status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>sex</th>
      <th>native-country</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>32561</td>
      <td>32561</td>
      <td>32561</td>
      <td>32561</td>
      <td>32561</td>
      <td>32561</td>
      <td>32561</td>
      <td>32561</td>
      <td>32561</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>9</td>
      <td>16</td>
      <td>7</td>
      <td>15</td>
      <td>6</td>
      <td>5</td>
      <td>2</td>
      <td>42</td>
      <td>2</td>
    </tr>
    <tr>
      <th>top</th>
      <td>Private</td>
      <td>HS-grad</td>
      <td>Married-civ-spouse</td>
      <td>Prof-specialty</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>22696</td>
      <td>10501</td>
      <td>14976</td>
      <td>4140</td>
      <td>13193</td>
      <td>27816</td>
      <td>21790</td>
      <td>29170</td>
      <td>24720</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-1463ebee-c30d-4a68-9065-c2651a7ff1e9')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-1463ebee-c30d-4a68-9065-c2651a7ff1e9 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-1463ebee-c30d-4a68-9065-c2651a7ff1e9');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




### 6. df.columns
prints the column names of a DataFrame


```python
df.columns
```




    Index(['age', 'workclass', 'fnlwgt', 'education', 'education-num',
           'marital-status', 'occupation', 'relationship', 'race', 'sex',
           'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
           'salary'],
          dtype='object')



### 7. df.astype()
To change column data type<br/>
For instance to change age column which is a `int64` to `float` datatype, we can do this operation.


```python
df["age"] = df["age"].astype("float")
```

we can now run the `df.info()` describe column to verify if the conversion happened on the `age` column.


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 32561 entries, 0 to 32560
    Data columns (total 15 columns):
     #   Column          Non-Null Count  Dtype  
    ---  ------          --------------  -----  
     0   age             32561 non-null  float64
     1   workclass       32561 non-null  object 
     2   fnlwgt          32561 non-null  int64  
     3   education       32561 non-null  object 
     4   education-num   32561 non-null  int64  
     5   marital-status  32561 non-null  object 
     6   occupation      32561 non-null  object 
     7   relationship    32561 non-null  object 
     8   race            32561 non-null  object 
     9   sex             32561 non-null  object 
     10  capital-gain    32561 non-null  int64  
     11  capital-loss    32561 non-null  int64  
     12  hours-per-week  32561 non-null  int64  
     13  native-country  32561 non-null  object 
     14  salary          32561 non-null  object 
    dtypes: float64(1), int64(5), object(9)
    memory usage: 3.7+ MB


### 8. df.value_counts()
 To analyze the distribution of values in your dataset:


```python
df["sex"].value_counts()
```




    Male      21790
    Female    10771
    Name: sex, dtype: int64



To calculate fractions, pass normalize=True to the value_counts function. You can use this to determine the percentage distribution.


```python
df["sex"].value_counts(normalize=True)
```




    Male      0.67
    Female    0.33
    Name: sex, dtype: float64




```python
df.value_counts()
```




    age  workclass         fnlwgt  education     education-num  marital-status      occupation         relationship   race   sex     capital-gain  capital-loss  hours-per-week  native-country  salary
    25   Private           195994  1st-4th       2              Never-married       Priv-house-serv    Not-in-family  White  Female  0             0             40              Guatemala       <=50K     3
    23   Private           240137  5th-6th       3              Never-married       Handlers-cleaners  Not-in-family  White  Male    0             0             55              Mexico          <=50K     2
    38   Private           207202  HS-grad       9              Married-civ-spouse  Machine-op-inspct  Husband        White  Male    0             0             48              United-States   >50K      2
    30   Private           144593  HS-grad       9              Never-married       Other-service      Not-in-family  Black  Male    0             0             40              ?               <=50K     2
    49   Self-emp-not-inc  43479   Some-college  10             Married-civ-spouse  Craft-repair       Husband        White  Male    0             0             40              United-States   <=50K     2
                                                                                                                                                                                                          ..
    31   Private           128567  HS-grad       9              Married-civ-spouse  Craft-repair       Husband        White  Male    0             0             40              United-States   <=50K     1
                           128493  HS-grad       9              Divorced            Other-service      Not-in-family  White  Female  0             0             25              United-States   <=50K     1
                           128220  7th-8th       4              Widowed             Adm-clerical       Not-in-family  White  Female  0             0             35              United-States   <=50K     1
                           127610  Bachelors     13             Married-civ-spouse  Prof-specialty     Wife           White  Female  0             0             40              United-States   >50K      1
    90   Self-emp-not-inc  282095  Some-college  10             Married-civ-spouse  Farming-fishing    Husband        White  Male    0             0             40              United-States   <=50K     1
    Length: 32537, dtype: int64



## Sorting

- DataFrame can be sorted by the value of one of the variables (i.e columns).
- For example, we can sort by age (use ascending=False to sort in descending order):




```python
df.sort_values(by="age", ascending=False).head()
```





  <div id="df-7fc73f46-c125-4ab9-b67a-db940b6dc593">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>workclass</th>
      <th>fnlwgt</th>
      <th>education</th>
      <th>education-num</th>
      <th>marital-status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>sex</th>
      <th>capital-gain</th>
      <th>capital-loss</th>
      <th>hours-per-week</th>
      <th>native-country</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5406</th>
      <td>90.0</td>
      <td>Private</td>
      <td>51744</td>
      <td>Masters</td>
      <td>14</td>
      <td>Never-married</td>
      <td>Exec-managerial</td>
      <td>Not-in-family</td>
      <td>Black</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>50</td>
      <td>United-States</td>
      <td>&gt;50K</td>
    </tr>
    <tr>
      <th>6624</th>
      <td>90.0</td>
      <td>Private</td>
      <td>313986</td>
      <td>11th</td>
      <td>7</td>
      <td>Married-civ-spouse</td>
      <td>Craft-repair</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>20610</th>
      <td>90.0</td>
      <td>Private</td>
      <td>206667</td>
      <td>Masters</td>
      <td>14</td>
      <td>Married-civ-spouse</td>
      <td>Prof-specialty</td>
      <td>Wife</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&gt;50K</td>
    </tr>
    <tr>
      <th>1040</th>
      <td>90.0</td>
      <td>Private</td>
      <td>137018</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Never-married</td>
      <td>Other-service</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>1935</th>
      <td>90.0</td>
      <td>Private</td>
      <td>221832</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>45</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-7fc73f46-c125-4ab9-b67a-db940b6dc593')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-7fc73f46-c125-4ab9-b67a-db940b6dc593 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-7fc73f46-c125-4ab9-b67a-db940b6dc593');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




We can also sort by multiple columns:


```python
df.sort_values(by=["age","education-num"], ascending=[False,True]).head()
```





  <div id="df-a710032a-7736-49ed-86cd-cc8fd0ce70bb">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>workclass</th>
      <th>fnlwgt</th>
      <th>education</th>
      <th>education-num</th>
      <th>marital-status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>sex</th>
      <th>capital-gain</th>
      <th>capital-loss</th>
      <th>hours-per-week</th>
      <th>native-country</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>24238</th>
      <td>90.0</td>
      <td>?</td>
      <td>166343</td>
      <td>1st-4th</td>
      <td>2</td>
      <td>Widowed</td>
      <td>?</td>
      <td>Not-in-family</td>
      <td>Black</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>19747</th>
      <td>90.0</td>
      <td>Private</td>
      <td>226968</td>
      <td>7th-8th</td>
      <td>4</td>
      <td>Married-civ-spouse</td>
      <td>Machine-op-inspct</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>25303</th>
      <td>90.0</td>
      <td>?</td>
      <td>175444</td>
      <td>7th-8th</td>
      <td>4</td>
      <td>Separated</td>
      <td>?</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>15</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>32367</th>
      <td>90.0</td>
      <td>Local-gov</td>
      <td>214594</td>
      <td>7th-8th</td>
      <td>4</td>
      <td>Married-civ-spouse</td>
      <td>Protective-serv</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>2653</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>5272</th>
      <td>90.0</td>
      <td>Private</td>
      <td>141758</td>
      <td>9th</td>
      <td>5</td>
      <td>Never-married</td>
      <td>Adm-clerical</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-a710032a-7736-49ed-86cd-cc8fd0ce70bb')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-a710032a-7736-49ed-86cd-cc8fd0ce70bb button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-a710032a-7736-49ed-86cd-cc8fd0ce70bb');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




## Indexing and retrieving data

### 1. Calculate mean


```python
df['hours-per-week'].mean()
```




    40.437455852092995



### 2. Column Indexing
- DataFrames can be indexed by column name (label) or row name (index).
- The loc method is used for indexing by name, while iloc() is used for indexing by number.

| loc() | iloc() |
|----------|----------|
|   used for indexing by name  |   used for indexing by number  |
|   df.loc['row_index'] will return the row with the specified index label  |   df.iloc[row_number] will return the row at the specified integer position  |
|   df.loc[row_label, column_label] will return the value at the specified row and column labels  |   df.iloc[row_number, column_number] will return the value at the specified row and column positions  |


Fetch rows 2 to 4 (inclusive)


```python
df.loc[2:4]
```





  <div id="df-36934431-5acb-4dc1-b2da-1286deb100f6">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>workclass</th>
      <th>fnlwgt</th>
      <th>education</th>
      <th>education-num</th>
      <th>marital-status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>sex</th>
      <th>capital-gain</th>
      <th>capital-loss</th>
      <th>hours-per-week</th>
      <th>native-country</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>38</td>
      <td>Private</td>
      <td>215646</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Divorced</td>
      <td>Handlers-cleaners</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>3</th>
      <td>53</td>
      <td>Private</td>
      <td>234721</td>
      <td>11th</td>
      <td>7</td>
      <td>Married-civ-spouse</td>
      <td>Handlers-cleaners</td>
      <td>Husband</td>
      <td>Black</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>4</th>
      <td>28</td>
      <td>Private</td>
      <td>338409</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Prof-specialty</td>
      <td>Wife</td>
      <td>Black</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>Cuba</td>
      <td>&lt;=50K</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-36934431-5acb-4dc1-b2da-1286deb100f6')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-36934431-5acb-4dc1-b2da-1286deb100f6 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-36934431-5acb-4dc1-b2da-1286deb100f6');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




Fetch rows 2 to 4 (not inclusive). Can use iloc[] as well.


```python
df.iloc[2:4]
```





  <div id="df-84ae7c17-19b3-4602-8ecd-070e1034600d">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>workclass</th>
      <th>fnlwgt</th>
      <th>education</th>
      <th>education-num</th>
      <th>marital-status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>sex</th>
      <th>capital-gain</th>
      <th>capital-loss</th>
      <th>hours-per-week</th>
      <th>native-country</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>38</td>
      <td>Private</td>
      <td>215646</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Divorced</td>
      <td>Handlers-cleaners</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>3</th>
      <td>53</td>
      <td>Private</td>
      <td>234721</td>
      <td>11th</td>
      <td>7</td>
      <td>Married-civ-spouse</td>
      <td>Handlers-cleaners</td>
      <td>Husband</td>
      <td>Black</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-84ae7c17-19b3-4602-8ecd-070e1034600d')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-84ae7c17-19b3-4602-8ecd-070e1034600d button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-84ae7c17-19b3-4602-8ecd-070e1034600d');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




Display rows 2 to 4 (inclusive) and columns from education to relationship.


```python
df.loc[2:4, "education":"relationship"]
```





  <div id="df-3bd773c7-ea59-4543-9589-17137eab614f">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>education</th>
      <th>education-num</th>
      <th>marital-status</th>
      <th>occupation</th>
      <th>relationship</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>HS-grad</td>
      <td>9</td>
      <td>Divorced</td>
      <td>Handlers-cleaners</td>
      <td>Not-in-family</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11th</td>
      <td>7</td>
      <td>Married-civ-spouse</td>
      <td>Handlers-cleaners</td>
      <td>Husband</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bachelors</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Prof-specialty</td>
      <td>Wife</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-3bd773c7-ea59-4543-9589-17137eab614f')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-3bd773c7-ea59-4543-9589-17137eab614f button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-3bd773c7-ea59-4543-9589-17137eab614f');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




Display rows 2 to 4 (inclusive) and first three columns.


```python
df.iloc[2:4, 0:3]
```





  <div id="df-16e75f3f-6837-4c03-bc2d-af98e5783cb1">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>workclass</th>
      <th>fnlwgt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>38</td>
      <td>Private</td>
      <td>215646</td>
    </tr>
    <tr>
      <th>3</th>
      <td>53</td>
      <td>Private</td>
      <td>234721</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-16e75f3f-6837-4c03-bc2d-af98e5783cb1')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-16e75f3f-6837-4c03-bc2d-af98e5783cb1 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-16e75f3f-6837-4c03-bc2d-af98e5783cb1');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




Note: You can cross the methods. For example, cannot use numerical representation on loc() or use alphabetic representation on iloc()

If we need the first or the last line of the data frame, we can use the df[:1] or df[-1:] construction:
- df[:1]: start at 0. end at 1 (non-inclusive).
- Result: displays first row alone.


```python
df[:1]
```





  <div id="df-d9f4cb8a-7ee7-4308-866a-fa3fb71ddf97">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>workclass</th>
      <th>fnlwgt</th>
      <th>education</th>
      <th>education-num</th>
      <th>marital-status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>sex</th>
      <th>capital-gain</th>
      <th>capital-loss</th>
      <th>hours-per-week</th>
      <th>native-country</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>39</td>
      <td>State-gov</td>
      <td>77516</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Never-married</td>
      <td>Adm-clerical</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>2174</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-d9f4cb8a-7ee7-4308-866a-fa3fb71ddf97')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-d9f4cb8a-7ee7-4308-866a-fa3fb71ddf97 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-d9f4cb8a-7ee7-4308-866a-fa3fb71ddf97');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




Indexing can be done in loops as well. i.e., you can use negative numbers as well. e.g., df[-7:-2]
- df[-1:] : start at -1(last row). end at 0 (non-inclusive)
- Result: displays last row alone.


```python
df[-1:]
```





  <div id="df-cea33295-b3fe-415f-a686-c37693754376">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>workclass</th>
      <th>fnlwgt</th>
      <th>education</th>
      <th>education-num</th>
      <th>marital-status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>sex</th>
      <th>capital-gain</th>
      <th>capital-loss</th>
      <th>hours-per-week</th>
      <th>native-country</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>32560</th>
      <td>52</td>
      <td>Self-emp-inc</td>
      <td>287927</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Wife</td>
      <td>White</td>
      <td>Female</td>
      <td>15024</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&gt;50K</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-cea33295-b3fe-415f-a686-c37693754376')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-cea33295-b3fe-415f-a686-c37693754376 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-cea33295-b3fe-415f-a686-c37693754376');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




### 3. Boolean indexing

* Boolean indexing with one column is also very convenient.
* The syntax is df[P(df['Name'])], where P is some logical condition that is checked for each element of the Name column.
* The result of such indexing is the DataFrame consisting only of the rows that satisfy the P condition on the Name column.

**What are the average values of numerical features for male users?**

Here we’ll resort to an additional method select_dtypes to select all numeric columns.


*   df.select_dtypes(include=np.number) selects columns in the DataFrame that have numeric data types. It filters out columns with non-numeric data types.

*   [df["sex"] == "male"] filters the DataFrame rows based on a condition. It selects only the rows where the "sex" column has a value of "male".

*   .mean() calculates the mean (average) value for each column in the filtered DataFrame.

* Therefore, the code calculates the mean value for each numeric column in the DataFrame (df) but only for the rows where the "sex" column is "male". This provides the average values of the numeric columns specifically for the male subset of the data.






```python
df.select_dtypes(include=np.number)[df["sex"] == "Male"].mean()
```




    age                   39.43
    fnlwgt            191771.45
    education-num         10.10
    capital-gain        1329.37
    capital-loss         100.21
    hours-per-week        42.43
    dtype: float64



* What is the mean age of Females in the survey?
  - Boolean condtion P => df["sex"] == "Female"
  - df[P]["age"]
  - using the method .mean() to calculate mean in the column of interest.


```python
df[df["sex"] == "Female"]["age"].mean()
```




    36.85823043357163



* What is the max age of males who have a capital gain greater than 15,000?
  - Boolean condtion P => (df["sex"] == "Male") & (df["capital-gain"] > 15000 )
  - df[P]["age"]
  - using the method .max() to calculate max in the column of interest.


```python
df[(df["sex"] == "Male") & (df["capital-gain"] > 15000 )]["age"].max()
```




    90.0



Example: Finding The proportion of German citizens (native-country feature):
- df.shape[0] gives the number of rows in DataFrame
- Dividing the sum of True values by the total number of rows gives the percentage of individuals with Germany as their native country.







```python
float((df["native-country"] == "Germany").sum()) / df.shape[0]
```




    0.004207487485028101



Example: What are mean value and standard deviation of the age of those who receive more than 50K per year (salary feature) and those who receive less than 50K per year?


```python
ages1 = df[df["salary"] == ">50K"]["age"]
ages2 = df[df["salary"] == "<=50K"]["age"]
print(
    "The average age of the rich: {0} +- {1} years, poor - {2} +- {3} years.".format(
        round(ages1.mean()),
        round(ages1.std(), 1),
        round(ages2.mean()),
        round(ages2.std(), 1),
    )
)
```

    The average age of the rich: 44 +- 10.5 years, poor - 37 +- 14.0 years.


 Example: Is it true that people who earn more than 50K have at least high school education? (education – Bachelors, Prof-school, Assoc-acdm, Assoc-voc, Masters or Doctorate feature)


```python
df[df["salary"] == ">50K"]["education"].unique()  # No
```




    array(['HS-grad', 'Masters', 'Bachelors', 'Some-college', 'Assoc-voc',
           'Doctorate', 'Prof-school', 'Assoc-acdm', '7th-8th', '12th',
           '10th', '11th', '9th', '5th-6th', '1st-4th'], dtype=object)



Example: Among whom is the proportion of those who earn a lot (>50K) greater: married or single men (marital-status feature)? Consider as married those who have a marital-status starting with Married (Married-civ-spouse, Married-spouse-absent or Married-AF-spouse), the rest are considered bachelors.



```python
# married men
df[(df["sex"] == "Male")
     & (df["marital-status"].str.startswith("Married"))][
    "salary"
].value_counts(normalize=True)
```




    <=50K    0.56
    >50K     0.44
    Name: salary, dtype: float64




```python
# single men
df[
    (df["sex"] == "Male")
    & ~(df["marital-status"].str.startswith("Married"))
]["salary"].value_counts(normalize=True)
```




    <=50K    0.92
    >50K     0.08
    Name: salary, dtype: float64



### 4. Apply Method
- df.apply() method can be used to apply functions on Cells, Rows or Columns

For example, find the max value across all columns.


```python
df.apply(np.max)
```




    age                             90
    workclass              Without-pay
    fnlwgt                     1484705
    education             Some-college
    education-num                   16
    marital-status             Widowed
    occupation        Transport-moving
    relationship                  Wife
    race                         White
    sex                           Male
    capital-gain                 99999
    capital-loss                  4356
    hours-per-week                  99
    native-country          Yugoslavia
    salary                        >50K
    dtype: object



Lambda functions are very convenient to apply using apply() method.
- For eg, select the native-countries starting with Y.
- To break this down:
  - `df["native-country"]` selects the  native-country.
  - `df["native-country"].apply(lambda state: state[0] == "Y")`  applies lambda function to include only the rows where the "native-country" column starts with the letter "Y"
  - `df[].head()` - then returns the first few rows of the filtered DataFrame.


```python
df[df["native-country"].apply(lambda country: country[0] == "Y")].head()
```





  <div id="df-6ed1191c-7465-4102-908e-ca811735f0be">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>workclass</th>
      <th>fnlwgt</th>
      <th>education</th>
      <th>education-num</th>
      <th>marital-status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>sex</th>
      <th>capital-gain</th>
      <th>capital-loss</th>
      <th>hours-per-week</th>
      <th>native-country</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1025</th>
      <td>56</td>
      <td>Private</td>
      <td>169133</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Married-civ-spouse</td>
      <td>Other-service</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>50</td>
      <td>Yugoslavia</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>4447</th>
      <td>25</td>
      <td>Private</td>
      <td>191230</td>
      <td>Some-college</td>
      <td>10</td>
      <td>Never-married</td>
      <td>Exec-managerial</td>
      <td>Own-child</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>Yugoslavia</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>6328</th>
      <td>20</td>
      <td>Private</td>
      <td>175069</td>
      <td>Some-college</td>
      <td>10</td>
      <td>Never-married</td>
      <td>Adm-clerical</td>
      <td>Own-child</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>Yugoslavia</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>7287</th>
      <td>35</td>
      <td>Private</td>
      <td>164526</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Married-civ-spouse</td>
      <td>Other-service</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>Yugoslavia</td>
      <td>&gt;50K</td>
    </tr>
    <tr>
      <th>12506</th>
      <td>40</td>
      <td>Local-gov</td>
      <td>183096</td>
      <td>9th</td>
      <td>5</td>
      <td>Married-civ-spouse</td>
      <td>Other-service</td>
      <td>Wife</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>Yugoslavia</td>
      <td>&gt;50K</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-6ed1191c-7465-4102-908e-ca811735f0be')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-6ed1191c-7465-4102-908e-ca811735f0be button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-6ed1191c-7465-4102-908e-ca811735f0be');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




### 5. Map & Replace Method

map method can be used to replace values in a column by passing a dictionary of the form {old_value: new_value} as its argument:


```python
d = {"Male": 1, "Female": 2}
df["sex_1"] = df["sex"].map(d)
df.head()
```





  <div id="df-11e7a0bd-e04d-4c9b-a172-507c9920a8a6">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>workclass</th>
      <th>fnlwgt</th>
      <th>education</th>
      <th>education-num</th>
      <th>marital-status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>sex</th>
      <th>capital-gain</th>
      <th>capital-loss</th>
      <th>hours-per-week</th>
      <th>native-country</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>39</td>
      <td>State-gov</td>
      <td>77516</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Never-married</td>
      <td>Adm-clerical</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>NaN</td>
      <td>2174</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>1</th>
      <td>50</td>
      <td>Self-emp-not-inc</td>
      <td>83311</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Husband</td>
      <td>White</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>13</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>2</th>
      <td>38</td>
      <td>Private</td>
      <td>215646</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Divorced</td>
      <td>Handlers-cleaners</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>3</th>
      <td>53</td>
      <td>Private</td>
      <td>234721</td>
      <td>11th</td>
      <td>7</td>
      <td>Married-civ-spouse</td>
      <td>Handlers-cleaners</td>
      <td>Husband</td>
      <td>Black</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>4</th>
      <td>28</td>
      <td>Private</td>
      <td>338409</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Prof-specialty</td>
      <td>Wife</td>
      <td>Black</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>Cuba</td>
      <td>&lt;=50K</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-11e7a0bd-e04d-4c9b-a172-507c9920a8a6')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-11e7a0bd-e04d-4c9b-a172-507c9920a8a6 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-11e7a0bd-e04d-4c9b-a172-507c9920a8a6');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




same thing can be done with the replace method.



```python
df['sex_2'] = df['sex'].replace(d)
df.head()
```





  <div id="df-c3c51eff-af59-420f-bdc9-fc953b0a51a7">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>workclass</th>
      <th>fnlwgt</th>
      <th>education</th>
      <th>education-num</th>
      <th>marital-status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>sex</th>
      <th>capital-gain</th>
      <th>capital-loss</th>
      <th>hours-per-week</th>
      <th>native-country</th>
      <th>salary</th>
      <th>sex_2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>39</td>
      <td>State-gov</td>
      <td>77516</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Never-married</td>
      <td>Adm-clerical</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>2174</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>50</td>
      <td>Self-emp-not-inc</td>
      <td>83311</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>13</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>38</td>
      <td>Private</td>
      <td>215646</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Divorced</td>
      <td>Handlers-cleaners</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>53</td>
      <td>Private</td>
      <td>234721</td>
      <td>11th</td>
      <td>7</td>
      <td>Married-civ-spouse</td>
      <td>Handlers-cleaners</td>
      <td>Husband</td>
      <td>Black</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>28</td>
      <td>Private</td>
      <td>338409</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Prof-specialty</td>
      <td>Wife</td>
      <td>Black</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>Cuba</td>
      <td>&lt;=50K</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-c3c51eff-af59-420f-bdc9-fc953b0a51a7')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-c3c51eff-af59-420f-bdc9-fc953b0a51a7 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-c3c51eff-af59-420f-bdc9-fc953b0a51a7');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




However, note `replace` method will not do anything with values not found in the mapping dictionary, while while `map` will change them to NaNs.


```python
# Create a Series
s = pd.Series(['A', 'B', 'C', 'D'])

# Create a mapping dictionary
mapping = {'A': 'Apple', 'B': 'Banana'}

# Using replace method
replaced = s.replace(mapping)
print('replace method will not do anything with values not found in the mapping dictionary')
print(replaced)

# Using map method
mapped = s.map(mapping)
print('map method will change values not found in the mapping dictionary to NaNs')
print(mapped)

```

    replace method will not do anything with values not found in the mapping dictionary
    0     Apple
    1    Banana
    2         C
    3         D
    dtype: object
    map method wwill change values not found in the mapping dictionary to NaNs
    0     Apple
    1    Banana
    2       NaN
    3       NaN
    dtype: object


### 6. Grouping

Grouping data in Pandas works as follows:
`df.groupby(by=grouping_columns)[columns_to_show].function()`



1.   First, the groupby method divides the grouping_columns by their values. They become a new index in the resulting dataframe.

2.   Then, columns of interest are selected (columns_to_show). If columns_to_show is not included, all non groupby clauses will be included.

3.   Finally, one or several functions are applied to the obtained groups per selected columns.



To understand this more intuitively, we can start with this small example.

*   We can group by `education`. By doing this we are creating a new dataframe with the `education` column as index.
* However, this dataframe as no series values. For this we can apply as simple function like `count()`
* The resulting dataframe shows the count each `education` value across all other columns. The data is repeating because, `count()` is simply counting the number of entries for each education type in age, workclass, etc which obviously will be same.


```python
result = df.groupby('education').count()
print(result)
```

                    age  workclass  fnlwgt  education-num  marital-status  \
    education                                                               
    10th            933        933     933            933             933   
    11th           1175       1175    1175           1175            1175   
    12th            433        433     433            433             433   
    1st-4th         168        168     168            168             168   
    5th-6th         333        333     333            333             333   
    7th-8th         646        646     646            646             646   
    9th             514        514     514            514             514   
    Assoc-acdm     1067       1067    1067           1067            1067   
    Assoc-voc      1382       1382    1382           1382            1382   
    Bachelors      5355       5355    5355           5355            5355   
    Doctorate       413        413     413            413             413   
    HS-grad       10501      10501   10501          10501           10501   
    Masters        1723       1723    1723           1723            1723   
    Preschool        51         51      51             51              51   
    Prof-school     576        576     576            576             576   
    Some-college   7291       7291    7291           7291            7291   
    
                  occupation  relationship   race    sex  capital-gain  \
    education                                                            
    10th                 933           933    933    933           933   
    11th                1175          1175   1175   1175          1175   
    12th                 433           433    433    433           433   
    1st-4th              168           168    168    168           168   
    5th-6th              333           333    333    333           333   
    7th-8th              646           646    646    646           646   
    9th                  514           514    514    514           514   
    Assoc-acdm          1067          1067   1067   1067          1067   
    Assoc-voc           1382          1382   1382   1382          1382   
    Bachelors           5355          5355   5355   5355          5355   
    Doctorate            413           413    413    413           413   
    HS-grad            10501         10501  10501  10501         10501   
    Masters             1723          1723   1723   1723          1723   
    Preschool             51            51     51     51            51   
    Prof-school          576           576    576    576           576   
    Some-college        7291          7291   7291   7291          7291   
    
                  capital-loss  hours-per-week  native-country  salary  sex_2  
    education                                                                  
    10th                   933             933             933     933    933  
    11th                  1175            1175            1175    1175   1175  
    12th                   433             433             433     433    433  
    1st-4th                168             168             168     168    168  
    5th-6th                333             333             333     333    333  
    7th-8th                646             646             646     646    646  
    9th                    514             514             514     514    514  
    Assoc-acdm            1067            1067            1067    1067   1067  
    Assoc-voc             1382            1382            1382    1382   1382  
    Bachelors             5355            5355            5355    5355   5355  
    Doctorate              413             413             413     413    413  
    HS-grad              10501           10501           10501   10501  10501  
    Masters               1723            1723            1723    1723   1723  
    Preschool               51              51              51      51     51  
    Prof-school            576             576             576     576    576  
    Some-college          7291            7291            7291    7291   7291  


This will start to make more sense by using a more meaningful function like .mean() which calculates the mean value for all numerical columns under each `education` type.


```python
result = df.groupby('education').mean()
print(result)
```

                    age     fnlwgt  education-num  capital-gain  capital-loss  \
    education                                                                   
    10th          37.43  196832.47            6.0        404.57         56.85   
    11th          32.36  194928.08            7.0        215.10         50.08   
    12th          32.00  199097.51            8.0        284.09         32.34   
    1st-4th       46.14  239303.00            2.0        125.88         48.33   
    5th-6th       42.89  232448.33            3.0        176.02         68.25   
    7th-8th       48.45  188079.17            4.0        233.94         65.67   
    9th           41.06  202485.07            5.0        342.09         29.00   
    Assoc-acdm    37.38  193424.09           12.0        640.40         93.42   
    Assoc-voc     38.55  181936.02           11.0        715.05         72.75   
    Bachelors     38.90  188055.91           13.0       1756.30        118.35   
    Doctorate     47.70  186698.76           16.0       4770.15        262.85   
    HS-grad       38.97  189538.74            9.0        576.80         70.47   
    Masters       44.05  179852.36           14.0       2562.56        166.72   
    Preschool     42.76  235889.37            1.0        898.39         66.49   
    Prof-school   44.75  185663.71           15.0      10414.42        231.20   
    Some-college  35.76  188742.92           10.0        598.82         71.64   
    
                  hours-per-week  sex_2  
    education                            
    10th                   37.05   1.32  
    11th                   33.93   1.37  
    12th                   35.78   1.33  
    1st-4th                38.26   1.27  
    5th-6th                38.90   1.25  
    7th-8th                39.37   1.25  
    9th                    38.04   1.28  
    Assoc-acdm             40.50   1.39  
    Assoc-voc              41.61   1.36  
    Bachelors              42.61   1.30  
    Doctorate              46.97   1.21  
    HS-grad                40.58   1.32  
    Masters                43.84   1.31  
    Preschool              36.65   1.31  
    Prof-school            47.43   1.16  
    Some-college           38.85   1.38  


    <ipython-input-28-a0bc15c64f52>:1: FutureWarning: The default value of numeric_only in DataFrameGroupBy.mean is deprecated. In a future version, numeric_only will default to False. Either specify numeric_only or select only columns which should be valid for the function.
      result = df.groupby('education').mean()


In the below example:


*   we grouped the DataFrame df by the 'education' column using the groupby method.
*   Then, we selected the 'age' column to calculate the average age for each education level.
*   The result is a Series that shows the average age for each education category.






```python
# result = df.groupby('education')['age'].mean()
result = df.groupby('education')['age'].mean()
print(result)
```

    education
    10th            37.43
    11th            32.36
    12th            32.00
    1st-4th         46.14
    5th-6th         42.89
    7th-8th         48.45
    9th             41.06
    Assoc-acdm      37.38
    Assoc-voc       38.55
    Bachelors       38.90
    Doctorate       47.70
    HS-grad         38.97
    Masters         44.05
    Preschool       42.76
    Prof-school     44.75
    Some-college    35.76
    Name: age, dtype: float64


Here is another example. Suppose we want to groupby Salary and determine how people of various occupations are collecting salary, we can do the following:

*   we can group the DataFrame df by both the 'occupation' and 'salary' columns using the groupby method. This is like first creating an index for `occupation` and then create index `salary` under **each** occupation type.
*   Then, we applied the size function to count the number of occurrences for each combination of occupation and salary.
*   The result is a Series that shows the count for each occupation and salary category.




```python
# Grouping by 'occupation' and 'salary' columns and counting the number of occurrences
result = df.groupby(['occupation', 'salary']).size()
print(result)
```

    occupation         salary
    ?                  <=50K     1652
                       >50K       191
    Adm-clerical       <=50K     3263
                       >50K       507
    Armed-Forces       <=50K        8
                       >50K         1
    Craft-repair       <=50K     3170
                       >50K       929
    Exec-managerial    <=50K     2098
                       >50K      1968
    Farming-fishing    <=50K      879
                       >50K       115
    Handlers-cleaners  <=50K     1284
                       >50K        86
    Machine-op-inspct  <=50K     1752
                       >50K       250
    Other-service      <=50K     3158
                       >50K       137
    Priv-house-serv    <=50K      148
                       >50K         1
    Prof-specialty     <=50K     2281
                       >50K      1859
    Protective-serv    <=50K      438
                       >50K       211
    Sales              <=50K     2667
                       >50K       983
    Tech-support       <=50K      645
                       >50K       283
    Transport-moving   <=50K     1277
                       >50K       320
    dtype: int64


We can extend this to show percentage by doing the following:


```python
# Grouping by 'occupation' and 'salary' columns and calculating the percentage
result = df.groupby(['occupation', 'salary']).size() / df.groupby(['occupation']).size() * 100
print(result)
```

    occupation         salary
    ?                  <=50K     89.64
                       >50K      10.36
    Adm-clerical       <=50K     86.55
                       >50K      13.45
    Armed-Forces       <=50K     88.89
                       >50K      11.11
    Craft-repair       <=50K     77.34
                       >50K      22.66
    Exec-managerial    <=50K     51.60
                       >50K      48.40
    Farming-fishing    <=50K     88.43
                       >50K      11.57
    Handlers-cleaners  <=50K     93.72
                       >50K       6.28
    Machine-op-inspct  <=50K     87.51
                       >50K      12.49
    Other-service      <=50K     95.84
                       >50K       4.16
    Priv-house-serv    <=50K     99.33
                       >50K       0.67
    Prof-specialty     <=50K     55.10
                       >50K      44.90
    Protective-serv    <=50K     67.49
                       >50K      32.51
    Sales              <=50K     73.07
                       >50K      26.93
    Tech-support       <=50K     69.50
                       >50K      30.50
    Transport-moving   <=50K     79.96
                       >50K      20.04
    dtype: float64


In this example, we can compare the normal describe method vs the describe method with groupby. As we can see, the group by method is indexed on occupation, where as the regular df is indexed on 0 to N row index.


```python
df.describe(percentiles=[])
```





  <div id="df-aa19063a-0427-40a9-94c4-5c55be66680b">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>fnlwgt</th>
      <th>education-num</th>
      <th>capital-gain</th>
      <th>capital-loss</th>
      <th>hours-per-week</th>
      <th>sex_2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>32561.00</td>
      <td>3.26e+04</td>
      <td>32561.00</td>
      <td>32561.00</td>
      <td>32561.00</td>
      <td>32561.00</td>
      <td>32561.00</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>38.58</td>
      <td>1.90e+05</td>
      <td>10.08</td>
      <td>1077.65</td>
      <td>87.30</td>
      <td>40.44</td>
      <td>1.33</td>
    </tr>
    <tr>
      <th>std</th>
      <td>13.64</td>
      <td>1.06e+05</td>
      <td>2.57</td>
      <td>7385.29</td>
      <td>402.96</td>
      <td>12.35</td>
      <td>0.47</td>
    </tr>
    <tr>
      <th>min</th>
      <td>17.00</td>
      <td>1.23e+04</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>37.00</td>
      <td>1.78e+05</td>
      <td>10.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>40.00</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>max</th>
      <td>90.00</td>
      <td>1.48e+06</td>
      <td>16.00</td>
      <td>99999.00</td>
      <td>4356.00</td>
      <td>99.00</td>
      <td>2.00</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-aa19063a-0427-40a9-94c4-5c55be66680b')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-aa19063a-0427-40a9-94c4-5c55be66680b button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-aa19063a-0427-40a9-94c4-5c55be66680b');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
df.groupby(["occupation"]).describe(percentiles=[])
```





  <div id="df-75b5f33e-6a6b-4add-b15b-4f9d89aeb844">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="6" halign="left">age</th>
      <th colspan="4" halign="left">fnlwgt</th>
      <th>...</th>
      <th colspan="4" halign="left">hours-per-week</th>
      <th colspan="6" halign="left">sex_2</th>
    </tr>
    <tr>
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>50%</th>
      <th>max</th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>...</th>
      <th>std</th>
      <th>min</th>
      <th>50%</th>
      <th>max</th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>50%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>occupation</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>?</th>
      <td>1843.0</td>
      <td>40.88</td>
      <td>20.34</td>
      <td>17.0</td>
      <td>35.0</td>
      <td>90.0</td>
      <td>1843.0</td>
      <td>188658.67</td>
      <td>107089.08</td>
      <td>12285.0</td>
      <td>...</td>
      <td>14.91</td>
      <td>1.0</td>
      <td>36.0</td>
      <td>99.0</td>
      <td>1843.0</td>
      <td>1.46</td>
      <td>0.50</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>Adm-clerical</th>
      <td>3770.0</td>
      <td>36.96</td>
      <td>13.36</td>
      <td>17.0</td>
      <td>35.0</td>
      <td>90.0</td>
      <td>3770.0</td>
      <td>192043.40</td>
      <td>103163.89</td>
      <td>19302.0</td>
      <td>...</td>
      <td>9.59</td>
      <td>1.0</td>
      <td>40.0</td>
      <td>80.0</td>
      <td>3770.0</td>
      <td>1.67</td>
      <td>0.47</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>Armed-Forces</th>
      <td>9.0</td>
      <td>30.22</td>
      <td>8.09</td>
      <td>23.0</td>
      <td>29.0</td>
      <td>46.0</td>
      <td>9.0</td>
      <td>215425.89</td>
      <td>83315.89</td>
      <td>76313.0</td>
      <td>...</td>
      <td>14.07</td>
      <td>8.0</td>
      <td>40.0</td>
      <td>60.0</td>
      <td>9.0</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Craft-repair</th>
      <td>4099.0</td>
      <td>39.03</td>
      <td>11.61</td>
      <td>17.0</td>
      <td>38.0</td>
      <td>90.0</td>
      <td>4099.0</td>
      <td>192132.60</td>
      <td>107434.09</td>
      <td>19491.0</td>
      <td>...</td>
      <td>9.05</td>
      <td>1.0</td>
      <td>40.0</td>
      <td>99.0</td>
      <td>4099.0</td>
      <td>1.05</td>
      <td>0.23</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>Exec-managerial</th>
      <td>4066.0</td>
      <td>42.17</td>
      <td>11.97</td>
      <td>17.0</td>
      <td>41.0</td>
      <td>90.0</td>
      <td>4066.0</td>
      <td>184414.01</td>
      <td>103314.96</td>
      <td>19914.0</td>
      <td>...</td>
      <td>11.11</td>
      <td>1.0</td>
      <td>40.0</td>
      <td>99.0</td>
      <td>4066.0</td>
      <td>1.29</td>
      <td>0.45</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>Farming-fishing</th>
      <td>994.0</td>
      <td>41.21</td>
      <td>15.07</td>
      <td>17.0</td>
      <td>39.0</td>
      <td>90.0</td>
      <td>994.0</td>
      <td>170190.18</td>
      <td>116925.90</td>
      <td>20795.0</td>
      <td>...</td>
      <td>17.32</td>
      <td>2.0</td>
      <td>40.0</td>
      <td>99.0</td>
      <td>994.0</td>
      <td>1.07</td>
      <td>0.25</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>Handlers-cleaners</th>
      <td>1370.0</td>
      <td>32.17</td>
      <td>12.37</td>
      <td>17.0</td>
      <td>29.0</td>
      <td>90.0</td>
      <td>1370.0</td>
      <td>204391.01</td>
      <td>111934.10</td>
      <td>19214.0</td>
      <td>...</td>
      <td>10.58</td>
      <td>2.0</td>
      <td>40.0</td>
      <td>95.0</td>
      <td>1370.0</td>
      <td>1.12</td>
      <td>0.32</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>Machine-op-inspct</th>
      <td>2002.0</td>
      <td>37.72</td>
      <td>12.07</td>
      <td>17.0</td>
      <td>36.0</td>
      <td>90.0</td>
      <td>2002.0</td>
      <td>195040.88</td>
      <td>98159.93</td>
      <td>13769.0</td>
      <td>...</td>
      <td>7.59</td>
      <td>1.0</td>
      <td>40.0</td>
      <td>96.0</td>
      <td>2002.0</td>
      <td>1.27</td>
      <td>0.45</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>Other-service</th>
      <td>3295.0</td>
      <td>34.95</td>
      <td>14.52</td>
      <td>17.0</td>
      <td>32.0</td>
      <td>90.0</td>
      <td>3295.0</td>
      <td>188608.45</td>
      <td>109452.80</td>
      <td>19752.0</td>
      <td>...</td>
      <td>12.71</td>
      <td>1.0</td>
      <td>40.0</td>
      <td>99.0</td>
      <td>3295.0</td>
      <td>1.55</td>
      <td>0.50</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>Priv-house-serv</th>
      <td>149.0</td>
      <td>41.72</td>
      <td>18.63</td>
      <td>17.0</td>
      <td>40.0</td>
      <td>81.0</td>
      <td>149.0</td>
      <td>201107.52</td>
      <td>102595.40</td>
      <td>24384.0</td>
      <td>...</td>
      <td>16.18</td>
      <td>4.0</td>
      <td>35.0</td>
      <td>99.0</td>
      <td>149.0</td>
      <td>1.95</td>
      <td>0.23</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>Prof-specialty</th>
      <td>4140.0</td>
      <td>40.52</td>
      <td>12.02</td>
      <td>17.0</td>
      <td>40.0</td>
      <td>90.0</td>
      <td>4140.0</td>
      <td>185296.61</td>
      <td>100135.45</td>
      <td>14878.0</td>
      <td>...</td>
      <td>12.54</td>
      <td>1.0</td>
      <td>40.0</td>
      <td>99.0</td>
      <td>4140.0</td>
      <td>1.37</td>
      <td>0.48</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>Protective-serv</th>
      <td>649.0</td>
      <td>38.95</td>
      <td>12.82</td>
      <td>17.0</td>
      <td>36.0</td>
      <td>90.0</td>
      <td>649.0</td>
      <td>202039.95</td>
      <td>101910.29</td>
      <td>19302.0</td>
      <td>...</td>
      <td>12.33</td>
      <td>3.0</td>
      <td>40.0</td>
      <td>99.0</td>
      <td>649.0</td>
      <td>1.12</td>
      <td>0.32</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>Sales</th>
      <td>3650.0</td>
      <td>37.35</td>
      <td>14.19</td>
      <td>17.0</td>
      <td>35.0</td>
      <td>90.0</td>
      <td>3650.0</td>
      <td>190885.89</td>
      <td>103779.87</td>
      <td>19410.0</td>
      <td>...</td>
      <td>13.24</td>
      <td>2.0</td>
      <td>40.0</td>
      <td>99.0</td>
      <td>3650.0</td>
      <td>1.35</td>
      <td>0.48</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>Tech-support</th>
      <td>928.0</td>
      <td>37.02</td>
      <td>11.32</td>
      <td>17.0</td>
      <td>36.0</td>
      <td>73.0</td>
      <td>928.0</td>
      <td>192098.30</td>
      <td>113987.28</td>
      <td>19847.0</td>
      <td>...</td>
      <td>10.58</td>
      <td>3.0</td>
      <td>40.0</td>
      <td>99.0</td>
      <td>928.0</td>
      <td>1.38</td>
      <td>0.48</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>Transport-moving</th>
      <td>1597.0</td>
      <td>40.20</td>
      <td>12.45</td>
      <td>17.0</td>
      <td>39.0</td>
      <td>90.0</td>
      <td>1597.0</td>
      <td>190366.36</td>
      <td>109240.30</td>
      <td>18827.0</td>
      <td>...</td>
      <td>12.72</td>
      <td>5.0</td>
      <td>40.0</td>
      <td>99.0</td>
      <td>1597.0</td>
      <td>1.06</td>
      <td>0.23</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
<p>15 rows × 42 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-75b5f33e-6a6b-4add-b15b-4f9d89aeb844')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-75b5f33e-6a6b-4add-b15b-4f9d89aeb844 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-75b5f33e-6a6b-4add-b15b-4f9d89aeb844');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




We can do the same thing using `agg()` function.


```python
df.groupby(["occupation"]).agg([np.mean, np.std, np.min, np.max])
```

    <ipython-input-32-f3544e70bb41>:1: FutureWarning: ['workclass', 'education', 'marital-status', 'relationship', 'race', 'sex', 'native-country', 'salary'] did not aggregate successfully. If any error is raised this will raise in a future version of pandas. Drop these columns/ops to avoid this warning.
      df.groupby(["occupation"]).agg([np.mean, np.std, np.min, np.max])






  <div id="df-ba502fbc-1d20-4208-8c3d-9d027fb7303a">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="4" halign="left">age</th>
      <th colspan="4" halign="left">fnlwgt</th>
      <th colspan="2" halign="left">education-num</th>
      <th>...</th>
      <th colspan="2" halign="left">capital-loss</th>
      <th colspan="4" halign="left">hours-per-week</th>
      <th colspan="4" halign="left">sex_2</th>
    </tr>
    <tr>
      <th></th>
      <th>mean</th>
      <th>std</th>
      <th>amin</th>
      <th>amax</th>
      <th>mean</th>
      <th>std</th>
      <th>amin</th>
      <th>amax</th>
      <th>mean</th>
      <th>std</th>
      <th>...</th>
      <th>amin</th>
      <th>amax</th>
      <th>mean</th>
      <th>std</th>
      <th>amin</th>
      <th>amax</th>
      <th>mean</th>
      <th>std</th>
      <th>amin</th>
      <th>amax</th>
    </tr>
    <tr>
      <th>occupation</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>?</th>
      <td>40.88</td>
      <td>20.34</td>
      <td>17</td>
      <td>90</td>
      <td>188658.67</td>
      <td>107089.08</td>
      <td>12285</td>
      <td>981628</td>
      <td>9.25</td>
      <td>2.60</td>
      <td>...</td>
      <td>0</td>
      <td>4356</td>
      <td>31.91</td>
      <td>14.91</td>
      <td>1</td>
      <td>99</td>
      <td>1.46</td>
      <td>0.50</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Adm-clerical</th>
      <td>36.96</td>
      <td>13.36</td>
      <td>17</td>
      <td>90</td>
      <td>192043.40</td>
      <td>103163.89</td>
      <td>19302</td>
      <td>1033222</td>
      <td>10.11</td>
      <td>1.70</td>
      <td>...</td>
      <td>0</td>
      <td>3770</td>
      <td>37.56</td>
      <td>9.59</td>
      <td>1</td>
      <td>80</td>
      <td>1.67</td>
      <td>0.47</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Armed-Forces</th>
      <td>30.22</td>
      <td>8.09</td>
      <td>23</td>
      <td>46</td>
      <td>215425.89</td>
      <td>83315.89</td>
      <td>76313</td>
      <td>344415</td>
      <td>10.11</td>
      <td>2.03</td>
      <td>...</td>
      <td>0</td>
      <td>1887</td>
      <td>40.67</td>
      <td>14.07</td>
      <td>8</td>
      <td>60</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Craft-repair</th>
      <td>39.03</td>
      <td>11.61</td>
      <td>17</td>
      <td>90</td>
      <td>192132.60</td>
      <td>107434.09</td>
      <td>19491</td>
      <td>1455435</td>
      <td>9.11</td>
      <td>2.04</td>
      <td>...</td>
      <td>0</td>
      <td>3004</td>
      <td>42.30</td>
      <td>9.05</td>
      <td>1</td>
      <td>99</td>
      <td>1.05</td>
      <td>0.23</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Exec-managerial</th>
      <td>42.17</td>
      <td>11.97</td>
      <td>17</td>
      <td>90</td>
      <td>184414.01</td>
      <td>103314.96</td>
      <td>19914</td>
      <td>1484705</td>
      <td>11.45</td>
      <td>2.14</td>
      <td>...</td>
      <td>0</td>
      <td>4356</td>
      <td>44.99</td>
      <td>11.11</td>
      <td>1</td>
      <td>99</td>
      <td>1.29</td>
      <td>0.45</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Farming-fishing</th>
      <td>41.21</td>
      <td>15.07</td>
      <td>17</td>
      <td>90</td>
      <td>170190.18</td>
      <td>116925.90</td>
      <td>20795</td>
      <td>663394</td>
      <td>8.61</td>
      <td>2.76</td>
      <td>...</td>
      <td>0</td>
      <td>2457</td>
      <td>46.99</td>
      <td>17.32</td>
      <td>2</td>
      <td>99</td>
      <td>1.07</td>
      <td>0.25</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Handlers-cleaners</th>
      <td>32.17</td>
      <td>12.37</td>
      <td>17</td>
      <td>90</td>
      <td>204391.01</td>
      <td>111934.10</td>
      <td>19214</td>
      <td>889965</td>
      <td>8.51</td>
      <td>2.20</td>
      <td>...</td>
      <td>0</td>
      <td>2824</td>
      <td>37.95</td>
      <td>10.58</td>
      <td>2</td>
      <td>95</td>
      <td>1.12</td>
      <td>0.32</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Machine-op-inspct</th>
      <td>37.72</td>
      <td>12.07</td>
      <td>17</td>
      <td>90</td>
      <td>195040.88</td>
      <td>98159.93</td>
      <td>13769</td>
      <td>1033222</td>
      <td>8.49</td>
      <td>2.29</td>
      <td>...</td>
      <td>0</td>
      <td>3900</td>
      <td>40.76</td>
      <td>7.59</td>
      <td>1</td>
      <td>96</td>
      <td>1.27</td>
      <td>0.45</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Other-service</th>
      <td>34.95</td>
      <td>14.52</td>
      <td>17</td>
      <td>90</td>
      <td>188608.45</td>
      <td>109452.80</td>
      <td>19752</td>
      <td>1366120</td>
      <td>8.78</td>
      <td>2.30</td>
      <td>...</td>
      <td>0</td>
      <td>3770</td>
      <td>34.70</td>
      <td>12.71</td>
      <td>1</td>
      <td>99</td>
      <td>1.55</td>
      <td>0.50</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Priv-house-serv</th>
      <td>41.72</td>
      <td>18.63</td>
      <td>17</td>
      <td>81</td>
      <td>201107.52</td>
      <td>102595.40</td>
      <td>24384</td>
      <td>549430</td>
      <td>7.36</td>
      <td>3.11</td>
      <td>...</td>
      <td>0</td>
      <td>1602</td>
      <td>32.89</td>
      <td>16.18</td>
      <td>4</td>
      <td>99</td>
      <td>1.95</td>
      <td>0.23</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Prof-specialty</th>
      <td>40.52</td>
      <td>12.02</td>
      <td>17</td>
      <td>90</td>
      <td>185296.61</td>
      <td>100135.45</td>
      <td>14878</td>
      <td>747719</td>
      <td>12.91</td>
      <td>2.03</td>
      <td>...</td>
      <td>0</td>
      <td>3900</td>
      <td>42.39</td>
      <td>12.54</td>
      <td>1</td>
      <td>99</td>
      <td>1.37</td>
      <td>0.48</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Protective-serv</th>
      <td>38.95</td>
      <td>12.82</td>
      <td>17</td>
      <td>90</td>
      <td>202039.95</td>
      <td>101910.29</td>
      <td>19302</td>
      <td>857532</td>
      <td>10.18</td>
      <td>1.87</td>
      <td>...</td>
      <td>0</td>
      <td>2444</td>
      <td>42.87</td>
      <td>12.33</td>
      <td>3</td>
      <td>99</td>
      <td>1.12</td>
      <td>0.32</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Sales</th>
      <td>37.35</td>
      <td>14.19</td>
      <td>17</td>
      <td>90</td>
      <td>190885.89</td>
      <td>103779.87</td>
      <td>19410</td>
      <td>1226583</td>
      <td>10.30</td>
      <td>2.18</td>
      <td>...</td>
      <td>0</td>
      <td>2824</td>
      <td>40.78</td>
      <td>13.24</td>
      <td>2</td>
      <td>99</td>
      <td>1.35</td>
      <td>0.48</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Tech-support</th>
      <td>37.02</td>
      <td>11.32</td>
      <td>17</td>
      <td>73</td>
      <td>192098.30</td>
      <td>113987.28</td>
      <td>19847</td>
      <td>1268339</td>
      <td>10.99</td>
      <td>1.80</td>
      <td>...</td>
      <td>0</td>
      <td>2444</td>
      <td>39.43</td>
      <td>10.58</td>
      <td>3</td>
      <td>99</td>
      <td>1.38</td>
      <td>0.48</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Transport-moving</th>
      <td>40.20</td>
      <td>12.45</td>
      <td>17</td>
      <td>90</td>
      <td>190366.36</td>
      <td>109240.30</td>
      <td>18827</td>
      <td>1184622</td>
      <td>8.77</td>
      <td>2.04</td>
      <td>...</td>
      <td>0</td>
      <td>2824</td>
      <td>44.66</td>
      <td>12.72</td>
      <td>5</td>
      <td>99</td>
      <td>1.06</td>
      <td>0.23</td>
      <td>1</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>15 rows × 28 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-ba502fbc-1d20-4208-8c3d-9d027fb7303a')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-ba502fbc-1d20-4208-8c3d-9d027fb7303a button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-ba502fbc-1d20-4208-8c3d-9d027fb7303a');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




Example: Display age statistics for each race (race feature) and each gender (sex feature). Use groupby() and describe(). Find the maximum age of men of Amer-Indian-Eskimo race


```python
for (race, sex), sub_df in df.groupby(["race", "sex"]):
    print("Race: {0}, sex: {1}".format(race, sex))
    print(sub_df["age"].describe())
```

    Race: Amer-Indian-Eskimo, sex: Female
    count    119.00
    mean      37.12
    std       13.11
    min       17.00
    25%       27.00
    50%       36.00
    75%       46.00
    max       80.00
    Name: age, dtype: float64
    Race: Amer-Indian-Eskimo, sex: Male
    count    192.00
    mean      37.21
    std       12.05
    min       17.00
    25%       28.00
    50%       35.00
    75%       45.00
    max       82.00
    Name: age, dtype: float64
    Race: Asian-Pac-Islander, sex: Female
    count    346.00
    mean      35.09
    std       12.30
    min       17.00
    25%       25.00
    50%       33.00
    75%       43.75
    max       75.00
    Name: age, dtype: float64
    Race: Asian-Pac-Islander, sex: Male
    count    693.00
    mean      39.07
    std       12.88
    min       18.00
    25%       29.00
    50%       37.00
    75%       46.00
    max       90.00
    Name: age, dtype: float64
    Race: Black, sex: Female
    count    1555.00
    mean       37.85
    std        12.64
    min        17.00
    25%        28.00
    50%        37.00
    75%        46.00
    max        90.00
    Name: age, dtype: float64
    Race: Black, sex: Male
    count    1569.00
    mean       37.68
    std        12.88
    min        17.00
    25%        27.00
    50%        36.00
    75%        46.00
    max        90.00
    Name: age, dtype: float64
    Race: Other, sex: Female
    count    109.00
    mean      31.68
    std       11.63
    min       17.00
    25%       23.00
    50%       29.00
    75%       39.00
    max       74.00
    Name: age, dtype: float64
    Race: Other, sex: Male
    count    162.00
    mean      34.65
    std       11.36
    min       17.00
    25%       26.00
    50%       32.00
    75%       42.00
    max       77.00
    Name: age, dtype: float64
    Race: White, sex: Female
    count    8642.00
    mean       36.81
    std        14.33
    min        17.00
    25%        25.00
    50%        35.00
    75%        46.00
    max        90.00
    Name: age, dtype: float64
    Race: White, sex: Male
    count    19174.00
    mean        39.65
    std         13.44
    min         17.00
    25%         29.00
    50%         38.00
    75%         49.00
    max         90.00
    Name: age, dtype: float64


Example:Count the average time of work (hours-per-week) those who earning a little and a lot (salary) for each country (native-country).


```python
for (country, salary), sub_df in df.groupby(["native-country", "salary"]):
    print(country, salary, round(sub_df["hours-per-week"].mean(), 2))
```

    ? <=50K 40.16
    ? >50K 45.55
    Cambodia <=50K 41.42
    Cambodia >50K 40.0
    Canada <=50K 37.91
    Canada >50K 45.64
    China <=50K 37.38
    China >50K 38.9
    Columbia <=50K 38.68
    Columbia >50K 50.0
    Cuba <=50K 37.99
    Cuba >50K 42.44
    Dominican-Republic <=50K 42.34
    Dominican-Republic >50K 47.0
    Ecuador <=50K 38.04
    Ecuador >50K 48.75
    El-Salvador <=50K 36.03
    El-Salvador >50K 45.0
    England <=50K 40.48
    England >50K 44.53
    France <=50K 41.06
    France >50K 50.75
    Germany <=50K 39.14
    Germany >50K 44.98
    Greece <=50K 41.81
    Greece >50K 50.62
    Guatemala <=50K 39.36
    Guatemala >50K 36.67
    Haiti <=50K 36.33
    Haiti >50K 42.75
    Holand-Netherlands <=50K 40.0
    Honduras <=50K 34.33
    Honduras >50K 60.0
    Hong <=50K 39.14
    Hong >50K 45.0
    Hungary <=50K 31.3
    Hungary >50K 50.0
    India <=50K 38.23
    India >50K 46.48
    Iran <=50K 41.44
    Iran >50K 47.5
    Ireland <=50K 40.95
    Ireland >50K 48.0
    Italy <=50K 39.62
    Italy >50K 45.4
    Jamaica <=50K 38.24
    Jamaica >50K 41.1
    Japan <=50K 41.0
    Japan >50K 47.96
    Laos <=50K 40.38
    Laos >50K 40.0
    Mexico <=50K 40.0
    Mexico >50K 46.58
    Nicaragua <=50K 36.09
    Nicaragua >50K 37.5
    Outlying-US(Guam-USVI-etc) <=50K 41.86
    Peru <=50K 35.07
    Peru >50K 40.0
    Philippines <=50K 38.07
    Philippines >50K 43.03
    Poland <=50K 38.17
    Poland >50K 39.0
    Portugal <=50K 41.94
    Portugal >50K 41.5
    Puerto-Rico <=50K 38.47
    Puerto-Rico >50K 39.42
    Scotland <=50K 39.44
    Scotland >50K 46.67
    South <=50K 40.16
    South >50K 51.44
    Taiwan <=50K 33.77
    Taiwan >50K 46.8
    Thailand <=50K 42.87
    Thailand >50K 58.33
    Trinadad&Tobago <=50K 37.06
    Trinadad&Tobago >50K 40.0
    United-States <=50K 38.8
    United-States >50K 45.51
    Vietnam <=50K 37.19
    Vietnam >50K 39.2
    Yugoslavia <=50K 41.6
    Yugoslavia >50K 49.5


Better way to do the same, using crosstab


```python
pd.crosstab(
    df["native-country"],
    df["salary"],
    values=df["hours-per-week"],
    aggfunc=np.mean,
).T
```





  <div id="df-d76df4a7-a141-40a5-8eb1-8bb2b25a6998">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>native-country</th>
      <th>?</th>
      <th>Cambodia</th>
      <th>Canada</th>
      <th>China</th>
      <th>Columbia</th>
      <th>Cuba</th>
      <th>Dominican-Republic</th>
      <th>Ecuador</th>
      <th>El-Salvador</th>
      <th>England</th>
      <th>...</th>
      <th>Portugal</th>
      <th>Puerto-Rico</th>
      <th>Scotland</th>
      <th>South</th>
      <th>Taiwan</th>
      <th>Thailand</th>
      <th>Trinadad&amp;Tobago</th>
      <th>United-States</th>
      <th>Vietnam</th>
      <th>Yugoslavia</th>
    </tr>
    <tr>
      <th>salary</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>&lt;=50K</th>
      <td>40.16</td>
      <td>41.42</td>
      <td>37.91</td>
      <td>37.38</td>
      <td>38.68</td>
      <td>37.99</td>
      <td>42.34</td>
      <td>38.04</td>
      <td>36.03</td>
      <td>40.48</td>
      <td>...</td>
      <td>41.94</td>
      <td>38.47</td>
      <td>39.44</td>
      <td>40.16</td>
      <td>33.77</td>
      <td>42.87</td>
      <td>37.06</td>
      <td>38.80</td>
      <td>37.19</td>
      <td>41.6</td>
    </tr>
    <tr>
      <th>&gt;50K</th>
      <td>45.55</td>
      <td>40.00</td>
      <td>45.64</td>
      <td>38.90</td>
      <td>50.00</td>
      <td>42.44</td>
      <td>47.00</td>
      <td>48.75</td>
      <td>45.00</td>
      <td>44.53</td>
      <td>...</td>
      <td>41.50</td>
      <td>39.42</td>
      <td>46.67</td>
      <td>51.44</td>
      <td>46.80</td>
      <td>58.33</td>
      <td>40.00</td>
      <td>45.51</td>
      <td>39.20</td>
      <td>49.5</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 42 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-d76df4a7-a141-40a5-8eb1-8bb2b25a6998')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-d76df4a7-a141-40a5-8eb1-8bb2b25a6998 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-d76df4a7-a141-40a5-8eb1-8bb2b25a6998');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




## Summary tables

### 1. crosstab()
Panda's `crosstab()` -allows you to compute a cross-tabulation of two or more factors, also known as a *contingency table*. It provides a convenient way to analyze the relationship between multiple categorical variables in a tabular format.

When you have two or more categorical variables in your DataFrame, crosstab helps you create a table that shows the frequency distribution of each variable combination. It counts the occurrences of different categories from each variable and arranges them in a grid-like structure.


```python
# Compute the cross-tabulation
result = pd.crosstab(df['sex'], df['salary'])
print(result)
```

    salary  <=50K  >50K
    sex                
    Female   9592  1179
    Male    15128  6662


 `margins=True` parameter in pd.crosstab() adds row and column totals to the resulting cross-tabulation table.


```python
# Compute the cross-tabulation
result = pd.crosstab(df['sex'], df['salary'], margins=True)
print(result)
```

    salary  <=50K  >50K    All
    sex                       
    Female   9592  1179  10771
    Male    15128  6662  21790
    All     24720  7841  32561


You can also add more then one value to compare against. The resulting table shows the counts for each combination of 'sex', 'race', and 'salary'.


```python
# Compute the cross-tabulation
result = pd.crosstab(df['sex'], [df['race'],df['salary']])
print(result)
```

    race   Amer-Indian-Eskimo      Asian-Pac-Islander      Black      Other       \
    salary              <=50K >50K              <=50K >50K <=50K >50K <=50K >50K   
    sex                                                                            
    Female                107   12                303   43  1465   90   103    6   
    Male                  168   24                460  233  1272  297   143   19   
    
    race    White        
    salary  <=50K  >50K  
    sex                  
    Female   7614  1028  
    Male    13085  6089  


The normalize=True parameter is passed to compute the cross-tabulation as percentages. The resulting table displays the percentage distribution of each salary level for each gender category.



```python
# Compute the cross-tabulation as percentage
result = pd.crosstab(df['sex'], df['salary'], normalize=True, margins=True)*100
print(result)
```

    salary  <=50K   >50K     All
    sex                         
    Female  29.46   3.62   33.08
    Male    46.46  20.46   66.92
    All     75.92  24.08  100.00


From this we can interpret:
*  almost 30% females have a salary less than or equal to 50K, while only around 4% have a salary greater than 50K.
* a higher proportion of males (around 46%) have a salary less than or equal to 50K, while a significant proportion (around 20%) have a salary greater than 50K.



### 2. pivot_table()

* pivot_table() allows you to create a Excel-style pivot table from a DataFrame
* way to summarize and analyze data by grouping and aggregating values based on one or more variables.

Here is how to make pivot tables work:
- `index` parameter - a list of variables to group data by,
- Specify the columns using the `columns` parameter.
- Choose the values to compute summary statistics.
- Select the aggregation function with `aggfunc` - what statistics we need to calculate for groups, e.g. sum, mean, maximum, minimum or something else.
- Customize with additional parameters like margins and fill_value.

Here is how we can calculate the average capital-gain, capital-loss and hours-per-week by sex.


```python
# Create the pivot table
pivot_table = pd.pivot_table(df, index='sex', values=['capital-gain', 'capital-loss', 'hours-per-week'], aggfunc='mean')

print(pivot_table)
```

            capital-gain  capital-loss  hours-per-week
    sex                                               
    Female        568.41         61.19           36.41
    Male         1329.37        100.21           42.43


On average, males have higher values for 'capital-gain', 'capital-loss', and 'hours-per-week' compared to females.

Here is a more complex example. Suppose we want to provide a comprehensive view of the average hours worked per week for various education levels and marital statuses, broken down by different races.

We want to:
* Set the index to ['education', 'marital-status']. Required to group and organize the data based on different education levels and marital statuses. Note, education becomes the primary index while marital status becomes the secondary index.
* Specify the columns of the pivot table to 'race'. Required so we can create separate columns in the pivot table for each unique race value.
* Choose the 'hours-per-week' column as the values to be analyzed using the values parameter.
* Select the 'mean' as the aggregation function using the aggfunc parameter.
* Fill any missing values in the pivot table with 0 using the fill_value parameter.


```python
# Create the pivot table
pivot_table = pd.pivot_table(df, index=['education', 'marital-status'], columns='race', values='hours-per-week', aggfunc='mean', fill_value=0)

print(pivot_table)
```

    race                                Amer-Indian-Eskimo  Asian-Pac-Islander  \
    education    marital-status                                                  
    10th         Divorced                            42.00               40.00   
                 Married-civ-spouse                  35.00               43.00   
                 Married-spouse-absent               40.00               38.50   
                 Never-married                       46.00               35.00   
                 Separated                            0.00                0.00   
    ...                                                ...                 ...   
    Some-college Married-civ-spouse                  45.37               45.41   
                 Married-spouse-absent               40.00               35.75   
                 Never-married                       37.14               32.52   
                 Separated                           35.50               46.67   
                 Widowed                             24.50                5.00   
    
    race                                Black  Other  White  
    education    marital-status                              
    10th         Divorced               37.43  42.00  41.51  
                 Married-civ-spouse     41.07  45.50  41.81  
                 Married-spouse-absent  37.33   0.00  44.33  
                 Never-married          34.33  31.67  30.31  
                 Separated              45.79   0.00  38.37  
    ...                                   ...    ...    ...  
    Some-college Married-civ-spouse     42.28  40.00  43.53  
                 Married-spouse-absent  36.05  29.00  37.88  
                 Never-married          35.82  33.50  33.74  
                 Separated              38.44  46.00  40.74  
                 Widowed                38.90   0.00  32.34  
    
    [101 rows x 5 columns]


## Data Transformations tables

### 1. Adding Columns

The `df.insert()` function is to insert a new column into a DataFrame at a specific position. It allows you to specify the index location where you want to insert the new column and provide the column name and its corresponding values.

Here's the syntax for `df.insert()`:

```python
df.insert(loc, column, value, allow_duplicates=False)
```

- `loc`: The index location where you want to insert the new column.
- `column`: The name of the new column.
- `value`: The values to be assigned to the new column.
- `allow_duplicates` (optional): If set to `True`, allows duplicate column names; default is `False`.

For example, we can create a new age group column (pandas series) based on pd.cut() and insert at postion 2.



```python
# Create the age_group serie/column based on age values
age_group = pd.cut(df['age'], bins=[0, 18, 30, 50, 100], labels=['<18', '18-30', '31-50', '51+'])

# Insert the age_group column at position 2
df.insert(2, 'age_group', age_group)
df.head()
```





  <div id="df-8b4c81a0-76d4-4c25-b163-a94303e3abb1">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>workclass</th>
      <th>age_group</th>
      <th>fnlwgt</th>
      <th>education</th>
      <th>education-num</th>
      <th>marital-status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>sex</th>
      <th>capital-gain</th>
      <th>capital-loss</th>
      <th>hours-per-week</th>
      <th>native-country</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>39</td>
      <td>State-gov</td>
      <td>31-50</td>
      <td>77516</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Never-married</td>
      <td>Adm-clerical</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>2174</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>1</th>
      <td>50</td>
      <td>Self-emp-not-inc</td>
      <td>31-50</td>
      <td>83311</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>13</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>2</th>
      <td>38</td>
      <td>Private</td>
      <td>31-50</td>
      <td>215646</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Divorced</td>
      <td>Handlers-cleaners</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>3</th>
      <td>53</td>
      <td>Private</td>
      <td>51+</td>
      <td>234721</td>
      <td>11th</td>
      <td>7</td>
      <td>Married-civ-spouse</td>
      <td>Handlers-cleaners</td>
      <td>Husband</td>
      <td>Black</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>4</th>
      <td>28</td>
      <td>Private</td>
      <td>18-30</td>
      <td>338409</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Prof-specialty</td>
      <td>Wife</td>
      <td>Black</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>Cuba</td>
      <td>&lt;=50K</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-8b4c81a0-76d4-4c25-b163-a94303e3abb1')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-8b4c81a0-76d4-4c25-b163-a94303e3abb1 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-8b4c81a0-76d4-4c25-b163-a94303e3abb1');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




If we dont need inserting at the a specific position, we simply add a new column by specifying it. The new column gets added at the end.


```python
df['education-num-group'] = pd.cut(df['education-num'], bins=[0, 5, 10, 15, 20], labels=['<5', '5-10', '15', '15+'])
df.head()
```





  <div id="df-2579a75b-daa7-4cc4-a765-805601dc554a">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>workclass</th>
      <th>age_group</th>
      <th>fnlwgt</th>
      <th>education</th>
      <th>education-num</th>
      <th>marital-status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>sex</th>
      <th>capital-gain</th>
      <th>capital-loss</th>
      <th>hours-per-week</th>
      <th>native-country</th>
      <th>salary</th>
      <th>education-num-group</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>39</td>
      <td>State-gov</td>
      <td>31-50</td>
      <td>77516</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Never-married</td>
      <td>Adm-clerical</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>2174</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
      <td>15</td>
    </tr>
    <tr>
      <th>1</th>
      <td>50</td>
      <td>Self-emp-not-inc</td>
      <td>31-50</td>
      <td>83311</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>13</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
      <td>15</td>
    </tr>
    <tr>
      <th>2</th>
      <td>38</td>
      <td>Private</td>
      <td>31-50</td>
      <td>215646</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Divorced</td>
      <td>Handlers-cleaners</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
      <td>5-10</td>
    </tr>
    <tr>
      <th>3</th>
      <td>53</td>
      <td>Private</td>
      <td>51+</td>
      <td>234721</td>
      <td>11th</td>
      <td>7</td>
      <td>Married-civ-spouse</td>
      <td>Handlers-cleaners</td>
      <td>Husband</td>
      <td>Black</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
      <td>5-10</td>
    </tr>
    <tr>
      <th>4</th>
      <td>28</td>
      <td>Private</td>
      <td>18-30</td>
      <td>338409</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Prof-specialty</td>
      <td>Wife</td>
      <td>Black</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>Cuba</td>
      <td>&lt;=50K</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-2579a75b-daa7-4cc4-a765-805601dc554a')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-2579a75b-daa7-4cc4-a765-805601dc554a button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-2579a75b-daa7-4cc4-a765-805601dc554a');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




### 2. Deleting Columns or Rows

* To delete columns or rows, use the drop method, passing the required indexes and the axis parameter (1 if you delete columns, and nothing or 0 if you delete rows).
* The inplace argument tells whether to change the original DataFrame.
* With inplace=False, the drop method doesn’t change the existing DataFrame and returns a new one with dropped rows or columns.
* With inplace=True, it alters the DataFrame.


```python
# get rid of just created columns
df.drop(["education-num-group", "age_group"], axis=1, inplace=True)
df.head()
```





  <div id="df-a1a6ab54-aa4b-4799-8c77-8d7f0528dc06">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>workclass</th>
      <th>fnlwgt</th>
      <th>education</th>
      <th>education-num</th>
      <th>marital-status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>sex</th>
      <th>capital-gain</th>
      <th>capital-loss</th>
      <th>hours-per-week</th>
      <th>native-country</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>39</td>
      <td>State-gov</td>
      <td>77516</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Never-married</td>
      <td>Adm-clerical</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>2174</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>1</th>
      <td>50</td>
      <td>Self-emp-not-inc</td>
      <td>83311</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>13</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>2</th>
      <td>38</td>
      <td>Private</td>
      <td>215646</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Divorced</td>
      <td>Handlers-cleaners</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>3</th>
      <td>53</td>
      <td>Private</td>
      <td>234721</td>
      <td>11th</td>
      <td>7</td>
      <td>Married-civ-spouse</td>
      <td>Handlers-cleaners</td>
      <td>Husband</td>
      <td>Black</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>4</th>
      <td>28</td>
      <td>Private</td>
      <td>338409</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Prof-specialty</td>
      <td>Wife</td>
      <td>Black</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>Cuba</td>
      <td>&lt;=50K</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-a1a6ab54-aa4b-4799-8c77-8d7f0528dc06')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-a1a6ab54-aa4b-4799-8c77-8d7f0528dc06 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-a1a6ab54-aa4b-4799-8c77-8d7f0528dc06');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
# and here’s how you can delete rows
print('Before change:{}'.format(df.shape))
df.drop([1, 2], inplace=True)
print('After change:{}'.format(df.shape))
```

    Before change:(32561, 15)
    After change:(32559, 15)


## References
* [Pandas Cheat Sheet](https://github.com/pandas-dev/pandas/blob/main/doc/cheatsheet/Pandas_Cheat_Sheet.pdf)
* [Kaggle Pandas Course](https://www.kaggle.com/learn/pandas)
* [Pandas User Guide](https://pandas.pydata.org/docs/user_guide/index.html)
