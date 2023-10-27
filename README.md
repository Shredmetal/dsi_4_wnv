# Problem Statement

## PROBLEM:

The west nile virus is a mosquito-borne disease plaguing the Chicago area. It results in multiple costs including but not limited to: 

1. Medical treatment required;

2. Lost productivity as people miss work due to being ill; and

3. Pain and suffering of people afflicted by it.

There may be a way to reduce the costs inflicted by the virus by targeting mosquito populations with pesticides or otherwise. However, we should target mosquito populations when west nile virus is most prevalent.

## OBJECTIVES:

1. Build a model which predicts when west nile virus is present in mosquito populations.

2. If the model predicts the presence of west nile virus, action can be taken to reduce or exterminate mosquito populations.

## SCOPE:

1. Use the data to feed various machine learning models to achieve the objectives.

2. Select the model with the best ROC-AUC score.

## DATA:

1. Mosquito trap data containing whether or not west nile virus was found, along with coordinates.

2. Data as to where and when pesticide spraying took place.

3. Weather data.

## METHODS AND TOOLS:

1. Logistic Regression.

2. Support Vector Classifier.

3. Random Forest.

4. Neural Network.

## SUCCESS METRICS:

1. ROC-AUC Score.

# Data Dictionary

| Field                     | Description                                                                                                        | Data Type    |
|---------------------------|--------------------------------------------------------------------------------------------------------------------|--------------|
| Station                   | Weather station index (1 or 2).                                                                                    | int64        |
| Date                      | Date of the weather record.                                                                                        | object       |
| Tmax                      | Maximum temperature for the day.                                                                                   | int64        |
| Tmin                      | Minimum temperature for the day.                                                                                   | int64        |
| Tavg                      | Average temperature.                                                                                               | object       |
| Depart                    | Temperature departure from the normal.                                                                             | object       |
| DewPoint                  | Dew point temperature.                                                                                             | int64        |
| WetBulb                   | Wet bulb temperature.                                                                                              | object       |
| Heat                      | Heating degree days.                                                                                               | object       |
| Cool                      | Cooling degree days.                                                                                               | object       |
| CodeSum                   | Weather phenomena code summary.                                                                                    | object       |
| Depth                     | Snow depth.                                                                                                        | object       |
| Water1                    | Water equivalent measure.                                                                                          | object       |
| SnowFall                  | Snowfall measure.                                                                                                  | object       |
| PrecipTotal               | Total precipitation.                                                                                               | object       |
| StnPressure               | Station pressure.                                                                                                  | object       |
| SeaLevel                  | Sea-level pressure.                                                                                                | object       |
| ResultSpeed               | Wind speed.                                                                                                        | float64      |
| ResultDir                 | Wind direction.                                                                                                    | int64        |
| AvgSpeed                  | Average wind speed.                                                                                                | object       |
| Id                        | The ID of the record.                                                                                              | Not Available|
| Date                      | Date that the WNV test is performed.                                                                               | object       |
| Address                   | Approximate address of the location of the trap (used for GeoCoder).                                               | object       |
| Species                   | The species of mosquitoes.                                                                                         | object       |
| Block                     | Block number of address.                                                                                           | int64        |
| Street                    | Street name.                                                                                                       | object       |
| Trap                      | ID of the trap.                                                                                                    | object       |
| AddressNumberAndStreet    | Approximate address returned from GeoCoder.                                                                        | object       |
| Latitude                  | Latitude of mosquito trap.                                                                                         | float64      |
| Longitude                 | Longitude of mosquito trap.                                                                                        | float64      |
| Latitude (Weather)        | Latitude returned from GeoCoder for weather data.                                                                  | float64      |
| Longitude (Weather)       | Longitude returned from GeoCoder for weather data.                                                                 | float64      |
| AddressAccuracy           | Accuracy returned from GeoCoder.                                                                                   | int64        |
| NumMosquitos              | Number of mosquitoes caught in this trap.                                                                          | int64        |
| WnvPresent                | Whether West Nile Virus was present in these mosquitoes. 1 means WNV is present, and 0 means not present.          | int64        |
| Date (Spray)              | The date of the spray.                                                                                             | object       |
| Time                      | The time of the spray.                                                                                             | object       |
| Latitude (Spray)          | The Latitude of the spray.                                                                                         | float64      |
| Longitude (Spray)         | The Longitude of the spray.                                                                                        | float64      |

# Requirements


### Technical Report (Data Wrangling) Notebook requirements:

imbalanced-learn==0.11.0

imblearn==0.0

keras==2.14.0

keras-core==0.1.7

numpy==1.25.2

pandas==2.1.0

python-dateutil==2.8.2

pytz==2023.3.post1

scikit-learn==1.3.0

### Technical Report (EDA) Notebook requirements:

branca              0.6.0

folium              0.14.0

matplotlib          3.7.1

mpl_toolkits        NA

numpy               1.24.3

pandas              1.5.3

scipy               1.10.1

seaborn             0.12.2

session_info        1.0.0

sklearn             1.3.0

### Technical Report (Modelling) Notebook requirements:

anyio               NA
asttokens           NA
attr                23.1.0
babel               2.11.0
backcall            0.2.0
brotli              NA
certifi             2023.07.22
charset_normalizer  2.0.4
colorama            0.4.6
comm                0.1.2
cython_runtime      NA
dateutil            2.8.2
debugpy             1.6.7
decorator           5.1.1
entrypoints         0.4
executing           0.8.3
fastjsonschema      NA
idna                3.4
ipykernel           6.25.0
ipython_genutils    0.2.0
jedi                0.18.1
jinja2              3.1.2
json5               NA
jsonschema          4.17.3
jupyter_server      1.23.4
jupyterlab_server   2.22.0
markupsafe          2.1.1
nbformat            5.9.2
numpy               1.26.1
packaging           23.1
parso               0.8.3
pickleshare         0.7.5
pkg_resources       NA
platformdirs        3.10.0
prometheus_client   NA
prompt_toolkit      3.0.36
psutil              5.9.0
pure_eval           0.2.2
pvectorc            NA
pydev_ipython       NA
pydevconsole        NA
pydevd              2.9.5
pydevd_file_utils   NA
pydevd_plugins      NA
pydevd_tracing      NA
pygments            2.15.1
pyrsistent          NA
pythoncom           NA
pywin32_system32    NA
pywintypes          NA
requests            2.31.0
rfc3339_validator   0.1.4
rfc3986_validator   0.1.1
send2trash          NA
six                 1.16.0
sniffio             1.2.0
socks               1.7.1
stack_data          0.2.0
terminado           0.17.1
tornado             6.3.3
traitlets           5.7.1
urllib3             1.26.16
wcwidth             0.2.5
websocket           0.58.0
win32api            NA
win32com            NA
win32con            NA
win32trace          NA
winerror            NA
winpty              2.0.10
zmq                 23.2.0

### .py files requirements:

imbalanced-learn==0.11.0

imblearn==0.0

joblib==1.3.2

numpy==1.26.0

pandas==2.1.1

python-dateutil==2.8.2

pytz==2023.3.post1

scikit-learn==1.3.1

scipy==1.11.3

six==1.16.0

threadpoolctl==3.2.0

tzdata==2023.3

### Model Performance

| Model for NumMoSquitos     | WnV Model                 | AUC (Train Data Set) | Kaggle Score | Best Params WNV Model                                                                                                                         |
|----------------------------|---------------------------|----------------------|--------------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| Neural Net Regression      | Neural Net Classifier     | 0.87                 | 0.54         | N.A                                                                                                                                           |
| Neural Net Regression      | RandomForestClassifier    | 0.86                 | 0.56         | rc__max_depth': None, 'rc__max_samples': None, 'rc__min_samples_leaf': 5, 'rc__min_samples_split': 10, 'rc__n_estimators': 100                |
| Neural Net Regression      | Support Vector Classifier | 0.86                 | 0.51         | 'svc__C': 100, 'svc__gamma': 'scale', 'svc__kernel': 'rbf'                                                                                    |
| Neural Net Regression      | Logistic Regression       | 0.87                 | 0.54         | 'lr__C': 100, 'lr__l1_ratio': 0.75, 'lr__penalty': 'l2'                                                                                       |
| Neural Net Regression      | XGB Classifier            | 0.87                 | 0.57         | 'xgc__booster': 'gbtree', 'xgc__gamma': 0, 'xgc__learning_rate': 0.1, 'xgc__n_estimators': 500, 'xgc__reg_alpha': 1, 'xgc__reg_lambda': 1     |
| RandomForestRegression     | Neural Net Classifier     | 0.87                 | 0.53         | N.A                                                                                                                                           |
| RandomForestRegression     | RandomForestClassifier    | 0.87                 | 0.56         | 'rc__max_depth': None, 'rc__max_samples': None, 'rc__min_samples_leaf': 5, 'rc__min_samples_split': 10, 'rc__n_estimators': 100               |
| RandomForestRegression     | Support Vector Classifier | 0.86                 | 0.53         | 'svc__C': 100, 'svc__gamma': 'scale', 'svc__kernel': 'rbf'                                                                                    |
| RandomForestRegression     | Logistic Regression       | 0.87                 | 0.55         | 'lr__C': 100, 'lr__l1_ratio': 0.25, 'lr__penalty': 'l2'                                                                                       |
| **RandomForestRegression** | **XGB Classifier**        | **0.87**             | **0.59**     | **'xgc__booster': 'gbtree', 'xgc__gamma': 0, 'xgc__learning_rate': 0.1, 'xgc__n_estimators': 500, 'xgc__reg_alpha': 1, 'xgc__reg_lambda': 1** |
| XGB Regression             | Neural Net Classifier     | 0.87                 | 0.53         | N.A                                                                                                                                           |
| XGB Regression             | RandomForestClassifier    | 0.87                 | 0.56         | 'rc__max_depth': None, 'rc__max_samples': None, 'rc__min_samples_leaf': 5, 'rc__min_samples_split': 10, 'rc__n_estimators': 300               |
| XGB Regression             | Support Vector Classifier | 0.86                 | 0.52         | 'svc__C': 100, 'svc__gamma': 'scale', 'svc__kernel': 'rbf'                                                                                    |
| XGB Regression             | Logistic Regression       | 0.87                 | 0.54         | 'lr__C': 100, 'lr__l1_ratio': 0.25, 'lr__penalty': 'elasticnet'                                                                               |
| XGB Regression             | XGB Classifier            | 0.87                 | 0.52         | 'xgc__booster': 'gbtree', 'xgc__gamma': 0, 'xgc__learning_rate': 0.1, 'xgc__n_estimators': 500, 'xgc__reg_alpha': 1, 'xgc__reg_lambda': 1     |

# Recommendations

### Conclusion
    
Based on historical data (year 2007, 2009, 2011 and 2013) , the presence of WNV virus is based on these factors:

* Numbers of mosquitoes.
* Temperature and humidity is higher than normal days.
* Lower average wind speed.
* During the months from May to August.
* Negligence of spraying at some of the areas with WNV presence.
* Mosquitoes species belonging to Culex Pipiens and Culex Restuans.


The top locations with highest WNV presence of at least 14 instances are:

* ORD Terminal 5, O'Hare International Airport, Chicago, IL 60666, USA
* South Doty Avenue, Chicago, IL, USA 
* 4100 North Oak Park Avenue, Chicago, IL 60634, USA
* South Stony Island Avenue, Chicago, IL, USA
* 4600 Milwaukee Avenue, Chicago, IL 60630, USA

The top traps located that has high mosquitoes count and WNV presence are T900 and T115.

Thunderstorms and mist might contribute to no. of WNV cases.

Limitations:
* The sprayed data is not consistent.
* Unavailability of manpower cost and data for cost analysis.

Based on the reports/articles found online, the total material cost for Zenivex is **293,415.87 USD**.

The total cost for treating WNV in Chicago is estimated at **517,502.67 USD**. Also, people who have been infected with West Nile Virus (WNV) may experience a reduction in their income, resulting in a cumulative income loss of **166,355.09 USD** in the city.
The total estimated loss in Chicago is **683,857.76 USD**.


However, there are other factors that may further contribute to the city's economy negatively such as lower workplace productivity, reduced tourism and increase of moving to other cities due to the presence of WNV.

Even though there are other factors such as manpower cost is not included in the cost analysis due to the unavailbilty of data, the officials should not underestimate the long term negative effect for not spraying insecticide. Hence, the efforts for spraying the city should not be stopped.


### Recomendation

* To use data driven approach to reduce WNV by implementing preventive measures (refer to the first two points below).

* The best period for spraying insecticide is during the month of July to August as these two months are the peak of the mosquitoes numbers.

* Ensure thorough spraying, as there are regions with the presence of West Nile Virus that have not been treated with spray and we do not assume that the pressence of WNV is not caused by mosquitoes.

* The officials should also be cautious that the wrong concentration amount of insecticides may cause the mosquitoes to grow resistance ([*source*](https://www.vox.com/health/23814358/west-nile-virus-symptoms-mosquito-repellant-disease-bite-insecticide-resistance)).

* Residents to wear loose-fitting clothes that can cover arms and legs during the month of July and August([*source*](https://www.cdc.gov/westnile/prevention/index.html)).

* Educate the residents to prevent mosquitoes breeding, such as draining water out from anything that can collect rainwater.

* Enforce the laws and regulations for mosquitoes breeding.
