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

### Data Dictionary

# Modelling Conclusions

### Model Performance and Time Taken

# Recommendations