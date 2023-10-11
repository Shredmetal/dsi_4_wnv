import pandas as pd
import numpy as np
from scipy import stats
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class DataCleaner:
    """
    Cleans the data in train.csv provided in GA DSIF Project 4.

    This class takes in a dataframe and a chosen feature list created from the DSIF Project 4 train.csv file and
    creates an object with the attributes df, features, X, y, X_train, y_train, X_test, y_test.

    These attributes are initialised as None, and are filled by executing the clean() method on the object. The clean
    method contains a series of functions which start by removing outliers and populating the X and y attributes
    of the object based on the list of features passed in.

    Each function thereafter is programmed to handle exceptions where the column has been removed due to it not being
    in X as a result of the features passed in.

    Please refer to the comments relating to each function below for a further explanation of what each function does.
    """

    def __init__(self, dataframe, features):
        self.df = dataframe
        self.features = features
        self.ohe = None
        self.y = None
        self.X = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    # Please refer to the class DocString for a fuller explanation of the clean method.
    # The inline comments refer to the expected output of each function.
    # bool - column contains only Y and N, Y is converted to 1 and N is converted to 0.
    # dummies - column contains categorical variable. Variable is dummified and first column is dropped.
    # scaler - column contains continuous variable. Variable is scaled by standard scaler.
    # Where scaler is applied, it must be fit on train data first before transforming both train and test data.
    # Therefore, train-test-split is applied prior to the execution of the scaler functions.

    def clean(self):
        self.df['PrecipTotal'] = self.df['PrecipTotal'].fillna(0)
        self.df.dropna(axis=0, how="any", inplace=True)
        class_0_df = self.df[self.df["WnvPresent"] == 0]
        no_class_0 = class_0_df["WnvPresent"].value_counts().tolist()[0]
        class_1_df = self.df[self.df["WnvPresent"] == 1]
        class_1_df = class_1_df.sample(n=no_class_0, replace=True, random_state=42)
        self.df = pd.concat([class_0_df, class_1_df], axis=0)
        self.X = self.df[self.features]
        self.y = self.df["WnvPresent"]
        self.clean_species()
        self.clean_wind_dir()
        self.clean_gps_cat()
        self.clean_sprayed()
        self.clean_sunrise()
        self.clean_sunset()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,
                                                                                self.y,
                                                                                test_size=0.2,
                                                                                random_state=42)
        self.clean_num_mosquitoes()
        self.clean_tmax()
        self.clean_tmin()
        self.clean_tavg()
        self.clean_dewpoint()
        self.clean_wetbulb()
        self.clean_heat()
        self.clean_cool()
        self.clean_preciptotal()
        self.clean_windspeed()
        self.clean_pressure()
        self.clean_sealevel()

    def clean_species(self):
        try:
            self.X = pd.get_dummies(self.X, columns=["Species"], dtype=int, prefix="species", drop_first=True)
        except KeyError:
            pass

    def clean_wind_dir(self):
        try:
            self.X = pd.get_dummies(self.X, columns=["ResultDir"], dtype=int, prefix="wind_dir", drop_first=True)
        except KeyError:
            pass

    def clean_gps_cat(self):
        try:
            self.X = pd.get_dummies(self.X, columns=["gps_cat"], dtype=int, prefix="gps_cat", drop_first=True)
        except KeyError:
            pass

    def clean_sprayed(self):
        try:
            self.X = pd.get_dummies(self.X, columns=["sprayed"], dtype=int, prefix="sprayed", drop_first=True)
        except KeyError:
            pass

    def clean_sunrise(self):
        try:
            self.X = pd.get_dummies(self.X, columns=["Sunrise"], dtype=int, prefix="sunrise", drop_first=True)
        except KeyError:
            pass

    def clean_sunset(self):
        try:
            self.X = pd.get_dummies(self.X, columns=["Sunset"], dtype=int, prefix="Sunset", drop_first=True)
        except KeyError:
            pass

    def clean_num_mosquitoes(self):
        try:
            sc = StandardScaler()
            sc.fit(self.X_train[["NumMosquitos"]])
            self.X_train["NumMosquitos"] = sc.transform(self.X_train[["NumMosquitos"]])
            self.X_test["NumMosquitos"] = sc.transform(self.X_test[["NumMosquitos"]])
        except KeyError:
            pass

    def clean_tmax(self):
        try:
            sc = StandardScaler()
            sc.fit(self.X_train[["Tmax"]])
            self.X_train["Tmax"] = sc.transform(self.X_train[["Tmax"]])
            self.X_test["Tmax"] = sc.transform(self.X_test[["Tmax"]])
        except KeyError:
            pass

    def clean_tmin(self):
        try:
            sc = StandardScaler()
            sc.fit(self.X_train[["Tmin"]])
            self.X_train["Tmin"] = sc.transform(self.X_train[["Tmin"]])
            self.X_test["Tmin"] = sc.transform(self.X_test[["Tmin"]])
        except KeyError:
            pass

    def clean_tavg(self):
        try:
            sc = StandardScaler()
            sc.fit(self.X_train[["Tavg"]])
            self.X_train["Tavg"] = sc.transform(self.X_train[["Tavg"]])
            self.X_test["Tavg"] = sc.transform(self.X_test[["Tavg"]])
        except KeyError:
            pass

    def clean_dewpoint(self):
        try:
            sc = StandardScaler()
            sc.fit(self.X_train[["DewPoint"]])
            self.X_train["DewPoint"] = sc.transform(self.X_train[["DewPoint"]])
            self.X_test["DewPoint"] = sc.transform(self.X_test[["DewPoint"]])
        except KeyError:
            pass

    def clean_wetbulb(self):
        try:
            sc = StandardScaler()
            sc.fit(self.X_train[["WetBulb"]])
            self.X_train["WetBulb"] = sc.transform(self.X_train[["WetBulb"]])
            self.X_test["WetBulb"] = sc.transform(self.X_test[["WetBulb"]])
        except KeyError:
            pass

    def clean_heat(self):
        try:
            sc = StandardScaler()
            sc.fit(self.X_train[["Heat"]])
            self.X_train["Heat"] = sc.transform(self.X_train[["Heat"]])
            self.X_test["Heat"] = sc.transform(self.X_test[["Heat"]])
        except KeyError:
            pass

    def clean_cool(self):
        try:
            sc = StandardScaler()
            sc.fit(self.X_train[["Cool"]])
            self.X_train["Cool"] = sc.transform(self.X_train[["Cool"]])
            self.X_test["Cool"] = sc.transform(self.X_test[["Cool"]])
        except KeyError:
            pass

    def clean_preciptotal(self):
        try:
            sc = StandardScaler()
            sc.fit(self.X_train[["PrecipTotal"]])
            self.X_train["PrecipTotal"] = sc.transform(self.X_train[["PrecipTotal"]])
            self.X_test["PrecipTotal"] = sc.transform(self.X_test[["PrecipTotal"]])
        except KeyError:
            pass

    def clean_windspeed(self):
        try:
            sc = StandardScaler()
            sc.fit(self.X_train[["ResultSpeed"]])
            self.X_train["ResultSpeed"] = sc.transform(self.X_train[["ResultSpeed"]])
            self.X_test["ResultSpeed"] = sc.transform(self.X_test[["ResultSpeed"]])
        except KeyError:
            pass

    def clean_pressure(self):
        try:
            sc = StandardScaler()
            sc.fit(self.X_train[["StnPressure"]])
            self.X_train["StnPressure"] = sc.transform(self.X_train[["StnPressure"]])
            self.X_test["StnPressure"] = sc.transform(self.X_test[["StnPressure"]])
        except KeyError:
            pass

    def clean_sealevel(self):
        try:
            sc = StandardScaler()
            sc.fit(self.X_train[["SeaLevel"]])
            self.X_train["SeaLevel"] = sc.transform(self.X_train[["SeaLevel"]])
            self.X_test["SeaLevel"] = sc.transform(self.X_test[["SeaLevel"]])
        except KeyError:
            pass
