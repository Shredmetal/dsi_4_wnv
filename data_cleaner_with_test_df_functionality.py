import pandas as pd
import numpy as np
from scipy import stats
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split


class DataCleaner:
    """
    Cleans the data provided in GA DSIF Project 4.

    This class takes in a dataframe and a chosen feature list created from the DSIF Project 4 train.csv file and
    creates an object with the attributes df, features, X, y, X_train, y_train, X_test, y_test.

    These attributes are initialised as None, and are filled by executing the clean() method on the object. The clean
    method contains a series of functions which start by removing outliers and populating the X and y attributes
    of the object based on the list of features passed in.

    Each function thereafter is programmed to handle exceptions where the column has been removed due to it not being
    in X as a result of the features passed in.

    Please refer to the comments relating to each function below for a further explanation of what each function does.
    """

    def __init__(self, dataframe, features, test_df=None):
        self.df = dataframe
        self.test_df = test_df
        self.features = features
        self.categorical = ["month", "Species", "ResultDir", "gps_cat", "sprayed", "Sunrise", "Sunset"]
        self.cat_features = []
        self.ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.encoded_features = []
        self.encoded_test_features = []
        self.kaggle_test = None
        self.y = None
        self.X = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.clean_executed = False
        self.num_mosquito_ss = None
        self.tmax_ss = None
        self.tmin_ss = None
        self.tavg_ss = None
        self.dewpoint_ss = None
        self.wetbulb_ss = None
        self.heat_ss = None
        self.cool_ss = None
        self.preciptotal_ss = None
        self.windspeed_ss = None
        self.pressure_ss = None
        self.sealevel_ss = None

    # Please refer to the class DocString for a fuller explanation of the clean method.

    def clean(self):
        self.df['PrecipTotal'] = self.df['PrecipTotal'].fillna(0)
        self.clean_categorical()
        self.df.dropna(axis=0, how="any", inplace=True)
        self.X = self.df[self.encoded_features]
        self.y = self.df["WnvPresent"]
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
        self.clean_executed = True

    def clean_test(self):
        if not self.clean_executed:
            print("clean_test() failed. Please run the clean() method on the train dataframe before running c"
                  "lean_test() on the test dataframe.")
        elif self.test_df is None:
            print("No test dataframe was passed when the DataCleaner object was instantiated. please instantiate "
                  "a new DataCleaner object with the appropriate arguments passed.")
        else:
            self.clean_test_categorical()
            self.test_df = self.test_df[self.encoded_test_features]
            self.clean_test_num_mosquitoes()
            self.clean_test_tmax()
            self.clean_test_tmin()
            self.clean_test_tavg()
            self.clean_test_dewpoint()
            self.clean_test_wetbulb()
            self.clean_test_heat()
            self.clean_test_cool()
            self.clean_test_preciptotal()
            self.clean_test_windspeed()
            self.clean_test_pressure()
            self.clean_test_sealevel()

    def clean_categorical(self):
        for feature in self.features:
            if feature in self.categorical:
                self.cat_features.append(feature)
            else:
                pass
        encoded_features = self.ohe.fit_transform(self.df[self.cat_features])
        encoded_columns = self.ohe.get_feature_names_out(self.cat_features)
        encoded_df = pd.DataFrame(encoded_features, columns=encoded_columns)
        df_no_cat = self.df.drop(self.cat_features, axis=1)
        self.df = pd.concat([df_no_cat, encoded_df], axis=1)
        for feature in encoded_columns:
            self.encoded_features.append(feature)
        for feature in self.features:
            if feature not in self.categorical:
                self.encoded_features.append(feature)
            else:
                pass

    def clean_num_mosquitoes(self):
        try:
            self.num_mosquito_ss = StandardScaler()
            self.num_mosquito_ss.fit(self.X_train[["NumMosquitos"]])
            self.X_train["NumMosquitos"] = self.num_mosquito_ss.transform(self.X_train[["NumMosquitos"]])
            self.X_test["NumMosquitos"] = self.num_mosquito_ss.transform(self.X_test[["NumMosquitos"]])
        except KeyError:
            pass

    def clean_tmax(self):
        try:
            self.tmax_ss = StandardScaler()
            self.tmax_ss.fit(self.X_train[["Tmax"]])
            self.X_train["Tmax"] = self.tmax_ss.transform(self.X_train[["Tmax"]])
            self.X_test["Tmax"] = self.tmax_ss.transform(self.X_test[["Tmax"]])
        except KeyError:
            pass

    def clean_tmin(self):
        try:
            self.tmin_ss = StandardScaler()
            self.tmin_ss.fit(self.X_train[["Tmin"]])
            self.X_train["Tmin"] = self.tmin_ss.transform(self.X_train[["Tmin"]])
            self.X_test["Tmin"] = self.tmin_ss.transform(self.X_test[["Tmin"]])
        except KeyError:
            pass

    def clean_tavg(self):
        try:
            self.tavg_ss = StandardScaler()
            self.tavg_ss.fit(self.X_train[["Tavg"]])
            self.X_train["Tavg"] = self.tavg_ss.transform(self.X_train[["Tavg"]])
            self.X_test["Tavg"] = self.tavg_ss.transform(self.X_test[["Tavg"]])
        except KeyError:
            pass

    def clean_dewpoint(self):
        try:
            self.dewpoint_ss = StandardScaler()
            self.dewpoint_ss.fit(self.X_train[["DewPoint"]])
            self.X_train["DewPoint"] = self.dewpoint_ss.transform(self.X_train[["DewPoint"]])
            self.X_test["DewPoint"] = self.dewpoint_ss.transform(self.X_test[["DewPoint"]])
        except KeyError:
            pass

    def clean_wetbulb(self):
        try:
            self.wetbulb_ss = StandardScaler()
            self.wetbulb_ss.fit(self.X_train[["WetBulb"]])
            self.X_train["WetBulb"] = self.wetbulb_ss.transform(self.X_train[["WetBulb"]])
            self.X_test["WetBulb"] = self.wetbulb_ss.transform(self.X_test[["WetBulb"]])
        except KeyError:
            pass

    def clean_heat(self):
        try:
            self.heat_ss = StandardScaler()
            self.heat_ss.fit(self.X_train[["Heat"]])
            self.X_train["Heat"] = self.heat_ss.transform(self.X_train[["Heat"]])
            self.X_test["Heat"] = self.heat_ss.transform(self.X_test[["Heat"]])
        except KeyError:
            pass

    def clean_cool(self):
        try:
            self.cool_ss = StandardScaler()
            self.cool_ss.fit(self.X_train[["Cool"]])
            self.X_train["Cool"] = self.cool_ss.transform(self.X_train[["Cool"]])
            self.X_test["Cool"] = self.cool_ss.transform(self.X_test[["Cool"]])
        except KeyError:
            pass

    def clean_preciptotal(self):
        try:
            self.preciptotal_ss = StandardScaler()
            self.preciptotal_ss.fit(self.X_train[["PrecipTotal"]])
            self.X_train["PrecipTotal"] = self.preciptotal_ss.transform(self.X_train[["PrecipTotal"]])
            self.X_test["PrecipTotal"] = self.preciptotal_ss.transform(self.X_test[["PrecipTotal"]])
        except KeyError:
            pass

    def clean_windspeed(self):
        try:
            self.windspeed_ss = StandardScaler()
            self.windspeed_ss.fit(self.X_train[["ResultSpeed"]])
            self.X_train["ResultSpeed"] = self.windspeed_ss.transform(self.X_train[["ResultSpeed"]])
            self.X_test["ResultSpeed"] = self.windspeed_ss.transform(self.X_test[["ResultSpeed"]])
        except KeyError:
            pass

    def clean_pressure(self):
        try:
            self.pressure_ss = StandardScaler()
            self.pressure_ss.fit(self.X_train[["StnPressure"]])
            self.X_train["StnPressure"] = self.pressure_ss.transform(self.X_train[["StnPressure"]])
            self.X_test["StnPressure"] = self.pressure_ss.transform(self.X_test[["StnPressure"]])
        except KeyError:
            pass

    def clean_sealevel(self):
        try:
            self.sealevel_ss = StandardScaler()
            self.sealevel_ss.fit(self.X_train[["SeaLevel"]])
            self.X_train["SeaLevel"] = self.sealevel_ss.transform(self.X_train[["SeaLevel"]])
            self.X_test["SeaLevel"] = self.sealevel_ss.transform(self.X_test[["SeaLevel"]])
        except KeyError:
            pass

    def clean_test_categorical(self):
        encoded_features = self.ohe.transform(self.test_df[self.cat_features])
        encoded_columns = self.ohe.get_feature_names_out(self.cat_features)
        encoded_df = pd.DataFrame(encoded_features, columns=encoded_columns)
        df_no_cat = self.test_df.drop(self.cat_features, axis=1)
        self.test_df = pd.concat([df_no_cat, encoded_df], axis=1)
        for feature in encoded_columns:
            self.encoded_test_features.append(feature)
        for feature in self.features:
            if feature not in self.categorical:
                self.encoded_test_features.append(feature)
            else:
                pass

    def clean_test_num_mosquitoes(self):
        if "NumMosquitos" in self.features:
            try:
                self.test_df["NumMosquitos"] = self.num_mosquito_ss.transform(self.test_df[["NumMosquitos"]])
            except KeyError:
                pass
        else:
            pass

    def clean_test_tmax(self):
        if "Tmax" in self.features:
            try:
                self.test_df["Tmax"] = self.tmax_ss.transform(self.test_df[["Tmax"]])
            except KeyError:
                pass
        else:
            pass

    def clean_test_tmin(self):
        if "Tmin" in self.features:
            try:
                self.test_df["Tmin"] = self.tmin_ss.transform(self.test_df[["Tmin"]])
            except KeyError:
                pass
        else:
            pass

    def clean_test_tavg(self):
        if "Tavg" in self.features:
            try:
                self.test_df["Tavg"] = self.tavg_ss.transform(self.test_df[["Tavg"]])
            except KeyError:
                pass
        else:
            pass

    def clean_test_dewpoint(self):
        if "DewPoint" in self.features:
            try:
                self.test_df["DewPoint"] = self.dewpoint_ss.transform(self.test_df[["DewPoint"]])
            except KeyError:
                pass
        else:
            pass

    def clean_test_wetbulb(self):
        if "WetBulb" in self.features:
            try:
                self.test_df["WetBulb"] = self.wetbulb_ss.transform(self.test_df[["WetBulb"]])
            except KeyError:
                pass
        else:
            pass

    def clean_test_heat(self):
        if "Heat" in self.features:
            try:
                self.test_df["Heat"] = self.heat_ss.transform(self.test_df[["Heat"]])
            except KeyError:
                pass
        else:
            pass

    def clean_test_cool(self):
        if "Cool" in self.features:
            try:
                self.test_df["Cool"] = self.cool_ss.transform(self.test_df[["Cool"]])
            except KeyError:
                pass
        else:
            pass

    def clean_test_preciptotal(self):
        if "PrecipTotal" in self.features:
            try:
                self.test_df["PrecipTotal"] = self.preciptotal_ss.transform(self.test_df[["PrecipTotal"]])
            except KeyError:
                pass
        else:
            pass

    def clean_test_windspeed(self):
        if "ResultSpeed" in self.features:
            try:
                self.test_df["ResultSpeed"] = self.windspeed_ss.transform(self.test_df[["ResultSpeed"]])
            except KeyError:
                pass
        else:
            pass

    def clean_test_pressure(self):
        if "StnPressure" in self.features:
            try:
                self.test_df["StnPressure"] = self.pressure_ss.transform(self.test_df[["StnPressure"]])
            except KeyError:
                pass
        else:
            pass

    def clean_test_sealevel(self):
        if "SeaLevel" in self.features:
            try:
                self.test_df["SeaLevel"] = self.sealevel_ss.transform(self.test_df[["SeaLevel"]])
            except KeyError:
                pass
        else:
            pass