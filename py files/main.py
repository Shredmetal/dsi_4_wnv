from collections import OrderedDict
import pandas as pd
import misc_func
from data_cleaner import DataCleaner
from log_reg_brain import LogRegBrain
import json
import time

start_time = time.time()

# List of features to consider.

# features_1 = ['Species', 'Latitude', 'Longitude', 'NumMosquitos', 'Tmax', 'Tmin', 'Tavg', 'DewPoint', 'WetBulb',
#                   'Heat', 'Cool', 'Sunrise', 'Sunset', 'PrecipTotal', 'ResultSpeed', 'ResultDir', 'StnPressure',
#                   'SeaLevel', 'gps_cat', 'sprayed']
#
# features_2 = ['Species', 'Latitude', 'Longitude', 'NumMosquitos', 'Tmax', 'Tmin', 'Tavg', 'DewPoint', 'WetBulb', 'Heat',
#               'Cool', 'Sunrise', 'Sunset', 'PrecipTotal', 'ResultSpeed', 'StnPressure', 'SeaLevel', 'gps_cat',
#               'sprayed']
#
# features_3 = ['Species', 'Latitude', 'Longitude', 'NumMosquitos', 'Tmax', 'Tmin', 'Tavg', 'DewPoint', 'WetBulb', 'Heat',
#             'Cool', 'Sunrise', 'Sunset', 'ResultSpeed', 'StnPressure', 'SeaLevel', 'gps_cat', 'sprayed']
#
# features_4 = ['Species', 'Latitude', 'NumMosquitos', 'Tmax', 'Tmin', 'Tavg', 'DewPoint', 'WetBulb', 'Heat', 'Cool',
#             'Sunrise', 'Sunset', 'ResultSpeed', 'StnPressure', 'SeaLevel', 'gps_cat', 'sprayed']
#
# features_5 = ['Latitude', 'NumMosquitos', 'Tmax', 'Tmin', 'Tavg', 'DewPoint', 'WetBulb', 'Heat', 'Cool', 'Sunrise',
#             'Sunset', 'ResultSpeed', 'StnPressure', 'SeaLevel', 'gps_cat', 'sprayed']
#
# features_6 = ['Latitude', 'NumMosquitos', 'Tmax', 'Tmin', 'DewPoint', 'WetBulb', 'Heat', 'Cool', 'Sunrise', 'Sunset',
#             'ResultSpeed', 'StnPressure', 'SeaLevel', 'gps_cat', 'sprayed']
#
# features_7 = ['Latitude', 'NumMosquitos', 'Tmax', 'Tmin', 'DewPoint', 'WetBulb', 'Heat', 'Cool', 'Sunrise', 'Sunset',
#             'ResultSpeed', 'StnPressure', 'gps_cat', 'sprayed']
#
# features_8 = ['NumMosquitos', 'Tmax', 'Tmin', 'DewPoint', 'WetBulb', 'Heat', 'Cool', 'Sunrise', 'Sunset', 'ResultSpeed',
#             'StnPressure', 'gps_cat', 'sprayed']
#
# features_9 = ['NumMosquitos', 'Tmax', 'Tmin', 'DewPoint', 'Heat', 'Cool', 'Sunrise', 'Sunset', 'ResultSpeed',
#             'StnPressure', 'gps_cat', 'sprayed']
#
# features_10 = ['NumMosquitos', 'Tmax', 'Tmin', 'DewPoint', 'Heat', 'Cool', 'Sunrise', 'Sunset', 'StnPressure',
#                'gps_cat', 'sprayed']
#
# features_11 = ['NumMosquitos', 'Tmax', 'Tmin', 'DewPoint', 'Heat', 'Sunrise', 'Sunset', 'StnPressure', 'gps_cat',
#             'sprayed']
#
# features_12 = ['NumMosquitos', 'Tmax', 'Tmin', 'Heat', 'Sunrise', 'Sunset', 'StnPressure', 'gps_cat', 'sprayed']
#
# features_13 = ['NumMosquitos', 'Tmin', 'Heat', 'Sunrise', 'Sunset', 'StnPressure', 'gps_cat', 'sprayed']
#
# features_14 = ['NumMosquitos', 'Tmin', 'Heat', 'Sunrise', 'Sunset', 'gps_cat', 'sprayed']
#
# features_15 = ['NumMosquitos', 'Tmin', 'Sunrise', 'Sunset', 'gps_cat', 'sprayed']
#
# features_16 = ['NumMosquitos', 'Tmin', 'Sunrise', 'Sunset', 'gps_cat']

features = ['NumMosquitos', 'Sunrise', 'Sunset', 'gps_cat']

start_feat_len = len(features)

# Turn off certain pd warnings which will clog up our terminal.

pd.options.mode.chained_assignment = None

# Create a series of variables which will be updates / referred to when iterating over all the possible combinations
# of our features.

best_score = 0.2
best_features = ""
counter = 0
score_dict = {}

# Get the combinations of features of no. features minus one.

combinations = misc_func.combos(features)

# Read in the train.csv and get rid of nan values.

df = pd.read_csv("train_added_cols_1000m.csv", low_memory=False)

# Main loop focusing on the RMSE of each linear model created from each combination of features returned by
# combinations.

for combination in combinations:

    # Convert each combination tuple into a list.

    features_to_use = list(combination)

    # Instantiate DataCleaner object.

    cleaned = DataCleaner(df, features_to_use)

    # Clean the DataCleaner object.

    cleaned.clean()

    # Print statement to show progress.

    print(f"{counter}, Feat len: {len(features_to_use)}")

    # Instantiate LinRegBrain object to obtain RMSE and various metrics.

    brain = LogRegBrain(cleaned.X_train, cleaned.y_train, cleaned.X_test, cleaned.y_test)

    # If test_score is highest seen so far, save the information into a dictionary.

    if brain.test_score > best_score:
        best_score = brain.test_score
        score_dict[counter] = {"best_test": brain.test_score,
                               "best_train": brain.train_score,
                               "best_features": features_to_use,
                               "feature_len": len(features_to_use)}
        print(f"Best test: {brain.test_score}")
        print(f"Train: {brain.train_score}")
        print(f"Features: {features_to_use}")
        print(f"Feat Len: {len(features_to_use)}")
    else:
        pass

    # Add counter for the purposes of tracking progress, counter gets printed every iteration.

    counter += 1

# Organise the dictionary of high scores, showing the best test first.

results = OrderedDict(sorted(score_dict.items(), key=lambda x: x[1]['best_test']))

# Save the dictionary into a JSON to access it easily later.

with open("scores.json", "w") as file:
    json.dump(results, file)

# Print the amount of time the program ran for.

print(f"Run time: {time.time() - start_time}")