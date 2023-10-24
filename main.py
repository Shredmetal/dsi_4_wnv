from collections import OrderedDict
import pandas as pd
import misc_func
from data_cleaner_with_test_df_functionality import DataCleaner
from log_reg_brain import LogRegBrain
import json
import time

start_time = time.time()

# List of features to consider. Due to time constraints, this was not completely automated. The program was run to
# eliminate features one at a time. Further development can be done to fully automate this process. Removing one at
# a time was chosen as the preferred method due to the huge number of potential combinations which would have resulted
# from simply taking every possible combination.

features = ['Species', 'Tmax', 'Tmin', 'Tavg', 'DewPoint', 'WetBulb', 'Heat', 'Cool', 'PrecipTotal',
              'ResultSpeed', 'ResultDir', 'StnPressure', 'SeaLevel', 'gps_cat', 'sprayed', 'month']

# features_1 = ['Species', 'NumMosquitos', 'Tmax', 'Tmin', 'Tavg', 'DewPoint', 'WetBulb', 'Heat', 'Cool', 'PrecipTotal',
#               'ResultSpeed', 'ResultDir', 'StnPressure', 'SeaLevel', 'sprayed', 'month']
#
# features_2 = ['Species', 'NumMosquitos', 'Tmax', 'Tmin', 'Tavg', 'DewPoint', 'WetBulb', 'Cool', 'PrecipTotal',
#               'ResultSpeed', 'ResultDir', 'StnPressure', 'SeaLevel', 'sprayed', 'month']
#
# features_3 = ['Species', 'NumMosquitos', 'Tmax', 'Tmin', 'Tavg', 'DewPoint', 'WetBulb', 'Cool', 'PrecipTotal',
#               'ResultSpeed', 'ResultDir', 'StnPressure', 'SeaLevel', 'month']
#
# features_4 = ['Species', 'NumMosquitos', 'Tmax', 'Tmin', 'Tavg', 'WetBulb', 'Cool', 'PrecipTotal', 'ResultSpeed',
#               'ResultDir', 'StnPressure', 'SeaLevel', 'month']
#
# features_5 = ['Species', 'NumMosquitos', 'Tmax', 'Tmin', 'Tavg', 'Cool', 'PrecipTotal', 'ResultSpeed', 'ResultDir',
#               'StnPressure', 'SeaLevel', 'month']
#
# features_6 = ['Species', 'NumMosquitos', 'Tmax', 'Tmin', 'Tavg', 'PrecipTotal', 'ResultSpeed', 'ResultDir',
#               'StnPressure', 'SeaLevel', 'month']
#
# features_7 = ['NumMosquitos', 'Tmax', 'Tmin', 'Tavg', 'PrecipTotal', 'ResultSpeed', 'ResultDir', 'StnPressure',
#               'SeaLevel', 'month']
#
# features_8 = ['NumMosquitos', 'Tmin', 'Tavg', 'PrecipTotal', 'ResultSpeed', 'ResultDir', 'StnPressure', 'SeaLevel',
#               'month']
#
# features_9 = ['NumMosquitos', 'Tmin', 'Tavg', 'PrecipTotal', 'ResultSpeed', 'ResultDir', 'StnPressure', 'month']
#
# features_10 = ['NumMosquitos', 'Tmin', 'Tavg', 'PrecipTotal', 'ResultSpeed', 'ResultDir', 'month']
#
# features_11 = ['NumMosquitos', 'Tmin', 'Tavg', 'PrecipTotal', 'ResultDir', 'month']
#
# features_12 = ['NumMosquitos', 'Tavg', 'PrecipTotal', 'ResultDir', 'month']
#
# features_13 = ['NumMosquitos', 'PrecipTotal', 'ResultDir', 'month']
#
# features_14 = ['NumMosquitos', 'ResultDir', 'month']
#
# features_15 = ['NumMosquitos', 'month']

# features = ['NumMosquitos', 'Sunrise', 'Sunset', 'gps_cat']

start_feat_len = len(features)

# Turn off certain pd warnings which will clog up our terminal.

pd.options.mode.chained_assignment = None

# Create a series of variables which will be updates / referred to when iterating over all the possible combinations
# of our features.

best_roc = 0.5
best_features = ""
counter = 0
score_dict = {}

# Get the combinations of features of no. features minus one.

combinations = misc_func.combos(features)

# Read in the train.csv and get rid of nan values.

df = pd.read_csv("train_sprayed_month_engineered.csv", low_memory=False)

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

    if brain.test_roc > best_roc:
        best_roc = brain.test_roc
        score_dict[counter] = {
            "best_test_roc": brain.test_roc,
            "best_train_roc": brain.train_roc,
            "best_features": features_to_use,
            "feature_len": len(features_to_use)
        }
        print(f"Test ROC: {brain.test_roc}")
        print(f"Train ROC: {brain.train_roc}")
        print(f"Features: {features_to_use}")
        print(f"Feat Len: {len(features_to_use)}")
    else:
        pass

    # Add counter for the purposes of tracking progress, counter gets printed every iteration.

    counter += 1

# Organise the dictionary of high scores, showing the best test first.

results = OrderedDict(sorted(score_dict.items(), key=lambda x: x[1]['best_test_roc']))

# Save the dictionary into a JSON to access it easily later.

with open("scores.json", "w") as file:
    json.dump(results, file)

# Print the amount of time the program ran for.

print(f"Run time: {time.time() - start_time}")