import tensorflow as tf
import pandas as pd
import numpy as np
from IPython import display
from matplotlib import pyplot as plt
from sklearn import metrics
import math


# Too much output...
tf.logging.set_verbosity(tf.logging.ERROR)


def preprocess_features(california_housing_dataframe):
    """
    Prepares input features from California housing data set.
    :param california_housing_dataframe: A Pandas Dataframe containing data from the California housing data set.
    :return: A Dataframe containing the features to be used by the model, including synthetic features.
    """
    # Note the double brackets
    selected_features = california_housing_dataframe[[
        "latitude",
        "longitude",
        "housing_median_age",
        "total_rooms",
        "total_bedrooms",
        "population",
        "households",
        "median_income"
    ]]
    # Need to make a deep copy, or the original Dataframe will be changed as well
    processed_features = selected_features.copy()
    # Create a syntheic feature
    processed_features["rooms_per_person"] = california_housing_dataframe["total_rooms"] / \
                                             california_housing_dataframe["population"]
    return processed_features


def preprocess_targets(california_housing_dataframe):
    """
    Prepares target features (i.e. labels) from California housing data set.
    :param california_housing_dataframe: A Pandas Dataframe containing data from the California housing data set.
    :return: A Dataframe containing the target feature.
    """
    output_targets = pd.DataFrame()
    # Scale the targets to be in units of thousands of dollars
    output_targets["median_house_value"] = california_housing_dataframe["median_house_value"] / 1000.0
    return output_targets


def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """
    Trans a linear regression model of multiple features.
    :param features: Pandas Dataframe of features
    :param targets: Pandas Dataframe of targets
    :param batch_size: Size of batches to be passed to the Model
    :param shuffle: True or False. Whether to shuffle the data.
    :param num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely.
    :return: Tuple of (features, labels) for next data batch
    """
    # Convert pandas data into a dict of np arrays.
    # Convert each row into an array; then features = {index: features_array}
    features = {key:np.array(value) for key, value in dict(features).items()}
    # Construct a tf dataset, and configure batching/repeating.
    ds = tf.data.Dataset.from_tensor_slices((features, targets))
    ds = ds.batch(batch_size).repeat(num_epochs)
    # Shuffle the data if specified
    # The buffer size for RandomShuffleQueue is 10000
    if shuffle:
        ds = ds.shuffle(10000)
    # Returns the next batch of data
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


def construct_feature_columns(input_features):
    """
    Construct the TensorFlow Feature Columns
    :param input_features: The names of the numerical input features to use
    (Actually just pass in the whole columns...)
    :return: A set of feature columns
    """
    return set(tf.feature_column.numeric_column(my_feature) for my_feature in input_features)


def train_model(learning_rate, steps, batch_size, training_samples, training_targets, validation_samples,
                validation_targets):
    """
    Trains a linear regression model of multiple features. In addition to training, this function also prints training
    progress information, as well as a plot of the training and validation loss over time.
    :param learning_rate:
    :param steps:
    :param batch_size:
    :param training_samples:
    :param training_targets:
    :param validation_samples:
    :param validation_targets:
    :return: A `LinearRegressor` object trained on the training data.
    """

    periods = 10
    steps_per_period = steps / periods

    # Create a LinearRegressor object
    # Still not so familier with the idea of Estimators and optimizers...?
    my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    linear_regressor = tf.estimator.LinearRegressor(feature_columns=construct_feature_columns(training_samples),
                                                    optimizer=my_optimizer)

    # Create input functions
    # So we use the batch size here...
    # And for prediction, do not repeat or shuffle
    training_input_fn = lambda: my_input_fn(training_samples, training_targets, batch_size=batch_size)
    predict_training_input_fn = lambda: my_input_fn(training_samples, training_targets, num_epochs=1,
                                                    shuffle=False)
    predict_validation_input_fn = lambda: my_input_fn(validation_samples, validation_targets, num_epochs=1,
                                                      shuffle=False)

    # Train the model
    print("Training model...")
    training_rmse = []
    validation_rmse = []
    for period in range(0, periods):
        # Train the model
        linear_regressor.train(input_fn=training_input_fn, steps=steps_per_period)
        # Make predictions
        training_predictions = linear_regressor.predict(input_fn=predict_training_input_fn)
        training_predictions = np.array([item["predictions"][0] for item in training_predictions])
        validation_predictions = linear_regressor.predict(input_fn=predict_validation_input_fn)
        validation_predictions = np.array([item["predictions"][0] for item in validation_predictions])
        # Calculate RMSE
        training_root_mean_squared_error = math.sqrt(metrics.mean_squared_error(
            training_targets, training_predictions))
        validation_root_mean_squared_error = math.sqrt(metrics.mean_squared_error(
            validation_targets, validation_predictions))
        training_rmse.append(training_root_mean_squared_error)
        validation_rmse.append(validation_root_mean_squared_error)
        print("  period %02d: Training RMSE=%0.2f, Validation RMSE=%0.2f" %
              (period, training_root_mean_squared_error, validation_root_mean_squared_error))

    print("Model training has finished.")
    # Output a graph of loss metrics over periods
    plt.figure(2)
    plt.ylabel("RMSE")
    plt.xlabel("periods")
    plt.title("RMSE vs. Periods")
    plt.tight_layout()
    plt.plot(training_rmse, label="training")
    plt.plot(validation_rmse, label="validation")
    plt.legend()
    plt.show()

    return linear_regressor

# Read in the data and randomize
california_housing_dataframe = pd.read_csv("../data/california_housing_train.csv")
california_housing_dataframe = california_housing_dataframe.reindex(
    np.random.permutation(california_housing_dataframe.index))
# Divide 12000 traning samples and 5000 validation samples
training_samples = preprocess_features(california_housing_dataframe.head(12000))
# display.display(training_samples.describe())
training_targets = preprocess_targets(california_housing_dataframe.head(12000))
# display.display(training_targets.describe())
validation_samples = preprocess_features(california_housing_dataframe.tail(5000))
# display.display(validation_samples.describe())
validation_targets = preprocess_targets(california_housing_dataframe.tail(5000))
# display.display(validation_targets.describe())

# Display latituge/longitude pictures
# Constrain the scales
plt.figure(1)
ax = plt.subplot(1, 2, 1)
ax.set_title("Validation Data")
ax.set_autoscaley_on(False)
ax.set_ylim([32, 43])
ax.set_autoscalex_on(False)
ax.set_xlim([-126, -112])
plt.scatter(validation_samples["longitude"], validation_samples["latitude"], cmap="coolwarm",
            c=validation_targets["median_house_value"] / validation_targets["median_house_value"].max())

ax = plt.subplot(1, 2, 2)
ax.set_title("Training Data")
ax.set_autoscaley_on(False)
ax.set_ylim([32, 43])
ax.set_autoscalex_on(False)
ax.set_xlim([-126, -112])
plt.scatter(training_samples["longitude"], training_samples["latitude"], cmap="coolwarm",
            c=training_targets["median_house_value"] / training_targets["median_house_value"].max())

plt.show()

linear_regressor = train_model(learning_rate=0.00001, steps=100, batch_size=1,
                               training_samples=training_samples, training_targets=training_targets,
                               validation_samples=validation_samples, validation_targets=validation_targets)