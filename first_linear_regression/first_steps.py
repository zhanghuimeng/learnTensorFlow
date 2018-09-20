import tensorflow as tf
from tensorflow.data import Dataset
from sklearn import metrics  # for calculating MSE (compatiable with numpy)
import pandas as pd
from matplotlib import pyplot as plt  # matplotlib is visualization for Numpy
from matplotlib import cm  # for colors
from IPython import display  # for displaying describe data
import numpy as np
import math


# The input function
def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """
    Trains a linear regression model of one feature
    :param features: pandas Dataframe of features
    :param targets: pandas Dataframe of targets
    :param batch_size: size of batches to be passed to the model
    :param shuffle: True or False; whether to shuffle the data
    :param num_epochs: Number of epochs for which data should be repeated. None=repeat indefinitely
    :return: tuple of (features, labels) for next data batch
    """

    # Convert pandas data into a dict of np arrays
    # {column: n*2 matrix (index and data)}
    features = {key: np.array(value) for key, value in dict(features).items()}

    # Construct a dataset, and configure batching/repeating.
    # The idea of Dataset is quite interesting...
    ds = Dataset.from_tensor_slices((features, targets))
    ds = ds.batch(batch_size).repeat(num_epochs)

    # Shuffle the data if needed
    if shuffle:
        ds = ds.shuffle(buffer_size=10000)

    # Return the next batch of data
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


def train_model(learning_rate, steps, batch_size, input_feature="total_rooms", input_label="median_house_value"):
    """
    Trains a linear regression model of one feature
    :param input_label: A string specifying a column from 'california_housing_dataframe'
    :param learning_rate: A float, the learning rate
    :param steps: A non-zero int, the total number of training steps.
    :param batch_size: A non-zero int, the batch size
    :param input_feature: A string specifying a column from 'california_housing_dataframe'
    """

    # Let's suppose that steps can be divided by 10
    periods = 10
    steps_per_period = steps / periods

    # Start to use LinearRegressor Model

    # 1. Define the feature
    # Because it is a feature, it should have 1*n dimensions
    my_feature = input_feature
    my_feature_data = california_housing_dataframe[[my_feature]]
    # A numerical feature column
    feature_columns = [tf.feature_column.numeric_column(my_feature)]

    # 2. Define the label
    my_label = input_label
    targets = california_housing_dataframe[my_label]

    # 3. Configure LinearRegressor
    # For SGD
    my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    # For Gradient Clipping
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    # Interesting that we only configured feature_columns and optimizer here
    linear_regressor = tf.estimator.LinearRegressor(feature_columns=feature_columns, optimizer=my_optimizer)

    # 4. Define input method
    # Construct a data iterator for LinearRegressor
    training_input_fn = lambda: my_input_fn(my_feature_data, targets, batch_size=batch_size)
    predicting_input_fn = lambda: my_input_fn(my_feature_data, targets, num_epochs=1, shuffle=False)

    # Plot the state of model's line in each period
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    plt.title("Learned Line by Period")
    plt.ylabel(my_label)
    plt.xlabel(my_feature)
    sample = california_housing_dataframe.sample(n=300)
    plt.scatter(sample[my_feature], sample[my_label])
    # seems like a color setting...?
    # This method seems deprecated
    # colors = [cm.coolwarm(x) for x in np.linspace(-1, 1, periods)]
    cm = plt.get_cmap('coolwarm')

    # 5. Train the model and periodically draw the lines
    print("Training model...")
    print("RMSE (on training data):")
    root_mean_squared_errors = []
    predictions = []
    root_mean_squared_error = 0.0
    for period in range(0, periods):
        # Train
        linear_regressor.train(input_fn=training_input_fn, steps=steps_per_period)
        # Predict and evaluate
        predictions = linear_regressor.predict(input_fn=predicting_input_fn)
        predictions = np.array([item['predictions'][0] for item in predictions])
        # Compute loss
        root_mean_squared_error = math.sqrt(metrics.mean_squared_error(targets, predictions))
        print("  period %02d : %0.2f" % (period, root_mean_squared_error))
        root_mean_squared_errors.append(root_mean_squared_error)

        # Ensure the data and line can be plotted
        y_extents = np.array([0, sample[my_label].max()])

        weight = linear_regressor.get_variable_value("linear/linear_model/%s/weights" % input_feature)[0]
        bias = linear_regressor.get_variable_value("linear/linear_model/bias_weights")
        # ??
        x_extents = (y_extents - bias) / weight
        x_extents = np.maximum(np.minimum(x_extents, sample[my_feature].max()), sample[my_feature].min())
        y_extents = weight * x_extents + bias
        # print(str(x_extents))
        # print(str(y_extents))
        lines = plt.plot(x_extents, y_extents, label=str(period))
        lines[0].set_color(cm(period / periods))

    # Draw RMSEs with periods
    print("Model training finished.")
    plt.subplot(1, 2, 2)
    plt.ylabel("RMSE")
    plt.xlabel("Periods")
    plt.title("Root Mean Squared Error vs. Periods")
    plt.tight_layout()
    plt.plot(root_mean_squared_errors)

    # Output calibration data
    print("\nStatics for prediction and target:")
    calibration_data = pd.DataFrame()
    calibration_data["predictions"] = pd.Series(predictions)
    calibration_data["targets"] = pd.Series(targets)
    display.display(calibration_data.describe())

    print("Final RMSE (on training data): %0.2f" % root_mean_squared_error)

    plt.show()


california_housing_dataframe = pd.read_csv('./california_housing_train.csv', sep=',')
# print(str(california_housing_dataframe))
# Then we get some 9 Series of data:
# ["longitude","latitude","housing_median_age","total_rooms","total_bedrooms",
# "population","households","median_income","median_house_value"]

# For the sake of learn rate (though I don't know why now)
california_housing_dataframe['median_house_value'] /= 1000.0

# run the training model
train_model(learning_rate=0.00001, steps=100, batch_size=1)