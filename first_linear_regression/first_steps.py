import tensorflow as tf
from tensorflow.data import Dataset
from sklearn import metrics
import pandas as pd
from matplotlib import pyplot as plt
# matplotlib is visualization for Numpy
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


california_housing_dataframe = pd.read_csv('./california_housing_train.csv', sep=',')
# print(str(california_housing_dataframe))
# Then we get some 9 Series of data:
# ["longitude","latitude","housing_median_age","total_rooms","total_bedrooms",
# "population","households","median_income","median_house_value"]

# For the sake of learn rate (though I don't know why now)
california_housing_dataframe['median_house_value'] /= 1000.0

# The data size is 17000 rows * 9 columns
# Try to draw a plot for (total_roomes, median_house_value) to get some understanding of the data
sample = california_housing_dataframe.sample(300)
plt.figure(1)
plt.ylabel("median_house_value / 1000.0")
plt.xlabel("total_rooms")
plt.scatter(sample["total_rooms"], sample["median_house_value"])
plt.show()

# Also use Dataset.describe() to understand the data
# print(california_housing_dataframe.describe())

# Randomize the data so SGD can work well
california_housing_dataframe = california_housing_dataframe.reindex(
    np.random.permutation(california_housing_dataframe.index))

# Start to use LinearRegressor Model

# 1. Define the feature
# Because it is a feature, it should have 1*n dimensions
my_feature = california_housing_dataframe[["total_rooms"]]
# A numerical feature column
feature_columns = [tf.feature_column.numeric_column("total_rooms")]

# 2. Define the label
targets = california_housing_dataframe["median_house_value"]

# 3. Configure LinearRegressor
# For SGD
my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0000001)
# For Gradient Clipping
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
# Interesting that we only configured feature_columns and optimizer here
linear_regressor = tf.estimator.LinearRegressor(feature_columns=feature_columns, optimizer=my_optimizer)

# 4. Define input method
# Construct a data iterator for LinearRegressor
# def my_input_fn
# print(str(dict(my_feature).items()))

# 5. Train the model
# First train for 100 steps
# input_fn is a lambda (very interesting)
# What's that _ here?
_ = linear_regressor.train(input_fn=lambda:my_input_fn(my_feature, targets), steps=100)

# 6. Evaluate the model
# Only testing on train data currently.
# Another lambda for input_fn, no repeat or shuffle this time
prediction_input_fn = lambda: my_input_fn(my_feature, targets, num_epochs=1, shuffle=False)
# Call predict() on the linear_regressor to make predictions
predictions = linear_regressor.predict(input_fn=prediction_input_fn)
# for item in predictions:
#    print(str(item['predictions']))
# The predictions looks like a list of dicts of lists
# Format predictions as a NumPy array to calculate error metrics
predictions = np.array([item['predictions'][0] for item in predictions])
# Print Mean Squared Error and Root Mean Squared Error
mean_squared_error = metrics.mean_squared_error(predictions, targets)
root_mean_error = math.sqrt(mean_squared_error)
print("Mean Squared Error (on training data): %0.3f" % mean_squared_error)
print("Root Mean Squared Error (on training data): %0.3f" % root_mean_error)
# Interpret the Mean Squared Error: using min & max
min_house_value = california_housing_dataframe["median_house_value"].min()
max_house_value = california_housing_dataframe["median_house_value"].max()
min_max_difference = max_house_value - min_house_value
print("Min of Median House Value: %0.3f" % min_house_value)
print("Max of Median House Value: %0.3f" % max_house_value)
print("Difference between Min and Max: %0.3f" % min_max_difference)

# Draw the model on plt
plt.figure(2)
x_0 = sample["total_rooms"].min()
x_1 = sample["total_rooms"].max()
# Retrieve the final weight and bias of LinearRegressor model
# Currently I don't know how to use the name correctly...?
weight = linear_regressor.get_variable_value("linear/linear_model/total_rooms/weights")[0]
bias = linear_regressor.get_variable_value("linear/linear_model/bias_weights")
y_0 = weight * x_0 + bias
y_1 = weight * x_1 + bias
plt.plot([x_0, x_1], [y_0, y_1], c='r')
plt.ylabel("median_house_value / 1000.0")
plt.xlabel("total_rooms")
plt.scatter(sample["total_rooms"], sample["median_house_value"])
plt.show()