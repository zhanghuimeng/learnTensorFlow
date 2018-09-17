import tensorflow as tf
import pandas as pd

california_housing_dataframe = pd.read_csv('./california_housing_train.csv', sep = ',')  # using ',' as sep
# Then we get some 9 Series of data:
# "longitude","latitude","housing_median_age","total_rooms","total_bedrooms",
# "population","households","median_income","median_house_value"

# print(str(california_housing_dataframe))
# The data size is 17000 rows * 9 columns

# Randomize the data so SGD can work well
# Also, tweak the median_house_value