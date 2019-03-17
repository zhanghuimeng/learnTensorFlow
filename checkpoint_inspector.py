from tensorflow.python import pywrap_tensorflow
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d', type=str, help='checkpoint path')
args = parser.parse_args()

reader = pywrap_tensorflow.NewCheckpointReader(args.d)
var_to_shape_map = reader.get_variable_to_shape_map()
for key in var_to_shape_map:
    print("tensor_name: ", key)
