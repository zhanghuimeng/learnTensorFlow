import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--corpus', type=str, nargs=2, help='The two corpus files')
parser.add_argument('--suffix', type=str, help='The suffix appended to the new file')
args = parser.parse_args()