import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--corpus', type=str, nargs=2, help='The two corpus files')
parser.add_argument('--suffix', type=str, help='The suffix appended to the new file')
args = parser.parse_args()

print('Reading file ' + args.corpus[0] + ' ...')
with open(args.corpus[0], 'r') as f:
    c1 = f.read().splitlines()
print('Read %d lines.' % len(c1))

print('Reading file ' + args.corpus[1] + ' ...')
with open(args.corpus[1], 'r') as f:
    c2 = f.read().splitlines()
print('Read %d lines.' % len(c2))

if not len(c1) == len(c2):
    raise ValueError('The number of lines of the two files are not equal.')

print('Shuffling...')
n = len(c1)
ind = np.arange(0, n)
np.random.shuffle(ind)

print('Writing to %s.%s ...' % (args.corpus[0], args.suffix))
with open(args.corpus[0] + '.' + args.suffix, 'w') as f:
    for i in ind:
        f.write(c1[i] + '\n')

print('Writing to %s.%s ...' % (args.corpus[1], args.suffix))
with open(args.corpus[1] + '.' + args.suffix, 'w') as f:
    for i in ind:
        f.write(c2[i] + '\n')
