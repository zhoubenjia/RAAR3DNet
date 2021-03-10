from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal1 normal_concat1 normal2 normal_concat2 normal3 normal_concat3')

PRIMITIVES_INCEPTION = [
    'none',
    'skip_connect',
    'conv_1x1x1',
    'conv_3x3x3',
    'dil_3x3x3',
    'conv_1x3x3',
    'conv_3x1x1',
]
