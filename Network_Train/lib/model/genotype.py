from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal1 normal_concat1 normal2 normal_concat2 normal3 normal_concat3')

'''
Searched Information:
    data: 2020.10 
    Dataset:IsoGD, 
    Total Epoch: 80, 
    Current Epoch: 62
    Valid Acc: 52.62%
'''
genotype = Genotype(
    normal1=[('conv_1x3x3', 1), ('skip_connect', 0), ('conv_1x3x3', 0), ('conv_3x3x3', 2), ('dil_3x3x3', 0),
             ('conv_3x1x1', 2), ('conv_3x3x3', 4), ('conv_1x3x3', 2)], normal_concat1=range(2, 6),
    normal2=[('skip_connect', 1), ('conv_1x1x1', 0), ('conv_3x1x1', 1), ('conv_1x1x1', 2), ('skip_connect', 1),
             ('skip_connect', 2), ('conv_3x1x1', 1), ('conv_3x3x3', 2)], normal_concat2=range(2, 6),
    normal3=[('conv_1x1x1', 0), ('dil_3x3x3', 1), ('dil_3x3x3', 1), ('conv_1x1x1', 2), ('conv_3x1x1', 3),
             ('dil_3x3x3', 1), ('dil_3x3x3', 4), ('conv_1x1x1', 3)], normal_concat3=range(2, 6))