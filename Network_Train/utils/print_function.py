import logging

def print_func(info):
    '''
    :param info: {name: value}
    :return:
    '''
    txts = []
    for name, value in info.items():
        txts.append('{}: {}'.format(name, value))
    logging.info('\t'.join(txts))