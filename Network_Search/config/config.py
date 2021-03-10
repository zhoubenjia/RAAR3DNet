import yaml
# from easydict import EasyDict as edict
def Config(args):

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for dic in config:
        for k, v in config[dic].items():
            setattr(args, k, v)
    return args