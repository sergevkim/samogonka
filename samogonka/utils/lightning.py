from collections import OrderedDict


def process_lightning_state_dict(state_dict):
    # model.feature_extractor.0.weight -> feature_extractor.0.weight
    return OrderedDict((k[6:], v) for k, v in state_dict.items())
