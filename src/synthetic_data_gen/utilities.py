import random

def choice(d):
    values = list(d.keys())
    weights = list(d.values())
    return random.choices(values, weights=weights, k=1)[0]

def synonym(options):
    return random.choice(options)