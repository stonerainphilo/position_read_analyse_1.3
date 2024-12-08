import numpy as np

def calc_weight(Standard, a, b):
    delta_a = np.abs(Standard - a)
    delta_b = np.abs(Standard - b)
    weight_a = delta_b/(delta_b+delta_a)
    weight_b = delta_a/(delta_a+delta_b)
    return weight_a, weight_b

def weighted_average(Standard, a, b):
    weight = calc_weight(Standard, a, b)
    return a*weight[0]+b*weight[1]

def weighted_average_for_anther_varible(Standard, a, b, Varible_a, Varible_b):
    weight = calc_weight(Standard, a, b)
    return Varible_a*weight[0]+Varible_b*weight[1]