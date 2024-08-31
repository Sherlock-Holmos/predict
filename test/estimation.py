# 质量预测算法
def weight(length, width):
    x = length
    z = width
    # weight-length
    y = 1 / ( 0 + 163.1664135544264 * 0.9807226646189438**x )
    # weight-width
    w = 1 / ( 0 + 336.1868514764623 * 0.9396616862616823**z )
    calculated_weight = (y + w) / 2
    return calculated_weight
