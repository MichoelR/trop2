import numpy as np

import unicodedata

n = 10
x = [[[[x * n + y + 1] for y in range(n)] for x in range(n)]]
x = np.array(x)

unicodedata.name('֖')
def unicode_name(char):
    name = " ".join(unicodedata.name(char).split()[2:])
    return name

