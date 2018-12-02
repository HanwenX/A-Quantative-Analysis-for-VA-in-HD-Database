import numpy as np

class Point:
    def __init__(self, coords, rid):
        self.mbb = np.vstack([coords, coords])
        self.rid = rid
        self.is_node = False
    def __str__(self):
        return 'Point at:' + str(self.mbb[0])
