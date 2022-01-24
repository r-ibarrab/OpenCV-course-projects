import cv2 as cv
import numpy as np

def order(sum, xy_points):
    pass


points = [
    [34, 56],
    [123, 34],
    [450, 200],
    [245, 180]
]

x = np.array(points)
sums = x.sum(1)

z= [ x for x in sorted(zip(sums, x), key= lambda coor: coor[0])]
points = [x[1] for x in z]

diff = np.diff(sums)

print(sums)
print(x)
print(diff)
print(z)
print(points)

