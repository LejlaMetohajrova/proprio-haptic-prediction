from __future__ import print_function
import numpy as np
from operator import itemgetter
import heapq

"""
Scripts used to produce final dataset.
"""

with open("leftArm.log", 'r') as f:
     file_lines = [''.join(['l ', x.strip(), '\n']) for x in f]

with open("leftArm.txt", 'w') as f:
     f.writelines(file_lines)

with open("rightArm.log", 'r') as f:
     file_lines = [''.join(['r ', x.strip(), '\n']) for x in f]

with open("rightArm.txt", 'w') as f:
     f.writelines(file_lines)

with open("skin_events.log", 'r') as f:
     file_lines = [''.join(['s ', x.strip(), '\n']) for x in f]

with open("skin_events.txt", 'w') as f:
     f.writelines(file_lines)

def extract_timestamp(line):
    return line.split()[3]

with open("skin_events.txt") as f0, open("leftArm.txt") as f1, open("rightArm.txt") as f2:
    sources = [f0, f1, f2]
    with open("merged.txt", "w") as dest:
        decorated = [
            ((extract_timestamp(line), line) for line in f)
            for f in sources]
        merged = heapq.merge(*decorated)
        undecorated = map(itemgetter(-1), merged)
        dest.writelines(undecorated)

with open("merged.txt", "r") as input, open("data.txt", "w") as out:
    prev = input.readline()
    for line in input:
        if prev[0] != line[0]:
            out.write(prev)
        prev = line    

taxel = np.array([[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
[12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
[24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35],
[36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47],
[48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
[96, 97, 98, 99, 102, 103, 120, 129, 130],
[100, 101, 104, 105, 106, 113, 116, 117],
[108, 109, 110, 111, 112, 114, 115, 118, 142, 143],
[121, 122, 123, 124, 125, 126, 127, 128],
[132, 133, 134, 135, 136, 137, 138, 140, 141]],

[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
[12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
[24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35],
[36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47],
[48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
[96, 97, 98, 100, 101, 110, 111, 112],
[99, 102, 103, 104, 105, 106, 127, 129, 130],
[108, 109, 113, 114, 115, 116, 117, 118, 142, 143],
[120, 121, 122, 123, 124, 125, 126, 128],
[132, 133, 134, 135, 136, 137, 138, 140, 141]],

[[288, 289, 290, 291, 292, 293, 295, 296, 297, 299, 300, 301, 302, 303, 304, 305, 307, 308, 309, 311, 348, 349, 350, 351, 352, 353, 355, 356, 357, 359],
[204, 205, 206, 207, 208, 209, 211, 212, 213, 215, 336, 337, 338, 339, 340, 341, 343, 344, 345, 347],
[252, 253, 254, 255, 256, 257, 259, 260, 261, 263, 312, 313, 314, 315, 316, 317, 319, 320, 321, 323],
[0, 1, 2, 3, 4, 5, 7, 8, 9, 11, 180, 181, 182, 183, 184, 185, 187, 188, 189, 191],
[24, 25, 26, 27, 28, 29, 31, 32, 33, 35, 12, 13, 14, 15, 16, 17, 19, 20, 21, 23],
[156, 157, 158, 159, 160, 161, 163, 164, 165, 167, 144, 145, 146, 147, 148, 149, 151, 152, 153, 155],
[132, 133, 134, 135, 136, 137, 139, 140, 141, 143, 168, 169, 170, 171, 172, 173, 175, 176, 177, 179],
[120, 121, 122, 123, 124, 125, 127, 128, 129, 131, 60, 61, 62, 63, 64, 65, 67, 68, 69, 71],
[96, 97, 98, 99, 100, 101, 103, 104, 105, 107, 108, 109, 110, 111, 112, 113, 115, 116, 117, 119],
[84, 85, 86, 87, 88, 89, 91, 92, 93, 95, 72, 73, 74, 75, 76, 77, 79, 80, 81, 83],
[36, 37, 38, 39, 40, 41, 43, 44, 45, 47, 48, 49, 50, 51, 52, 53, 55, 56, 57, 59]]])

record = []

with open('data.txt') as f:
    for line in f:
        record.append([x.split() for x in [l.replace('(', '').replace(')', '') for l in line.strip().split(' ((')]])

def binarize(contact_vector):
    b = np.zeros(42)
    
    if contact_vector == []:
        return b

    for v in contact_vector:
        if v[3] == 1:
            for t in v[19:-1]:
                b[[i for i in range(len(taxel[0])) if t in taxel[0][i]]] = 1
        elif v[3] == 2:
            for t in v[19:-1]:
                b[[i + 10 for i in range(len(taxel[2])) if t in taxel[2][i]]] = 1
        elif v[3] == 4:
            for t in v[19:-1]:
                b[[i + 20 for i in range(len(taxel[1])) if t in taxel[1][i]]] = 1
        elif v[3] == 5:
            for t in v[19:-1]:
                b[[i + 30 for i in range(len(taxel[2])) if t in taxel[2][i]]] = 1
    return b

with open('d.txt', 'w') as f:
    r = None
    l = None
    for rec in record:
        if rec[0][0] == 's':
            if l is not None and r is not None:
                print(" ".join(str(item) for item in np.append(binarize([[float(x) for x in y] for y in rec[1:]]), [l, r])), file=f)
        elif rec[0][0] == 'r':
            r = rec[0][4:11]
        else:
            l = rec[0][4:11]
