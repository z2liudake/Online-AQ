import numpy as np

path_to_log = './log_paper/log_caltech_2_update_beamsearch.txt'
# path_to_log = './log_shuffle.txt'

with open(path_to_log, 'r') as f:
    lines = f.readlines()
    for line in lines:
        if 'encoding step' in line:
            *recall_k, recall_v = line.strip().split(' ')
            print(float(recall_v), end=' ')
print()