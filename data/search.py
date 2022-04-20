import os


def check_exist(path):
    if not os.path.exists(path):
        print(path)


file = './list_test_cufsf.txt'
with open(file, 'r') as f:
    lines = f.readlines()
for l in lines:
    s, p, m, _ = l.strip().split('||')
    check_exist('../'+s)
    check_exist('../'+p)
    check_exist('../'+m)
    # break
