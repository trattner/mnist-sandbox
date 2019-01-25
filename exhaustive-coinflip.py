all_possible = []

def recursive_enumerate(s, l):
    if len(s) == l:
        all_possible.append(s)
    else:
        recursive_enumerate(s+'h', l)
        recursive_enumerate(s+'t', l)

recursive_enumerate('', 10)

print 'checking all possible strings generated: ' + str(len(all_possible)==2**10)

#data = {}

output = 0
no_match = 0
tails = 0

for s in all_possible:
    n_h = 0 # number of heads found between H-H sandwich
    n_t = 0 # number of tails found between H-H sandwich
    for i in range(10):
        if s[(i-1)%9]=='h' and s[(i+1)%9]=='h':
            if s[i] == 'h':
                n_h += 1
            elif s[i] == 't':
                n_t += 1
    total_sw = n_h + n_t # how many total sandwiches in the string
    if total_sw == 0:
        no_match += 1
    else:
        output += n_h / float(total_sw)
        tails += n_t / float(total_sw)

print no_match
print output
print tails



''' #to record all the options...

    if total_sw == 0:
        if 0 in data:
            data[0] += 1
        else:
            data[0] = 1
    else:
        if total_sw in data:
            if n_h in data[total_sw]:
                if n_t in data[total_sw][n_h]:
                    data[total_sw][n_h][n_t] += 1
                else:
                    data[total_sw][n_h][n_t] = 1
            else:
                data[total_sw][n_h] = {}
                data[total_sw][n_h][n_t] = 1
        else:
            data[total_sw] = {}
            data[total_sw][n_h] = {}
            data[total_sw][n_h][n_t] = 1

for k in data.keys():
    print data[k]

'''
