import random
import numpy as np

def flip():
    r = random.random()
    if r > 0.5:
        return 'h'
    else:
        return 't'

def choose_rand(l):
    if len(l) == 1:
        return l[0]
    return l[random.randint(0,len(l)-1)]

n = 10
trials = 1000
samples = 30000
p_list = []

for j in range(samples):
    outcomes = []
    no_match_counter = 0
    head_count = 0
    total_count = 0
    for i in range(trials):
        students = []
        for i in range(n):
            students.append(flip())

        candidate_i = []
        for i in range(n):
            if students[(i-1)%n] == 'h' and students[(i+1)%n] == 'h':
                candidate_i.append(i)


        if len(candidate_i)>=1:
            k = choose_rand(candidate_i)
            outcomes.append(students[k])
            if students[k] == 'h':
                head_count += 1
            total_count += 1

            #print 'student ' + str(k)+ ' had ' + str(students[k])
        else:
            #print 'no match'
            no_match_counter += 1
    p = head_count / float(total_count)
    p_list.append(p)

print np.average(p_list)
print np.std(p_list)
