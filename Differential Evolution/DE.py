import math as m
import random as r

RAND = lambda x,y:r.uniform(x,y)
egg_holder = lambda x,y:((-1)*(y+47)*m.sin(m.sqrt(abs((x/2)+(y+47)))))-((x)*m.sin(m.sqrt(abs(x-(y+47)))))
holder_table_function = lambda x,y:(-1)*abs(m.sin(x)*m.cos(y)*m.exp(abs(1-(m.sqrt((x*x)+(y*y))/3.14))))

def initialization(LB,UB,pop_size,vector_size):
    POP = list()
    for i in range(pop_size):
        temp = list()
        for j in range(vector_size):
            temp.append(RAND(LB[j],UB[j]))
        POP.append(temp)
    return POP

def COP(l):         #For copying the list
    nl = list()
    for i in range(len(l)):
        nl.append(l[i])
    return nl

def mvg(POP,F_LB,F_UB,K):
    F = RAND(F_LB,F_LB)
    M = list()
    for i in range(len(POP)):
        temp = list()
        r1 = r.randint(0,len(POP)-1)
        r2 = r.randint(0,len(POP)-1)
        r3 = r.randint(0,len(POP)-1)
        for j in range(len(POP[0])):
            temp.append(POP[i][j]+(K*(POP[r1][j]-POP[i][j]))+(F*(POP[r2][j]-POP[r3][j])))
        M.append(temp)
    return M

def tvg(POP,MV,Cr):
    tv = list()
    for i in range(len(POP)):
        temp = list()
        r = RAND(0,1)
        for j in range(len(POP[0])):
            r = RAND(0,1)
            if(r<=Cr):
                temp.append(MV[i][j])
            else:
                temp.append(POP[i][j])
        tv.append(temp)
    return tv

def eval(LB,UB,initial_pop,pop_size,F_LB,F_UB,K,Cr):
    eva = list()
    C = 0
    while(C!=pop_size):
        B = mvg(initial_pop,F_LB,F_UB,K)
        trail = tvg(initial_pop,B,Cr)
        for i in range(pop_size):
            if((LB[0]<trail[i][0] and trail[i][0]<UB[0]) and (LB[1]<trail[i][1] and trail[i][1]<UB[1])):
                eva.append(trail[i])
                C+=1
            if(C==pop_size):
                break
    return eva

def elitism(POP,tvg,function):
    new_gen = list()
    for i in range(len(POP)):
        A = function(POP[i][0],POP[i][1])
        B = function(tvg[i][0],tvg[i][1])
        print(tvg[i][0],tvg[i][1])
        if(B<A):
            new_gen.append(tvg[i])
        else:
            new_gen.append(POP[i])
    return new_gen

def Differential_Evolution(NOG,LB,UB,pop_size,vector_size,F_LB,F_UB,K,Cr,function):
    Generations = list()
    initial_pop = initialization(LB,UB,pop_size,vector_size)
    Generations.append(initial_pop)
    for i in range(NOG):
        A = eval(LB,UB,initial_pop,pop_size,F_LB,F_UB,K,Cr)
        D = elitism(initial_pop,A,function)
        Generations.append(D)
        initial_pop = COP(D)
    return Generations

#Constraints
LB = [-512,-512]   # Lower Bound
UB = [512,512]     # Upper Bound
K = 0.5            # Relaxation Parameter-1
Cr = 0.80         # Cross over Probability
F_LB = -2
F_UB = 2
Generations = list()
NOG = 80
pop_size = 20
vector_size = 2

Differential_Evolution(NOG,LB,UB,pop_size,vector_size,F_LB,F_UB,K,Cr,egg_holder)
