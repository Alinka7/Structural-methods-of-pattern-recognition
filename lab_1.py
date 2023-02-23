from pynput.keyboard import Key, Listener
import time
import pandas as pd
import numpy as np
from scipy.stats import t, f
import matplotlib.pyplot as plt
import scipy


def enteringPhrase(attempts, nameFile, phrase):
    allAttempts = []
    def on_press(key):
        global previous_time
        current_time = time.time()
        l.append(current_time)
    def on_release(key):
        if key ==  Key.enter:
            return False
    previous_time = time.time()
    i=0
    while i < attempts:
        l = []
        with Listener(on_press=on_press, on_release=on_release) as listener:
            foo = input('input a string: ')
            listener.join()
            allAttempts.append(l)  
        i += 1
    result = [l[:-1] for l in allAttempts]
    total = []
    for elem in result:
        total.append([round((elem[i + 1] - elem[i]),7) for i in range(len(elem)-1)])
    total = [i for i in total if len(i)==len(phrase)-1]
    with open(f'{nameFile}.txt', 'w') as f:
        for line in total:
            f.write("%s\n" % line)
            
    with open(f'{nameFile}.txt', 'r') as my_file:
        text = my_file.read()
        text = text.replace("[", "")
        text = text.replace("]", "")
        text = text.replace(",", "")
        
    with open(f'{nameFile}.txt', 'w') as my_file:
        my_file.write(text)
    f = open(f'{nameFile}.txt')
    text = f.read()
    f.close()
    f = open(f'{nameFile}.txt', 'w')
    f.write(' '.join([str(x) for x in list(range(len(phrase)-1))] ) + '\n')
    f.write(text)
    f.close()


def chekcEl(alpha, nameFile, nameFileStats, phrase):
    df = pd.read_csv(f"{nameFile}.txt", sep=" ")

    # checking normal distribution
    result = pd.DataFrame(columns = df.columns)
    for index, row in df.iterrows():
        stat, p = scipy.stats.shapiro(row)
        if p > alpha:
            result.loc[len(result.index)] = row
    df  = result 
    expct = pd.DataFrame()
    for i in df.columns:
        expct[i] = (df.drop(i, axis=1).sum(axis=1))/(len(list(df.columns))-1)
    disp = pd.DataFrame()
    for i in df.columns:
        disp[i] = ((df.drop(i, axis=1).subtract(expct.drop(i, axis=1)))**2).sum(axis=1)/(len(list(df.columns))-2)
    var = np.sqrt(disp)
    alpha = 0.1
    x = len(phrase) - 3                            
    tVal = t.ppf(1 - alpha/2, x) 
    tVal = round(tVal, 2)
    false = []
    for index, row in df.iterrows():
        for i in row.index:
            mi = (expct.iloc[[index]][i]).values[0]
            di = (var.iloc[[index]][i]).values[0]
            temp = abs((row.iloc[[int(i)]]-mi)/di).values[0]
            if temp > tVal:
                false.append(index)
    if len(set(false)) != 0:
        df.drop(false, inplace=True)
    if nameFile == 'stud':
        for i in range(df.shape[0]):
            plt.plot(list(df.columns),list(df.iloc[i]))
        # plt.savefig('plot.jpg')
        plt.savefig('D:/lab1/plot.jpg')

    tot = []
    for index, row in df.iterrows():
        e = (sum(list(row))/len(list(row)))
        e = np.array([e]*len(list(row)))
        row = np.array(row)
        d = sum((row - e)**2)/(len(list(row))-2)
        tot.append([round(e[0], 7), round(d, 7)])


    with open(f'{nameFileStats}.txt', 'w') as f:
        for line in tot:
            f.write("%s\n" % line)
    with open(f'{nameFileStats}.txt', 'r') as my_file:
        text = my_file.read()
        text = text.replace("[", "")
        text = text.replace("]", "")
        text = text.replace(",", "")
        
    with open(f'{nameFileStats}.txt', 'w') as my_file:
        my_file.write(text)
    f = open(f'{nameFileStats}.txt')
    text = f.read()
    f.close()
    f = open(f'{nameFileStats}.txt', 'w')
    f.write('1 2\n')
    f.write(text)
    f.close()



def testT(alpha, attemptsA, attemptsS, statS, statA, phrase):
    stud = pd.read_csv(f"{statS}.txt", sep=" ")
    authen = pd.read_csv(f"{statA}.txt", sep=" ")
    stud.columns = ['m', 'd']
    authen.columns = ['m', 'd']
    sD, aD, sM, aM = list(stud['d']), list(authen['d']), list(stud['m']), list(authen['m'])
    fVal = f.ppf(q=1-alpha, dfn=len(phrase)-1, dfd=len(phrase)-1)
    satisf = 0
    for i in sD:
        for j in aD:
            maxVal = max(i, j)
            minVal = min(i, j)
            F = maxVal/minVal
            if F < fVal:
                satisf= satisf + 1
    if satisf/((len(sD)) * len(aD)) > 0.95:
        print('Дисперсії однорідні')
    else:
        print('Диспресії неоднорідні', f'{satisf/((len(sD)) * len(aD))}%')
    n = len(phrase) - 1
    total_S = []
    for i in sD:
        for j in aD:
            dis = (((i**2 + j **2) * (n - 1) )/(2*n - 1))**(1/2)
            total_S.append(round(dis, 7))
    total_T = []
    for i in sM:
        for j in range(len(aM)):
            tp = abs(aM[j]-i)/((total_S[j]*((2/n)**(1/2))))
            total_T.append((j, tp))
    x = len(phrase) - 2                            
    tVal = t.ppf(1 - alpha/2, x) 
    tVal = round(tVal, 2)
    c = 0
    flse = []
    for i in total_T:
        if i[1] >= tVal:
            c = c + 1
        else:
            flse.append(i[0])
    print('p:', c/((len(sD)) * len(aD)))
    p1 = (len(set(flse)))/((len(sD)) * len(aD))
    p2 = (attemptsA-len(set(flse)))/((len(sD)) * len(aD))
    print('p1 ', p1)
    print('p2 ', p2)

    


if __name__ == '__main__':
    alpha = 0.05
    attemptsS, attemptsA = 5,5
    phrase='ozitem'

    # studying
    print('Studying')
    enteringPhrase(attemptsS, 'stud', phrase)
    chekcEl(alpha, 'stud', 'statsStud', phrase)


    # authentification
    print('authentification')
    enteringPhrase(attemptsA, 'auth', phrase)
    chekcEl(alpha, 'auth', 'statsAuth', phrase)

    # testing
    testT(alpha, attemptsA, attemptsS, 'statsStud', 'statsAuth', phrase)
    
