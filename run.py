import sys

if __name__=='__main__':
    list = ['em01','em02','em06','em07','em08','em09']
    for exp in list:
        sys.argv=['',exp, 'debug']
        execfile('main.py')