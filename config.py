ARR_DATA_DIR = '/Users/rahul/Documents/G1/BrainInspiredComputing/TermProject/mit-bih-arrhythmia-database-1.0.0'
NSR_DATA_DIR = '/Users/rahul/Documents/G1/BrainInspiredComputing/TermProject/mit-bih-normal-sinus-rhythm-database-1.0.0'

# USE THE AAMI EC57 STANDARD: https://arxiv.org/html/2503.07276v1
BEAT_TO_ENCODING = {
    '!': 4,
    '"': -1,
    '+': -1,
    '/': 4,
    'A': 1,
    'E': 2,
    'F': 3,
    'J': 1,
    'L': 0,
    'N': 0,
    'Q': 4,
    'R': 0,
    'S': 1,
    'V': 2,
    '[': 4,
    ']': 4,
    'a': 1,
    'e': 0,
    'f': 0,
    'j': 0,
    'x': 4,
    '|': -1,
    '~': -1,
}

ARRHYTHMIA_TO_ENCODING = {
    '(AB': 2,
    '(AFIB': 1,
    '(AFIB': 1,
    '(AFL': 2,
    '(B': 3,
    '(BII': 5,
    '(IVR': 3,
    '(N': 0,
    '(NOD': 2,
    '(P': 4,
    '(PREX': 4,
    '(SBR': 0,
    '(SVTA': 2,
    '(T': 3,
    '(VFL': 3,
    '(VT': 3,
    'MISSB': -1,
    'PSE': -1,
    'TS': -1, 
    '': -1,
}

NSR_TO_ENCODING = {
    'F': -1, 
    'J': -1, 
    'N': 0, 
    'S': -1, 
    'V': -1, 
    '|': -1, 
    '~': -1
}
