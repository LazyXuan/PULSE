#!/usr/bin/python

import numpy as np
from keras.utils import np_utils
from keras.models import model_from_json
from utils import seq2mat
from keras.optimizers import SGD
import sys
import getopt

def usage():
    print 'Usage:%s [-h|-s|-i|-o] [--help|--species|--input|--output] args....' % sys.argv[0]
    print 'the species are allowed in \'human\' and \'mus\'...'
    print 'the input sequences should be in one-line fasta format...'

def get_opt():
    sp = ''
    out = ''
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'hs:i:o:', ['help', 'species=', 'input=', 'output='])
    except getopt.GetoptError:
        print 'Please provide right parameters!'
        usage()
        sys.exit(1)
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            usage()
            sys.exit(1)
        elif opt in ('-s', '--species'):
            sp = arg
        elif opt in ('-i', '--input'):
            ipath = arg
        elif opt in ('-o', '--output'):
            opath = arg
        else:
            usage()
            sys.exit(1)
    return sp, ipath, opath

def load_seqs(ipath):
    #print 'loading sequences...'
    genes = []
    fin = open(ipath, 'r')
    while 1:
        line = fin.readline().strip()
        if len(line) == 0:
            break
        elif '>' in line:
            name = line[1:]
            seq = fin.readline().strip()
        else:
            continue
        genes.append((name, seq))
    fin.close()
    #print 'Sequences were loaded!'
    return genes

def coding(gene):
    #print 'Sequences are coding into numeric data......'
    flank = 50
    data = []
    name = gene[0]
    seq = gene[1]
    l = len(seq)
    seq = 'N'*flank+seq+'N'*flank
    for i in xrange(l):
        mp = flank + i
        subseq = ''
        if seq[mp] == 'T':
            subseq = seq[mp-flank:mp+flank+1]
            mat = seq2mat([subseq])
            data.append([mat, i, name, l])
        else:
            pass

    X = np.array([data[j][0] for j in xrange(len(data))])
    idxs = [data[j][1] for j in xrange(len(data))]
    names = [data[j][2] for j in xrange(len(data))]
    lens = [data[j][3] for j in xrange(len(data))]

    #print 'Sequence coded!'
    return X, idxs, names, lens

def load_cnn(sp):
    print 'loading trained cnn model...'
    model = model_from_json(open('cnn_structure.json').read())
    if sp == 'mus':
        model.load_weights('./mPULSE_weights.h5')
    elif sp == 'human':
        model.load_weights('./hPULSE_weights.h5')
    else:
        print 'ERROR! Please check your species parameter!'
        usage()
        exit(1)
    sgd = SGD(lr=0.02, momentum=0.92, decay=1e-6, nesterov=False)
    model.compile(loss='binary_crossentropy', optimizer=sgd)
    print 'cnn model loaded!'
    return model

def prediction(model, X, idxs, names, lens, opath):
    fout = open(opath, 'a')
    preds = model.predict(X)[:, 1]
    for i in xrange(len(idxs)):
        op = '\t'.join([names[i], str(idxs[i]), str(lens[i]), str(preds[i])])
        fout.write(op)
        fout.write('\n')
    #print 'Current gene predicted!'
    fout.close()

if __name__ == '__main__':
    window = 101
    sp, ipath, opath = get_opt()
    model = load_cnn(sp)
    genes = load_seqs(ipath)
    
    fout = open(opath, 'a')
    fout.write('gene_name\tpsi_position\tgene_length\tpsi_potential')
    fout.write('\n')
    fout.close()
    
    for gene in genes:
        if 'T' not in gene[1]:
            continue
        X, idxs, names, lens = coding(gene)
        prediction(model, X, idxs, names, lens, opath)
    print 'Job Done!'
