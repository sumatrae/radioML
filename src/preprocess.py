import pickle
import numpy as np
import pandas as pd
from radioDSP import RadioSignal
import matplotlib.pyplot as plot
from scipy import signal


file_path = "./RML2016.10a/RML2016.10a_dict.pkl"
with open(file_path, 'rb') as f:
    Xd = pickle.load(f, encoding='bytes')
    print(Xd)

    snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1, 0])

    X = []
    lbl = []
    for mod in mods:
        for snr in snrs:
            X.append(Xd[(mod, snr)])

            for i in range(Xd[(mod, snr)].shape[0]):
                lbl.append((mod, snr))
    #print(X)
    X = np.vstack(X)


df = pd.DataFrame(X[0,:,:])
print(df)
# Partition the data
#  into training and test sets of the form we can train/test on
#  while keeping SNR and Mod labels handy for each
np.random.seed(2016)
print(X.shape)
#
# n_examples = X.shape[0]
# n_train = n_examples * 0.5
# train_idx = np.random.choice(range(0,n_examples), size=int(n_train), replace=False)
#
# test_idx = list(set(range(0,n_examples))-set(train_idx))
#
# X_train = X[train_idx]
# y_train = [lbl[i] for i in train_idx]
# X_test =  X[test_idx]
offset = 19900
sig = RadioSignal(bit_width=8, fs=640)
for index in range(0,219999,20000):
    i = X[index + offset,0,:]
    q = X[index + offset,1,:]
    sig.load(i,q)
    sig.make_wave()
    # Plot the signal
    plot.subplot(211)
    plot.title(lbl[index + offset])
    plot.plot(sig.power_db)
    plot.xlabel('Sample')
    plot.ylabel('Amplitude')

    # Plot the spectrogram
    plot.subplot(212)
    plot.specgram(sig.power_db, Fs=640)
    # plot.magnitude_spectrum()
    plot.xlabel('Time')
    plot.ylabel('Frequency')
    plot.show()

