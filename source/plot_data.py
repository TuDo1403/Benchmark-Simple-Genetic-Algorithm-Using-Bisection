import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.ticker

def plot(data, label, title, plot_eval=False, hold=False):
    X = data[:, 0]
    y = data[:, 1] if not plot_eval else data[:, 2]
    y_err = data[:, 3] if not plot_eval else data[:, 4]

    
    ax.errorbar(X, y, yerr=y_err, label=label, fmt='-o')

    ax.set_xlabel('Problem size (l)')
    ylabel = 'MRPS (mean)' if not plot_eval else 'Average number of fitness function calls'
    ax.set_ylabel(ylabel)  

    ax.set_yscale('log')
    ax.set_xscale('log')

    ax.set_xticks([10 * 2**i for i in range(5)])

    ax.set_xlim(9, 170)
    ax.set_ylim(10, 10**7)
    
    # Keep tick from log scale
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())

    plt.legend(loc='upper right')

    ax.set_title(title)
    
    if not hold:
        plt.show()

fig, ax = plt.subplots()
filename = 'sGA-UX-OneMax.csv'
df = pd.read_csv('../report/{}'.format(filename))
df = df[df.MRPS != '-']
ux = df.to_numpy()
ux = np.array(ux, dtype=np.float)

plot(ux, 'UX', 'OneMax Benchmark (ECGA vs POPOP)', True, hold=True)

filename = 'sGA-1X-OneMax.csv'
df = pd.read_csv('../report/{}'.format(filename))
df = df[df.MRPS != '-']
onepoint = df.to_numpy()
onepoint = np.array(onepoint, dtype=np.float)

plot(onepoint, '1X', 'OneMax Benchmark (ECGA vs POPOP)', True, hold=True)

filename = 'ECGA-OneMax.csv'
df = pd.read_csv('../report/{}'.format(filename))
df = df[df.MRPS != '-']
ecga = df.to_numpy()
ecga = np.array(ecga, dtype=np.float)

plot(ecga, 'ECGA', 'OneMax Benchmark (ECGA vs POPOP)', True, hold=False)

