from matplotlib.backends.backend_pdf import PdfPages
import pickle
import matplotlib.pyplot as plt


def check_figs(figs):
    if figs is None:
        figs = [plt.figure(i) for i in plt.get_fignums()]
    return figs


def save_fig(*figs):
    figs = check_figs(figs)
    pickle.dump(figs, open("plot.pickle", "wb"))


def save_pdf(*figs):
    figs = check_figs(figs)
    pp = PdfPages('plot.pdf')
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()


def save_image(*figs):
    figs = check_figs(figs)
    for i, fig in enumerate(figs):
        fig.savefig('plot.image_{}'.format(i))


ax = pickle.load(open("plot.pickle", "rb"))
plt.show()
