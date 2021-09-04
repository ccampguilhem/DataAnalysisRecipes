from functools import partial
import os
import subprocess as sp

import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rcParams
from matplotlib.colors import Colormap, to_rgba
from IPython.display import HTML
import numpy as np



BLUE = "#3498DB"
RED = "#E74C3C"
GREEN = "#2ECC71"


# This function svaes the state of a neural network
# It does use pickle to make it usable whatever the Python or sklearn version

def save_network(model, filename):
    # Initializations
    params = []
    # Iterate through the layers
    for i in range(1, model.n_layers_):
        layer_index = i - 1
        params.append({"name": "w{}".format(layer_index), "attr": "coefs_", "index": layer_index})
        params.append({"name": "b{}".format(layer_index), "attr": "intercepts_", "index": layer_index})
    # Save to files
    with open(filename, "w") as fobj:
        for param in params:
            array = getattr(model, param["attr"])[param["index"]]
            _filename = "{}_{}.csv".format(os.path.splitext(filename)[0], param["name"])
            fobj.write("{};{};{}\n".format(param["name"], _filename, ",".join([str(v) for v in array.shape])))
            np.savetxt(_filename, array)
        

# Load neural network from disk
        
def load_network(model, filename):
    # Initialization
    coefs = []
    intercepts = []
    # Read the master file
    with open(filename, "r") as fobj:
        for line in fobj:
            fields = line.strip().split(";")
            if fields[0][0] == "w":
                value = np.loadtxt(fields[1])
                shape = tuple(int(v) for v in fields[2].split(","))
                coefs.append(value.reshape(shape))
            elif fields[0][0] == "b":
                value = np.loadtxt(fields[1])
                shape = tuple(int(v) for v in fields[2].split(","))                
                intercepts.append(value.reshape(shape))
    # Load the model
    model.coefs_ = coefs
    model.intercepts_ = intercepts


# A custom color map for matplotlib

class MyColorMap(Colormap):
    def __init__(self):
        super(MyColorMap, self).__init__("clf", 2)
        self.monochrome = False
    
    def __call__(self, x, alpha=None, bytes=False):
        if hasattr(x, "__len__"):
            return self._map_array(x, alpha, bytes)
        else:
            return self._map_scalar(x, alpha, bytes)
    
    def _map_array(self, x, alpha=None, bytes=False):
        y = [to_rgba(BLUE, alpha) if _x < 0.5 else to_rgba(RED, alpha) for _x in x]
        if bytes:
            y = [np.floor(_y * 255.).astype("int") for _y in y]
        return np.array(y)
    
    def _map_scalar(self, x, alpha=None, bytes=False):
        y = to_rgba(BLUE, alpha) if x < 0.5 else to_rgba(RED, alpha)
        if bytes:
            y = np.floor(y * 255.).astype("int")
        return y
    
    
# A ploting function for lines, datasets, neural network classifiers and regressors

def plot_line_and_points(w=None, x=None, c=None, clf=None, reg=None, autofit=False, levels=2, figsize=(6, 6),
                         sampling=None):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    if sampling:
        indices = np.arange(0, x.shape[0], 1)
        np.random.shuffle(indices)
        if x is not None:
            x = x[indices[:sampling], :]
        if c is not None:
            c = c[indices[:sampling]]
    if w is not None:
        _x = np.linspace(-1., 1., 5)
        _a, _b, _c = w[0], w[1], w[2]
        if np.abs(_b) <= 1.e-8:
            _b = 1.e-8
        _y = - (_a * _x + _c) / _b
        ax.plot(_x, _y, color='k')
        _X, _Y = np.meshgrid(np.linspace(-1., 1., 11), np.linspace(-1., 1., 11))
        _x = _X.flatten()
        _y = _Y.flatten()
        _z = _a * _x + _b * _y + _c
        _Z = _z.reshape(_X.shape)
        if levels == 2:
            ax.contourf(_X, _Y, _Z, levels=[-100., 0., 100.], colors=[BLUE, RED], alpha=0.75)
        else:
            cont = ax.contourf(_X, _Y, _Z, levels=levels, cmap="bwr")
            fig.colorbar(cont, ax=ax)
    if clf is not None:
        _X, _Y = np.meshgrid(np.linspace(-1., 1., 501), np.linspace(-1., 1., 501))
        _x = np.vstack((_X.flatten(), _Y.flatten())).T
        _z = clf.predict(_x)
        if _z.min() >= 0.:
            # the model predicts output to be in the range [0, 1] (training used a cross entropy loss)
            _z = _z * 2. - 1.  # scale in the range [-1., 1.]
        _Z = _z.reshape(_X.shape)
        ax.contourf(_X, _Y, _Z, levels=[-1., 0., 1.], colors=[BLUE, RED], alpha=0.75)
    if reg is not None:
        _x = np.linspace(-1., 1., 50)
        _y = reg.predict(_x.reshape(50, 1))
        ax.plot(_x, _y, color='w')
    if x is not None:
        if c is None:
            c = GREEN
            cmap = None
        else:
            cmap = MyColorMap()
        if len(x.shape) > 1:
            _x = x[:, 0]
            _y = x[:, 1]
        else:
            _x = x[0]
            _y = x[1]
        ax.scatter(_x, _y, c=c, edgecolor='w', lw=1, s=50, zorder=10, cmap=cmap)
    if not autofit:
        ax.set_xlim((-1., 1.))
        ax.set_ylim((-1., 1.))
    ax.set_xlabel("$x_1$", fontsize=14)
    ax.set_ylabel("$x_2$", fontsize=14)
    ax.set_aspect('equal')
    return fig


# An animated plot for transformations

class AnimatedScatter():
    def __init__(self, generator, interval=100, nb_frames=100, autofit=False,
                 figure_kwargs=None, scatter_kwargs=None):
        figure_kwargs_defaults = {
            "figsize": (6, 6),
            "xlim": (-1., 1.),
            "ylim": (-1., 1.),
            "facecolor": "#212121",
        }
        self.figure_kwargs = figure_kwargs_defaults
        if figure_kwargs is not None:
            self.figure_kwargs.update(figure_kwargs)        
        scatter_kwargs_defaults = {
            "c": "w",
            "edgecolor": "w",
        }
        self.scatter_kwargs = scatter_kwargs_defaults
        if scatter_kwargs is not None:
            self.scatter_kwargs.update(scatter_kwargs)
        self.autofit = autofit
        self.fig = plt.figure(figsize=self.figure_kwargs["figsize"])
        self.interval = interval
        self.anim = matplotlib.animation.FuncAnimation(
            self.fig, self.update, generator, interval=self.interval, 
            init_func=self.init_func, save_count=nb_frames, blit=True
        )
        
    def init_func(self):
        self.ax = self.fig.add_subplot(111)
        self.title = self.ax.set_title(None)
        self.scatter = self.ax.scatter([], [], [])
        self.ax.set_ylim(*self.figure_kwargs["ylim"])
        self.ax.set_xlim(*self.figure_kwargs["xlim"])
        return self.scatter, self.title
    
    def update(self, data):
        title, xy, s, c = data
        self.title.set_text(title)
        self.scatter.set_offsets(xy)
        self.scatter.set_sizes(s)
        self.scatter.set_color(self.scatter_kwargs["cmap"](c))
        self.scatter.set_edgecolor(self.scatter_kwargs["edgecolor"])
        if self.autofit:
            xmin, ymin = xy.min(axis=0)
            xmax, ymax = xy.max(axis=0)
            self.ax.set_xlim(xmin, xmax)
            self.ax.set_ylim(ymin, ymax)
        plt.close()
        return self.scatter, self.title
    
    def display(self):
        return HTML(self.anim.to_jshtml())
    
    def save(self, filename, dpi, *args, **kwargs):
        writervideo = matplotlib.animation.FFMpegWriter(fps=1000./self.interval, *args, **kwargs)
        self.anim.save(filename, writervideo, dpi=dpi, 
                       savefig_kwargs={"facecolor": self.figure_kwargs["facecolor"]})
        

# An animated plot for classification and regression

class AnimatedNetwork():
    def __init__(self, generator, interval=100, nb_frames=100, figure_kwargs=None,
                 scatter_kwargs=None, plot_kwargs=None, contour_kwargs=None):
        figure_kwargs_defaults = {
            "figsize": (6, 6),
            "facecolor": "#212121",
        }
        self.figure_kwargs = figure_kwargs_defaults
        if figure_kwargs is not None:
            self.figure_kwargs.update(figure_kwargs)        
        scatter_kwargs_defaults = {
            "cmap": None,
            "edgecolor": "w",
            "c": "w",
        }
        self.scatter_kwargs = scatter_kwargs_defaults
        if scatter_kwargs is not None:
            self.scatter_kwargs.update(scatter_kwargs)
        plot_kwargs_defaults = {
            "c": "k",
        }
        self.plot_kwargs = plot_kwargs_defaults
        if plot_kwargs is not None:
            self.plot_kwargs.update(plot_kwargs)  
        contour_kwargs_defaults = {
            "levels": 8,
            "alpha": 1.0,
            "cmap": matplotlib.cm.viridis,
        }
        self.contour_kwargs = contour_kwargs_defaults
        if contour_kwargs is not None:
            self.contour_kwargs.update(contour_kwargs)
        self.interval = interval
        self.fig = plt.figure(figsize=self.figure_kwargs["figsize"])
        self.anim = matplotlib.animation.FuncAnimation(
            self.fig, self.update, generator, interval=self.interval, 
            init_func=self.init_func, save_count=nb_frames, blit=False
        )
        
    def init_func(self):
        self.ax = self.fig.add_subplot(111)
        # Video export does not work great with text outside of plot area
#         self.title = self.ax.set_title(None)
        self.title = self.ax.annotate("", (0.5, 0.95), xycoords='axes fraction', va="top", ha="center", 
                                      zorder=100, c="k", fontsize=12, 
                                      bbox=dict(facecolor='white', alpha=0.75, edgecolor='white',
                                                boxstyle='round,pad=0.2'))
        self.scatter = self.ax.scatter([], [], [])
        self.plot, = self.ax.plot([], [])
        x, y = np.meshgrid(np.linspace(0., 1., 5), np.linspace(0., 1., 5))
        z = np.zeros(x.shape)
        self.contour = self.ax.contourf(x, y, z)
        self.ax.set_ylim(-1., 1.)
        self.ax.set_xlim(-1., 1.)
        return self.title, self.scatter, self.plot, self.contour
    
    def update(self, data):
        title, scatter_xy, scatter_sizes, scatter_color, plot_x, plot_y, contour_x, contour_y, contour_z = data
        self.title.set_text(title)
        self.contour = self.ax.contourf(contour_x, contour_y, contour_z, cmap=self.contour_kwargs["cmap"])
        self.scatter.set_offsets(scatter_xy)
        self.scatter.set_sizes(scatter_sizes)
        if self.scatter_kwargs["cmap"] is None:
            self.scatter.set_color(self.scatter_kwargs["c"])
        else:
            self.scatter.set_color(self.scatter_kwargs["cmap"](scatter_color))
        self.scatter.set_edgecolor(self.scatter_kwargs["edgecolor"])
        self.scatter.set_zorder(10)
        self.plot.set_xdata(plot_x)
        self.plot.set_ydata(plot_y)
        self.plot.set_color(self.plot_kwargs["c"])
        self.plot.set_zorder(5)
        plt.close()
        return self.title, self.scatter, self.plot, self.contour
    
    def display(self):
        return HTML(self.anim.to_jshtml())
    
    def save(self, filename, dpi, *args, **kwargs):
        writervideo = matplotlib.animation.FFMpegWriter(fps=1000./self.interval, *args, **kwargs)
        self.anim.save(filename, writervideo, dpi=dpi, 
                       savefig_kwargs={"facecolor": self.figure_kwargs["facecolor"]})
        
        
# An animated plot for loss

class AnimatedLoss():
    def __init__(self, generator, interval=100, nb_frames=100, figure_kwargs=None,
                 scatter_kwargs=None):
        figure_kwargs_defaults = {
            "figsize": (18, 9),
            "facecolor": "#212121",
            "titles": [None] * 10,
        }
        self.figure_kwargs = figure_kwargs_defaults
        if figure_kwargs is not None:
            self.figure_kwargs.update(figure_kwargs)        
        scatter_kwargs_defaults = {
            "cmap": "cool_r",
            "size": 50,
        }
        self.scatter_kwargs = scatter_kwargs_defaults
        if scatter_kwargs is not None:
            self.scatter_kwargs.update(scatter_kwargs)
        self.interval = interval
        self.fig = plt.figure(figsize=self.figure_kwargs["figsize"], constrained_layout=True)
        self.anim = matplotlib.animation.FuncAnimation(
            self.fig, self.update, generator, interval=self.interval, 
            init_func=self.init_func, save_count=nb_frames, blit=False
        )
        self.i = 0
        
    def init_func(self):
        gs = self.fig.add_gridspec(3, 6)
        self.ax1 = self.fig.add_subplot(gs[0, 0])
        self.sp1 = self.ax1.scatter([], [], [], c=[], cmap=self.scatter_kwargs["cmap"])
        self.ax1.set_title(self.figure_kwargs["titles"][0])
        self.ax2 = self.fig.add_subplot(gs[0, 1], sharey=self.ax1)
        self.sp2 = self.ax2.scatter([], [], [], c=[], cmap=self.scatter_kwargs["cmap"])
        self.ax2.set_title(self.figure_kwargs["titles"][1])
        self.ax3 = self.fig.add_subplot(gs[0, 2], sharey=self.ax1)
        self.sp3 = self.ax3.scatter([], [], [], c=[], cmap=self.scatter_kwargs["cmap"])
        self.ax3.set_title(self.figure_kwargs["titles"][2])
        self.ax4 = self.fig.add_subplot(gs[1, 0], sharey=self.ax1)
        self.sp4 = self.ax4.scatter([], [], [], c=[], cmap=self.scatter_kwargs["cmap"])
        self.ax4.set_title(self.figure_kwargs["titles"][3])
        self.ax5 = self.fig.add_subplot(gs[1, 1], sharey=self.ax1)
        self.sp5 = self.ax5.scatter([], [], [], c=[], cmap=self.scatter_kwargs["cmap"])
        self.ax5.set_title(self.figure_kwargs["titles"][4])
        self.ax6 = self.fig.add_subplot(gs[1, 2], sharey=self.ax1)
        self.sp6 = self.ax6.scatter([], [], [], c=[], cmap=self.scatter_kwargs["cmap"])
        self.ax6.set_title(self.figure_kwargs["titles"][5])
        self.ax7 = self.fig.add_subplot(gs[2, 0], sharey=self.ax1)
        self.sp7 = self.ax7.scatter([], [], [], c=[], cmap=self.scatter_kwargs["cmap"])
        self.ax7.set_title(self.figure_kwargs["titles"][6])
        self.ax8 = self.fig.add_subplot(gs[2, 1], sharey=self.ax1)
        self.sp8 = self.ax8.scatter([], [], [], c=[], cmap=self.scatter_kwargs["cmap"])
        self.ax8.set_title(self.figure_kwargs["titles"][7])
        self.ax9 = self.fig.add_subplot(gs[2, 2], sharey=self.ax1)
        self.sp9 = self.ax9.scatter([], [], [], c=[], cmap=self.scatter_kwargs["cmap"])
        self.ax9.set_title(self.figure_kwargs["titles"][8])
        self.ax10 = self.fig.add_subplot(gs[0:, 3:], sharey=self.ax1)
        self.sp10 = self.ax10.scatter([], [], c=[], cmap=self.scatter_kwargs["cmap"])
        self.ax10.set_title(self.figure_kwargs["titles"][9])
        self.ax10.set_xlabel("epochs")
        return (self.sp1, self.sp2, self.sp3, self.sp4, self.sp5, self.sp6, self.sp7, self.sp8, 
                self.sp9, self.sp10)
    
    def update(self, data):
        xs, ys, cs, xlim, ylim = data
        for i, (sp, ax) in enumerate([
            (self.sp1, self.ax1),
            (self.sp2, self.ax2),
            (self.sp3, self.ax3),
            (self.sp4, self.ax4),
            (self.sp5, self.ax5),
            (self.sp6, self.ax6),
            (self.sp7, self.ax7),
            (self.sp8, self.ax8),
            (self.sp9, self.ax9),
            (self.sp10, self.ax10),
        ]):
            sp.set_offsets(np.vstack((xs[i], ys[i])).T)
            sp.set_array(cs[i])
            if i < 9:
                sp.set_sizes([50]*len(cs[i]))
            else:
                sp.set_sizes([300]*len(cs[i]))
            ax.set_xlim(xlim[i])
            ax.set_ylim(ylim[i])
        self.i += 1
        self.fig.savefig("frame-{:03d}.jpg".format(self.i), dpi=60, facecolor="#212121")
        plt.close()
        return (self.sp1, self.sp2, self.sp3, self.sp4, self.sp5, self.sp6, self.sp7, self.sp8, 
                self.sp9, self.sp10)
    
    def display(self):
        return HTML(self.anim.to_jshtml())
    
#     def save(self, filename, dpi, *args, **kwargs):
#         writervideo = matplotlib.animation.FFMpegWriter(fps=1000./self.interval, *args, **kwargs)
#         self.anim.save(filename, writervideo, dpi=dpi, 
#                        savefig_kwargs={"facecolor": self.figure_kwargs["facecolor"]})
    def save(self, filename):
        args = ["ffmpeg", "-y", "-framerate", "{}".format(np.floor(1000./self.interval)), 
                "-i", "frame-%03d.jpg", filename]
        p = sp.Popen(args=args)
        p.wait()
        

# Animation to visualize the influence of one parameter on the neural network        

class AnimatedParameter():
    def __init__(self, generator, interval=100, nb_frames=100, figure_kwargs=None,
                 scatter_kwargs=None, plot_kwargs=None, contour_kwargs=None):
        figure_kwargs_defaults = {
            "figsize": (18, 6),
            "facecolor": "#212121",
        }
        self.figure_kwargs = figure_kwargs_defaults
        if figure_kwargs is not None:
            self.figure_kwargs.update(figure_kwargs)        
        scatter_kwargs_defaults = {
            "cmap": None,
            "edgecolor": "w",
            "c": "w",
        }
        self.scatter_kwargs = scatter_kwargs_defaults
        if scatter_kwargs is not None:
            self.scatter_kwargs.update(scatter_kwargs)
        plot_kwargs_defaults = {
            "c": "k",
        }
        self.plot_kwargs = plot_kwargs_defaults
        if plot_kwargs is not None:
            self.plot_kwargs.update(plot_kwargs)  
        contour_kwargs_defaults = {
            "levels": 8,
            "alpha": 1.0,
            "cmap": matplotlib.cm.viridis,
        }
        self.contour_kwargs = contour_kwargs_defaults
        if contour_kwargs is not None:
            self.contour_kwargs.update(contour_kwargs)
        self.interval = interval
        self.fig = plt.figure(figsize=self.figure_kwargs["figsize"])
        self.anim = matplotlib.animation.FuncAnimation(
            self.fig, self.update, generator, interval=self.interval, 
            init_func=self.init_func, save_count=nb_frames, blit=False
        )
        self.fig.subplots_adjust(wspace=0.25)
        
    def init_func(self):
        # Create three subplots
        gs = self.fig.add_gridspec(1, 3)
        self.ax1 = self.fig.add_subplot(gs[0, 0])
        self.ax2 = self.fig.add_subplot(gs[0, 1])
        self.ax3 = self.fig.add_subplot(gs[0, 2])
        self.plot, = self.ax1.plot([], [])
        self.ax1.set_xlabel("value")
        self.ax1.set_ylabel("loss")
        self.ax1.set_title("parameter")
        self.scatter1 = self.ax2.scatter([], [], [])
        self.ax2.set_xlim(-1., 1.)
        self.ax2.set_ylim(-1., 1.)
        self.ax2.set_title("transformation of hidden layer")
        self.ax2.set_xlabel("$x^{(1)}_1$")
        self.ax2.set_ylabel("$x^{(1)}_2$")
        self.scatter2 = self.ax3.scatter([], [], [])
        self.ax3.set_xlim(-1., 1.)
        self.ax3.set_ylim(-1., 1.)
        self.ax3.set_title("classification")
        self.ax3.set_xlabel("$x_1$")
        self.ax3.set_ylabel("$x_2$")
        x, y = np.meshgrid(np.linspace(0., 1., 5), np.linspace(0., 1., 5))
        z = np.zeros(x.shape)
        self.contour = self.ax3.contourf(x, y, z)
        return self.plot, self.scatter1, self.scatter2, self.contour
    
    def update(self, data):
        plot_x = data[0]
        plot_y = data[1]
        plot_xlim = data[2]
        plot_ylim = data[3]
        scatter1_xy = data[4]
        scatter1_sizes = data[5]
        scatter1_c = data[6]
        scatter2_xy = data[7]
        scatter2_sizes = data[8]
        scatter2_c = data[9]
        contour_x = data[10]
        contour_y = data[11]
        contour_z = data[12]
        self.plot.set_xdata(plot_x)
        self.plot.set_ydata(plot_y)
        self.ax1.set_xlim(plot_xlim)
        self.ax1.set_ylim(plot_ylim)
        self.scatter1.set_offsets(scatter1_xy)
        self.scatter1.set_sizes(scatter1_sizes)
        if self.scatter_kwargs["cmap"] is None:
            self.scatter1.set_color(self.scatter_kwargs["c"])
        else:
            self.scatter1.set_color(self.scatter_kwargs["cmap"](scatter1_c))
        self.scatter1.set_edgecolor(self.scatter_kwargs["edgecolor"])
        self.scatter1.set_zorder(10)
        self.scatter2.set_offsets(scatter2_xy)
        self.scatter2.set_sizes(scatter2_sizes)
        if self.scatter_kwargs["cmap"] is None:
            self.scatter2.set_color(self.scatter_kwargs["c"])
        else:
            self.scatter2.set_color(self.scatter_kwargs["cmap"](scatter2_c))
        self.scatter2.set_edgecolor(self.scatter_kwargs["edgecolor"])
        self.scatter2.set_zorder(10)
        self.contour = self.ax3.contourf(contour_x, contour_y, contour_z, cmap=self.contour_kwargs["cmap"])
        plt.close()
        return self.plot, self.scatter1, self.scatter2, self.contour
    
    def display(self):
        return HTML(self.anim.to_jshtml())
    
    def save(self, filename, dpi, *args, **kwargs):
        writervideo = matplotlib.animation.FFMpegWriter(fps=1000./self.interval, *args, **kwargs)
        self.anim.save(filename, writervideo, dpi=dpi, 
                       savefig_kwargs={"facecolor": self.figure_kwargs["facecolor"]})
        
#     def save(self, filename):
#         args = ["ffmpeg", "-y", "-framerate", "{}".format(np.floor(1000./self.interval)), 
#                 "-i", "frame-%03d.jpg", filename]
#         p = sp.Popen(args=args)
#         p.wait()
        
        
# A factory function for creating animation for regression 

def make_regression_animation(xy, hist_w):
    def generator(xy, hist_w):
        s = np.ones((xy.shape[0],)) * 50.
        xplot = np.linspace(-1., 1., 11)
        xgrid, ygrid = np.meshgrid(np.linspace(-1., 1., 101), np.linspace(-1., 1., 101))
        i = 0
        while True:
            legend = "{:3d} passes through dataset".format(i)
            a, b, c = hist_w[i]
            yplot = - a / b * xplot - c / b
            zgrid = np.zeros(xgrid.shape)
            mask = a * xgrid + b * ygrid + c > 0
            zgrid[mask] = 1.0
            yield legend, xy, s, None, xplot, yplot, xgrid, ygrid, zgrid
            i += 1
    anim = AnimatedNetwork(partial(generator, xy, hist_w), interval=50, nb_frames=len(hist_w),
                           scatter_kwargs={"c": GREEN},
                           contour_kwargs={"cmap": MyColorMap()})
    return anim


# A factory function for creating animation for classification 

def make_classification_animation(xy, color, hist_w):
    def generator(xy, color, hist_w):
        s = np.ones((xy.shape[0],)) * 50.
        xplot = np.linspace(-1., 1., 11)
        xgrid, ygrid = np.meshgrid(np.linspace(-1., 1., 101), np.linspace(-1., 1., 101))
        i = 0
        while True:
            legend = "{:3d} passes through dataset".format(i)
            a, b, c = hist_w[i]
            yplot = - a / b * xplot - c / b
            zgrid = np.zeros(xgrid.shape)
            mask = a * xgrid + b * ygrid + c > 0
            zgrid[mask] = 1.0
            yield legend, xy, s, color, xplot, yplot, xgrid, ygrid, zgrid
            i += 1
    anim = AnimatedNetwork(partial(generator, xy, color, hist_w), interval=50, nb_frames=len(hist_w),
                           scatter_kwargs={"cmap": MyColorMap()},
                           contour_kwargs={"cmap": MyColorMap()})
    return anim


# A factory function for creating animation for non-linear transformation 

def make_transform_animation(xy_before, xy_after, color):
    def generator(xy_before, xy_after, color):
        legend = None
        s = np.ones((xy_before.shape[0],)) * 50.
        # we need to scale xy_after in the range [-1., 1.] for both x and y
        xy_after = (xy_after - xy_after.min(axis=0)) / (xy_after.max(axis=0) - xy_after.min(axis=0)) * 2 - 1.
        i = 0
        while i < 100:
            progress = i / 100.
            xy = (1 - progress) * xy_before + progress * xy_after
            yield legend, xy, s, color
            i += 1
    anim = AnimatedScatter(partial(generator, xy_before, xy_after, color), interval=50, nb_frames=101,
                           scatter_kwargs={"cmap": MyColorMap()})
    return anim


# A factory function for creating animation for loss

def make_loss_animation(hist, group_size=10, interval=50, figure_kwargs=None, scatter_kwargs=None):
    params = ["w0[0, 0]", "w0[1, 0]", "b0[0, 0]", 
              "w0[0, 1]", "w0[1, 1]", "b0[0, 1]", 
              "w1[0, 0]", "w1[1, 0]", "b1[0, 0]"]
    xlim = []
    ylim = []
    for name in params:
        xmin = xmax = None
        ymin = ymax = None
        param = hist["param_records"][name]
        for record, loss in zip(param, hist["loss_records"]):
            if xmin is None or record.min() < xmin:
                xmin = record.min()
            if xmax is None or record.max() > xmax:
                xmax = record.max()
            if ymin is None or loss.min() < ymin:
                ymin = loss.min()
            if ymax is None or loss.max() > ymax:
                ymax = loss.max()
        xlim.append((xmin-0.1, xmax+0.1))
        ylim.append((ymin-0.1, ymax+0.1))
    xlim.append((0., len(hist["loss_records"])))
    ylim.append(ylim[-1])
    def generator(hist, group_size):
        i = -(group_size - 2)
        while True:
            _i = max(0, i)
            _group_size = min(group_size + i, group_size)            
            loss = hist["loss_records"][_i: _i + _group_size]
            epochs = np.array(range(len(hist["loss_records"])))[_i: _i + _group_size]
            xs = []
            ys = []
            colors = []
            for name in params:
                xs.append(hist["param_records"][name][_i: _i + _group_size])
                ys.append(loss)
                colors.append(np.linspace(0., 1., _group_size))
            xs.append(epochs)
            ys.append(loss)
            colors.append(np.linspace(0., 1., _group_size))
            yield xs, ys, colors, xlim, ylim
            i += 1
    nb_frames = len(hist["loss_records"]) - 1
    if figure_kwargs is None:
        figure_kwargs = {"titles": params + ["$\epsilon$"]}
    else:
        figure_kwargs.update({"titles": params + ["$\epsilon$"]})
    anim = AnimatedLoss(partial(generator, hist, group_size), interval=interval, nb_frames=nb_frames, 
                       figure_kwargs=figure_kwargs, scatter_kwargs=scatter_kwargs)
    return anim


# A factory function for creating animation for parameter

def make_parameter_animation(model, param_name, index, values, x, y):
    def generator(model, param_name, index, values, x, y):
        # Data for the plot
        plot_x = []
        plot_y = []
        # Data for contour
        contour_x, contour_y = np.meshgrid(np.linspace(-1., 1.0, 50), np.linspace(-1., 1.0, 50))
        # Gegerator
        for v in values:
            # Replace the parameter with given value
            param = getattr(model, param_name)
            param[index] = v
            # Make a forward pass for the whole dataset
            y_hat = model.predict(x)
            # Calculate the loss
            if model.loss_function == "cross-entropy":
                loss = model.cross_entropy(y_hat, y) / x.shape[0]
            # Update plot
            plot_x.append(v)
            plot_y.append(loss)
            plot_xlim = (values.min(), values.max())
            plot_ylim = (0., max(plot_y))
            # Update scatter
            scatter1_xy = (model.x1 - model.x1.min(axis=0)) / (model.x1.max(axis=0) - model.x1.min(axis=0)) * 2 - 1.
            scatter1_sizes = np.array([50]*x.shape[0])
            # Update contour
            contour_z = np.ones(contour_x.shape)
            contour_xy = np.vstack((contour_x.flatten(), contour_y.flatten())).T
            contour_z = model.predict(contour_xy).reshape(contour_x.shape)
            yield (plot_x, plot_y, plot_xlim, plot_ylim, 
                   scatter1_xy, scatter1_sizes, y,
                   x, scatter1_sizes, y,
                   contour_x, contour_y, contour_z)
                
    anim = AnimatedParameter(partial(generator, model, param_name, index, values, x, y), interval=50, nb_frames=len(values),
                           scatter_kwargs={"cmap": MyColorMap()},
                           contour_kwargs={"cmap": MyColorMap()})
    return anim


