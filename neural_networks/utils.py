from functools import partial

import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rcParams
from matplotlib.colors import Colormap, to_rgba
from IPython.display import HTML
import numpy as np



BLUE = "#3498DB"
RED = "#E74C3C"
GREEN = "#2ECC71"


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

