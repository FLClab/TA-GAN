
'''This module contains the functions/dialogues to interact with users.
Ex. Ask user to pick background/foreground points.
'''

import numpy

from matplotlib import pyplot, widgets, patches, rcParams
from matplotlib.widgets import Button
from mpl_toolkits.axes_grid1 import make_axes_locatable
#pyplot.ion()

try:
    import microscope_newspecpy as microscope
except:
    print("Could not load microscope interface. Some functions may not be available.")
import utils


class LinePicker:
    def __init__(self, figure):
        figure.canvas.mpl_connect("button_press_event", self.on_press)
        figure.canvas.mpl_connect("button_release_event", self.on_release)
        self.pressed = False
        self.start = []
        self.end = []

    def on_press(self, event):
        self.start.append((int(event.xdata), int(event.ydata)))
        self.pressed = True

    def on_release(self, event):
        if self.pressed:
            point = (int(event.xdata), int(event.ydata))
            if point == self.start[-1]:
                self.start.pop(-1)
                print("Pick lines, not points!")
            else:
                self.end.append(point)
            self.pressed = False

    def get_lines(self):
        if not self.pressed:
            return self.start, self.end


class SidePicker:
    def __init__(self, figure):
        figure.canvas.mpl_connect("button_press_event", self.on_press)
        figure.canvas.mpl_connect("button_release_event", self.on_release)
        self.pressed = False
        self.start = []
        self.end = []
        self.side = []

    def on_press(self, event):
        self.start.append((int(event.xdata), int(event.ydata)))
        self.side.append((int(event.xdata), int(event.ydata)))
        self.pressed = True

    def on_release(self, event):
        if self.pressed:
            point = (int(event.xdata), int(event.ydata))
            if point == self.start[-1] and self.end != []:
                self.start.pop(-1)
                print("Pick lines, not points!")
            elif point == self.start[-1] and self.end == []:
                self.side.append(point)
            else:
                self.end.append(point)
            self.pressed = False

    def get_lines(self):
        if not self.pressed:
            return self.start, self.end

    def get_point(self):
        if not self.pressed:
            return self.point



class PointPicker:
    def __init__(self, figure):
        figure.canvas.mpl_connect("button_press_event", self.on_press)
        self.points = []

    def on_press(self, event):
        self.points.append((int(event.xdata), int(event.ydata)))


class RectPicker:
    def __init__(self, figure):
        figure.canvas.mpl_connect("button_press_event", self.on_press)
        figure.canvas.mpl_connect("button_release_event", self.on_release)
        figure.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.pressed = False
        self.start = []
        self.end = []
        self.ax = pyplot.gca()
        self.rect = patches.Rectangle((0,0), 1, 1, fill=False, linewidth=2, linestyle='dashed', edgecolor="#FFFFFF")
        self.ax.add_patch(self.rect)
        self.patches = []

    def on_press(self, event):
        if event.inaxes == self.ax:
            self.start.append((int(event.xdata), int(event.ydata)))
            self.pressed = True

    def on_release(self, event):
        if self.pressed and event.inaxes == self.ax:
            point = (int(event.xdata), int(event.ydata))
            if point == self.start[-1]:
                self.start.pop(-1)
                print("Pick rectangles, not points!")
            else:
                self.end.append(point)
                for p in [ patches.Rectangle((self.start[i][0], self.start[i][1]),
                                              self.end[i][0] - self.start[i][0],
                                              self.end[i][1] - self.start[i][1],
                                              fill = False, linewidth = 2,
                                              linestyle = "dashed", edgecolor = "#FFFFFF")
                                              for i in list(range(0, len(self.start)))
                         ]:
                        self.ax.add_patch(p)
                        self.patches.append(p)
                self.pressed = False
            return self.patches

    def on_motion(self, event):
        if self.pressed and event.inaxes == self.ax:
            # print("x position : {}, y position {}".format(event.xdata, event.ydata))
            #for i in list(range(0, len(self.start))):
            i = len(self.start) - 1
            try:
                self.rect.set_width(event.xdata - self.start[i][0])
                self.rect.set_height(event.ydata - self.start[i][1])
                self.rect.set_xy((self.start[i][0], self.start[i][1]))
                self.ax.figure.canvas.draw()
            except TypeError:
                print("Oops! You went out of the image... Come back in!")

    def update(self):
        self.patches[-1].remove()
        self.ax.figure.canvas.draw()

    def get_rectangle(self):
        if not self.pressed:
            return self.start, self.end


def get_lines(img, n):
    lines = []
    lines_left = n
    while lines_left > 0:
        fig = pyplot.figure(figsize=(20, 20))
        fig.canvas.set_window_title("Pick at least {} lines".format(lines_left))
        lp = LinePicker(fig)
        # pyplot.imshow(img, interpolation=None, vmin=0, vmax=int(0.9*numpy.max(img)), picker=True)
        pyplot.imshow(img, interpolation=None, picker=True)
        pyplot.colorbar()
        pyplot.grid(True)
        pyplot.show(block=True)
        lines.extend([(s, e) for s, e in zip(*lp.get_lines())])
        lines_left -= len(lines)
        if not lines:
            print("early return in function get_lines (user.py)")
            break
    return lines


def get_side(img, n):
    lines = []
    lines_left = n
    while lines_left > 0:
        fig = pyplot.figure(figsize=(8, 8))
        fig.canvas.set_window_title("Pick at least {} lines".format(lines_left))
        lp = SidePicker(fig)
        print(lp.get_point())
        # pyplot.imshow(img, interpolation=None, vmin=0, vmax=int(0.9*numpy.max(img)), picker=True)
        pyplot.imshow(img, interpolation=None, picker=True)
        pyplot.colorbar()
        pyplot.grid(True)
        pyplot.show(block=True)
        lines.extend([(s, e) for s, e in zip(*lp.get_lines())])
        lines_left -= len(lines)
        if not lines:
            print("early return in function get_lines (user.py)")
            break
    return lines


def get_points(img, n, label="", rectangles=None):
    points = []
    points_left = n
    while points_left > 0:
        fig = pyplot.figure()
        fig.canvas.set_window_title("Pick at least {} {}".format(points_left, label))
        if rectangles is not None:
            ax = pyplot.gca()
            for p in [ patches.Rectangle((start[0], start[1]),
                                          end[0] - start[0],
                                          end[1] - start[1],
                                          fill = False, linewidth = 2,
                                          linestyle = "dashed", edgecolor = "#FFFFFF")
                                          for start, end in rectangles
                             ]:
                            ax.add_patch(p)
            string = [str(x) for x in range(len(rectangles))]
            [ax.annotate(string[itt], xy=start, xytext=start, horizontalalignment="right",
                verticalalignment="bottom", color="white") for itt, (start, end) in enumerate(rectangles)]
        pp = PointPicker(fig)
        image = pyplot.imshow(img, interpolation=None, picker=True)
        image.set_clim([0, 0.3*numpy.amax(img)])
        pyplot.colorbar()
        pyplot.grid(True)
        pyplot.show(block=True)
        points.extend(pp.points)
        points_left -= len(points)
        if not points:
            print("early return in function get_points (user.py)")
            break
    return points


def get_rectangles(img, n, label=""):
    answer = "y"
    while answer == "y":
        rect = []
        rect_left = n
        while rect_left > 0:
            rcParams['toolbar'] = "None"
            fig = pyplot.figure(figsize=(12,12))
            fig.canvas.set_window_title("Pick at least {} rectangle(s) {}.".format(rect_left, label))
            fig.subplots_adjust(bottom=0.2)
            rc = RectPicker(fig)
            image = pyplot.imshow(img, interpolation=None, picker=True)
            image.set_clim([0, 0.3*numpy.amax(img)])
            pyplot.colorbar()
            pyplot.grid(True)

            class button_action():
                def close_window(self, event):
                    pyplot.close()

                def clear_selections(self, event):
                    rc.start.pop()
                    rc.end.pop()
                    rc.update()

            callback = button_action()

            button1 = pyplot.axes([0.25, 0.02, 0.5, 0.1])
            divider = make_axes_locatable(button1)
            button2 = divider.append_axes("right", size="100%", pad=0.1)

            clear_button = Button(button1, "Clear last selection\nOnly one (1) !!")
            close_button = Button(button2, "Done!")

            clear_button.on_clicked(callback.clear_selections)
            close_button.on_clicked(callback.close_window)

            pyplot.show(block=True)
            rect.extend([(s, e) for s, e in zip(*rc.get_rectangle())])

            rect_left -= len(rect)
            if not rect:
                print("early return in function get_rectangle (user.py)")
                break
        answer = input("Would you like to restart your selections? (y) If not press enter.")
    return rect


def get_regions(n=1, label=None, rectangles=None, actin=False):
    '''Ask the user to select regions and return their offsets.
    '''
    if actin:
        config = microscope.get_config("\nSelect the STED image :)")
        img = microscope.get_image(config)
        points = get_points(img, n, " subregions within the overview {}".format(label))
    else:
        config = microscope.get_config("Setting configuration for overview")
        img = microscope.get_overview(config)
        points = get_points(img, n, " subregions within the overview {}".format(label), rectangles=rectangles)
    regions = utils.points2regions(points, microscope.get_pixelsize(config), microscope.get_resolution(config))
    x_offset, y_offset = microscope.get_offsets(config)
    regions_offset = [(x + x_offset, y + y_offset) for (x, y) in regions]
    return regions_offset


def get_rect_regions(n=1,overview=None,config_overview=None):
    '''Ask the user to select rectangle regions and returns their positions
    '''
    if overview is None:
        config = microscope.get_config("Setting configuration for overview")
        img = microscope.get_overview(config)
    else:
        config = config_overview
        img = microscope.get_overview(config_overview, name=overview)
    rectangles = get_rectangles(img, n, " subregions within the overview") # Select rectangles
    regions = utils.rect2regions(rectangles, microscope.get_pixelsize(config)) # New window size
    # points = utils.get_rect_center(rectangles, microscope.get_pixelsize(config))
    points = utils.get_rect_center(rectangles, microscope.get_pixelsize(config), microscope.get_resolution(config))
    x_offset, y_offset = microscope.get_offsets(config)
    rect_region_offset = [(x + x_offset, y + y_offset) for (x, y) in points]
    return rect_region_offset, regions, rectangles # returns the offset and the regions dimensions


def select(thetas, objectives):
    '''Return the index of the option selected by a user.
    '''
    print("Asking user to select best option...")

    fig = pyplot.figure(figsize=(10, 10))
    ax = fig.gca()

    ax.set_title("Pick the best option by clicking on the point.")

    # 3 points tolerance
    if len(objectives) > 2:
        sc = ax.scatter(thetas[0], thetas[1], s=100, c=thetas[2], marker="o", alpha=0.5, picker=3)
        pyplot.colorbar(sc, ax=ax)
    else:
        ax.scatter(thetas[0], thetas[1], s=200, marker="o", alpha=0.5, picker=3)
    ax.grid(True)
    ax.set_xlabel(objectives[0].label)
    ax.set_ylabel(objectives[1].label)

    def onpick(event):
        global index
        # handle the situation where several points overlap
        if len(objectives) > 2:
            print("Selected points:", event.ind)
            # objectives are minimized (see objectives.py)
            min_z = numpy.min(thetas[2][event.ind])
            candidates = event.ind[thetas[2][event.ind] == min_z]
        else:
            candidates = event.ind
        print("Picking at random in", candidates)
        index = numpy.random.choice(candidates)

    fig.canvas.mpl_connect("pick_event", onpick)
    pyplot.show(block=False)
    while pyplot.waitforbuttonpress():
        pass
    pyplot.close()
    assert index is not None, "User did not pick any point!"
    return index


def select_double(params1, params2, options1, options2, shared, objectives):
    '''Return the indices of options selected by a user.
    '''
    print("Asking user to select best option...")

    fig, axarr = pyplot.subplots(ncols=2, figsize=(25, 10))

    axarr[0].set_title("Pick the best option by clicking on the point:")

    xthetas = [objectives[0].as_objective(theta[0]) for theta in options1]
    ythetas = [objectives[1].as_objective(theta[1]) for theta in options1]

    # 3 points tolerance
    if len(objectives) > 2:
        zthetas = [objectives[2].as_objective(theta[2]) for theta in options1]
        if objectives[2].bounds is None:
            sc1 = axarr[0].scatter(xthetas, ythetas, s=100, c=zthetas, marker="o", alpha=0.5,
                                  vmin=0, picker=3)
        else:
            sc1 = axarr[0].scatter(xthetas, ythetas, s=100, c=zthetas, marker="o", alpha=0.5,
                                  vmin=objectives[2].bounds[0], vmax=objectives[2].bounds[1], picker=3)
        pyplot.colorbar(sc1, ax=axarr[0])
    else:
        sc1 = axarr[0].scatter(xthetas, ythetas, s=200, marker="o", alpha=0.5, picker=3, c=["r"]*len(options1))
    axarr[0].grid(True)
    axarr[0].set_xlabel(objectives[0].label_objective)
    axarr[0].set_ylabel(objectives[1].label_objective)
    if objectives[0].bounds is not None:
        axarr[0].set_xlim(objectives[0].bounds)
    if objectives[1].bounds is not None:
        axarr[0].set_ylim(objectives[1].bounds)

    xthetas2 = [objectives[0].as_objective(theta[0]) for theta in options2]
    ythetas2 = [objectives[1].as_objective(theta[1]) for theta in options2]

    axarr[1].set_title("Pick the best valid option by clicking on the point:")

    # 3 points tolerance
    sc2 = axarr[1].scatter(xthetas2, ythetas2, s=200, marker="o", alpha=0.5, picker=3, c=["r"]*len(options2))
    axarr[1].grid(True)
    axarr[1].set_xlabel(objectives[0].label_objective)
    axarr[1].set_ylabel(objectives[1].label_objective)
    if objectives[0].bounds is not None:
        axarr[1].set_xlim(objectives[0].bounds)
    if objectives[1].bounds is not None:
        axarr[1].set_ylim(objectives[1].bounds)

    params, options, scs = [params1, params2], [options1, options2], [sc1, sc2]

    global idxi
    global idxj
    global valids
    idxi, idxj, valids = None, None, None

    def onpick(event):
        global idxi
        global idxj
        global valids
        global mainax
        global axi
        global axj

        if valids is None:
            mainax = event.artist.axes
            if mainax == axarr[0]:
                axi, axj = 0, 1
            else:
                axi, axj = 1, 0

        if event.artist.axes == mainax:
            # handle the situation where several points overlap
            if len(objectives) > 2:
                print("Selected points:", event.ind)
                # objectives are minimized (see objectives.py)
                min_z = numpy.min(options[axi][:, 2][event.ind])
                candidates = event.ind[options[axi][:, 2][event.ind] == min_z]
            else:
                candidates = event.ind
            print("Picking at random in", candidates)
            idxi = numpy.random.choice(candidates)

            scs[axi]._facecolors[:] = (1, 0, 0, 0.5)
            scs[axi]._facecolors[idxi] = (0, 1, 0, 0.5)
            # for i in range(len(options[axi])):
            #     scs[axi]._facecolors[i, :] = (1, 0, 0, 0.5)
            # scs[axi]._facecolors[idxi, :] = (0, 1, 0, 0.5)

            valids = numpy.all(params[axj][:, shared] == params[axi][idxi, shared], axis=1)
            valids_idx = numpy.where(valids)[0]
            if len(valids_idx) == 1:
                idxj = valids_idx[0]

            scs[axj]._facecolors[:] = (1, 0, 0, 0.05)
            scs[axj]._facecolors[valids_idx] = (0, 1, 0, 1)
            # for i, valid in enumerate(valids):
            #     if valid:
            #         scs[axj]._facecolors[i, :] = (0, 1, 0, 0.5)
            #     else:
            #         scs[axj]._facecolors[i, :] = (1, 0, 0, 0.5)
            fig.canvas.draw()
        else:
            # handle the situation where several points overlap
            candidates = [c for c in event.ind if valids[c]]
            print("Picking at random in", candidates)
            idxj = numpy.random.choice(candidates)

            scs[axj]._facecolors[:] = (1, 0, 0, 0.05)
            scs[axj]._facecolors[idxj] = (0, 1, 0, 1)
            # for i in range(len(options[axj])):
            #     scs[axj]._facecolors[i,:] = (1, 0, 0, 0.5)
            # scs[axj]._facecolors[idxj,:] = (0, 1, 0, 0.5)
            fig.canvas.draw()

    fig.canvas.mpl_connect("pick_event", onpick)
    pyplot.show(block=True)
    assert idxi is not None and idxj is not None, "User did not pick any point!"
    if axi == 0:
        return idxi, idxj
    else:
        return idxj, idxi


def give_score(img, label):
    fig = pyplot.figure()
    fig.canvas.set_window_title("Score image")

    pyplot.subplots_adjust(left=0.25, bottom=0.25)

    # pyplot.imshow(img, interpolation=None, vmin=0, vmax=int(0.9*numpy.max(img)))
    pyplot.imshow(img, interpolation=None, vmax=int(0.9*numpy.max(img)))
    pyplot.colorbar()
    pyplot.grid(True)

    axslider = pyplot.axes([0.25, 0.1, 0.65, 0.03])
    slider = widgets.Slider(axslider, label, 0, 100, valinit=50)

    pyplot.show(block=True)
    return slider.val / 100


if __name__ == "__main__":
    import numpy
    import objectives

    objs = [objectives.Bleach(), objectives.ScoreInv("Quality")]

    grid = numpy.meshgrid(*[numpy.linspace(0, 1, 5) for i in range(3)])
    params = numpy.vstack(map(numpy.ravel, grid)).T

    options1 = numpy.random.random((len(params), 2))
    options2 = numpy.random.random((len(params), 2))

    select_double(params, params, options1, options2, objs)
