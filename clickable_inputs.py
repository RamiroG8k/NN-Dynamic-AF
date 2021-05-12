import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor

class ClickableInputs:
    def get_data(self):
        res = {
            "inputs": np.array(self.__inputs),
            "desired": np.array(self.__d)
        }
        return res

    def on_click(self, event):
        if event.dblclick:
            x, y = event.xdata, event.ydata
            if event.button == 1:
                plt.plot(x, y, 'bo')
                self.__d.append(0)
            elif event.button == 3:
                plt.plot(x, y, 'go')
                self.__d.append(1)
            # Round prediction of axis
            self.__inputs.append([round(x,3), round(y,3)])

    def open_clickable_inputs(self):
        self.__inputs = []
        self.__d = []
        fig, ax = plt.subplots()
        plt.ylim(top=5, bottom=-5)
        plt.xlim(left=-5, right=5)
        # plt.axhline(0, color="red")
        # plt.axvline(0, color="red")
        # cursor = Cursor(ax, horizOn=True, vertOn=True, color='white', linewidth=2.0)
        cursor = Cursor(ax, horizOn=False, vertOn=False)
        fig.canvas.mpl_connect('button_press_event', self.on_click)
        plt.grid()
        plt.show()
