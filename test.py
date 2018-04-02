from matplotlib.pyplot import show, plot
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
'''plt.axis([0, 10, 0, 1])
plt.ion()

for i in range(10):
    y = np.random.random()
    plt.scatter(i, y)
    plt.pause(0.1)
'''
#while True:
#    plt.pause(0.05)
'''
fo  = open('test.txt','r')
pulse=fo.read().split(' ')
del pulse[-1],pulse[0:1]
print pulse'''



fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = plt.plot([1,2], [0.5,0.6], 'ro', animated=True)

def init():
    ax.set_xlim(0, 2*np.pi)
    ax.set_ylim(-1, 1)
    return ln,

def update(frame):
    xdata.append(frame)
    ydata.append(np.sin(frame))
    ln.set_data(xdata, ydata)
    return ln,

ani = FuncAnimation(fig, update, frames=np.linspace(0, 2*np.pi, 128),
                    init_func=init, blit=True)
plt.show()
