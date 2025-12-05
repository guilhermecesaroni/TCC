import matplotlib.pyplot as plt

def live_axes():
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    line_rpm, = ax.plot([], [], label="RPM/100")
    line_temp, = ax.plot([], [], label="Coolant °C")
    limit_temp, = ax.plot([], [], linestyle="--", label="Limite 100°C")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("valor")
    ax.legend(loc="upper left")
    return fig, ax, line_rpm, line_temp, limit_temp
