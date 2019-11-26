
###########################
# Auxilary Plot Functions #
###########################

def plot_section(a, h, b, c):

    sideDown = Rectangle((- b / 2., -(h / 2. - c)), b, c, linewidth=1, edgecolor='g', facecolor='none', alpha=0.7)
    sideUp = Rectangle((- b / 2., (h / 2.)), b, c, linewidth=1, edgecolor='g', facecolor='none', alpha=0.7)
    center = Rectangle((- a / 2., (h / 2. - c)), a, h, linewidth=1, edgecolor='g', facecolor='none', alpha=0.7)

    return sideDown, sideUp, center

def plot_profile(x, y, h):

    perimeterS1 = [] # Side 1
    perimeterS2 = [] # Side 2

    # First Point
    P = np.array([x[0], y[0], 0.0])
    V = np.array([x[0], y[0], 0.0]) - np.array([x[1], y[1], 0.0])

    V = np.cross(V, np.array([0.0, 0.0, 1.0])) / np.linalg.norm(V)

    PS1 = P + h[0] * V
    PS2 = P - h[0] * V

    perimeterS1.append(PS1[:2])
    perimeterS2.append(PS2[:2])

    # Middle Points
    for i in range(1, len(x) - 2):

        P = np.array([x[i], y[i], 0.0])

        V0 = np.array([x[i], y[i], 0.0]) - np.array([x[i - 1], y[i - 1], 0.0])
        V1 = np.array([x[i + 1], y[i + 1], 0.0]) - np.array([x[i], y[i], 0.0])
        
        V0 = np.cross(V0, np.array([0.0, 0.0, 1.0])) / np.linalg.norm(V0)
        V1 = np.cross(V1, np.array([0.0, 0.0, 1.0])) / np.linalg.norm(V1)

        V = (V0 + V1) / np.linalg.norm(V0 + V1)

        PS1 = P + h[i] * V
        PS2 = P - h[i] * V

        perimeterS1.append(PS1[:2])
        perimeterS2.append(PS2[:2])

    # Last Point
    P = np.array([x[-1], y[-1], 0.0])
    V = np.array([x[-1], y[-1], 0.0]) - np.array([x[-2], y[-2], 0.0])

    V = np.cross(V, np.array([0.0, 0.0, 1.0])) / np.linalg.norm(V)

    PS1 = P + h[-1] * V
    PS2 = P - h[-1] * V

    perimeterS1.append(PS1[:2])
    perimeterS2.append(PS2[:2])

    perimeter = perimeterS2

    return np.array(perimeter)

def plot_constrainAreas(areas):

    circles = []
    for i in range(len(rm.constrainAreas)):
        circle = plt.Circle(rm.constrainAreas[i][0:2], rm.constrainAreas[i][2], color='r', alpha=0.4)
        circles.append(circle)

def plot(beam, x, y):

    gridsize = (3,5)
    fig = plt.figure(figsize=(20,10))
    plt.axis('equal')
    axis = plt.gca()

    for i in range(beam.N):

        ax = plt.subplot2grid(gridsize, (0, i), colspan=1, rowspan=1)

        a = beam.a[i]
        h = beam.h[i]
        b = beam.b[i]
        c = beam.c[i]

        sideDown, sideUp, center = plot_section(a, h, b, c)

        axis.add_patch(sideDown)
        axis.add_patch(sideUp)
        axis.add_patch(center)
    
    ax = plt.subplot2grid(gridsize, (1, 0), colspan=5, rowspan=2)
    
    perimeter = plot_profile(x, y, beam.h)
    print(perimeter)
    ax.plot(perimeter[:, 0], perimeter[:, 1])

    plt.show()
