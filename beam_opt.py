import scipy.optimize as opt
import numpy as np
from math import *
import matplotlib.pyplot as plt
from random import random as rd


class RM (object):
    
    def __init__(self):
        
        self.COEF_SECURITY = 3.
        self.COEF_SECURITY_DIST = 1.6

        self.N = 5 # First and last point are fixed and a third one is fixed for the load
        
        # state values
        self.x_start = 0
        self.y_start = self.x_start + self.N
        self.section_start = self.y_start + self.N
        
        # cte values
        self.Fx_start = 0
        self.Fy_start = self.Fx_start + self.N
        self.M_start = self.Fy_start + self.N
                        
        self.set_init_forces()
            
        self.a = 0.001  # Wide Center Bar
        self.h = 0.001  # Global Height
        self.b = 0.001  # Global Longitud
        self.c = 0.001  # Wide Side Bar
        
        self.section_accumulated = []
        self.tension_accumulated = []
        self.cost_accumulated = []
        
        self.area = 0.001
        self.I_X = 0.001
        self.I_Y = 0.001
                
        self.du = 30
        
        self.constrainAreas = [(-0.360,0.06,0.035), (-0.140,0.06,0.025)] # X, Y, R
        self.tensionLimit = 200*1e6 / float(self.COEF_SECURITY) #[MPa] 
        self.density = 0.0
        
    def init_values(self):
        self.area = self.get_area()
        self.I_X , self.I_Y = self.get_inertia()
        
    def get_area(self):
        return (self.h - 2 * self.c) * self.a + 2 * (self.c * self.b) 
        
    def get_inertia(self):
        
        I = lambda h, b: 1./12. * b * h ** 3
        
        I_X_center = I(self.h - 2*self.c, self.a)
        I_Y_center = I(self.a, self.h - 2*self.c)
        
        I_X_side = I(self.a, self.b)
        I_Y_side = I(self.b, self.a)
        
        I_X_global = I_X_center + 2*(I_X_side + (self.b * self.c) * ((self.h + self.c) / 2.)**2)
        I_Y_global = I_Y_center + 2*(I_Y_side)
        
        return I_X_global, I_Y_global
    
    def get_init_state(self):
        
        rand_values_x = [0.0] * 2
        rand_values_y = [0.0] * 2

        rand_values_x[0] = -(rd() * (0.45 - 0.21) + 0.21)
        rand_values_x[1] = -(rd() * (0.49 - abs(rand_values_x[0])) + abs(rand_values_x[0]))

        rand_values_y[0] = (rd() * (0.25 - 0.0) + 0.0)
        rand_values_y[1] = (rd() * (0.3 - rand_values_y[0]) + rand_values_y[0])

        ret = np.array([
            0.0,
            -0.175,
            rand_values_x[0],
            rand_values_x[1],
            -0.5,
            0.0,
            0.0,
            rand_values_y[0],
            rand_values_y[1],
            0.12,
            7/1000.,
            20/1000.,
            20/1000.,
            7/1000.
        ])
        
        # a, h, b, c
        
        return ret
        
    def get_boundaries(self):
        
        ret = np.zeros([self.N * 2 + 4, 2])

        for i in range(ret.shape[0]):
            ret[i, 0] = -0.5
            ret[i, 1] = 0.5
        
        ret[self.x_start, 0] = 0.0
        ret[self.x_start, 1] = 0.0

        ret[self.y_start, 0] = 0.0
        ret[self.y_start, 1] = 0.0

        ret[self.x_start + self.N - 1, 0] = -0.5
        ret[self.x_start + self.N - 1, 1] = -0.5

        ret[self.y_start + self.N - 1, 0] = 0.12
        ret[self.y_start + self.N - 1, 1] = 0.12
        
        ret[self.x_start + 1, 0] = -0.175
        ret[self.x_start + 1, 1] = -0.175

        #ret[self.y_start + 1, 0] = 0.0
        #ret[self.y_start + 1, 1] = 0.0
        
        # a, h, b, c
        ret[self.section_start, 0] = 3/1000.
        ret[self.section_start, 1] = 30/1000.
        
        ret[self.section_start + 1, 0] = 3/1000.
        ret[self.section_start + 1, 1] = 30/1000.
        
        ret[self.section_start + 2, 0] = 3/1000.
        ret[self.section_start + 2, 1] = 30/1000.
        
        ret[self.section_start + 3, 0] = 3/1000.
        ret[self.section_start + 3, 1] = 30/1000.
        
        return ret
    
    def set_init_forces(self):
        
        RB = (300.*175.) / 500.
        RA = (300.*(500.-175.)) / 500.
        F = -300
        
        self.cte_values = np.zeros(3 * self.N)
        
        self.cte_values[self.Fy_start] = RA
        self.cte_values[self.Fy_start + self.N - 1] = RB
        self.cte_values[self.Fy_start + 1] = F
        
    
    def model(self, u, i):
        
        deltaX = u[self.x_start + 1 + i] - u[self.x_start + i]
        deltaY = u[self.y_start + 1 + i] - u[self.y_start + i]
                
        alpha = atan2(deltaY, deltaX)
        N = cos(alpha) * self.cte_values[self.Fx_start + i] - sin(alpha) * self.cte_values[self.Fy_start + i]
        T = sin(alpha) * self.cte_values[self.Fx_start + i] + cos(alpha) * self.cte_values[self.Fy_start + i]
        
        M = T * sqrt(deltaX ** 2 + deltaY ** 2) + self.cte_values[self.M_start + i]
        
        tensionMaxSection = abs(N / self.area) + abs(M / self.I_X * (self.h + self.c) / 2.)
        
        self.cte_values[self.Fx_start + 1 + i] += cos(alpha) * N + sin(alpha) * T
        self.cte_values[self.Fy_start + 1 + i] += - sin(alpha) * N + cos(alpha) * T
        self.cte_values[self.M_start + 1 + i] += M
                
        return tensionMaxSection
    
        
    def constrains(self, u):
        
        self.a, self.h, self.b, self.c = u[self.section_start: self.section_start + 4]
        
        # Set initial forces, inertia and section area
        self.set_init_forces()
        self.init_values()
        
        dist = lambda x, y, x_circle, y_circle : (x - x_circle)**2 + (y - y_circle)**2
        res = []
        tension = []
                
        for i in range(self.N - 1):

            P0 = np.array([u[self.x_start + i], u[self.y_start + i]])
            P1 = np.array([u[self.x_start + 1 + i], u[self.y_start + 1 + i]])

            # No access zone:
            vector = P1 - P0
            
            for k in range(self.du):
                
                k = k / float(self.du)
                
                P = P0 + vector * k
                
                for j in range(len(self.constrainAreas)):
                    x, y, r = self.constrainAreas[j]
                    
                    constrainRadius = ((r + self.h / 2.) * self.COEF_SECURITY_DIST ) ** 2
                    distanceZone = dist(P[0], P[1], x, y)
                    
                    res.append(distanceZone - constrainRadius)

            # Limit Tension         
            tension.append(self.model(u, i))
                        
        # Area constrains
        section_area = [0.0] * 2
        section_area[0] = self.h - 2 * self.c
        section_area[1] = self.b - self.a
        section_area = np.array(section_area)
        
        tension = np.array(tension)

        self.tension_accumulated.append(tension)
        self.section_accumulated.append(u[self.section_start: self.section_start + 4])

        tension = np.ones(self.N - 1) * self.tensionLimit - abs(tension)

        res = np.array(res)
        res = np.concatenate((tension, res))
        res = np.concatenate((section_area, res))
        
        return res
                                         

    def cost_function(self, u):
                        
        self.a, self.h, self.b, self.c = u[self.section_start: self.section_start + 4]
        
        self.init_values() 
        self.set_init_forces()

        lon = 0.
        for i in range(self.N - 1):

            P0 = np.array([u[self.x_start + i], u[self.y_start + i]])
            P1 = np.array([u[self.x_start + 1 + i], u[self.y_start + 1 + i]])

            lon += np.linalg.norm(P1 - P0)
            
        self.cost_accumulated.append(lon * self.area)
        
        return lon * self.area


    def find_opt (self):

        funConstrains = {"type": "ineq", "fun": self.constrains}
        
        initSolution = self.get_init_state()
        boundaries = self.get_boundaries()
        
        res = opt.minimize(self.cost_function, initSolution, bounds = boundaries, method = 'SLSQP', constraints = funConstrains, options={'maxiter': 10000, 'ftol': 1e-06, 'iprint': 1, 'disp': False, 'eps': 1.4901161193847656e-08})

        return res
    

if __name__ == "__main__":

    optimal_cost = 100000
    for i in range(10000):
        rm = RM()
        res = rm.find_opt()
        print(i)

        if rm.cost_accumulated[-1] < optimal_cost:
            optimal_cost = rm.cost_accumulated[-1]
            print(optimal_cost)
            solutions = res

    res = solutions

    np.set_printoptions(precision=7)

    fig, ax = plt.subplots()
    plt.axis('equal')

    posx = []
    posy = []

    posx1 = []
    posy1 = []

    init = rm.get_init_state()
    for i in range(rm.N):
        x = res.x[i]
        y = res.x[rm.N + i]

        x1 = init[i]
        y1 = init[rm.N + i]    
        
        posx1.append(x1)
        posy1.append(y1)
        
        posx.append(x)
        posy.append(y)
        
    posx = np.array(posx)
    posy = np.array(posy)

    posx1 = np.array(posx1) 
    posy1 = np.array(posy1)

    for j in range(len(rm.constrainAreas)):
        circle = plt.Circle(rm.constrainAreas[j][0:2], rm.constrainAreas[j][2], color='r', alpha=0.4)
        ax.add_artist(circle)

    print "section sol"
    print res.x[rm.section_start: rm.section_start + 4]
    print "section init"
    print init[rm.section_start: rm.section_start + 4]

    print posx1, posy1
    print posx, posy
    

    plt.plot(posx, posy, "-b", linestyle="--")
    plt.plot(posx, posy + np.array([rm.h]*rm.N), "-b")
    plt.plot(posx, posy - np.array([rm.h]*rm.N), "-b")
    
    plt.scatter(posx, posy)

    plt.scatter(posx1, posy1)
    plt.plot(posx1, posy1, "-r")
    plt.show()

    rm.section_accumulated = np.array(rm.section_accumulated)
    plt.plot(range(len(rm.section_accumulated)), rm.section_accumulated)
    plt.show()

    rm.cost_accumulated = np.array(rm.cost_accumulated)
    plt.plot(range(len(rm.cost_accumulated)), rm.cost_accumulated)
    plt.show()

    rm.tension_accumulated = np.array(rm.tension_accumulated)
    plt.plot(range(len(rm.tension_accumulated)), rm.tension_accumulated[:,0])
    plt.show()

    plt.plot(range(len(rm.tension_accumulated)), rm.tension_accumulated[:,1])
    plt.show()

    plt.plot(range(len(rm.tension_accumulated)), rm.tension_accumulated[:,2])
    plt.show()

    plt.plot(range(len(rm.tension_accumulated)), rm.tension_accumulated[:,3])
    plt.show()