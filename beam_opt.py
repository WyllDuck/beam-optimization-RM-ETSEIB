import scipy.optimize as opt
import numpy as np
from math import *
import matplotlib.pyplot as plt
from random import random as rd


class RM (object):
    
    def __init__(self):
        
        # Security Constants
        self.COEF_SECURITY = 3.
        self.COEF_SECURITY_DIST = 1.6

        self.N = 5 # First and last point are fixed and a third one is fixed on Y for the load
        
        # State Values
        self.x_start = 0
        self.y_start = self.x_start + self.N
        self.section_a_start = self.y_start + self.N
        self.section_h_start = self.section_a_start + self.N
        self.section_b_start = self.section_h_start + self.N
        self.section_c_start = self.section_b_start + self.N
        
        # Forces Values
        self.Fx_start = 0
        self.Fy_start = self.Fx_start + self.N
        self.M_start = self.Fy_start + self.N
        
        # Section Design Values
        self.a = [0.001] * self.N  # Wide Center Bar
        self.h = [0.001] * self.N  # Height
        self.b = [0.001] * self.N  # Longitud
        self.c = [0.001] * self.N  # Wide Side Bar

        self.lowerLimitSection = 2.
        self.upperLimitSection = 35.
        
        # Properties of the beam 
        self.area = [0.001] * self.N
        self.I_X  = [0.001] * self.N
        self.I_Y  = [0.001] * self.N
        
        self.tensionLimit = 200*1e6 / float(self.COEF_SECURITY) # [MPa] 
        self.density = 7850 # [kg/m**3] Steel ASTM A36
        
        # Constrain Areas
        self.du = 30
        self.constrainAreas = [(-0.360,0.06,0.035), (-0.140,0.06,0.025)] # X, Y, R


    #######################
    # Auxiliary Functions #
    #######################

    def init_values(self):

        for i in range(self.N):

            a = self.a[i]
            h = self.h[i]
            b = self.b[i]
            c = self.c[i]

            area = self.get_area(a, h, b, c)
            I_X , I_Y = self.get_inertia(a, h, b, c)

            self.area[i] = area
            self.I_X[i] = I_X
            self.I_Y[i] = I_Y
            
    def get_area(self, a, h, b, c):
        return (h - 2 * c) * a + 2 * (c * b) 
        
    def get_inertia(self, a, h, b, c):
        
        I = lambda h_, b_: 1./12. * b_ * h_ ** 3
        
        I_X_center = I(h - 2*c, a)
        I_Y_center = I(a, h - 2*c)
        
        I_X_side = I(a, b)
        I_Y_side = I(b, a)
        
        I_X_global = I_X_center + 2*(I_X_side + (b * c) * ((h + c) / 2.)**2)
        I_Y_global = I_Y_center + 2*(I_Y_side)
        
        return I_X_global, I_Y_global

    def set_init_forces(self):
        
        RB = (300.*175.) / 500.
        RA = (300.*(500.-175.)) / 500.
        F = -300
        
        self.cte_values = np.zeros(3 * self.N)
        
        self.cte_values[self.Fy_start] = RA
        self.cte_values[self.Fy_start + self.N - 1] = RB
        self.cte_values[self.Fy_start + 1] = F
        

    ##############################
    # Auxiliary Functions SOLVER #
    ##############################
    
    def get_init_state(self):
        
        # x, y
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
        ])

        # a, h, b, c
        section = np.array([0.0] * self.N * 4)
        for i in range(self.N * 4):
            section[i] = (self.upperLimitSection - self.lowerLimitSection) * rd() + self.lowerLimitSection

        ret = np.concatenate((ret, section))

        return ret
        
    def get_boundaries(self):
        
        # x, y
        ret = np.zeros([self.N * 6, 2])
        for i in range(self.N):
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
        for i in range(self.N):
            ret[self.section_a_start + i, 0] = self.lowerLimitSection / 1000.
            ret[self.section_a_start + i, 1] = self.upperLimitSection / 1000.
            
        for i in range(self.N):
            ret[self.section_h_start + i, 0] = self.lowerLimitSection / 1000.
            ret[self.section_h_start + i, 1] = self.upperLimitSection / 1000.
        
        for i in range(self.N):
            ret[self.section_b_start + i, 0] = self.lowerLimitSection / 1000.
            ret[self.section_b_start + i, 1] = self.upperLimitSection / 1000.

        for i in range(self.N):
            ret[self.section_c_start + i, 0] = self.lowerLimitSection / 1000.
            ret[self.section_c_start + i, 1] = self.upperLimitSection / 1000.
        
        return ret


    ##########
    # SOLVER #
    ##########

    def model(self, u, i):
        
        # Set the vector that describes the beam
        deltaX = u[self.x_start + 1 + i] - u[self.x_start + i]
        deltaY = u[self.y_start + 1 + i] - u[self.y_start + i]

        # Change the orientation of the forces from global to the local of the beam       
        alpha = atan2(deltaY, deltaX)
        N = cos(alpha) * self.cte_values[self.Fx_start + i] - sin(alpha) * self.cte_values[self.Fy_start + i]
        T = sin(alpha) * self.cte_values[self.Fx_start + i] + cos(alpha) * self.cte_values[self.Fy_start + i]
        
        # Get the moment of the beam at it's end
        M = T * sqrt(deltaX ** 2 + deltaY ** 2) + self.cte_values[self.M_start + i]

        # Get maximum tension of the beam generated by the moment and the normal        
        tensionMaxSection = abs(N / self.area[i + 1]) + abs(M / self.I_X[i + 1] * (self.h[i + 1] + self.c[i + 1]) / 2.)
        
        # Update the forces for the next iteration
        self.cte_values[self.Fx_start + 1 + i] += cos(alpha) * N + sin(alpha) * T
        self.cte_values[self.Fy_start + 1 + i] += - sin(alpha) * N + cos(alpha) * T
        self.cte_values[self.M_start + 1 + i] += M
                
        return tensionMaxSection
    
        
    def constrains(self, u):

        # Auxilary Fucntion
        dist = lambda x, y, x_circle, y_circle : (x - x_circle) ** 2 + (y - y_circle) ** 2

        # Save section values to get properties of the beam in each point
        self.a = u[self.section_a_start: self.section_a_start + self.N]
        self.h = u[self.section_h_start: self.section_h_start + self.N]
        self.b = u[self.section_b_start: self.section_b_start + self.N]
        self.c = u[self.section_c_start: self.section_c_start + self.N]
        
        # Set initial forces, inertia and section area
        self.set_init_forces()
        self.init_values()
        
        res = np.array([])

        areaConstrains    = []
        sectionConstrains = []
        tensionConstrains = []
                
        ### AREA CONSTRAINS (X, Y, R)
        for i in range(self.N - 1):
            
            # Determine if the straight line enters the constrain area at any point
            P0 = np.array([u[self.x_start + i], u[self.y_start + i]])
            P1 = np.array([u[self.x_start + 1 + i], u[self.y_start + 1 + i]])

            vector = P1 - P0
            vector_dh = self.h[1 + i] - self.h[i]
            
            for k in range(self.du):
                
                # Get the intermediary point P and height h
                k = k / float(self.du)
                P = P0 + vector * k
                dh = self.h[i] + vector_dh * k
                
                # Iterate through every constrain area
                for j in range(len(self.constrainAreas)):
                    x, y, r = self.constrainAreas[j]
                    
                    constrainRadius = ((r + dh / 2.) * self.COEF_SECURITY_DIST ) ** 2
                    distanceZone = dist(P[0], P[1], x, y)
                    
                    areaConstrains.append(distanceZone - constrainRadius)

        ### LIMIT TENSION
        for i in range(self.N - 1): # range(self.N - 2):
            tensionConstrains.append(self.model(u, i))

        ### SECTION CONSTRAINS
        for i in range(self.N):
            sectionConstrains.append(self.h[i] - 2 * self.c[i])
            sectionConstrains.append(self.b[i] - self.a[i])

            # h cannot be bigger than 3 times b | buckling prevention
            sectionConstrains.append(self.h[i] / self.b[i] - 3)

        # All the constrains must be positive
        sectionConstrains = np.array(sectionConstrains)
        tensionConstrains = np.array(tensionConstrains)
        tensionConstrains = np.ones(self.N - 1) * self.tensionLimit - abs(tensionConstrains)
        areaConstrains = np.array(areaConstrains)

        # Concatenate all the constrains
        res = np.concatenate((areaConstrains, res))
        res = np.concatenate((tensionConstrains, res))
        res = np.concatenate((sectionConstrains, res))
        
        return res
                                         

    def cost_function(self, u):
                        
        # Save section values to get properties of the beam in each point
        self.a = u[self.section_a_start: self.section_a_start + self.N]
        self.h = u[self.section_h_start: self.section_h_start + self.N]
        self.b = u[self.section_b_start: self.section_b_start + self.N]
        self.c = u[self.section_c_start: self.section_c_start + self.N]
        
        # Set initial forces, inertia and section area
        self.set_init_forces()
        self.init_values()

        vol = 0.
        for i in range(self.N - 1):

            P0 = np.array([u[self.x_start + i], u[self.y_start + i]])
            P1 = np.array([u[self.x_start + 1 + i], u[self.y_start + 1 + i]])

            averageArea = 0.5 * (self.area[1 + i] + self.area[i])
            vol += np.linalg.norm(P1 - P0) * averageArea
                    
        return vol * self.density


    def find_opt (self):

        funConstrains = {"type": "ineq", "fun": self.constrains}
        initSolution = self.get_init_state()
        boundaries = self.get_boundaries()
        
        res = opt.minimize(self.cost_function, initSolution, bounds = boundaries, method = 'SLSQP', constraints = funConstrains, options={'maxiter': 10000, 'ftol': 1e-06, 'iprint': 1, 'disp': False, 'eps': 1.4901161193847656e-08})

        return res
    

if __name__ == "__main__":

    optimal_cost = 100000
    for i in range(1):
        rm = RM()
        res = rm.find_opt()
        print(i)

        """
        if rm.cost_accumulated[-1] < optimal_cost:
            optimal_cost = rm.cost_accumulated[-1]
            print(optimal_cost)
            solutions = res
    
    res = solutions
    """

    np.set_printoptions(precision=7)

    fig, ax = plt.subplots()
    plt.axis('equal')

    posx = []
    posy = []

    posx1 = []
    posy1 = []

    #init = rm.get_init_state()
    for i in range(rm.N):
        x = res.x[i]
        y = res.x[rm.N + i]

        #x1 = init[i]
        #y1 = init[rm.N + i]    
        
        #posx1.append(x1)
        #posy1.append(y1)
        
        posx.append(x)
        posy.append(y)
        
    posx = np.array(posx)
    posy = np.array(posy)

    #posx1 = np.array(posx1) 
    #posy1 = np.array(posy1)

    for j in range(len(rm.constrainAreas)):
        circle = plt.Circle(rm.constrainAreas[j][0:2], rm.constrainAreas[j][2], color='r', alpha=0.4)
        ax.add_artist(circle)

    #print "section sol"
    #print res.x[rm.section_start: rm.section_start + 4]
    #print "section init"
    #print init[rm.section_start: rm.section_start + 4]

    #print posx1, posy1
    print posx, posy
    

    plt.plot(posx, posy, "-b", linestyle="--")
    plt.plot(posx, posy + np.array([rm.h]*rm.N), "-b")
    plt.plot(posx, posy - np.array([rm.h]*rm.N), "-b")
    
    plt.scatter(posx, posy)
    plt.show()


    """
    plt.scatter(posx1, posy1)
    plt.plot(posx1, posy1, "-r")
    plt.show()
    """
    
    """
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
    """