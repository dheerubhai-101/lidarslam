import numpy as np
import math
from fractions import Fraction
import scipy.odr as sp

class featureDetection:
    def __init__(self) -> None:
        #variables
        self.epsilon = 10
        self.delta = 501
        self.Snum = 6
        self.Pmin = 20
        self.Gmax = 20
        self.SeedSegments = []
        self.LineSegments = []
        self.laserPoints = []
        self.lineParams = None
        self.Np = len(self.laserPoints) - 1
        self.Lmin = 20 # minimum length of line
        self.Lr = 0 # real length of line segment
        self.Pr = 0 


    def dist_Point2Point(self, point1, point2):
        Px2 = (point1[0] - point2[0])**2
        Py2 = (point1[1 - point2[1]])**2
        return math.sqrt(Px2 + Py2)
    

    def dist_Point2Line(self, params, point):
        a, b, c = params
        d = abs(a*point[0] + b*point[2] + c)/math.sqrt(a**2+b**2)
        return d
    
    def line_2points(self,m,b):
        x = 5
        y = m*x + b
        x2 = 2000
        y2 = m*x2 + b 
        return [(x,y),(x2,y2)]

    # general form to slope-intercept 
    def lineform_G2SI(self, a,b,c):
        m = -a/b
        B = -c/b
        return m,B
    
    def lineform_SI2G(self, m,B):
        a, b, c = -m, 1, -B
        # if a < 0:
        #     a,b,c = -a, -b, -c    
        # den_a = Fraction(a).limit_denominator(1000).as_integer_ratio()[1]
        # den_c = Fraction(c).limit_denominator(1000).as_integer_ratio()[1]

        # gcd = np.gcd(den_a, den_c)
        # lcm = den_a*den_c/gcd

        # a = a*lcm
        # b = b*lcm
        # c = c*lcm
        return a,b,c
    
    def line_intersect_general(self, params1, params2):
        a1, b1, c1 = params1
        a2, b2, c2 = params2 

        x = (b1*c2 - b2*c1)/(a1*b2 - a2*b1)
        y = (a2*c1 - a1*c2)/(a1*b2 - a2*b1)
        return x,y
    
    def points_2line(self, point1, point2):
        m, b= 0,0
        if point1[1] == point2[1]:
            m = 0
            b = point1[1]
        elif point1[0] == point2[0]:
            pass
        else:
            m = (point2[1] - point1[1])/(point2[0] - point1[0])
            b = point1[1] - m*point1[0]
        return m,b
    
    def projection_point2line(self, point, m, b):
        x, y = point
        m2 = -1/m
        b2 = y - m2*x
        x2 = (b2 - b)/(m - m2)
        y2 = m2*x2 + b2
        return x2, y2
    
    def AD2pos(self, distance, angle, robot_pos):
        x = distance*math.cos(angle) + robot_pos[0]
        y = distance*math.sin(angle) + robot_pos[1]
        return (x,y)
    
    def laser_points_set(self, data):
        self.laserPoints = []
        if not data:
            pass
        else:
            for datapoint in data:
                coordinates = self.AD2pos(datapoint[0], datapoint[1], datapoint[2])
                self.laserPoints.append([coordinates, datapoint[1]])
            self.Np = len(self.laserPoints) - 1
    
    def linear_func(self, p, x):
        m, b = p
        return m*x + b
    
    def odr_fit(self, laser_points):
        x = np.array([point[0][0] for point in laser_points])
        y = np.array([point[0][1] for point in laser_points])
        
        # model for fitting
        linear_model = sp.Model(self.linear_func)

        # RealData object from our initiated data
        data = sp.RealData(x, y)

        # Set up ODR with the model and data
        odr_model = sp.ODR(data, linear_model, beta0=[0., 0.])
        out = odr_model.run()
        m, b = out.beta
        return m, b
    
    def predict_point(self, line_params, sensed_point, robotpos):
        m, b = self.points_2line(robotpos, sensed_point)
        params1 = self.lineform_SI2G(m, b)
        predx, predy = self.line_intersect_general(params1, line_params)
        return predx, predy