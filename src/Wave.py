
'''
Ce code définit une classe appelée "Wave" qui représente une onde dans une simulation. 
'''


import numpy as np
import matplotlib.pyplot as plt
import math
import random
from matplotlib.patches import Circle



#===========================================================================
class Wave:
#===========================================================================
    '''
    Définit la classe Wave
    '''

    def __init__(self,scene):
        '''
        Définit la méthode de construction qui est appelée lorsque l'objet est créé. 
        La méthode prend deux arguments, "self" qui représente l'objet lui-même, 
        et "scene" qui représente la scène où l'onde est dessinée.
        '''
        self.v = 6000                               # Initialise la vitesse de l'onde à 6000 m/s
        self.f = 10                                 # Initialise la fréquence de l'onde à 10 Hz
        self.x0 = 0                                 # Initialise la position horizontale de la source ponctuelle de l'onde à 0.
        self.y0 = 0                                 # Initialise la position verticale de la source ponctuelle de l'onde à 0.
        self.T = 1/self.f                           # Calcule la période par défaut de l'onde à partir de sa fréquence.
        self.lambda0 = self.T * self.v              # Calcule la longueur d'onde de l'onde à partir de sa période et de sa vitesse.
        self.phase = 0                              # Initialise le dephasage de la source
        self.amplitude = 1
        self.isReflectedWave= False                 # Initialise un booléen qui indique si l'onde est réfléchie ou non.
        self.mirror=[0.5,0.5]                       # Initialise les coordonnées du miroir qui réfléchit l'onde. 
                                                    # Par défaut, l'onde est réfléchie symétriquement par rapport au centre de la scène.
        self.scene=scene                            # Initialise la scène où l'onde est dessinée.
        self.scene.waves.append(self)               # Ajoute l'onde à la liste des ondes de la scène.
        self.isLinear=False                         # Initialise un booléen qui indique si l'onde est linéaire ou non.
        self.linearAngle=0                          # Initialise l'angle de la source linéaire de l'onde (si elle est linéaire). 
                                                    # Par défaut, la source linéaire de l'onde est horizontale.
        self.isDrawRays = scene.isDrawRays          # Initialise un booléen qui indique si les rayons sont dessinés ou non, 
                                                    # en utilisant la valeur de "isDrawRays" de la scène.
        self.isDrawReflectedRays=False              # Initialise un booléen qui indique si les rayons réfléchis sont dessinés ou non.
        self.nrays= scene.nrays                     # Initialise le nombre de rayons à 25.
        self.isDrawCircles=scene.isDrawCircles      # Initialise un booléen qui indique si les cercles concentriques sont dessinés ou non.
        self.isDrawFocus=scene.isDrawFocus          # Initialise un booléen qui indique si le point focal est dessiné ou non.
        self.focusRadius = scene.focusRadius        # Initialise le rayon du point focal
        self.dashCounter = 0                        # Initialise un compteur pour les pointillés des rays sismiques
        self.isHidden = False                       # Initialise un booléen qui indique si l'onde est cachée ou non.
        self.isDrawClippedArea=False                # Initialise un booléen qui indique si la zone de clipping est dessinée ou non.
        self.delayTime = 0                          # Initialise le temps de retard de l'onde
        self.deltaT = 0                             # Initialise le temps de retard de l'onde
        self.sourceWave = None                      # Initialise la source de l'onde source à "None".
        self.isDrawSourceRay = False                # Initialise un booléen qui indique si le rayon de l'onde source est dessiné ou non.
        self.isMovableFocus = False                 # Initialise un booléen qui indique si le point focal se déplace ou non.
        self.isAttenuating = False
        self.attenuationFactor = 0
        self.isReflected = False
        self.isRefracted = False
        self.lifetime = None
        self.lifePeriods = None
        self.vrefracted = 7000
        self.makeReflected = False
        
        self.sourceCircleColor = '#1f77b4'          # bleu
        self.reflectedCircleColor = '#2ca02c'       # vert
        self.refractedCircleColor = '#d62728'       # rouge
        self.circleColor = self.sourceCircleColor
        
        if scene.randomPhase:
            phase = random.uniform (0, 360)
            self.setPhase (phase)    
    
    def displayInfo(self):
        print("\tx0            {}".format(self.x0))
        print("\ty0            {}".format(self.y0))
        print("\tv             {}".format(self.v))
        print("\tf             {}".format(self.f))
        print("\tlambda        {}".format(self.lambda0))
        print("\tphase         {}".format(self.phase * 180 / math.pi))
        print ("\tamplitude     {}".format(self.amplitude))    
        print("\tdrawRays      {}".format(self.isDrawRays))
        print("\tnRays         {}".format(self.nrays))
        print("\tdrawCircles   {}".format(self.isDrawCircles))
        print("\tdrawFocus     {}".format(self.isDrawFocus))
        print("\tfocusRadius   {}".format(self.focusRadius))
        print("\tdelayTime     {}".format(self.delayTime))
        print("\tdeltaT        {}".format(self.deltaT))
        print("\tisAttenuating {}".format(self.isAttenuating))
        print("\tattenuation   {}".format(self.attenuationFactor))
        print("\thidden        {}".format(self.isHidden))
        print("\tlifetime      {}".format(self.lifetime))
        print("\treflected     {}".format(self.isReflected))
        print("\trefracted     {}".format(self.isRefracted))
        print("\tlinear source {}".format(self.isLinear))
        print("\tlinear angle  {}".format(self.linearAngle))

    def setPhase(self,valueInDegree):
        self.phase = valueInDegree * np.pi / 180
        
    def setReflected(self):
        self.isReflected = True
        self.isRefracted = False
        self.circleColor = self.reflectedCircleColor

    def setRefracted(self):
        self.isReflected = False
        self.isRefracted = True
        self.circleColor = self.refractedCircleColor
        
    def setLifePeriods(self, nperiods):
        self.lifePeriods = nperiods
        self.lifetime = nperiods * self.T
        
    def setFrequence(self,f0):
        '''
        Cette méthode permet de définir la fréquence de l'onde en modifiant l'attribut "f". 
        Elle calcule également la période "T" et la longueur d'onde "lambda0" en fonction de la vitesse de l'onde "v".
        '''
        self.f = f0
        self.T = 1/self.f
        self.lambda0 = self.T * self.v
        
    def setAttenuation(self,factor):
        self.isAttenuating = True
        self.attenuationFactor = factor   

    def setRelativePosition (self,fx,fy):
        '''
        Cette méthode permet de définir la position relative de l'onde 
        dans la scène en modifiant les attributs "x0" et "y0". 
        Les arguments "fx" et "fy" sont des valeurs entre 0 et 1 qui indiquent 
        la position relative de l'onde dans la scène. 
        La méthode calcule les positions absolues à partir des limites de la scène 
        "xmin", "xmax", "ymin" et "ymax".
        '''
        self.x0 = self.scene.xmin + fx * (self.scene.xmax-self.scene.xmin)
        self.y0 = self.scene.ymin + fy * (self.scene.ymax-self.scene.ymin)

    def setPosition (self,x0,y0):
        '''
        Cette méthode permet de définir la position absolue 
        de l'onde en modifiant les attributs "x0" et "y0".
        '''
        self.x0 = x0
        self.y0 = y0
    
    def setMovableFocus (self,x00, y00, x01, y01):
        '''
        Cette méthode permet de définir un point focal mobile pour l'onde. 
        Elle active également le booléen "isMovableFocus" qui permet de déplacer le point focal. 
        Les arguments "x00", "y00", "x01" et "y01" sont les coordonnées des points 
        qui délimitent la zone dans laquelle le point focal peut être déplacé.        
        '''
        self.isMovableFocus = True 
        self.x00 = x00 
        self.x01 = x01
        self.y00 = y00
        self.y01 = y01
        
    def setDrawCircles(self,flag):
        self.isDrawCircles=flag    

    def setDrawFocus(self,flag):
        self.isDrawFocus=flag    
        
    def calculateMovableFocus(self,t):
        if self.isMovableFocus:
            self.x0 = self.x00 + (self.x01-self.x00) * (t-self.scene.tmin)/(self.scene.tmax-self.scene.tmin)
            self.y0 = self.y00 + (self.y01-self.y00) * (t-self.scene.tmin)/(self.scene.tmax-self.scene.tmin)
        
    def setReflectedWave(self,wave0,mirror):
        self.isReflectedWave=True
        self.mirror = mirror
        self.wave0 = wave0
        
        x0 = wave0.x0
        y0 = wave0.y0
        
        fa = mirror[0]
        fb = mirror[1]
        xa = self.scene.xmin
        ya = (self.scene.ymax - self.scene.ymin)*fa + self.scene.ymin
        xb = self.scene.xmax
        yb = (self.scene.ymax - self.scene.ymin)*fb + self.scene.ymin

        alpha = (yb-ya)/(xb-xa)

        var0 = 1 + alpha*alpha
        var1 =  x0 - alpha*alpha*x0 + 2 * alpha * y0 + 2 * alpha*alpha * xa - 2 * alpha * ya
        var2 = -y0 + alpha*alpha*y0 + 2 * alpha * x0 - 2 * alpha       * xa + 2 *         ya

        self.x0 = var1 / var0
        self.y0 = var2 / var0
        self.xa=xa
        self.xb=xb
        self.ya=ya
        self.yb=yb
        
        
            
    def drawReflectedRay(self,f):
        if self.isReflectedWave and self.isDrawReflectedRays:
            x3 = self.scene.xmin + f * (self.scene.xmax-self.scene.xmin)
            y3 = self.scene.ymin
            x0 = self.wave0.x0
            y0 = self.wave0.y0
            x1 = self.x0
            y1 = self.y0
            xa=self.xa
            ya=self.ya
            xb=self.xb
            yb=self.yb
            alpha = (y3-y1)/(x3-x1)
            beta = (yb-ya)/(xb-xa)
            x2 = ( y1 - ya - x1*alpha + xa*alpha)/(beta-alpha)            
            y2 = ( y1*beta-x1*alpha*beta-ya*alpha+xa*alpha*beta)/(beta-alpha)
            
            plt.plot ( [x0,x2], [y0,y2], 'k-')
            plt.plot ( [x1,x2], [y1,y2], 'k-')
            plt.plot ( [x2,x3], [y2,y3], 'k-')
            
    def drawReflectedRays(self):
        if self.isReflectedWave and self.isDrawReflectedRays:
            for i in range(11):
                f = i/10
                self.drawReflectedRay(f)
            
    def drawRays(self):
        if self.isDrawRays:
            self.dashCounter += 1
            if self.dashCounter % 3 == 0:
                dashes = [ 1, 3 ]
            elif self.dashCounter % 3 == 1:   
                dashes = [ 2, 2 ]    
            else:
                dashes = [ 3, 1 ]    
            if self.isLinear:
                nrays=self.nrays

                alpha = np.tan (math.pi*self.linearAngle/180)
                xa = self.scene.xmin
                xb = self.scene.xmax
                ya = self.y0 + alpha*(xa-self.x0) 
                yb = self.y0 + alpha*(xb-self.x0)

                d = np.sqrt ( (xa-xb)*(xa-xb) + (ya-yb)*(ya-yb))
                delta = d / (nrays+1)
                xmin = self.scene.xmin
                ymin = self.scene.ymin
                xmax = self.scene.xmax
                ymax = self.scene.ymax
                for k in range(1,nrays+1):
                    f = k * delta / d
                    x1 = xa + f * (xb - xa)
                    y1 = ya + f * (yb - ya)
                    y = ymin
                    x = x1 - (y-y1) * (yb-ya) / (xb-xa)
                    plt.plot ( [x1,x], [y1,y], linestyle="dotted",color='black',dashes=dashes)
                    y = ymax
                    x = x1 - (y-y1) * (yb-ya) / (xb-xa)
                    plt.plot ( [x1,x], [y1,y], linestyle="dotted",color='black',dashes=dashes)
            else:  
                nrays=self.nrays
                angle= 2*np.pi/nrays
                x0 = self.x0
                y0 = self.y0
                dx = (self.scene.xmax-self.scene.xmin)
                dy = (self.scene.ymax-self.scene.ymin)
                R = np.sqrt (dx*dx+dy*dy)
                for k in range(0,nrays+1):
                    x1 = x0 + R * np.cos(angle*k)
                    y1 = y0 + R * np.sin(angle*k)
#                    plt.plot( [x0,x1], [y0,y1], linestyle="dotted", color='black',dashes=dashes)
                    plt.plot( [x0,x1], [y0,y1],  color='black', linewidth=0.5 )

    def drawSourceRay(self):
        if self.sourceWave != None:
            x0 = self.sourceWave.x0
            y0 = self.sourceWave.y0
            x1 = self.x0
            y1 = self.y0
            plt.plot ([x0,x1], [y0,y1], color='black', linewidth=1)
            
    def setDrawRays(self,flag,nrays):
        self.isDrawRays = flag
        self.nrays = nrays        


    #----------------------------------------------------------------
    def drawCircles(self,ax,ti,beta,clipPath):
    #----------------------------------------------------------------
        '''
            Cette fonction dessine les fronts d'onde circulaire
        '''
        if self.isDrawCircles:
            alphamin = -ti*self.v / self.lambda0 
            alphamax = (self.scene.xmax-self.scene.xmin-ti*self.v)/ self.lambda0 
            alphamin = math.ceil(alphamin)
            alphamax = math.ceil(alphamax)
            for alpha in list(range(alphamin, alphamax)):
                if self.isReflected or self.isRefracted:
                    phase = -self.phase  
                else:
                    phase = self.phase    
                
                r = (ti-self.deltaT  -phase/(2*math.pi*self.f) ) * self.v + alpha*self.lambda0*beta 
                if self.scene.isTransient:
                    dmax = (ti-self.delayTime)*self.v
                    dmax += 1e-6
                    dmin = 0
                    if self.lifetime != None:
                        dmin = (ti-self.delayTime-self.lifetime) * self.v
                        dmin -= 1e-6
                        if dmin<=0:
                            dmin=0
                    if r>dmin and r<= dmax :
                        circle = Circle ( (self.x0, self.y0), r, 
                        color=self.circleColor, facecolor='none', linewidth=0.5, fill=False,
                        edgecolor='black')
                        ax.add_patch(circle)
                        if clipPath != None:
                            circle.set_clip_path (clipPath)
                    if r > dmax:
                         break    
                else:    
                    if r>0:
                        circle = Circle ( (self.x0, self.y0), r, 
                        color=self.circleColor, facecolor='none', linewidth=0.5, fill=False,
                        edgecolor='black')
                        ax.add_patch(circle)
                        if clipPath != None:
                            circle.set_clip_path (clipPath)


    def drawFocus (self):
        if self.isDrawFocus==True:
            angle = np.pi / 5
            r = self.focusRadius 
            points = np.zeros((11, 2)) 
            for i in range(10):
                if i % 2 == 0:
                    x = self.x0+r * np.cos(i * angle)
                    y = self.y0+r * np.sin(i * angle)
                else:
                    x = self.x0+r/2 * np.cos(i * angle)
                    y = self.y0+r/2 * np.sin(i * angle)
                points[i] = [x, y]
    
            points[10]=points[0]
            plt.fill(points[:,0], points[:,1],'r',zorder=3)
            plt.plot(points[:,0], points[:,1],'black',zorder=4)
            


    def setLinear(self):
            self.isLinear = True
            self.isDrawCircles = False

    def createWave(self,t):            
        X=self.scene.X
        Y=self.scene.Y
        self.calculateMovableFocus(t)
        
        A = np.ones_like (X) * self.amplitude
        
        
        if self.isLinear == False:
            if t>= self.delayTime or self.isHidden :
                D = (X-self.x0)**2 + (Y-self.y0)**2
                D = np.sqrt(D)
                if self.isAttenuating:
                    A = np.exp (-self.attenuationFactor/D)
                waveArray = A * np.sin ( 2*np.pi * ( (t-self.deltaT)/self.T - D/self.lambda0) + self.phase)
                if self.scene.isTransient:
                    dmax = (t-self.delayTime) * self.v
                    maskD = D > dmax
                    waveArray [maskD] = 0
                    
                    if self.lifetime != None:
                        dmin = (t-self.delayTime -self.lifetime) * self.v
                        maskA = D < dmin
                        waveArray [maskA] = 0
            else:
                waveArray = np.zeros_like (X)
                # waveArray = 0    
        else:
            if t>= self.delayTime or self.isHidden :
                alpha = np.tan (self.linearAngle)
                xa = self.scene.xmin
                xb = self.scene.xmax
                x0 = self.x0
                y0 = self.y0
                ya = self.y0 + math.tan(alpha)*(xa-x0) 
                yb = self.y0 + math.tan(alpha)*(xb-x0)
                    # X1 et Y1 are the projected points of X and Y onto
                    # the source plane
                X1 = ( alpha*alpha*x0 - alpha*y0 + X + alpha*Y ) / (1+alpha*alpha)
                Y1 = (-alpha*x0 + y0 + alpha*X + alpha*alpha*Y ) / (1+alpha*alpha)
                    # D is the distance between (X,Y) and (X1,Y1)
                D=(X-X1)*(X-X1)+(Y-Y1)*(Y-Y1)
                D = np.sqrt(D)
                if self.isAttenuating:
                    A = np.exp (-self.attenuationFactor/D)

                waveArray = A * np.sin ( 2*np.pi * ( (t-self.deltaT)/self.T - D/self.lambda0) + self.phase )            
            else:
                waveArray = np.zeros_like (X)
                # waveArray = 0    
            
        return waveArray

