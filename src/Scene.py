import os
import sys
import math
import random
import copy
import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import ImageSequenceClip
from functools import partial
import multiprocessing
import re
from matplotlib.patches import Polygon
import matplotlib as mpl
import matplotlib.colors as mcolors
from matplotlib.colors import LightSource, Normalize
from matplotlib.path import Path

from Wave import Wave


VIEW_SOURCE       = 0b000000001
CLIP_SOURCE_ABOVE = 0b000000010
CLIP_SOURCE_BELOW = 0b000000100
VIEW_REFLEX       = 0b000001000
CLIP_REFLEX_ABOVE = 0b000010000
CLIP_REFLEX_BELOW = 0b000100000
VIEW_REFRAC       = 0b001000000
CLIP_REFRAC_ABOVE = 0b010000000
CLIP_REFRAC_BELOW = 0b100000000


#===========================================================================
class Scene:
#===========================================================================
    def __init__(self):
        self.xmin = 0
        self.xmax = 3000
        self.ymin = 0
        self.ymax = 2000 
        self.tmin = 0
        self.tmax = 2/3
        self.nx = 2000
        self.ny = 2000
        self.fps = 30
        self.na = 100
        self.mirrors = []
        self.waves = []
        self.imagesFolderPath = "images"
        self.animationsFolderPath = "animations"
        self.focusRadius = 10
        self.isDrawRays = False
        self.nrays = 10
        self.isDrawFocus = False
        self.isDrawCircles = False
        self.isParallelizing = False
        self.nProcessors = 1
        self.randomPhase = False
        self.isTransient = False
        self.hasGrid = False
        self.hasColorMap = False
        self.colorMapName = 'RdBu'
        self.colorMapIndex = 0
        self.colorBar = False
        self.colorMap = None
        self.isAttenuating = False
        self.attenuationFactor = 0
        self.hidden = False
        self.onlyFrame = False
        self.selectedFrame = None
        self.refracted = False
        self.clipped = False
        self.equalAxis = False
        self.figWidth = 10
        self.figHeight = 10
        self.nsource = 50           # Nombre de sources ponctuelles de Huygens sur les mirroirs. Valeur par défaut.
        self.normalize = False
    
        self.aboveClipPoly = None
        self.belowClipPoly = None
        self.aboveClipPath = None
        self.belowClipPath = None
        self.maskClipAbove = None
        self.maskClipBelow = None

        
            # data for the source plane
        self.x0 = self.xmin
        self.y0 = self.ymin
        self.angle0 = 0
        self.length0 = self.xmax-self.xmin
        
        self.eraseImagesFolder()

        
    def setBounds(self,xmin,xmax,ymin,ymax):    
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        
    def prepare(self):    
        self.x = np.linspace(self.xmin, self.xmax, self.nx)
        self.y = np.linspace(self.ymin, self.ymax, self.ny)
        self.X, self.Y = np.meshgrid (self.x,self.y)
        self.na = round((self.tmax-self.tmin)*(10*self.fps))
        width = self.xmax - self.xmin
        height = self.ymax - self.ymin 
        self.figWidth = self.nx / 100
        self.figHeight = self.figWidth * height / width 
        self.points = np.column_stack (( self.X.ravel(), self.Y.ravel() ))
        
    def displayInfo(self):
        print ("nx            {}".format(self.nx))    
        print ("ny            {}".format(self.ny))    
        print ("na            {}".format(self.na))    
        print ("xmin          {}".format(self.xmin))    
        print ("xmax          {}".format(self.xmax))    
        print ("ymin          {}".format(self.ymin))    
        print ("ymax          {}".format(self.ymax))    
        print ("tmin          {}".format(self.tmin))    
        print ("tmax          {}".format(self.tmax))    
        print ("Draw rays     {}".format(self.isDrawRays))
        print ("Draw circles  {}".format(self.isDrawCircles))
        print ("Draw focus    {}".format(self.isDrawFocus))
        print ("n rays        {}".format(self.nrays))
        print ("Parallelizing {}".format(self.isParallelizing))
        print ("Transient     {}".format(self.isTransient))
        print ("Refracted     {}".format(self.refracted))
        print ("Clipped       {}".format(self.clipped))
        print("\tNormalisze   {}".format(self.normalize))

        k = 1
        for wave in self.waves:
            print ("Wave {}".format(k))
            wave.displayInfo()
            k += 1

    def setAttenuation(self,factor):
        self.isAttenuating = True
        for wave in self.waves:
            wave.setAttenuation(factor)
                        
    #---------------------------------------------------------------------------
    def eraseImagesFolder(self):
    #---------------------------------------------------------------------------
        if not os.path.exists(self.imagesFolderPath):
            os.makedirs(self.imagesFolderPath)
    
        files = os.listdir(self.imagesFolderPath)
        for file in files:
            filePath = os.path.join(self.imagesFolderPath, file)
            if os.path.exists(filePath):
                os.remove(filePath)

    #---------------------------------------------------------------------------
    def saveAnimation(self):
    #---------------------------------------------------------------------------
        
        if self.onlyFrame == False:
            if os.path.exists(self.imagesFolderPath + '/' + ".DS_Store"):
                os.remove(self.imagesFolderPath + '/' + ".DS_Store")
            
            if not os.path.exists(self.animationsFolderPath):
                os.makedirs(self.animationsFolderPath)

            clip = ImageSequenceClip(self.imagesFolderPath, fps=30)
            filename = self.animationsFolderPath + '/' + self.filename + '.mp4'
            clip.write_videofile(filename)

    #---------------------------------------------------------------------------
    def buildMasks(self,ax):
    #---------------------------------------------------------------------------
        if len(self.mirrors)>0:
            mirror = self.mirrors[0]
            
            self.aboveClipPoly = self.getClipArea('above',ax)
            self.belowClipPoly = self.getClipArea('below',ax)
            self.aboveClipPath = Path (self.aboveClipPoly.get_xy())
            self.belowClipPath = Path (self.belowClipPoly.get_xy())
            self.maskClipAbove = self.aboveClipPath.contains_points(self.points)
            self.maskClipBelow = self.belowClipPath.contains_points(self.points)
        
        

    #---------------------------------------------------------------------------
    def segmentMirrors (self):
    #---------------------------------------------------------------------------
        '''
                Cette fonction segmente les miroirs en (ns-1) segments
        '''
        if len(self.mirrors) > 0:
            for mirror in self.mirrors:
                mirror ["segmented"] = []
                ns = mirror["nsource"] if "nsource" in mirror else self.nsource
                if "fa" in mirror:
                    fa = mirror["fa"]
                    fb = mirror["fb"]
                    xa1 = self.xmin
                    ya1 = (self.ymax - self.ymin)*fa + self.ymin
                    xb1 = self.xmax
                    yb1 = (self.ymax - self.ymin)*fb + self.ymin
                    
                    for k in range(0,ns):
                        x1 = xa1 + (xb1-xa1) * k / (ns-1)
                        y1 = ya1 + (yb1-ya1) * k / (ns-1)
                        mirror["segmented"].append([x1,y1])
                elif "points" in mirror:
                    d = 0
                    points = mirror["points"]
                    point0 = points[0]
                    x0 = point0[0]
                    y0 = point0[1]
                    for point in points :
                        x1 = point[0]
                        y1 = point[1]
                        d += np.sqrt((x1-x0)*(x1-x0)+(y1-y0))
                        x0 = x1
                        y0 = y1
                    dseg = d / ns
                    
                    x0 = points[0][0]
                    y0 = points[0][1]
                    d0 = 0
                    mirror["segmented"].append ([x0,y0])
                    for point in points:
                        x1 = point[0]
                        y1 = point[1]
                        d = np.sqrt ((x1-x0)*(x1-x0)+(y1-y0)*(y1-y0))
                        while d0+d > dseg:
                            alpha = (dseg-d0)/d
                            xx = x0 + alpha * (x1 - x0)         
                            yy = y0 + alpha * (y1 - y0)
                            mirror["segmented"].append ([xx,yy])
                            d -= (dseg-d0)
                            d0 = 0
                            x0 = xx
                            y0 = yy
                        else:
                            d0 += d
                        x0 = x1
                        y0 = y1    
                    x0 = points[-1][0]
                    y0 = points[-1][1]
                    mirror["segmented"].append ([x0,y0])


    #-----------------------------------------------------------------------------------
    def buildDiscreteLinearSource(self, x0, y0, alpha, nsource, length, v, f, phase ):
    #-----------------------------------------------------------------------------------
        '''  Cette fonction crée une source discrète linéaire
             constituées de 'nsource' sources ponctuelles
             d'ondes synchrones de même fréquence 'f' et
             se propageant à la même vitesse 'v'
             
             "alpha" est l'angle fait par la source linéaire par rapport à l'horizontale
             "length" est la longueur de la source linéaire discrète
             "x0" et "y0" sont les coordonnées du milieu de la source linéaire
        '''
        cosalpha = np.cos(alpha)
        sinalpha = np.sin(alpha)
        r = length/2
        x1 = x0 - r * cosalpha
        y1 = y0 - r * sinalpha
        x2 = x0 + r * cosalpha
        y2 = y0 + r * sinalpha
        deltax = (x2-x1) / (nsource-1)
        deltay = (y2-y1) / (nsource-1)
        
        wavesList = []
        listk = range(0,nsource)
        for k in listk:
            x = x1 + deltax * k
            y = y1 + deltay * k
            wave = Wave (self)
            wave.setPosition (x,y)
            wave.v = v
            wave.vrefracted = v
            wave.setFrequence(f)
            wave.setPhase(phase)
            wavesList.append(wave)
        
        return wavesList    


    #---------------------------------------------------------------------------
    def calculateRupturePropagation(self, wavesList, vrupture):
    #---------------------------------------------------------------------------
        first = self.waves[0]
        x0 = first.x0
        y0 = first.y0
        x1 = x0
        y1 = y0
        d = np.sqrt ( (x1-x0)*(x1-x0) + (y1-y0)*(y1-y0) )
        i = 1
        delayTime0 = 0
        for wave in wavesList:
            x1 = wave.x0
            y1 = wave.y0
            waved = np.sqrt ( (x1-x0)*(x1-x0) + (y1-y0)*(y1-y0) )
            wave.delayTime = delayTime0 + waved / vrupture
            phase = wave.delayTime / wave.T
            phase = math.fmod (phase, 1) 
            wave.setPhase(-360*phase)
            delayTime0 = wave.delayTime
            print (f"Wave {i} \t {wave.delayTime} \t {phase*360}")
            x0 = x1
            y0 = y1
            i = i + 1

    
    #---------------------------------------------------------------------------
    def setProgressiveDephasing(self, wavesList, nw):
    #---------------------------------------------------------------------------
        '''
            Cette fonction fixe un décalage de nw périodes entre le 1er point
            et le dernier point d'une rangée de sources ponctuelles alignées
        '''
        first = self.waves[0]
        last = self.waves[-1]
        x0 = first.x0
        y0 = first.y0
        x1 = last.x0
        y1 = last.y0
        d = np.sqrt ( (x1-x0)*(x1-x0) + (y1-y0)*(y1-y0) )
        i = 1
        for wave in wavesList:
            waved = np.sqrt ( (wave.x0-x0)*(wave.x0-x0) + (wave.y0-y0)*(wave.y0-y0) )
            alpha = waved/d
            wave.deltaT =  nw*alpha*wave.T
            print (f"Wave {i} \t {alpha} \t {wave.deltaT}")
            i = i + 1
        
    
    
    def setDrawRays(self,flag,nrays):
        self.isDrawRays = flag
        self.nrays = nrays
        for wave in self.waves:
            wave.setDrawRays(flag,nrays)
    
    def setSourcePlane(self,x0,y0,length0,angle0):        
        self.x0 = x0
        self.y0 = y0
        self.length0 = length0
        self.angle0 = angle0
        
    def setDrawCircles(self,flag):
        for wave in self.waves:
            wave.setDrawCircles(flag)    
            
    def setDrawFocus(self,flag):
        for wave in self.waves:
            wave.setDrawFocus(flag)    
            
    #---------------------------------------------------------------------------
    def calculateProjectionPointOnLine(self,x1, y1, xa, ya, xb, yb):
    #---------------------------------------------------------------------------
        # Calcul du vecteur direction de la droite
        v = (xb - xa, yb - ya)

        # Calcul du vecteur qui relie le point (x0, y0) à l'un des points sur la droite
        u = (x1 - xa, y1 - ya)

        # Calcul de la projection de u sur v
        proj = (u[0]*v[0] + u[1]*v[1]) / (v[0]**2 + v[1]**2)
        proj_v = (proj*v[0], proj*v[1])

        # Calcul des coordonnées de la projection
        xp = xa + proj_v[0]
        yp = ya + proj_v[1]

        return xp, yp
    
    #---------------------------------------------------------------------------
    def drawMirrors(self):
    #---------------------------------------------------------------------------
        for mirror in self.mirrors:
            if "points" in mirror:
                points = mirror["points"]
                xm = []
                ym = []
                for point in points:
                    xm.append ( point[0] )
                    ym.append ( point[1] )
                plt.plot  ( xm, ym, color='black', linewidth=2 )   
                    
            elif "fa" in mirror:
                fa = mirror["fa"]
                fb = mirror["fb"]
                xa = self.xmin
                ya = (self.ymax - self.ymin)*fa + self.ymin
                xb = self.xmax
                yb = (self.ymax - self.ymin)*fb + self.ymin
                plt.plot ([xa,xb], [ya,yb], color='black', linewidth=2)
                

    #---------------------------------------------------------------------------
    def addWavesSourcesOnMirror(self):
    #---------------------------------------------------------------------------
        '''
        Cette fonction crée les sources ponctuelles de Huygens sur les miroirs
        '''
    
        if len(self.mirrors) > 0:
            for mirror in self.mirrors:
                
                for wave in self.waves:
                    if wave.makeReflected == True or wave.makeRefracted:
                        return 
                    # if wave.isReflected == False and wave.isRefracted == False :
                    #     wave.isHidden = True
                
                waves = copy.copy(self.waves)
                ns = mirror["nsource"] if "nsource" in mirror else self.nsource
                
                for wave in waves:
#                    if wave.isReflected == False and wave.isRefracted == False and wave.makeReflected == False :
                    if wave.isReflectedWave == False and wave.isRefractedWave == False and wave.makeReflected == False and wave.makeRefracted == False:
                        x0 = wave.x0
                        y0 = wave.y0
                        segmentationPoints = mirror["segmented"]
                        for point in segmentationPoints:
                            x1 = point[0]
                            y1 = point[1]
                        
                            if wave.isLinear == False:
                                d1 = np.sqrt( (x1-x0)*(x1-x0) + (y1-y0)*(y1-y0))
                            
                            else: 
                                alpha = math.tan(wave.linearAngle)
                                xa = self.xmin
                                xb = self.xmax
                                x0 = wave.x0
                                y0 = wave.y0
                                ya = self.y0 + alpha*(xa-x0) 
                                yb = self.y0 + alpha*(xb-x0)
                                # xp et yp are the projected points of x1 and y1 onto
                                # the source plane
                                # (xp,yp) = self.calculateProjectionPointOnLine(x1, y1, xa, ya, xb, yb)
                                
                                xp = ( alpha*alpha*x0 - alpha*y0 + x1 + alpha*y1 ) / (1+alpha*alpha)
                                yp = (-alpha*x0 + y0 + alpha*x1 + alpha*alpha*y1 ) / (1+alpha*alpha)
                                # d is the distance between (xp,yp) and (x1,y1)
                                d1=(xp-x1)*(xp-x1)+(yp-y1)*(yp-y1)
                                d1 = np.sqrt(d1)
                                       
                            phase1 = d1 / wave.lambda0
                            phase1 = math.fmod( phase1, 1)
                        
                            wave2 = Wave(self)
                            wave2.amplitude = 1 / ns
                            wave2.setPosition (x1,y1)
                            wave2.v = wave.vrefracted
                            wave2.setFrequence (wave.f)
                            wave2.setPhase(-360*phase1)
                            wave2.setRefracted()
                            wave2.isOnMirror = True
                            wave2.lifetime = wave.lifetime
                            wave2.viewOptions = wave.viewOptions
                            if self.isTransient:
                                wave2.delayTime = d1 / wave.v 
                            
                            wave1 = Wave(self)
                            wave1.amplitude = 1 / ns
                            wave1.setPosition (x1,y1)
                            wave1.v = wave.v
                            wave1.setFrequence (wave.f)
                            wave1.setPhase(-360*phase1)
                            wave1.setReflected()
                            wave1.lifetime = wave.lifetime
                            wave1.isOnMirror = True
                            wave1.viewOptions = wave.viewOptions
                            if self.isTransient:
                                wave1.delayTime = d1 / wave1.v 
                                
                            if wave.isLinear:
                                wave1.isSourceLinear = True
                                wave1.xp = xp
                                wave1.yp = yp    
                                    
                            
                            
    #---------------------------------------------------------------------------
    def addReflectedSources(self):
    #---------------------------------------------------------------------------
        if len(self.mirrors)>0:
            mirror=self.mirrors[0]
            if not "fa" in mirror:
                return
            if not "fb" in mirror:
                return
            fa=mirror['fa']
            fb=mirror['fb']
            xa = self.xmin
            ya = (self.ymax - self.ymin)*fa + self.ymin
            xb = self.xmax
            yb = (self.ymax - self.ymin)*fb + self.ymin
            alpha1 = (yb-ya)/(xb-xa)
            waves = copy.copy(self.waves)
            angle1 = math.tan (alpha1)

            i = 0
            for wave0 in waves:
                if wave0.makeReflected :
                    x0=wave0.x0
                    y0=wave0.y0
                    (xp,yp) = self.calculateProjectionPointOnLine (x0, y0, xa, ya, xb, yb)
                    
                    x1 = 2 * xp - x0
                    y1 = 2 * yp - y0
                    
                    wave1 = Wave(self)
                    wave1.amplitude = wave0.amplitude
                    wave1.setPosition (x1,y1)
                    wave1.setReflected()
                    wave1.v = wave0.v
                    wave1.setFrequence (wave0.f)
                    wave1.phase = wave0.phase
                    wave1.lifetime = wave0.lifetime
                    wave1.viewOptions = wave0.viewOptions
                    if self.isTransient:
                         wave1.delayTime = wave0.delayTime
                        
                    if wave0.isLinear:
                         angle0 = wave0.linearAngle
                         angle2 = angle1 - angle0
                         wave1.isLinear = True
                         wave1.linearAngle = angle2
                
                elif wave0.makeRefracted:                        
                    x0=wave0.x0
                    y0=wave0.y0
                    (xp,yp) = self.calculateProjectionPointOnLine (x0, y0, xa, ya, xb, yb)
                    
                    alpha = wave0.v / wave0.vrefracted 
                    
                    x1 = xp + (x0-xp) * alpha
                    y1 = yp + (y0-yp) * alpha

                    wave1 = Wave(self)
                    wave1.amplitude = wave0.amplitude
                    wave1.setPosition (x1,y1)
                    wave1.setRefracted()
                    wave1.v = wave0.vrefracted
                    wave1.vrefracted = wave0.v
                    wave1.setFrequence (wave0.f)
                    wave1.phase = wave0.phase
                    wave1.lifetime = wave0.lifetime
                    wave1.viewOptions = wave0.viewOptions
                    if self.isTransient:
                        wave1.delayTime = wave0.delayTime
                        
                    if wave0.isLinear:
                        angle0 = wave0.linearAngle
                        angle2 = angle1 + angle0
                        
                        sina = np.sin(angle0) * wave0.vrefracted/wave0.v
                        if abs(sina) <= 1.0:
                            angle2 = math.asin (sina)
                        
                        
                        wave1.isLinear = True
                        wave1.linearAngle = angle2 
                        wave1.phase = wave0.phaser

    #---------------------------------------------------------------------------
    def sumWavesArray (self,waveArray1,waveArray2):
    #---------------------------------------------------------------------------
        if len(waveArray1) == 0:
            return waveArray2
        if len(waveArray2) == 0:
            return waveArray1

        sumArray = []
        for a,b in zip(waveArray1,waveArray2):
            sumArray . append (a+b)
    
        return sumArray
    

    #---------------------------------------------------------------------------
    def getClipArea(self,flag,ax):  
    #---------------------------------------------------------------------------

        if len(self.mirrors) > 0:
            mirror = self.mirrors[0]
            if "fa" in mirror:
                fa = mirror['fa']
                fb = mirror['fb'] 
                xa = self.xmin
                ya = (self.ymax - self.ymin)*fa + self.ymin
                xb = self.xmax
                yb = (self.ymax - self.ymin)*fb + self.ymin

                if flag=='above':
                    x = [xa, self.xmin, self.xmax, xb, xa ]
                    y = [ya, self.ymin, self.ymin, yb, ya ]
                elif flag=='below':
                    x = [xa, self.xmin, self.xmax, xb, xa ]
                    y = [ya, self.ymax, self.ymax, yb, ya ]
            elif "points" in mirror:
                x = []
                y = []
                points = mirror["points"]
                for point in points:
                    xp = point[0]
                    yp = point[1]
                    x.append (xp)
                    y.append (yp)
                if flag=='above':
                    x.append (self.xmax,)
                    y.append (self.ymin)    
                    x.append (self.xmin)
                    y.append (self.ymin)    
                    point=points[0]
                    x.append (point[0])
                    y.append (point[1])
                elif flag=='below':
                    x.append (self.xmax)
                    y.append (self.ymax)    
                    x.append (self.xmin)
                    y.append (self.ymax)    
                    point=points[0]
                    x.append (point[0])
                    y.append (point[1])

            poly = Polygon ( list(zip(x,y)),transform=ax.transData)
            # poly = Polygon ( list(zip(x,y)) )
            return poly

    #---------------------------------------------------------------------------
    def buildColorMap(self):
    #---------------------------------------------------------------------------
        self.colorMap = plt.get_cmap(self.colorMapName)
        self.colorMap.set_bad(color='white')

        
    #---------------------------------------------------------------------------
    def createAnimationFrameImage(self,  ti, i):
    #---------------------------------------------------------------------------
        '''
        Cette méthode crée une image pour la séquence d'animation à l'aide
        de la bibliothèque Matplotlib.
            ti est le temps
            i est l'indice de l'image    
        '''
        fig, ax = plt.subplots(figsize=(self.figWidth,self.figHeight))
        if self.equalAxis:
            plt.axis('equal')
        ax.set_xlim(self.xmin,self.xmax)
        ax.set_ylim(self.ymin,self.ymax)
        ax.invert_yaxis()
        
        self.buildMasks(ax)
        
        waves = self.waves
        sourceWaveArray0 = []       # all view
        sourceWaveArray1 = []       # view above mirror 
        sourceWaveArray2 = []       # view below mirror
        reflectedWaveArray = []
        refractedWaveArray = []
        n = 0
        maxn = 0
        for wave in waves:
            if wave.isHidden == False:
                waveArray0 = wave.createWave (ti)
                # if wave.isReflected:
                #     reflectedWaveArray  = self.sumWavesArray (reflectedWaveArray, waveArray0)
                # elif wave.isRefracted:                    
                #     refractedWaveArray  = self.sumWavesArray (refractedWaveArray, waveArray0)
                # else:
                maxn += wave.amplitude
                if wave.viewOptions != 0:
                    if wave.isReflectedWave:
                        if wave.viewOptions & CLIP_REFLEX_ABOVE:
                            sourceWaveArray1 = self.sumWavesArray (sourceWaveArray1, waveArray0)    
                        if wave.viewOptions & CLIP_REFLEX_BELOW:
                            sourceWaveArray2 = self.sumWavesArray (sourceWaveArray2, waveArray0)    
                        if wave.viewOptions & VIEW_REFLEX :
                            sourceWaveArray0 = self.sumWavesArray (sourceWaveArray0, waveArray0)    
                    elif wave.isRefractedWave:
                        if wave.viewOptions & CLIP_REFRAC_ABOVE:
                            sourceWaveArray1 = self.sumWavesArray (sourceWaveArray1, waveArray0)    
                        if wave.viewOptions & CLIP_REFRAC_BELOW:
                            sourceWaveArray2 = self.sumWavesArray (sourceWaveArray2, waveArray0)    
                        if wave.viewOptions & VIEW_REFRAC :
                            sourceWaveArray0 = self.sumWavesArray (sourceWaveArray0, waveArray0)    
                    else:
                        if wave.viewOptions & CLIP_SOURCE_ABOVE:
                            sourceWaveArray1 = self.sumWavesArray (sourceWaveArray1, waveArray0)    
                        if wave.viewOptions & CLIP_SOURCE_BELOW:
                            sourceWaveArray2 = self.sumWavesArray (sourceWaveArray2, waveArray0)    
                        if wave.viewOptions & VIEW_SOURCE :
                            sourceWaveArray0 = self.sumWavesArray (sourceWaveArray0, waveArray0)    
                else:    
                    sourceWaveArray0 = self.sumWavesArray (sourceWaveArray0, waveArray0)    
            else:
                if wave.isReflectedWave == False and wave.isRefractedWave == False:
                    maxn += wave.amplitude

        min00 = min01 = min02 = max00 = max01 = max02 = 0
        if len(sourceWaveArray0) > 0:
            min00 = np.amin(sourceWaveArray0)
            max00 = np.amax(sourceWaveArray0)
        if len(sourceWaveArray1) > 0:
            min01 = np.amin(sourceWaveArray1)
            max01 = np.amax(sourceWaveArray1)
        if len(sourceWaveArray2) > 0:
            min02 = np.amin(sourceWaveArray2)
            max02 = np.amax(sourceWaveArray2)

        mintot = abs(min00) + abs(min01) + abs(min02)
        maxtot = max00 + max01 +  max02 
        if self.normalize:    
            norm = Normalize (vmin=-maxtot, vmax=maxtot)
        else:
            norm = 'linear'
      

        imageData0 = []
        imageData1 = []
        imageData2 = []
        finalImage = []

        if len(sourceWaveArray0) > 0:
            image = ax.imshow(sourceWaveArray0, extent=(self.xmin, self.xmax, self.ymin, self.ymax), cmap=self.colorMap, origin="lower",  norm=norm)
            imageData0 = image.get_array().data
            finalImage = imageData0

        if len(sourceWaveArray1) > 0:
            image = ax.imshow(sourceWaveArray1, extent=(self.xmin, self.xmax, self.ymin, self.ymax), cmap=self.colorMap, origin="lower",  norm=norm)
            
            imageData = image.get_array().data
            imageMask = np.ma.masked_array(imageData, ~self.maskClipAbove)
            imageData1 = np.ma.filled (imageMask, fill_value=0)

            if len(finalImage)==0:
                finalImage = imageData1
            else:    
                finalImage += imageData1

        if len(sourceWaveArray2) > 0:
            image = ax.imshow(sourceWaveArray2, extent=(self.xmin, self.xmax, self.ymin, self.ymax), cmap=self.colorMap, origin="lower",  norm=norm)

            imageData = image.get_array().data
            imageMask = np.ma.masked_array(imageData, ~self.maskClipBelow)
            imageData2 = np.ma.filled (imageMask, fill_value=0)
            if len(finalImage)==0:
                finalImage = imageData2
            else:    
                finalImage += imageData2

        if len(finalImage)>0:                                
            ax.imshow (finalImage, extent=(self.xmin, self.xmax, self.ymin, self.ymax), cmap=self.colorMap, origin="lower", norm=norm)       
            
        if self.colorBar:
            cp = ax.get_children()
            cp2 = cp[0]
            plt.colorbar( cp2, ax=ax)
        
    
        for wave in waves:
            if ti >= wave.delayTime:
                if wave.isRefractedWave:
                    wave.drawRays()
                    # wave.drawSourceRay()
                    # wave.drawReflectedRays ()
                    wave.drawCircles(ax, ti, 1.0, self.belowClipPoly)
                    wave.drawFocus()
        
        
        for wave in waves:
            if ti >= wave.delayTime:
                if wave.isReflectedWave:
                    wave.drawRays()
                    # wave.drawSourceRay()
                    # wave.drawReflectedRays ()
                    wave.drawCircles(ax, ti, 1.0, self.aboveClipPoly)
                    wave.drawFocus()

        for wave in waves:
            if ti >= wave.delayTime:
                if wave.isReflectedWave == False and wave.isRefractedWave == False :
                    wave.drawRays()
                    # wave.drawSourceRay()
                    # wave.drawReflectedRays ()
                    wave.drawCircles(ax, ti, 1.0, None)
                    wave.drawFocus()


        self.drawMirrors()

        rx = (self.xmax-self.xmin)*0.2
        ry = (self.ymax-self.ymin)*0.05
        rect = plt.Rectangle( (0,0), rx, ry, fc='#444444',alpha=0.8)
        ax.add_patch (rect)
        ax.text(rx*0.5,ry*0.5, "{} s".format(round(ti,4)), ha="center", va="center",color="white")

        if self.hasGrid:
            ax.grid(True, color='r', linestyle='--', linewidth=0.5)


    
        fig.savefig("{}/image{:04d}.png".format(self.imagesFolderPath,i))
        plt.close(fig)
        

    #---------------------------------------------------------------------------
    def buildAnimation(self):
    #---------------------------------------------------------------------------
        if self.isParallelizing:
            numProcesses = self.nProcessors
            pool = multiprocessing.Pool(numProcesses)
            taskFunc = partial (self.buildAnimationTask)
            taskArgs = range(self.na)
            results = pool.map ( taskFunc, taskArgs)
            pool.close()
            pool.join()
            
        else:
            if self.onlyFrame:
                ti = self.tmin + self.selectedFrame / (10 * self.fps )
                print ("{} \t {}".format(self.selectedFrame,ti))
                self.createAnimationFrameImage( ti, self.selectedFrame)
            else:        
                for i in range(self.na):
                    ti = self.tmin + i / (10 * self.fps )
                    print ("{} \t {}".format(i,ti))
                    self.createAnimationFrameImage( ti, i)

        self.saveAnimation ()
        

    #---------------------------------------------------------------------------
    def buildAnimationTask(self,i):
    #---------------------------------------------------------------------------
        if self.onlyFrame:
            duplicatedScene = copy.deepcopy(self)
            if i == duplicatedScene.selectedFrame:
                ti = duplicatedScene.tmin + duplicatedScene.selectedFrame / (10 * self.fps )
                print ("{} \t {}".format(duplicatedScene.selectedFrame,ti))
                duplicatedScene.createAnimationFrameImage( ti, i)
        else:    
            duplicatedScene = copy.deepcopy(self)
            ti = duplicatedScene.tmin + i / (10 * self.fps )
            print (i,ti)
            duplicatedScene.createAnimationFrameImage( ti, i)
        
        
    #---------------------------------------------------------------------------
    def buildHuygens(self,ns):
    #---------------------------------------------------------------------------
        angle0 = math.pi*self.angle0/180
        r = self.length0 / 2
        
        x0 = self.x0
        y0 = self.y0
        xa = x0 - r * math.cos(angle0)
        ya = y0 - r * math.sin(angle0)
        xb = x0 + r * math.cos(angle0)
        yb = y0 + r * math.sin(angle0)
        
        
        # xa = self.xmin
        # xb = self.xmax
        # ya = y0 + alpha*(xa-x0) 
        # yb = y0 + alpha*(xb-x0)
        
        # if ya<self.ymin or ya>self.ymax or yb<self.ymin or yb>self.ymax:
        #     if self.angle0 >= 0:
        #         ya=self.ymin
        #         yb=self.ymax
        #     else:    
        #         yb=self.ymin
        #         ya=self.ymax
        #     xa = (ya-y0)/alpha + x0
        #     xb = (yb-y0)/alpha + x0
        
        for i in range(0,ns):
            x = xa+i/ns*(xb-xa)
            y = ya+i/ns*(yb-ya)
            wave = Wave(self)
            wave.linearAngle=self.angle0    
            wave.setPosition(x,y)
            wave.isDrawCircles=False
            wave.isDrawRays=False


            

    #---------------------------------------------------------------------------
    def parseFile(self, fileName):
    #---------------------------------------------------------------------------
        with open ("scripts/"+fileName+".txt", "r") as file:
            fileContent = file.read()

        lines = fileContent.strip().split("\n")
        data = {}
        current_section = None
        for line in lines:
            line = line.strip()
            index = line.find('#')
            if index != -1:
                line = line [:index]
                line.strip()

            if line == '':
                continue
            
            tokens = line.split()
            
            if tokens[0] in ( "xmin", "xmax", "ymin", "ymax", "parallel", "nx", "ny", "na", "fps", "frame" ):
                key = tokens[0]
                data[key] = int ( tokens[1] )
                current_section = None
            elif tokens[0] in ( "tmin", "tmax" ):
                key = tokens[0]
                data[key] = float(eval(tokens[1]))
                current_section = None
            elif tokens[0] in ( "randomPhase", "transient", "grid",  "refracted", "equalAxis", "colorBar", "normalize" ):
                key = tokens[0]
                data[key] = True
                current_section = None
            
            
            elif tokens[0] == 'colorMap':
                self.hasColorMap = True
                ntokens = len (tokens)
                if ntokens == 1:
                    data["colorMapName"] = 'RdBu'
                elif ntokens == 2:  
                    cmaps = plt.colormaps() 
                    if tokens[1].isdigit():
                        data["colorMapIndex"] = int (tokens[1])
                        data["colorMapName"] = plt.colormaps()[ data["colorMapIndex"] ]
                    else:
                        data["colorMapName"] = tokens[1]    
            
            elif tokens [0] == "wave": 
                if "waves" not in data:
                    data["waves"] = []
                current_section = "wave"
                wave = {}
                data["waves"].append(wave)

            elif tokens[0] == "discreteLinear" :
                if "discreteLinears" not in data:
                    data["discreteLinears"] = []

                current_section = "discreteLinear"
                discreteLinear = {}
                data["discreteLinears"].append(discreteLinear)

            elif tokens[0] == "mirror": 
                if "mirrors" not in data:
                    data["mirrors"] = []
                
                current_section = "mirror"
                mirror = {}
                if len(tokens) == 3:
                    fa = float(tokens[1])
                    fb = float(tokens[2])
                    mirror ["fa"] = fa
                    mirror ["fb"] = fb
                data["mirrors"].append(mirror)

            elif tokens[0] in ( "x", "y", "v", "f" ):
                key = tokens[0]
                if current_section == "wave":
                    wave[key] = float(tokens[1])
                elif current_section == "discreteLinear":
                    discreteLinear[key] = float(tokens[1])



            elif tokens[0] == "linear": 
                if current_section == "wave":
                    wave["linear"] = True
            elif tokens[0] == "makeReflected": 
                if current_section == "wave":
                    wave["makeReflected"] = True
                elif current_section == "discreteLinear":
                    discreteLinear["makeReflected"] = True
            elif tokens[0] == "makeRefracted": 
                if current_section == "wave":
                    wave["makeRefracted"] = True
                elif current_section == "discreteLinear":
                    discreteLinear["makeRefracted"] = True
                    
            elif tokens[0] in ("clipped"):
                if current_section == "wave":
                    wave["clipped"] = True
                else:
                    data["clipped"] = True
            elif tokens[0] in ( "drawCircles", "drawFocus", "hidden" ):
                key = tokens[0]
                if current_section == "wave":
                    wave[key] = True
                elif current_section == "discreteLinear":
                    discreteLinear[key] = True
                else:   
                    data[key] = True
                    
            elif tokens[0] == 'viewOptions':        
                key = tokens[0]
                ntokens = len(tokens)
                value = 0
                if ntokens >= 2:
                    k = 1
                    kmax = ntokens -1
                    while k <= kmax:
                        if tokens[k].startswith('0b'):
                            value = int(tokens[1],2)
                        elif tokens[k].isdigit():
                            value = int(tokens[1])
                        else:
                            if tokens[k] == "viewSource":
                                value |= VIEW_SOURCE
                            elif tokens[k] == "clipSourceAbove":
                                value |= CLIP_SOURCE_ABOVE
                            elif tokens[k] == "clipSourceBelow":
                                value |= CLIP_SOURCE_BELOW
                            elif tokens[k] == "viewReflex":
                                value |= VIEW_REFLEX
                            elif tokens[k] == "clipReflexAbove":
                                value |= CLIP_REFLEX_ABOVE
                            elif tokens[k] == "clipReflexBelow":
                                value |= CLIP_REFLEX_BELOW
                            elif tokens[k] == "viewRefrac":
                                value |= VIEW_REFRAC
                            elif tokens[k] == "clipRefracAbove":
                                value |= CLIP_REFRAC_ABOVE
                            elif tokens[k] == "clipRefracBelow":
                                value |= CLIP_REFRAC_BELOW
                        k += 1        
                if current_section == "wave":
                    wave[key] = value
                elif current_section == "discreteLinear":
                    discreteLinear[key] = value

            elif tokens[0] in ( "drawRays" ):
                key = tokens[0]
                if current_section == "wave":
                    wave[key] = int (tokens[1])
                elif current_section == "discreteLinear":
                    discreteLinear[key] = int (tokens[1])
                else:   
                    data[key] = int (tokens[1])
            elif tokens[0] in ( "focusRadius" ):
                key = tokens[0]
                if current_section == "wave":
                    wave[key] = float (tokens[1])
                elif current_section == "discreteLinear":
                    discreteLinear[key] = float (tokens[1])
                else:   
                    data[key] = float (tokens[1])
            elif tokens[0] in ( "attenuation" ):
                key = tokens[0]
                if current_section == "wave":
                    wave[key] = float (tokens[1])
                elif current_section == "discreteLinear":
                    discreteLinear[key] = float (tokens[1])
                else:   
                    data[key] = float (tokens[1])
            elif tokens[0] in ( "lifetime" ):
                key = tokens[0]
                if current_section == "wave":
                    wave[key] = float (tokens[1])
                elif current_section == "discreteLinear":
                    discreteLinear[key] = float (tokens[1])
                else:   
                    data[key] = float (tokens[1])
            elif tokens[0] in ( "delayTime" ):
                key = tokens[0]
                if current_section == "wave":
                    wave[key] = float (tokens[1])
                elif current_section == "discreteLinear":
                    discreteLinear[key] = float (tokens[1])
                else:   
                    data[key] = float (tokens[1])
            elif tokens[0] in ( "stopTime" ):
                key = tokens[0]
                if current_section == "wave":
                    wave[key] = float (tokens[1])
                elif current_section == "discreteLinear":
                    discreteLinear[key] = float (tokens[1])
                else:   
                    data[key] = float (tokens[1])


            elif tokens[0] in ( "length", "progressive" ):
                key = tokens[0]
                if current_section == "discreteLinear":
                    discreteLinear[key] = float( tokens[1] )

            elif tokens[0] == "point":
                if current_section == "mirror":
                    if "points" not in mirror:
                        mirror["points"] = []
                    point = [ float(tokens[1]), float(tokens[2]) ]
                    mirror["points"].append(point)

            elif tokens[0] == "nsource": 
                if current_section == "discreteLinear":
                    discreteLinear["nsource"] = int(tokens[1])
                elif current_section == "mirror":
                    mirror["nsource"] = int (tokens[1])
                    
            elif tokens[0] == "vrupture":
                if current_section == "discreteLinear":
                    discreteLinear["vrupture"] = float(tokens[1])

            elif tokens[0] == "alpha": 
                if current_section == "discreteLinear":
                    discreteLinear["alpha"] = float(tokens[1])/180.0*math.pi
                elif current_section == "wave":
                    wave["alpha"] = float(tokens[1])/180.0*math.pi
            elif tokens[0] == "phase": 
                if current_section == "discreteLinear":
                    discreteLinear["phase"] = float(tokens[1])
                elif current_section == "wave":
                    wave["phase"] = float(tokens[1])
            elif tokens[0] == "phaser": 
                if current_section == "discreteLinear":
                    discreteLinear["phaser"] = float(tokens[1])
                elif current_section == "wave":
                    wave["phaser"] = float(tokens[1])

            elif tokens[0] == "vrefracted": 
                if current_section == "discreteLinear":
                    discreteLinear["vrefracted"] = float(tokens[1])
                elif current_section == "wave":
                    wave["vrefracted"] = float(tokens[1])
                
        return data


    def buildFromScript(self,filename):
        self.filename=filename
        data = self.parseFile(filename)
        self.buildScene(data)
        self.prepare()
        self.displayInfo()


    def buildScene(self,data):
        
        for key in ["xmin", "xmax", "ymin", "ymax", "nx", "ny", "na", "tmin", "tmax", "refracted", "clipped", "equalAxis", "colorBar", "normalize"]:
            if key in data:
                setattr(self, key, data[key])

        if "drawRays" in data:
            self.setDrawRays(True,data["drawRays"])     
        if "focusRadius" in data:
            self.focusRadius = data["focusRadius"]
        if "drawCircles" in data:
            self.isDrawCircles = True
        if "fps" in data:
            self.fps = data["fps"]
        if "parallel" in data:
            self.isParallelizing = True    
            self.nProcessors = data["parallel"]
        if "frame" in data:
            self.onlyFrame = True    
            self.selectedFrame = data["frame"]
        if "colorMapName" in data:
            self.colorMapName = data["colorMapName"]
        if "colorMapIndex" in data:
            self.colorMapIndex = data["colorMapIndex"]


        if "grid" in data:
            self.hasGrid = True 
        if "drawFocus" in data:
            self.isDrawFocus = True
        if "randomPhase" in data:
                self.randomPhase = True
        if "transient" in data:
                self.isTransient = True
        if "colorMap" in data:
                self.hasColorMap = True
                

        if "mirrors" in data:
            nMirrors = len(data["mirrors"])
            for k in range(0,nMirrors):
                mirror = data["mirrors"][k]
                self.mirrors.append(mirror)
            
        if "waves" in data:
            nWaves = len (data["waves"])
            for k in range(0,nWaves):
                wave = Wave(self)
                waveData = data["waves"][k]

                x = waveData["x"] if "x" in waveData else 0
                y = waveData["y"] if "y" in waveData else 0
                wave.setPosition(x, y)

                wave.v = waveData["v"]                         if "v"              in waveData else wave.v
                wave.setFrequence(waveData["f"])               if "f"              in waveData else None
                wave.setAttenuation(waveData["attenuation"])   if "attenuation"    in waveData else None
                wave.setLifePeriods(waveData["lifetime"])      if "lifetime"       in waveData else None
                wave.delayTime = waveData["delayTime"]         if "delayTime"      in waveData else wave.delayTime
                wave.stopTime = waveData["stopTime"]           if "stopTime"       in waveData else wave.stopTime
                wave.setLinear()                               if "linear"         in waveData else None
                wave.linearAngle = waveData["alpha"]           if "alpha"          in waveData else wave.linearAngle
                wave.setPhase(waveData["phase"])               if "phase"          in waveData else None
                wave.setPhaser(waveData["phaser"])             if "phaser"         in waveData else None
                wave.setDrawRays(True, waveData["drawRays"])   if "drawRays"       in waveData else None
                wave.setDrawCircles(True)                      if "drawCircles"    in waveData else None
                wave.setDrawFocus(True)                        if "drawFocus"      in waveData else None
                wave.focusRadius = waveData["focusRadius"]     if "focusRadius"    in waveData else wave.focusRadius
                wave.isDrawClippedArea = True                  if "clipped"        in waveData else wave.isDrawClippedArea
                wave.vrefracted = waveData["vrefracted"]       if "vrefracted"     in waveData else wave.v
                wave.makeReflected = waveData["makeReflected"] if "makeReflected"  in waveData else wave.makeReflected
                wave.makeRefracted = waveData["makeRefracted"] if "makeRefracted"  in waveData else wave.makeRefracted
                wave.viewOptions = waveData["viewOptions"]     if "viewOptions"    in waveData else wave.viewOptions
                wave.isHidden = True                           if "hidden"         in waveData else wave.isHidden

        if "discreteLinears" in data:
            nDiscreteLinears = len (data["discreteLinears"])
            for k in range(0,nDiscreteLinears):
                discreteLinear = data["discreteLinears"][k]
                
                x        = discreteLinear.get("x", 0)
                y        = discreteLinear.get("y", 0)
                alpha    = discreteLinear.get("alpha", 0)
                length   = discreteLinear.get("length", 100)
                v        = discreteLinear.get("v", 0)
                f        = discreteLinear.get("f", 0)
                nsource  = discreteLinear.get("nsource", 3)
                phase    = discreteLinear.get("phase", 0)
                phaser   = discreteLinear.get("phaser", 0)
                vrupture = discreteLinear.get("vrupture", None)

                wavesList = self.buildDiscreteLinearSource(x,y,alpha,nsource,length,v,f,phase) 

                for wave in wavesList:
                    if "drawRays" in discreteLinear:
                        wave.setDrawRays(True, discreteLinear["drawRays"])
                    if "drawCircles" in discreteLinear:
                        wave.setDrawCircles(True)
                    if "drawFocus" in discreteLinear:
                        wave.setDrawFocus(True)
                    if "focusRadius" in discreteLinear:
                        wave.focusRadius = discreteLinear["focusRadius"]
                    if "attenuation" in discreteLinear:
                        wave.setAttenuation (discreteLinear["attenuation"])
                    if "lifetime" in discreteLinear:
                        wave.setLifePeriods (discreteLinear["lifetime"])
                    if "delayTime" in discreteLinear:    
                        wave.delayTime = discreteLinear["delayTime"] 
                    if "stopTime" in discreteLinear:    
                        wave.stopTime = discreteLinear["stopTime"] 
                    if "phase" in discreteLinear:    
                        wave.setPhase ( discreteLinear["phase"] ) 
                    if "phaser" in discreteLinear:    
                        wave.setPhaser ( discreteLinear["phaser"] ) 
                    if "vrefracted" in discreteLinear:
                        wave.vrefracted = discreteLinear["vrefracted"]
                    if self.randomPhase:
                        wave.setPhase  ( random.uniform(0, 360) ) 
                    if "makeReflected" in discreteLinear:
                        wave.makeReflected = True
                    if "makeRefracted" in discreteLinear:
                        wave.makeRefracted = True
                    if "viewOptions" in discreteLinear:
                        wave.viewOptions = discreteLinear["viewOptions"]
                    if "hidden" in discreteLinear:
                        wave.isHidden = True     

                if "progressive" in discreteLinear:
                    self.setProgressiveDephasing(wavesList, discreteLinear["progressive"])

                if vrupture != None:
                    self.calculateRupturePropagation(wavesList, vrupture)                         

        self.segmentMirrors()
        self.addWavesSourcesOnMirror()       
        self.addReflectedSources()                     
                    
        if "attenuation" in data:
            self.setAttenuation ( data["attenuation"] )
            
        if "hidden" in data:
            for wave in self.waves:
                wave.isHidden = True

        # if "clipped" in data:
        #     for wave in self.waves:
        #         wave.isDrawClippedArea = True

        if "lifetime" in data:
            for wave in self.waves:
                wave.setLifePeriods (data["lifetime"])
                
        self.buildColorMap()        
            

        return self

