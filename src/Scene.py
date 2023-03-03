import os
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


from Wave import Wave

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
        self.progressiveRupture = False
        self.vrupture = 0
        self.isTransient = False
        self.hasGrid = False
        self.hasColorMap = False
        self.isAttenuating = False
        self.attenuationFactor = 0
        self.hidden = False
        self.onlyFrame = False
        self.selectedFrame = None
        self.refracted = False
        self.clipped = False
        
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
    def segmentMirrors (self):
    #---------------------------------------------------------------------------
        if len(self.mirrors) > 0:
            ns = 50
            for mirror in self.mirrors:
                mirror ["segmented"] = []
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
                    for point in mirror["points"]:
                        x1 = point[0]
                        y1 = point[1]
                        d += np.sqrt((x1-x0)*(x1-x0)+(y1-y0))
                    dseg = d / ns
                    
                    fa = points[0][1] / (self.ymax - self.ymin)
                    fb = points[-1][1] / (self.ymax - self.ymin)
                    mirror["fa"]=fa
                    mirror["fb"]=fb
                    xa1 = self.xmin
                    ya1 = (self.ymax - self.ymin)*fa + self.ymin
                    xb1 = self.xmax
                    yb1 = (self.ymax - self.ymin)*fb + self.ymin
                    
                    for k in range(0,ns):
                        x1 = xa1 + (xb1-xa1) * k / (ns-1)
                        y1 = ya1 + (yb1-ya1) * k / (ns-1)
                        mirror["segmented"].append([x1,y1])


    #---------------------------------------------------------------------------
    def buildDiscreteLinearSource(self, x0, y0, alpha, nsource, length, v, f, phase ):
    #---------------------------------------------------------------------------
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
            wave.setFrequence(f)
            wave.setPhase(phase)
            wavesList.append(wave)
        
        return wavesList    


    # #---------------------------------------------------------------------------
    # def buildLinearSourceWithDelay(self, x0, y0, alpha, nw, v, f, nc ):
    # #---------------------------------------------------------------------------
    #     T = 1/f
    #     lambda0 = v * T
    #     sinalpha = np.sin(alpha)
    #     listk = range(0,nc+1)
    #     for k in listk:
    #         listi = range(0,nw+1)
    #         for i in listi:
    #             x = x0 + i / nw * lambda0 / sinalpha
    #             y = y0
    #             wave = Wave (self)
    #             wave.setPosition (x,y)
    #             wave.v = v
    #             wave.set_frequence(f)
    #             wave.deltaT=i/nw*T
    #             # wave.delayTime=k*T+wave.deltaT
    #             print (f"k {k}\t i {i} \t x {x} \t y {y} \t {wave.deltaT}")
    #         x0 = x + lambda0 / ( nw * sinalpha )
    #         y0 = y    
            

    #---------------------------------------------------------------------------
    def calculateRupturePropagation(self, wavesList):
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
            wave.delayTime = delayTime0 + waved / self.vrupture
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
            Cette fonction fixe un décalage de nt périodes entre le 1er point
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
            
    # def setIncoherent(self,flag):
    #     self.isIncoherent=flag
    #     if flag==True:
    #         i = 1
    #         for wave in self.waves:
    #             wave.deltaT = random.uniform (0,wave.T)
    #             print (f"Onde {i} : {wave.deltaT} \t {wave.deltaT/wave.T*100} ")
    #             i = i + 1

    #---------------------------------------------------------------------------
    def drawLine(self,fa,fb):
    #---------------------------------------------------------------------------
        xa = self.xmin
        ya = (self.ymax - self.ymin)*fa + self.ymin
        xb = self.xmax
        yb = (self.ymax - self.ymin)*fb + self.ymin
        plt.plot ([xa,xb], [ya,yb], color='black', linewidth=2)
        
    #---------------------------------------------------------------------------
    def projection_point_droite(self,x0, y0, xa1, ya1, xb1, yb1):
    #---------------------------------------------------------------------------
        # Calcul du vecteur direction de la droite
        v = (xb1 - xa1, yb1 - ya1)

        # Calcul du vecteur qui relie le point (x0, y0) à l'un des points sur la droite
        u = (x0 - xa1, y0 - ya1)

        # Calcul de la projection de u sur v
        proj = (u[0]*v[0] + u[1]*v[1]) / (v[0]**2 + v[1]**2)
        proj_v = (proj*v[0], proj*v[1])

        # Calcul des coordonnées de la projection
        xp = xa1 + proj_v[0]
        yp = ya1 + proj_v[1]

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
                plt.plot     
                    
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
    
        if len(self.mirrors) > 0:
            for mirror in self.mirrors:
                fa = mirror["fa"]
                fb = mirror["fb"]
                xa1 = self.xmin
                ya1 = (self.ymax - self.ymin)*fa + self.ymin
                xb1 = self.xmax
                yb1 = (self.ymax - self.ymin)*fb + self.ymin
                alpha1 = (yb1-ya1)/(xb1-xa1)
                d = np.sqrt( (xb1-xa1)*(xb1-xa1) + (yb1-ya1)*(yb1-ya1) )
                
                for wave in self.waves:
                    wave.isHidden = True
                
                
                waves = copy.copy(self.waves)
                ns = 50
                
                for wave in waves:
                    x0 = wave.x0
                    y0 = wave.y0
                    segmentationPoints = mirror["segmented"]
                    for point in segmentationPoints:
                        x1 = point[0]
                        y1 = point[1]
#                     for k in range(0,ns):
#                         x1 = xa1 + (xb1-xa1) * k / (ns-1)
#                         y1 = ya1 + (yb1-ya1) * k / (ns-1)


                        # (xp,yp) = self.projection_point_droite (x0, y0, xa1, ya1, xb1, yb1)
                        # dp = np.sqrt ( (xp-x0)*(xp-x0) + (yp-y0)*(yp-y0))
                        # phasep = dp / wave.lambda0
                        # phasep = math.fmod( phasep, 1)
                        # u = ( x1 - xp, y1 - yp )
                        # v = ( x0 - xp, y0 - yp )
                        # cosalpha = ( u[0] * v[0] + u[1] * v[1] ) / ( np.sqrt ( u[0] * u[0] + u[1] * u[1]) * np.sqrt ( v[0] * v[0] + v[1] * v[1]) )
                        # alpha = math.acos(cosalpha)
                        
                        d1 = np.sqrt( (x1-x0)*(x1-x0) + (y1-y0)*(y1-y0))
                        phase1 = d1 / wave.lambda0
                        phase1 = math.fmod( phase1, 1)
                        
                        wave2 = Wave(self)
                        wave2.setPosition (x1,y1)
                        wave2.setFrequence (wave.f)
                        wave2.v = wave.vrefracted
                        wave2.setPhase(-360*phase1)
                        wave2.isRefracted = True
                        wave2.lifetime = wave.lifetime
                        if self.isTransient:
                            wave2.delayTime = d1 / wave.v 
                            
                        wave1 = Wave(self)
                        wave1.setPosition (x1,y1)
                        wave1.setFrequence (wave.f)
                        wave1.v = wave.v
                        wave1.setPhase(-360*phase1)
                        wave1.isReflected = True
                        wave1.lifetime = wave.lifetime
                        if self.isTransient:
                            wave1.delayTime = d1 / wave1.v 
                            
                            
                        
                        
                    
                        
    
    
        # if len(self.mirrors)>0:
        #     mirror=self.mirrors[0]
        #     fa=mirror[0]
        #     fb=mirror[1]
        #     xa1 = self.xmin
        #     ya1 = (self.ymax - self.ymin)*fa + self.ymin
        #     xb1 = self.xmax
        #     yb1 = (self.ymax - self.ymin)*fb + self.ymin
        #     alpha1 = (yb1-ya1)/(xb1-xa1)
        #     waves = copy.copy(self.waves)

        #     i = 0
        #     for wave0 in waves:
                
        #         wave0.isHidden=True
        #         wave1 = Wave(self)
        #         wave1.sourceWave = wave0
        #         wave1.isDrawSourceRay = True
        #         x0 = wave0.x0
        #         y0 = wave0.y0

        #         angle0 = wave0.linearAngle
        #         radangle0 = angle0 * math.pi / 180
        #         xa0 = self.xmin
        #         xb0 = self.xmax
        #         ya0 = y0 + math.tan(radangle0)*(xa0-x0) 
        #         yb0 = y0 + math.tan(radangle0)*(xb0-x0)
                
        #         alpha0 = (yb0-ya0)/(xb0-xa0)
                
        #         x1 = (x0 + alpha0*y0 + alpha0*alpha1*xa1 - alpha0*ya1) / (1+alpha0*alpha1)
        #         y1 = ( alpha1*x0 + alpha0*alpha1*y0 - xa1 * alpha1 + ya1 ) / (1+alpha0*alpha1)
                
        #         radangle1 =  math.atan(alpha1)
        #         angle1 = radangle1*180/math.pi
                
        #         d=(x1-xa1)*(x1-xa1)+(y1-ya1)*(y1-ya1)
        #         d = np.sqrt(d)
        #         wave1.deltaT = -d/wave1.v*math.sin(math.pi/180*(angle0-angle1))
                
        #         wave1.setPosition(x1,y1)
                
        #         wave1.isDrawRays=False
        #         wave1.isDrawCircles=True
        #         wave1.isDrawClippedArea=False
                
        #         if i==0:
        #             wave1.isReflectedWave=True
        #             wave1.wave0 = wave0
        #             wave1.xa = xa1
        #             wave1.ya = ya1
        #             wave1.xb = xb1
        #             wave1.yb = yb1
        #             wave1.isDrawReflectedRays=False
        #             wave1.isHidden=True
    
        #         i+=1

        #     for wave0 in waves:
        #         wave1 = Wave(self)
        #         wave1.setReflectedWave(wave0,mirror)
        #         wave1.isDrawRays=False
        #         wave1.isDrawCircles=False
        #         wave1.isDrawClippedArea=False
        #         wave1.isHidden=True

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

            poly = Polygon ( list(zip(x,y)),transform=ax.transData)
            return poly

        
    #---------------------------------------------------------------------------
    def createAnimationFrameImage(self,  ti, i):
    #---------------------------------------------------------------------------
        '''
        Cette méthode crée une image pour la séquence d'animation à l'aide
        de la bibliothèque Matplotlib.
            ti est le temps
            i est l'indice de l'image    
        '''
        fig, ax = plt.subplots(figsize=(15,15))
        plt.axis('equal')
        ax.set_xlim(self.xmin,self.xmax)
        ax.set_ylim(self.ymin,self.ymax)
        ax.invert_yaxis()
        
        waves = self.waves
        sourceWaveArray = []
        reflectedWaveArray = []
        refractedWaveArray = []
        for wave in waves:
            if wave.isHidden == False:
                waveArray0 = wave.createWave (ti)
                if wave.isReflected:
                    reflectedWaveArray  = self.sumWavesArray (reflectedWaveArray, waveArray0)
                elif wave.isRefracted:                    
                    refractedWaveArray  = self.sumWavesArray (refractedWaveArray, waveArray0)
                else:
                    sourceWaveArray = self.sumWavesArray (sourceWaveArray, waveArray0)    


        if len(sourceWaveArray) > 0:
            if self.hasColorMap:
                n = len(waves)
                sourceWaveArray[0][0] = n + 0.5
                sourceWaveArray[0][1] = -n - 0.5

            image = ax.imshow(sourceWaveArray, extent=(self.xmin, self.xmax, self.ymin, self.ymax), 
                      cmap="RdBu", origin="lower")


        if len(reflectedWaveArray) > 0:
            reflectedClipPath = self.getClipArea('above',ax)
            image = ax.imshow(reflectedWaveArray, extent=(self.xmin, self.xmax, self.ymin, self.ymax), 
                      cmap="RdBu", origin="lower")
            image.set_clip_path(reflectedClipPath)

        if len(refractedWaveArray) > 0:
            refractedClipPath = self.getClipArea('below',ax)
            image = ax.imshow(refractedWaveArray, extent=(self.xmin, self.xmax, self.ymin, self.ymax), 
                      cmap="RdBu", origin="lower")
            image.set_clip_path(refractedClipPath)

    
        if self.hasColorMap:
            cp = ax.get_children()
            cp2 = cp[0]
            plt.colorbar( cp2, ax=ax)
        
    
        for wave in waves:
            if ti >= wave.delayTime:
                if wave.isRefracted:
                    wave.drawRays()
                    wave.drawSourceRay()
                    wave.drawReflectedRays ()
                    wave.drawCircles(ax, ti, 1.0, refractedClipPath)
                    wave.drawFocus()
        
        
        for wave in waves:
            if ti >= wave.delayTime:
                if wave.isReflected:
                    wave.drawRays()
                    wave.drawSourceRay()
                    wave.drawReflectedRays ()
                    poly = self.getClipArea('above',ax)
                    wave.drawCircles(ax, ti, 1.0, reflectedClipPath)
                    wave.drawFocus()

        for wave in waves:
            if ti >= wave.delayTime:
                if wave.isReflected == False and wave.isRefracted == False :
                    wave.drawRays()
                    wave.drawSourceRay()
                    wave.drawReflectedRays ()
                    wave.drawCircles(ax, ti, 1.0, None)
                    wave.drawFocus()


        self.drawMirrors()

        # if len(self.mirrors)>0 :
        #     for mirror in self.mirrors:
        #         self.drawLine(mirror['fa'], mirror['fb'])

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
            elif tokens[0] in ( "tmin", "tmax", "vrupture" ):
                key = tokens[0]
                data[key] = float(eval(tokens[1]))
                current_section = None
            elif tokens[0] in ( "randomPhase", "transient", "grid", "colorMap", "refracted" ):
                key = tokens[0]
                data[key] = True
                current_section = None
            
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
        
        for key in ["xmin", "xmax", "ymin", "ymax", "nx", "ny", "na", "tmin", "tmax", "refracted", "clipped"]:
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
        if "vrupture" in data:
                self.vrupture = data["vrupture"]
                self.progressiveRupture = True

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

                wave.v = waveData["v"]                       if "v"           in waveData else wave.v
                wave.setFrequence(waveData["f"])             if "f"           in waveData else None
                wave.setAttenuation(waveData["attenuation"]) if "attenuation" in waveData else None
                wave.setLifePeriods(waveData["lifetime"])    if "lifetime"    in waveData else None
                wave.setLinear()                             if "linear"      in waveData else None
                wave.linearAngle = waveData["alpha"]         if "alpha"       in waveData else wave.linearAngle
                wave.setPhase(waveData["phase"])             if "phase"       in waveData else None
                wave.setDrawRays(True, waveData["drawRays"]) if "drawRays"    in waveData else None
                wave.setDrawCircles(True)                    if "drawCircles" in waveData else None
                wave.setDrawFocus(True)                      if "drawFocus"   in waveData else None
                wave.focusRadius = waveData["focusRadius"]   if "focusRadius" in waveData else wave.focusRadius
                wave.isDrawClippedArea = True                if "clipped"     in waveData else wave.isDrawClippedArea
                wave.vrefracted = waveData["vrefracted"]     if "vrefracted"  in waveData else wave.vrefracted

        if "discreteLinears" in data:
            nDiscreteLinears = len (data["discreteLinears"])
            for k in range(0,nDiscreteLinears):
                discreteLinear = data["discreteLinears"][k]
                
                x       = discreteLinear.get("x", 0)
                y       = discreteLinear.get("y", 0)
                alpha   = discreteLinear.get("alpha", 0)
                length  = discreteLinear.get("length", 100)
                v       = discreteLinear.get("v", 0)
                f       = discreteLinear.get("f", 0)
                nsource = discreteLinear.get("nsource", 3)
                phase   = discreteLinear.get("phase", 0)

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
                    if "vrefracted" in discreteLinear:
                        wave.vrefracted = discreteLinear["vrefracted"]
                    if self.randomPhase:
                        wave.setPhase  ( random.uniform(0, 360) ) 
                        

                if "progressive" in discreteLinear:
                    self.setProgressiveDephasing(wavesList, discreteLinear["progressive"])

                if self.progressiveRupture:
                    self.calculateRupturePropagation(wavesList)                         

        self.segmentMirrors()
        self.addWavesSourcesOnMirror()                            
                    
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
                
                
            

        return self

