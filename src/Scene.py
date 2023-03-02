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
        print ("n rays        {}".format(self.nrays))
        print ("Parallelizing {}".format(self.isParallelizing))
        print ("Transient     {}".format(self.isTransient))

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
        files = os.listdir(self.imagesFolderPath)
        for file in files:
            filePath = os.path.join(self.imagesFolderPath, file)
            if os.path.exists(filePath):
                os.remove(filePath)

    #---------------------------------------------------------------------------
    def saveAnimation(self):
    #---------------------------------------------------------------------------
        clip = ImageSequenceClip(self.imagesFolderPath, fps=30)
        filename = self.animationsFolderPath + '/' + self.filename + '.mp4'
        clip.write_videofile(filename)
        

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
    
    def appendMirror (self,mirror):
        fa = mirror[0]
        fb = mirror[1]
        self.mirrors.append([fa,fb])
        
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
    def addWavesSourcesOnMirror(self):
    #---------------------------------------------------------------------------
    
        if len(self.mirrors) > 0:
            for mirror in self.mirrors:
                fa = mirror[0]
                fb = mirror[1]
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
                    for k in range(0,ns):
                        x1 = xa1 + (xb1-xa1) * k / (ns-1)
                        y1 = ya1 + (yb1-ya1) * k / (ns-1)
                        (xp,yp) = self.projection_point_droite (x0, y0, xa1, ya1, xb1, yb1)
                        dp = np.sqrt ( (xp-x0)*(xp-x0) + (yp-y0)*(yp-y0))
                        phasep = dp / wave.lambda0
                        phasep = math.fmod( phasep, 1)
                        u = ( x1 - xp, y1 - yp )
                        v = ( x0 - xp, y0 - yp )
                        cosalpha = ( u[0] * v[0] + u[1] * v[1] ) / ( np.sqrt ( u[0] * u[0] + u[1] * u[1]) * np.sqrt ( v[0] * v[0] + v[1] * v[1]) )
                        alpha = math.acos(cosalpha)
                        
                        d1 = np.sqrt( (x1-x0)*(x1-x0) + (y1-y0)*(y1-y0))
                        phase1 = d1 / wave.lambda0
                        phase1 = math.fmod( phase1, 1)
                        wave1 = Wave(self)
                        wave1.setPosition (x1,y1)
                        wave1.setFrequence (wave.f)
                        wave1.v = wave.v
                        wave1.setPhase(-360*phase1)
                        wave1.isReflected = True
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
        waveArray = []
        for wave in waves:
            if wave.isHidden == False:
                waveArray0 = wave.createWave (ti)
                waveArray  = self.sumWavesArray (waveArray, waveArray0)

        if len(waveArray) > 0:
            if self.hasColorMap:
                n = len(waves)
                waveArray[0][0] = n + 0.5
                waveArray[0][1] = -n - 0.5

            image = ax.imshow(waveArray, extent=(self.xmin, self.xmax, self.ymin, self.ymax), 
                      cmap="RdBu", origin="lower")
    
        if self.hasColorMap:
            cp = ax.get_children()
            cp2 = cp[0]
            plt.colorbar( cp2, ax=ax)
        
    
        for wave in waves:
            if ti >= wave.delayTime:
                wave.drawRays()
                wave.drawSourceRay()
                wave.drawReflectedRays ()
                wave.drawCircles(ax, ti, 1.0)
                wave.drawClippedArea()
                wave.drawFocus()



        if len(self.mirrors)>0 :
            for f in self.mirrors:
                self.drawLine(f[0], f[1])

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
                ti = self.selectedFrame / (10 * self.fps )
                print ("{} \t {}".format(self.selectedFrame,ti))
                self.createAnimationFrameImage( ti, self.selectedFrame)
            else:        
                for i in range(self.na):
                    ti = i / (10 * self.fps )
                    print ("{} \t {}".format(i,ti))
                    self.createAnimationFrameImage( ti, i)

        self.saveAnimation ()
        

    #---------------------------------------------------------------------------
    def buildAnimationTask(self,i):
    #---------------------------------------------------------------------------
        duplicatedScene = copy.deepcopy(self)
        ti = i / (10 * self.fps )
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
            # if line.startswith("#"):
            #   # c'est un commentaire dans le script -> ne rien faire
            #     continue
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
            elif tokens[0] in ( "randomPhase", "transient", "grid", "colorMap" ):
                key = tokens[0]
                data[key] = True
                current_section = None
            
            elif line.startswith("wave") :
                if "waves" not in data:
                    data["waves"] = []
                current_section = "wave"
                wave = {}
                data["waves"].append(wave)

            elif line.startswith("discreteLinear"):
                if "discreteLinears" not in data:
                    data["discreteLinears"] = []

                current_section = "discreteLinear"
                discreteLinear = {}
                data["discreteLinears"].append(discreteLinear)

            elif line.startswith("mirror"):
                if "mirrors" not in data:
                    data["mirrors"] = []
                
                fa = float(tokens[1])
                fb = float(tokens[2])
                data["mirrors"].append( [fa,fb] )

            elif tokens[0] in ( "x", "y", "v", "f" ):
                key = tokens[0]
                if current_section == "wave":
                    wave[key] = float(tokens[1])
                elif current_section == "discreteLinear":
                    discreteLinear[key] = float(tokens[1])



            elif line.startswith("linear"):
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


            elif tokens[0] in ( "length", "progressive" ):
                key = tokens[0]
                if current_section == "discreteLinear":
                    discreteLinear[key] = float( tokens[1] )


            elif line.startswith("nsource"):
                if current_section == "discreteLinear":
                    discreteLinear["nsource"] = int(line.split()[1])

            elif line.startswith("alpha"):
                if current_section == "discreteLinear":
                    discreteLinear["alpha"] = float(line.split()[1])/180.0*math.pi
                elif current_section == "wave":
                    wave["alpha"] = float(line.split()[1])/180.0*math.pi
            elif line.startswith("phase"):
                if current_section == "discreteLinear":
                    discreteLinear["phase"] = float(line.split()[1])
                elif current_section == "wave":
                    wave["phase"] = float(line.split()[1])
                
        return data


    def buildFromScript(self,filename):
        self.filename=filename
        data = self.parseFile(filename)
        self.buildScene(data)
        self.displayInfo()
        self.prepare()


    def buildScene(self,data):
        
        for key in ["xmin", "xmax", "ymin", "ymax", "nx", "ny", "na", "tmin", "tmax"]:
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
                self.mirrors.append( mirror)
            
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
                wave.setLinear()                             if "linear"      in waveData else None
                wave.linearAngle = waveData["alpha"]         if "alpha"       in waveData else wave.linearAngle
                wave.setPhase(waveData["phase"])             if "phase"       in waveData else None
                wave.setDrawRays(True, waveData["drawRays"]) if "drawRays"    in waveData else None
                wave.setDrawCircles(True)                    if "drawCircles" in waveData else None
                wave.setDrawFocus(True)                      if "drawFocus"   in waveData else None
                wave.focusRadius = waveData["focusRadius"]   if "focusRadius" in waveData else wave.focusRadius
                wave.isDrawClippedArea = True                if "clipped"     in waveData else wave.isDrawClippedArea

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
                    if self.randomPhase:
                        wave.setPhase  ( random.uniform(0, 360) ) 

                if "progressive" in discreteLinear:
                    self.setProgressiveDephasing(wavesList, discreteLinear["progressive"])

                if self.progressiveRupture:
                    self.calculateRupturePropagation(wavesList)                         

        self.addWavesSourcesOnMirror()                            
                    
        if "attenuation" in data:
            self.setAttenuation ( data["attenuation"] )
            
        if "hidden" in data:
            for wave in self.waves:
                wave.isHidden = True

        if "clipped" in data:
            for wave in self.waves:
                wave.isDrawClippedArea = True
                
                
            

        return self

