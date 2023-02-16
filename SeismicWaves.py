import argparse
import sys
import os

import numpy as np
import math
import matplotlib.pyplot as plt
# import matplotlib.animation as animation
from matplotlib.patches import Circle
from moviepy.editor import ImageSequenceClip
from PIL import Image
import copy
import random

from matplotlib.animation import FuncAnimation
from functools import partial
import multiprocessing
from multiprocessing import freeze_support


#===========================================================================
class Wave:
#===========================================================================
    def __init__(self,scene):
        self.v = 6000
        self.f = 10
        self.x0 = 0
        self.y0 = 0
        self.T = 1/self.f
        self.lambda0 = self.T * self.v
        self.isReflectedWave= False
        self.mirror=[0.5,0.5]
        self.scene=scene
        self.scene.waves.append(self)
        self.isLinear=False
        self.linearAngle=0
        self.isDrawRays = scene.isDrawRays
        self.isDrawReflectedRays=False
        self.nrays= 25
        self.isDrawCircles=True
        self.isDrawFocus=True
        self.focusRadius = 10
        self.dashCounter = 0
        self.isHidden = False
        self.isDrawClippedArea=True
        self.delayTime = 0
        self.deltaT = 0
        self.sourceWave = None
        self.isDrawSourceRay = False
        self.isMovableFocus = False
    
    def displayInfo(self):
        print("\tx0 {}".format(self.x0))
        print("\ty0 {}".format(self.y0))
        print("\tv {}".format(self.v))
        print("\tlambda {}".format(self.lambda0))
        print("\tdrawRays {}".format(self.isDrawRays))
        print("\tnRays {}".format(self.nrays))
        print("\tdrawCircles {}".format(self.isDrawCircles))
        print("\tdrawFocus {}".format(self.isDrawFocus))
        
    def set_frequence(self,f0):
        self.f = f0
        self.T = 1/self.f
        self.lambda0 = self.T * self.v

    def setRelativePosition (self,fx,fy):
        self.x0 = self.scene.xmin + fx * (self.scene.xmax-self.scene.xmin)
        self.y0 = self.scene.ymin + fy * (self.scene.ymax-self.scene.ymin)

    def setPosition (self,x0,y0):
        self.x0 = x0
        self.y0 = y0
    
    def setMovableFocus (self,x00, y00, x01, y01):
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
        xa = scene.xmin
        ya = (scene.ymax - scene.ymin)*fa + scene.ymin
        xb = scene.xmax
        yb = (scene.ymax - scene.ymin)*fb + scene.ymin

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
        
        
    def drawClippedArea(self):  
        if self.isReflectedWave and self.isDrawClippedArea:
            fa = self.mirror[0]
            fb = self.mirror[1] 
            xa = self.scene.xmin
            ya = (self.scene.ymax - self.scene.ymin)*fa + self.scene.ymin
            xb = self.scene.xmax
            yb = (self.scene.ymax - self.scene.ymin)*fb + self.scene.ymin
            
            x = [xa, self.scene.xmin, self.scene.xmax, xb, xa ]
            y = [ya, self.scene.ymax, self.scene.ymax, yb, ya ]
            plt.fill(x,y, 'white', alpha=0.9)
            
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


    def drawCircles(self,ax,ti,beta):
        if self.isDrawCircles:
            alphamin = -ti*self.v / self.lambda0 
            alphamax = (self.scene.xmax-self.scene.xmin-ti*self.v)/ self.lambda0
            alphamin = math.ceil(alphamin)
            alphamax = math.ceil(alphamax)
            for alpha in list(range(alphamin, alphamax)):
                r = (ti-self.deltaT)*self.v + alpha*self.lambda0*beta
                if r>0:
                    circle = Circle ( (self.x0, self.y0), r, 
                     color='black', facecolor='none', linewidth=1, fill=False,
                     edgecolor='black')
                    ax.add_patch(circle)


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
            plt.fill(points[:,0], points[:,1],'r')
            plt.plot(points[:,0], points[:,1],'black')



    def setLinear(self):
            self.isLinear = True
            self.isDrawCircles = False

    def createWave(self,t):            
        X=self.scene.X
        Y=self.scene.Y
        self.calculateMovableFocus(t)
        
        if self.isLinear == False:
            if t>= self.delayTime:
                D = (X-self.x0)**2 + (Y-self.y0)**2
                D = np.sqrt(D)
                waveArray = np.sin ( 2*np.pi * ( (t-self.deltaT)/self.T - D/self.lambda0))
            else:
                waveArray = np.zeros_like (X)
                # waveArray = 0    
        else:
            if t>= self.delayTime:
                alpha = np.tan (math.pi*self.linearAngle/180)
                xa = self.scene.xmin
                xb = self.scene.xmax
                x0=self.x0
                y0=self.y0
                ya = self.y0 + math.tan(alpha)*(xa-x0) 
                yb = self.y0 + math.tan(alpha)*(xb-x0)
                    # X1 et Y1 are the projected points of X and Y onto
                    # the source plane
                X1 = ( alpha*alpha*x0 - alpha*y0 + X + alpha*Y ) / (1+alpha*alpha)
                Y1 = (-alpha*x0 + y0 + alpha*X + alpha*alpha*Y ) / (1+alpha*alpha)
                    # D is the distance between (X,Y) and (X1,Y1)
                D=(X-X1)*(X-X1)+(Y-Y1)*(Y-Y1)
                D = np.sqrt(D)
                waveArray = np.sin ( 2*np.pi * ( (t-self.deltaT)/self.T - D/self.lambda0))            
            else:
                waveArray = np.zeros_like (X)
                # waveArray = 0    
            
        return waveArray


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
        # self.nx = 200
        # self.ny = 400
        self.nx = 2000
        self.ny = 2000
        self.fps = 30
        self.na = 100
        self.mirrors = []
        self.waves = []
        self.imagesFolderPath = "images"
        self.animationFileName = "animation.mp4"
        self.isDrawRays = False
        self.isParallelizing = False
        self.nProcessors = 1
        
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
        print ("nx {}".format(self.nx))    
        print ("ny {}".format(self.ny))    
        print ("na {}".format(self.na))    
        print ("xmin {}".format(self.xmin))    
        print ("xmax {}".format(self.xmax))    
        print ("ymin {}".format(self.ymin))    
        print ("ymax {}".format(self.ymax))    
        print ("tmin {}".format(self.tmin))    
        print ("tmax {}".format(self.tmax))    
        print ("Draw rays {}".format(self.isDrawRays))
        print ("Parallezing {}".format(self.isParallelizing))
        k = 1
        for wave in self.waves:
            print ("Wave {}".format(k))
            wave.displayInfo()
            k += 1
        
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
        clip.write_videofile(self.animationFileName)
        

    #---------------------------------------------------------------------------
    def buildLinearSourceWithDelay(self, x0, y0, alpha, nw, v, f, nc ):
    #---------------------------------------------------------------------------
        T = 1/f
        lambda0 = v * T
        sinalpha = np.sin(alpha)
        listk = range(0,nc+1)
        for k in listk:
            listi = range(0,nw+1)
            for i in listi:
                x = x0 + i / nw * lambda0 / sinalpha
                y = y0
                wave = Wave (self)
                wave.setPosition (x,y)
                wave.v = v
                wave.set_frequence(f)
                wave.deltaT=i/nw*T
                # wave.delayTime=k*T+wave.deltaT
                print (f"k {k}\t i {i} \t x {x} \t y {y} \t {wave.deltaT}")
            x0 = x + lambda0 / ( nw * sinalpha )
            y0 = y    
            

    
    #---------------------------------------------------------------------------
    def setProgressiveDelay(self,nt):
    #---------------------------------------------------------------------------
        first = self.waves[0]
        last = self.waves[-1]
        x0 = first.x0
        y0 = first.y0
        x1 = last.x0
        y1 = last.y0
        d = np.sqrt ( (x1-x0)*(x1-x0) + (y1-y0)*(y1-y0) )
        i = 1
        for wave in self.waves:
            waved = np.sqrt ( (wave.x0-x0)*(wave.x0-x0) + (wave.y0-y0)*(wave.y0-y0) )
            alpha = waved/d
            wave.deltaT =  nt*alpha*wave.T
            print (f"Wave {i} \t {alpha} \t {wave.deltaT}")
            i = i + 1
        
    
    
    def setDrawRays(self,flag,nrays):
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
            
    def setUncoherent(self,flag):
        self.isUncoherent=flag
        if flag==True:
            i = 1
            for wave in self.waves:
                wave.deltaT = random.uniform (0,wave.T)
                print (f"Onde {i} : {wave.deltaT} \t {wave.deltaT/wave.T*100} ")
                i = i + 1

    #---------------------------------------------------------------------------
    def drawLine(self,fa,fb):
    #---------------------------------------------------------------------------
        xa = self.xmin
        ya = (self.ymax - self.ymin)*fa + self.ymin
        xb = self.xmax
        yb = (self.ymax - self.ymin)*fb + self.ymin
        plt.plot ([xa,xb], [ya,yb], color='black', linewidth=2)

    #---------------------------------------------------------------------------
    def addWavesSourcesOnMirror(self):
    #---------------------------------------------------------------------------
        if len(self.mirrors)>0:
            mirror=self.mirrors[0]
            fa=mirror[0]
            fb=mirror[1]
            xa1 = self.xmin
            ya1 = (self.ymax - self.ymin)*fa + self.ymin
            xb1 = self.xmax
            yb1 = (self.ymax - self.ymin)*fb + self.ymin
            alpha1 = (yb1-ya1)/(xb1-xa1)
            waves = copy.copy(self.waves)

            i = 0
            for wave0 in waves:
                
                wave0.isHidden=True
                wave1 = Wave(scene)
                wave1.sourceWave = wave0
                wave1.isDrawSourceRay = True
                x0 = wave0.x0
                y0 = wave0.y0

                angle0 = wave0.linearAngle
                radangle0 = angle0 * math.pi / 180
                xa0 = self.xmin
                xb0 = self.xmax
                ya0 = y0 + math.tan(radangle0)*(xa0-x0) 
                yb0 = y0 + math.tan(radangle0)*(xb0-x0)
                
                alpha0 = (yb0-ya0)/(xb0-xa0)
                
                x1 = (x0 + alpha0*y0 + alpha0*alpha1*xa1 - alpha0*ya1) / (1+alpha0*alpha1)
                y1 = ( alpha1*x0 + alpha0*alpha1*y0 - xa1 * alpha1 + ya1 ) / (1+alpha0*alpha1)
                
                radangle1 =  math.atan(alpha1)
                angle1 = radangle1*180/math.pi
                
                d=(x1-xa1)*(x1-xa1)+(y1-ya1)*(y1-ya1)
                d = np.sqrt(d)
                wave1.deltaT = -d/wave1.v*math.sin(math.pi/180*(angle0-angle1))
                
                wave1.setPosition(x1,y1)
                
                wave1.isDrawRays=False
                wave1.isDrawCircles=True
                wave1.isDrawClippedArea=False
                
                if i==0:
                    wave1.isReflectedWave=True
                    wave1.wave0 = wave0
                    wave1.xa = xa1
                    wave1.ya = ya1
                    wave1.xb = xb1
                    wave1.yb = yb1
                    wave1.isDrawReflectedRays=False
                    wave1.isHidden=True
    
                i+=1

            for wave0 in waves:
                wave1 = Wave(self)
                wave1.setReflectedWave(wave0,mirror)
                wave1.isDrawRays=False
                wave1.isDrawCircles=False
                wave1.isDrawClippedArea=False
                wave1.isHidden=True

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
        fig, ax = plt.subplots(figsize=(15,10))
        ax.set_xlim(self.xmin,self.xmax)
        ax.set_ylim(self.ymin,self.ymax)
        ax.invert_yaxis()

        waves = self.waves
        waveArray = []
        for wave in waves:
            if wave.isHidden == False:
                waveArray0 = wave.createWave (ti)
                waveArray  = self.sumWavesArray (waveArray, waveArray0)

        image = ax.imshow(waveArray, extent=(self.xmin, self.xmax, self.ymin, self.ymax), 
                      cmap="RdBu", origin="lower")
    
        for wave in waves:
            wave.drawRays()
            wave.drawSourceRay()
            wave.drawReflectedRays ()
            wave.drawCircles(ax, ti, 1.0)
            wave.drawFocus()
            wave.drawClippedArea()


        if len(self.mirrors)>0 :
            for f in self.mirrors:
                self.drawLine(f[0], f[1])

        rect = plt.Rectangle( (0,0), 500, 150, fc='#444444',alpha=0.8)
        ax.add_patch (rect)
        ax.text(250.0,75.0, "{} s".format(round(ti,4)), ha="center", va="center",color="white")
    
        fig.savefig("{}/image{:04d}.png".format(self.imagesFolderPath,i))
        

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
            for i in range(self.na):
                ti = i / (10 * self.fps )
                print (i,ti)
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


            

    def parseFile(self, fileName):
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
            elif line.startswith("xmin"):
                data["xmin"] = int(line.split()[1])
            elif line.startswith("xmax"):
                data["xmax"] = int(line.split()[1])
            elif line.startswith("ymin"):
                data["ymin"] = int(line.split()[1])
            elif line.startswith("ymax"):
                data["ymax"] = int(line.split()[1])
            elif line.startswith("parallel"):
                data["parallel"] = int(line.split()[1])

            elif line.startswith("nx"):
                data["nx"] = int(line.split()[1])
            elif line.startswith("ny"):
                data["ny"] = int(line.split()[1])
            elif line.startswith("na"):
                data["na"] = int(line.split()[1])
            elif line.startswith("fps"):
                data["fps"] = int(line.split()[1])
            elif line.startswith("tmin"):
                data["tmin"] = float(line.split()[1])
            elif line.startswith("tmax"):
                data["tmax"] = float(eval(line.split()[1]))


 
            elif line.startswith("wave") :
                if "waves" not in data:
                    data["waves"] = []
                current_section = "wave"
                wave = {}
                data["waves"].append(wave)

            elif line.startswith("x"):
                if current_section == "wave":
                    wave["x"] = float(line.split()[1])
            elif line.startswith("y"):
                if current_section == "wave":
                    wave["y"] = float(line.split()[1])

                
            elif line.startswith("drawRays"):
                if current_section == "wave":
                    wave["drawRays"] = int(line.split()[1])

            elif line.startswith("drawCircles"):
                if current_section == "wave":
                    wave["drawCircles"] = True
        return data


    def buildFromScript(self,filename):
        self.filename=filename
        data = self.parseFile(filename)
        self.buildScene(data)
        self.displayInfo()
        self.prepare()


    def buildScene(self,data):
        if "xmin" in data:
            self.xmin = data["xmin"]
        if "xmax" in data:
            self.xmax = data["xmax"]
        if "ymin" in data:
            self.ymin = data["ymin"]
        if "ymax" in data:
            self.ymax = data["ymax"]

        if "nx" in data:
            self.nx = data["nx"]
        if "ny" in data:
            self.ny = data["ny"]
        if "na" in data:
            self.ny = data["na"]
        if "tmin" in data:
            self.tmin = data["tmin"]
        if "tmax" in data:
            self.tmax = data["tmax"]
        if "fps" in data:
            self.fps = data["fps"]
        if "parallel" in data:
            self.isParallelizing = True    
            self.nProcessors = data["parallel"]
            
        if "waves" in data:
            nWaves = len (data["waves"])
            for k in range(0,nWaves):
                wave = Wave(self)
                x = 0
                y = 0
                if "x" in data["waves"][k]:
                    x = data["waves"][k]["x"]
                if "y" in data["waves"][k]:
                    y = data["waves"][k]["y"]
                wave.setPosition (x,y)
                if "drawRays" in data["waves"][k]:
                    wave.setDrawRays(True, data["waves"][k]["drawRays"])
                if "drawCircles" in data["waves"][k]:
                    wave.setDrawCircles(True)
            

        return self




#-------------------------------------------------------------------------
#   Main code
#-------------------------------------------------------------------------

if __name__ == '__main__':
    freeze_support()
    

    parser = argparse.ArgumentParser(description='Caculate seismic  waves')
    parser.add_argument('script',  type=str, help='the script file name')
    parser.add_argument('--parallel', type=int, help='Nombre de processus en parallèle')
    args = parser.parse_args()
    filename = args.script
    parallel = args.parallel
    
    print ("Process script {}".format(filename))

    scene = Scene()
    scene.buildFromScript(filename)
    
    if parallel != None:
        scene.isParallelizing = True
        scene.nProcessors = parallel

    
    scene.buildAnimation()


# scene.setBounds(0,5000,0,4000)

# wave = Wave(scene)
# wave.setPosition (3000, 1000)
#wave.setDrawRays(25)
#wave.setDrawCircles()




'''

mirror = [0.2,0.9]


            
flag = 3

if flag==1:
    wave = Wave(scene)
    wave.setPosition (3000, 1000)
    #wave.setDrawRays(25)
    #wave.setDrawCircles()

elif flag==2:    
    wave = Wave(scene)
    wave.setPosition (2500, 1000)
    wave.linearAngle=0
    wave.setLinear()
    wave.setDrawRays(True,25)
    
elif flag==3:    
    scene.setBounds(0,5000,0,4000)
    scene.buildLinearSourceWithDelay(1000,1000,np.pi/4,50,6000,10,4)
    # wave = Wave(scene)
    # wave.setPosition (1000, 1000)
    # wave.set_frequence(10)
    #wave.setMovableFocus(1000,1000,4000,1000)
    
    # wave = Wave(scene)
    # wave.setPosition (4000, 1000)
    
    # nn = 10
    # iterations = range (0,nn+1)
    # for i in iterations:
    #     x0 = -5000 + 10000 * i / nn 
    #     y0 = 1000  
    #     wave = Wave(scene)
    #     wave.setPosition (x0, y0)

    scene.setDrawFocus(True)
    scene.setDrawRays (False,20)
    scene.setDrawCircles (False)
    #  scene.setUncoherent (True)

    # scene.setProgressiveDelay(10000/600)
    # i= 0 
    # for wave in scene.waves:
    #   alpha = (wave.x0-1000)/3000
    #   wave.deltaT =  5*alpha*wave.T
    #   print (f"Wave {i} \t {alpha} \t {wave.deltaT}")
    #   i = i + 1
    
elif flag==4:    
    mirror = [0.2,0.6]
    scene.appendMirror (mirror)

    wave0 = Wave(scene)
    wave0.setPosition( (scene.xmin + scene.xmax) / 2, 1000 )
    wave0.isDrawRays=False
    wave1 = Wave(scene)
    wave1.setReflectedWave(wave0, mirror)
    wave1.isDrawRays=False

elif flag==5:
    scene.setBounds(0,10000,0,6000)
    scene.setSourcePlane(2500,0,4000,0)   
    scene.buildHuygens(20) 
    scene.appendMirror (mirror)

    waves = copy.copy(scene.waves)
    for wave0 in waves:
        wave1 = Wave(scene)
        wave1.setReflectedWave(wave0, mirror)
        wave1.isDrawRays=False
        wave1.isDrawCircles=False
        wave1.isDrawClippedArea=False
        wave0.isHidden=True

elif flag==6:
    scene.setBounds(0,10000,0,6000)
    scene.setSourcePlane(2500,0,4000,-20)   
    scene.buildHuygens(2) 
    scene.appendMirror (mirror)
    scene.addWavesSourcesOnMirror()
    
elif flag == 7:    



    # Génération des données
    x = np.linspace(0, 2*np.pi, 900)
    y = np.sin(x)

    # Tracé de la courbe
    plt.plot(x, y)

    plt.show()
    
elif flag == 8:    


    x = np.linspace(0, 6*np.pi, 300)
    y = np.sin(x)

    # Création du graphique
    fig, ax = plt.subplots()
    scat = ax.scatter(x[0], y[0], s=30, color='r')
    ax.set_xlim(0, 6*np.pi)
    ax.set_ylim(-1.4, 1.4)

    # Création de l'animation
    ani = FuncAnimation(fig, update, frames=range(1, len(x)), fargs=[x, y, scat], interval=1)

    # Affichage de l'animation
    plt.show()

elif flag == 9:    



    t = np.linspace(0, 5, 1000)
    x = np.linspace(0, 10*np.pi,1000)
    i = 0
    
    for tt in t:
        fig, ax = plt.subplots(figsize=(8,6))
        ax.set_xlim( 0, 10*np.pi)
        ax.set_ylim(-1.4,1.4)

        y = np.sin( 2*np.pi*(tt/0.7 - x/4) )
        ax.plot(x, y, lw=2, color='r')

        fig.savefig("{}/image{:04d}.png".format(scene.imagesFolderPath,i))
        i += 1
     
    scene.saveAnimation()





def update(num, x, y, scat):
    # scat.set_data(x[:num], y[:num])
    scat.set_offsets((x[num], y[num]))
    # Tracé de la courbe
    plt.plot(x, y, 'r')
    


def update1(t):
    x = np.linspace(0, 10, 1000)
    y = np.sin(x+2*np.pi*t)
    plt.plot(x,y,'r')

'''
    





