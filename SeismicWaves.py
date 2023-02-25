import argparse
import sys

sys.path.append("src")

from Scene import Scene
from multiprocessing import freeze_support

# import os

# import numpy as np
# import math
# import matplotlib.pyplot as plt
# from matplotlib.patches import Circle
# from moviepy.editor import ImageSequenceClip
# from PIL import Image
# import copy
# import random

# from matplotlib.animation import FuncAnimation
# from functools import partial
# import multiprocessing




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
    if parallel != None:
        scene.isParallelizing = True
        scene.nProcessors = parallel

    scene.buildFromScript(filename)
    scene.buildAnimation()



'''

mirror = [0.2,0.9]


            
flag = 3

    
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
    
    

'''
    





