import argparse
import sys

sys.path.append("src")

from Scene import Scene
from multiprocessing import freeze_support


#-------------------------------------------------------------------------
#   Main code
#-------------------------------------------------------------------------

if __name__ == '__main__':
    freeze_support()
    

    parser = argparse.ArgumentParser(description='Caculate seismic  waves')
    parser.add_argument('script',  type=str, help='the script file name')
    parser.add_argument('--parallel', type=int, help='Nombre de processus en parallèle')
    parser.add_argument('--frame', type=int, help='Image à calculer')
    args = parser.parse_args()
    filename = args.script
    parallel = args.parallel
    frame = args.frame
    
    print ("Process script {}".format(filename))

    scene = Scene()

    if parallel != None:
        scene.isParallelizing = True
        scene.nProcessors = parallel

    scene.buildFromScript(filename)
    
    if frame != None:
        scene.onlyFrame = True
        scene.selectedFrame = frame    
    
    scene.buildAnimation()



'''

mirror = [0.2,0.9]


            
    
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
    





