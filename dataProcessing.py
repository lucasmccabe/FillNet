#dependencies
import subprocess as sub
import cv2
import os
from PIL import Image
import csv
import numpy as np
#------------


#functions----------
def dlv(url):
    #given a youtube url, downloads the video
    print('Downloading video at: ' + url)
    commandline = 'youtube-dl '+url+' -c'
    sub.call(commandline.split(), shell=False) #saves link as mp4
    print('\tDone.')

def getVidList():
    #returns a list of all video in current directory
    return [f for f in os.listdir() if f.endswith('.mp4') or f.endswith('.webm')]

def vid2frames(f):
    #extracts frames from downloaded video
    print('Extracting frames from ' + f)
    id = f[:10].strip()
    capture = cv2.VideoCapture(f)
    bool,frame = capture.read()
    frameNum = 0
    bool = True
    while bool:
      cv2.imwrite(id+'_frame%d.jpg' % frameNum, frame)
      bool,frame = capture.read()
      frameNum += 1
    print('\tDone.')

def createDir(name):
    #if it doesn't already exist, creates a direcctory called name
    curr = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(curr + '/' + name):
        os.makedirs(curr + '/' + name)

def moveFrames(destDir):
    #Organize frames into destDir. Assumes you don't have other jpegs in the current directory
    print('Moving frames to ' + destDir)
    curr = os.path.dirname(os.path.abspath(__file__))
    frames = [f for f in os.listdir() if f.endswith('.jpg')]
    for f in frames:
        os.rename(curr + '/' + f, curr + '/' + destDir + '/' + f)
    print('\tDone.')

def greyify(dir):
    print('Greyifying frames in ' + dir)
    path = os.path.dirname(os.path.abspath(__file__)) + '/' + dir
    for f in os.listdir(path):
        img = Image.open(dir + '/' + f).convert('L')
        img.save(dir + '/' + f)
    print('\tDone.')

def frames_to_csv(dir, dest):
    print('Converting frames in ' + dir + ' to csv at ' + dest)
    path = os.path.dirname(os.path.abspath(__file__)) + '/' + dir

    data = []

    for f in os.listdir(dir):
        if f.endswith('.jpg'):
            img = Image.open(dir + '/' + f)
            img = img.resize((200, 100), Image.ANTIALIAS) #resize image
            img_v = np.array(img).flatten()/255.0 #flatten and normalize
            label = f[:-4] #just scrapes off the .jpg
            vals = img_v.tolist()
            data.append([label]+vals)

    with open(dest, 'w') as dest:
        writer = csv.writer(dest)
        writer.writerows(data)

    print('\tDone.')

#main method - just everything in sequence
def main():
    

    for url in url_list:
        dlv(url)

    for vid in getVidList():
        vid2frames(vid)

    createDir('Frames_bambi')
    moveFrames('Frames_bambi')
    greyify('Frames_bambi')
    
    frames_to_csv('Frames_bambi_subset', 'data_subset.csv')

    print('Done.')
#------------


#can add youtube links to videos here
url_list = ['https://www.youtube.com/watch?v=MaWnvvrngyQ']

#run the whole shabang
main()
