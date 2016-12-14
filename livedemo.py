import argparse
import logging
import sys
import os
import math
import time
import numpy as np
import matplotlib.pyplot as pylab
import scipy.misc
import functools

sys.path.append('./caffe_pp2/python/')
import caffe
caffe.set_mode_gpu()

import pygame.image
import pygame.surfarray
import threading
import json
import PIL

import ImageNetThumbnails

sys.path.append('./classification_framework2/')
import Classification
import Resize
import Crop
import Caffe


# In[2]:

# main function
parser = argparse.ArgumentParser()
parser.add_argument( '-c', '--categories', help='reduced list of categories as a JSON hash', default='data/all_categories.json' )
parser.add_argument( '--width', type=int, help='requested camera width', default=512 )
parser.add_argument( '--height', type=int, help='requested camera height', default=512 )
parser.add_argument( '--thumbdir', help='directory with thumbnail images for the synsets', default='./thumbnails' )
parser.add_argument( '--downloadthumbs', help='download non-existing thumbnail images', action='store_true')
parser.add_argument( '--threaded', help='use classification thread', action='store_true')
parser.add_argument( '--nocenteronly', help='disable center-only classification mode', action='store_true', default=False)
parser.add_argument( '--offlinemode', help='download|decode|directory', choices=['download', 'decode', 'directory'])
parser.add_argument( '--url', help='youtube video that will be downloaded in offline mode' )
parser.add_argument( '--videofile', help='video file that will be processed in offline mode' )
parser.add_argument( '--videodir', help='directory with PNG files that will be processed in offline mode' )
parser.add_argument( '--loglevel', help='log level', choices=['debug','info','warning','error','critical'], default='info')
parser.add_argument( '--delay', help='delay (0=no delay, negative value=button wait, positive value=milliseconds to wait)', type=float, default=0)
parser.add_argument( '--pooling', help='type of pooling used', choices=['avg', 'none', 'max'], default='none' )
parser.add_argument( '--poolingsize', help='pooling size', type=int, default=100 )
parser.add_argument( '--cnn_model_dir', help='Folder that contains the CNN model. This folder should contain a deploy.protoxt and a file called "model".', default='./model/alexnet_ep_fc2/')
parser.add_argument( '--fontsize', type=int, default=50)
args = parser.parse_args([])


# In[3]:

# We assume the blob names are the same as the layer name, which produce these blobs
global selected_blob, prob_blobs
prob_blobs = ['anytime_prob_{:02d}'.format(i) for i in [1,2,3,4,5]]
selected_blob = len(prob_blobs) - 1
print('Initially using blob {}'.format(prob_blobs[selected_blob]))


# In[4]:

proto = os.path.join(args.cnn_model_dir, 'deploy.prototxt')
cnn_model = os.path.join(args.cnn_model_dir, 'model')


# In[5]:

numeric_level = getattr(logging, args.loglevel.upper(), None)
assert isinstance(numeric_level, int)
logging.basicConfig(level=numeric_level)


# In[6]:

def draw_text(text, pos):
  myfont = pygame.font.SysFont("monospace", args.fontsize)
  myfont.set_bold(True)
  text_object = myfont.render(text, 1, (255,0,0), (0,0,0))
  screen.blit(text_object, pos)


# In[7]:

class SingleFunctionThread(threading.Thread):
  """ Class used for threading """

  def __init__(self, function_that_classifies):
    threading.Thread.__init__(self)
    self.runnable = function_that_classifies
    self.daemon = True

  def run(self):
    while True:
      self.runnable()



""" load, rescale, and store thumbnail images """
def create_thumbnail_cache(synsets, timgsize, thumbdir):
  timgsize = np.array(timgsize).astype(int)
  maxk = 3
  maxtries = 10

  logging.info("Loading thumbnails ...")
  for synset in synsets:
    logging.debug("Caching thumbnails for synset %s" % (synset))
    tryk = 0
    successk = 0
    while tryk < maxtries and successk < maxk:
      thumbfn = os.path.join(thumbdir, '%s'%synset, 'thumbnail_%04d.jpg'%tryk)
      try:
        timgbig = pygame.image.load( thumbfn )
      except:
        tryk = tryk + 1
        continue

      logging.debug("Storing image %s %d: %s" % ( synset, successk, thumbfn ))

      successk = successk + 1
      tryk = tryk + 1

      timg = pygame.transform.scale ( timgbig, timgsize )
      if not synset in thumbnail_cache:
        thumbnail_cache[synset] = []
      thumbnail_cache[synset].append(timg)




""" given a list of synsets, display the thumbnails """
def display_thumbnails(synsets, woffset, wsize, numthumbnails=3):
  screen.fill ( (0,0,0), pygame.Rect(woffset[0], woffset[1], wsize[0], wsize[1]) )
  timgsize = ( wsize[0] / len(synsets), wsize[1] / numthumbnails )
  for i in range(len(synsets)):
    synset = synsets[i]
    if synset in thumbnail_cache:
      for k in range( len(thumbnail_cache[synset]) ):
          y = timgsize[0] * k + woffset[0]
          x = timgsize[1] * i + woffset[1]
          screen.blit(thumbnail_cache[synset][k],(y,x))


def display_results(synsets, scores, woffset, wsize):
  # delete previous area
  screen.fill ( (0,0,0), pygame.Rect(woffset[0], woffset[1], wsize[0], wsize[1]) )

  myfont = pygame.font.SysFont("monospace", args.fontsize)
  myfont.set_bold(True)
  rowsep = int ( wsize[1] / len(synsets) )
  rowoffset = rowsep/2

  sumscores = 0
  for i in range(len(synsets)):
    sumscores = sumscores + scores[i]

  for i in range(len(synsets)):
    text = "{:>4d}% - {}".format(int(scores[i] * 100), synsets[i].split(',')[0])
    #text = synsets[i]
    label = myfont.render(text, 1, (255,0,0), (0,0,0) )
    screen.blit(label, (woffset[0], woffset[1] + i * rowsep + rowoffset ))


# In[8]:

global elapsed
elapsed = 0


# In[9]:

""" classify the image, over and over again """
def classify_image(center_only=True):
  if capturing:
    # transpose image :)
    camimg = np.transpose(pygame.surfarray.array3d(img), [1,0,2])
    
    logging.debug("Classification (image: %d x %d)" % (camimg.shape[1], camimg.shape[0]))
    
    camimg = camimg.astype(np.float32)/255
    scores = classifier.compute(camimg)
    # Ugly hack to get the pure CNN time
    start = time.time()
    net.get_net().forward(end=prob_blobs[selected_blob])
    elapsed = time.time() - start

    if pooling!='none':
      all_scores.append(scores)

      if len(all_scores)>pooling_size:
        all_scores.pop(0)

      logging.debug("Pool size: {0}".format(len(all_scores)))

      pooled_scores = all_scores[0]
      if pooling=='avg':
        for s in all_scores[1:]:
          pooled_scores = np.add( pooled_scores, s )
        pooled_scores = pooled_scores / len(all_scores)
      else:
        for s in all_scores[1:]:
          pooled_scores = np.fmax( pooled_scores, s )

      scores = pooled_scores

    if categories:
      top_class_ids = [class_id for class_id in np.argsort(-scores) if label_names[class_id] in categories]
    else:
      top_class_ids = np.argsort(-scores)
    
    top_scores = scores[top_class_ids]
    top_classes = label_names[top_class_ids]
    top_desc = label_desc[top_class_ids]
    
    logging.debug("ImageNet guesses (1000 categories): {0}".format(top_classes[0:5]))

    display_scores = top_scores[:5]
    display_synsets = top_classes[:5]
    display_descs = top_desc[:5]

    if categories:
      logging.debug("Reduced set ({0} categories): {1}".format(len(categories), display_descs))


    imgsize = (camimg.shape[1],camimg.shape[0])
    display_thumbnails( display_synsets[0:3], imgsize, imgsize )
    display_results ( display_descs[0:3], display_scores[0:3], (0,camimg.shape[0]), imgsize )
    draw_text('Calc time {:>3d} ms'.format(int(elapsed*1000)),( int(camimg.shape[1]*1.2),int(camimg.shape[0]*0.8)))


# In[10]:

requested_cam_size = (args.width,args.height)

# deep net init
global classifier, net
classifier = Classification.Classification()
cropsize = 224
classifier.add_algorithm(Resize.Resize((256,), mode='resize_smaller_side'))
classifier.add_algorithm(Crop.Crop((cropsize,cropsize),'center'))
net = Caffe.Caffe(proto, cnn_model,
                mean = np.float32([0,0,0]),
                outblob = prob_blobs[selected_blob],
                endlayer = prob_blobs[selected_blob],
                batchsize = 1)
classifier.add_algorithm(net)

# In[12]:

# pygame general initialization
ret = pygame.init()
logging.debug("PyGame result: {0}".format(ret))
logging.debug("PyGame driver: {0}".format(pygame.display.get_driver()))


# In[13]:

if args.offlinemode:
    from Camera.VideoCapture import Capture
    logging.info("Selecting the first camera")
    cam = Capture(requested_cam_size=requested_cam_size, url=args.url, videodir=args.videodir, mode=args.offlinemode, videofile=args.videofile)
    cam_size = requested_cam_size
else:
    from Camera.Capture import Capture
    logging.info("List of cameras:")
    logging.info(Capture.enumerateDevices())
    cam = Capture(index=len(Capture.enumerateDevices())-1, requested_cam_size=requested_cam_size)
    timg, width, height, orientation = cam.grabRawFrame()
    cam_size = (width, height)
    logging.info("Video camera size: {0}".format(cam_size))


# In[14]:

# pooling settings
global all_scores
all_scores = []
pooling = args.pooling
pooling_size = args.poolingsize


# In[15]:

# load categories
global categories
categories = {}
if args.categories:
  categories = json.load(open( args.categories))


# In[16]:

global label_names
label_names = np.array([ll.split(' ')[0] for ll in open('data/synset_descriptions.txt','rt').read().splitlines()])


# In[17]:

global label_desc
label_desc = np.array([ll[ll.find(' ')+1:] for ll in open('data/synset_descriptions.txt','rt').read().splitlines()])


# In[18]:

# preload synset thumbnails
logging.debug("Initialize thumbnails")
global thumbnail_cache
thumbnail_cache = {}

logging.debug("Pre-downloading thumbnails")
if args.downloadthumbs:
  for idx, synset in enumerate(categories):
    logging.info("%d/%d %s" % ( idx, len(categories), synset))
    #ImageNetThumbnails.download(synset, 6, verbose=True, overwrite=False, outputdir=args.thumbdir)
    ImageNetThumbnails.generate('/home/atlas1_ssd/simon/ilsvrc12-scaled/', synset, 6, verbose=True, overwrite=False, outputdir=args.thumbdir)
create_thumbnail_cache ( categories.keys(), (cam_size[0]/3, cam_size[1]/3), args.thumbdir )


# In[19]:

logging.debug("Initialize screen")
# open window
global screen
screen = pygame.display.set_mode( ( 2*cam_size[0], 2*cam_size[1] ), (pygame.RESIZABLE)   )


# In[20]:

# starting the threading
global img
global capturing
capturing = True


# In[21]:

if args.threaded:
  logging.debug("Initialize thread")
  thread = SingleFunctionThread(functools.partial(classify_image, not args.nocenteronly))
  thread.start()

if args.delay>0:
  pygame.time.set_timer(pygame.USEREVENT + 1, int(args.delay*1000))


# In[22]:

running = True
finished = False
webcam_image_buffer = cam.grabFrameNumpy()[0][:,:,::-1].copy()
reference_size = np.array(webcam_image_buffer.shape[:2][::-1])

pipeline = np.array(PIL.Image.open('./drawings/pipeline.png'))
scale_factor = webcam_image_buffer.shape[1] / min(pipeline.shape[:2]) * 0.25
pipeline = scipy.misc.imresize(pipeline, scale_factor)
pipeline_s = pygame.image.frombuffer(pipeline.copy(), pipeline.shape[:2][::-1], "RGBA")
pipeline_r = np.array(pipeline.shape[:2][::-1])

prediction = np.array(PIL.Image.open('./drawings/prediction.png'))
prediction = scipy.misc.imresize(prediction, scale_factor)
prediction_s = pygame.image.frombuffer(prediction.copy(), prediction.shape[:2][::-1], "RGBA")

while not finished:
  if running:
    screen.fill ( (0,0,0), screen.get_clip())
    
    logging.debug("Capture image")
    capturing = False
    webcam_image_buffer[...] = cam.grabFrameNumpy()[0][:,:,::-1]
    img = pygame.image.frombuffer(webcam_image_buffer, webcam_image_buffer.shape[:2][::-1], "RGB" )
    img = pygame.transform.flip(img, False, False)
    screen.blit(img,(0,0))
    capturing = True
    
    if not args.threaded:
      classify_image(center_only=(not args.nocenteronly))

    screen.blit(pipeline_s, (reference_size*np.array([1.2,0.1])).astype(int) )
    screen.blit(prediction_s, (reference_size*np.array([1.2,0.1]) + pipeline_r*np.array([-0.1,0.38]) + selected_blob*pipeline_r*np.array([0.19,0])).astype(int) )
    
    pygame.display.flip()

  blocking = True
  while blocking:
    for event in pygame.event.get():
      if event.type==pygame.QUIT:
        sys.exit()
    
      if event.type==pygame.KEYDOWN:
          if event.key==pygame.K_SPACE:
            logging.debug("Setting running flag to: {0}".format(running))
            running = not running
            pygame.event.clear(pygame.KEYUP)
            pygame.event.clear(pygame.KEYDOWN)
          finished = event.key==pygame.K_q
          if event.key in [pygame.K_RIGHT, pygame.K_UP, pygame.K_PAGEUP] and selected_blob<len(prob_blobs)-1:
            selected_blob += 1
            net.set_output(outblob = prob_blobs[selected_blob],
                endlayer = prob_blobs[selected_blob])
            logging.info("Switching to blob {}".format(prob_blobs[selected_blob]))
          if event.key in [pygame.K_LEFT, pygame.K_DOWN, pygame.K_PAGEDOWN] and selected_blob>0:
            selected_blob -= 1
            net.set_output(outblob = prob_blobs[selected_blob],
                endlayer = prob_blobs[selected_blob])
            logging.info("Switching to blob {}".format(prob_blobs[selected_blob]))        
      if event.type==pygame.USEREVENT+1:
        blocking = False

    if args.delay==0:
      blocking = False

#  if args.delay>0:
#   time.sleep(args.delay)


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



