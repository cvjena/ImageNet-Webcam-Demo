import sys
import os
import urllib3
import logging

import scipy.misc
import random
import glob
import numpy as np

def download(wnid, k, verbose=False, overwrite=True, outputdir='.', timeout=2):
  # template
  imgfntemplate = outputdir + os.path.sep + '%s_thumbnail_%04d.jpg'

  # download urls
  if not overwrite:
    thumbnails_available = True
    for i in range(k):
      imgfn = imgfntemplate % (wnid, i)
      if not os.path.isfile(imgfn):
        thumbnails_available = False
        break
    
    if thumbnails_available:
      if verbose:
        logging.info( "Thumbnails are already available for %s" % (wnid) )
      return

  http = urllib3.PoolManager()
  url = 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=%s' % (wnid)
  response = http.request('GET', url )
  urllist = response.data.decode('utf-8').splitlines()
  ini = 0
  outi = 0
  while ini < len(urllist) and outi < k:
    imgurl = urllist[ini].rstrip()
    if verbose:
      logging.info( "Thumbnail URL: %s" % (imgurl) )
    imgfn = imgfntemplate % (wnid, outi)
    ini = ini + 1
      
    if not overwrite and os.path.isfile(imgfn):
      logging.info( "Thumbnail %s already available" % (imgfn) )
      continue

    try:
      with open(imgfn, 'wb') as imgout:
        r = http.request('GET', imgurl, timeout=timeout )
        imgout.write( r.data )

      # check the file 
      scipy.misc.imread( imgfn )
      outi = outi + 1
    except:
      logging.error("Error: Unable to download image from %s ... skipping" % (imgurl))
      # clean up to avoid errors with the overwrite=False setting
      os.remove( imgfn )
      continue

    if verbose:
      logging.info("Thumbnail %s downloaded" % (imgfn))


def generate(imagenet_dir, wnid, k, verbose=False, overwrite=True, outputdir='.', timeout=2):
  # template
  name_template = os.path.join(outputdir,'{}', 'thumbnail_{:04d}.jpg')

  synset_images = list(glob.glob(os.path.join(imagenet_dir, wnid, '*')))
  random.shuffle(synset_images)

  for i in range(6):
    im = scipy.misc.imread(synset_images[i])
    offset = np.floor((np.array(im.shape[:2]) - min(im.shape[:2]))/2).astype(int)
    im = im[offset[0]:offset[0]+im.shape[0], offset[1]:offset[1]+im.shape[1]]
    im = scipy.misc.imresize(im, (250,250))
    outfile = name_template.format(wnid,i)
    # Create folder 
    try: 
      os.makedirs(os.path.split(name_template.format(wnid,i))[0]) 
    except: 
      pass
    scipy.misc.imsave(outfile, im)


if __name__ == "__main__":
  download('n04116512', 3, verbose=True)

