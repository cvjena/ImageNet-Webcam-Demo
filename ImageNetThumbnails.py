import sys
import os
import urllib3
import logging

import scipy.misc
import glob

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
  imgfntemplate = os.path.join(outputdir,'%s_thumbnail_%04d.jpg')

  synset_images = list(glob.glob(os.path.join(imagenet_dir, train, wnid, '*')))
  random.shuffle(synset_images)
  
  for i in range(k):
    im = scipy.misc.imread(synset_images[i])
    #im = scipy.misc.resize(

  http = urllib3.PoolManager()
  url = "http://www.image-net.org/api/text/imagenet.synset.geturls?wnid={}".format(wnid)
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





if __name__ == "__main__":
  download('n04116512', 3, verbose=True)

