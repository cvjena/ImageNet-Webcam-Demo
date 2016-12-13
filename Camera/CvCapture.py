"""Camera abstraction layer for Linux using OpenCV.

The Capture class provided from this module uses the OpenCV package
for Python to access video devices.

Author: Bjoern Barz
"""

import cv2
from PIL import Image


class Capture(object):
    """Provides access to video devices."""

    def __init__(self, index = 0, requested_cam_size=(640,480)):
        """Opens a video device for capturing.
        
        index - The number of the device to open.
        Throws an exception if the device can't be opened or if the given index
        is out of range.
        """
        print("CvCapture: initialize camera")
        object.__init__(self)
        self.capture = cv2.VideoCapture()
        self.capture.open(index)
        self.capture.set( cv2.CAP_PROP_FRAME_WIDTH, requested_cam_size[0] )
        self.capture.set( cv2.CAP_PROP_FRAME_HEIGHT, requested_cam_size[1] )

    def __del__(self):
        self.capture.release()


    def grabFrame(self):
        """Returns a snapshot from the device as PIL.Image.Image object."""
        
        data, w, h, orientation = self.grabRawFrame()
        return Image.fromstring("RGB", (w, h), data, "raw", "BGR", 0, orientation)


    def grabRawFrame(self):
        """Returns a snapshot from this device as raw pixel data.
        
        This function returns a 4-tuple consisting of the raw pixel data as string,
        the width and height of the snapshot and it's orientation, which is either
        1 (top-to-bottom) or -1 (bottom-to-top).
        """
        
        result, cimg = self.capture.read() # cimg will represent the image as numpy array
        height, width, depth = cimg.shape
        return cimg.tostring(), width, height, 1


    @staticmethod
    def enumerateDevices():
        """Lists all available video devices.
        
        Returns a tuple of 2-tuples, which contain the integral index
        and the display name (if available) of the video device.
        """
        
        devices = ()
        i = 0
        while True:
            try:
                cap = cv2.VideoCapture(i)
                if not cap.isOpened():
                    cap.release()
                    break
                cap.release()
                devices += ((i, 'Device #{0}'.format(i)),)
                i += 1
            except:
                break
        return devices
