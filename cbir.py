import numpy as np
import cv2
import scipy.spatial.distance as distance
import os, glob, math
import settings

# cv2.getGaborKernel(ksize, sigma, theta, lambda, gamma, psi, ktype)
# ksize  - size of gabor filter (n, n)
# sigma  - standard deviation of the gaussian function
# theta  - orientation of the normal to the parallel stripes
# lambda - wavelength of the sunusoidal factor
# gamma  - spatial aspect ratio
# psi    - phase offset
# ktype  - type and range of values that each pixel in the gabor kernel can hold
class TBIR(object):
    def __init__(self, **kargs):
        self.dataset_dir        = kargs.get("dataset_dir", None)
        self.descriptors_dir    = kargs.get("descriptors_dir", None)
    
    def _get_descriptor(self, image_file):
        kernel        = cv2.getGaborKernel((21, 21), 8.0, np.pi/4, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        kernel       /= math.sqrt((kernel * kernel).sum())
        image         = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        filtered_img  = cv2.filter2D(image,    cv2.CV_8UC3, kernel)
        heigth, width = kernel.shape
        
        # convert matrix to vector 
        descriptor = cv2.resize(filtered_img, (3*width, 3*heigth), interpolation=cv2.INTER_CUBIC)
        return np.hstack(descriptor)
    
    def save_array_to_file(self, **kargs):
        descriptor_to_save = kargs.get("descriptor", None)
        image_file         = kargs.get("image_path", None)
        image_file         = os.path.join(self.descriptors_dir, os.path.dirname(image_file),"{}.npy".format(os.path.basename(image_file)))
        
        # save descriptors in away that we can retrieve their images easily..
        if not os.path.exists(os.path.dirname(image_file)):
            os.makedirs(os.path.dirname(image_file)) 

        if settings.DEBUG:
            print("\r Saving descriptor: {} ...".format(image_file), end="")

        np.save(image_file, descriptor_to_save, allow_pickle=False)

    def _descriptor_distance(self, descriptors_file, inputed_descriptor):
        descriptor = np.load(descriptors_file)
        return abs(distance.euclidean(descriptor, inputed_descriptor))

    def offline_phase(self):
        all_descriptors = []
        for entry in glob.glob(os.path.join(self.dataset_dir, "**","*.jpg")):
            descriptor    = self._get_descriptor(entry)
            self.save_array_to_file(descriptor=descriptor, image_path=entry)
    
    def online_phase(self, test_iamge):
        test_descriptor = self._get_descriptor(test_iamge)
        best_fit_images = []

        for descriptor in glob.glob(os.path.join(self.descriptors_dir, self.dataset_dir, "**","*.npy")):
            entry = {"image_path": descriptor.replace(".npy","").replace(self.descriptors_dir,"."),
                     "distance"  : self._descriptor_distance(descriptor, test_descriptor)
                    }
            best_fit_images.append(entry)
            best_fit_images.sort(key= lambda x:x["distance"], reverse=False)
            best_fit_images = best_fit_images[:settings.ELITE_NUMBER]
        
        return best_fit_images
