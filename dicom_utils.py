"""This class contains utilities for handling DICOM files and associated contour files for segmentation purposes

Note: Code follows the following directory structure - please make modifications to section DATA STRUCTURE below
if data directory structure conventions changes.

* DICOM_PATH contains sub folders one for each patient.
    * Directory named after patient_id (SCD0000101). See link.scv for details
    * Each patient's folder contains ~200 DICOM files (1.dcm, 200.dcm)
* CONTOUR_PATH contains sub folders one for each patient.
    * Directory named after original_id (SC-HF-I-1). See link.csv for details
    * Each folder contains i-contours and o-contours sub folders
    * Each i-contour folder contains ~20 contour files (IM-0001-0040-icontour-manual.txt)
    * Each o-counter folder contains ~10 contour files (IM-0001-0040-ocontour-manual.txt)
    * Assumption: 0040 refers to the dcm file name. TODO:Check assumption    
* TODO:
    * Faster data_generator: Store the preprocessed dicom and mask arrays on disk (using pickle?) to
    * Setup instructions which includes required dependencies
    * Handle dicoms of sizes other than 256

"""
import csv
import dicom
from dicom.errors import InvalidDicomError
import glob
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import math
import random, re, os

# DATA STRUCTURE
DICOM_SUFFIX = 'dicoms/'
CONTOUR_SUFFIX = 'contourfiles/'
LINK_FILE = 'link.csv'
CFILE_REGEX_PATTERN = '(SC-HF-I-\d+)/[io]-contours/IM-\d+-(\d+)-[io]contour-manual.txt'
OCFILE_GLOB_PATTERN = '*/o-*/*'

SIZE = 256 #Change this if DICOM dimensions change

class DicomUtils:

    def __init__(self, data_path):
        self.validate(data_path)

        self.data_path = data_path
        self.dicom_path = data_path + DICOM_SUFFIX
        self.contour_path = data_path + CONTOUR_SUFFIX

        #linkdict is a map to connect contours with DICOMs (original_id -> patient_id)
        self.linkdict = {}

        with open(data_path + LINK_FILE, mode='r') as infile:
            reader = csv.reader(infile)
            next(reader)  # Skip header
            for rows in reader:
                self.linkdict[rows[1]] = rows[0]

    def validate(self, data_path):
        if (not os.path.exists(data_path)):
            raise ValueError('Path does not exist: {}'.format(data_path))
        if (not os.path.exists(data_path + DICOM_SUFFIX)):
            raise ValueError('Path does not exist: {} \n Please update DICOM_SUFFIX'.format(data_path + DICOM_SUFFIX))
        if (not os.path.exists(data_path + CONTOUR_SUFFIX)):
            raise ValueError('Path does not exist: {} \n Please update CONTOUR_SUFFIX'.format(data_path + CONTOUR_SUFFIX))
        if (not os.path.exists(data_path + LINK_FILE)):
            raise ValueError('Path does not exist: {} \n Please update LINK_FILE'.format(data_path + LINK_FILE))

    def parse_dicom_file(self, filename):
        """Parse the given DICOM filename

            :param filename: filepath to the DICOM file to parse
            :return: DICOM image data

            throws InvalidDicomError if Dicom is corrupted
            To understand slope and intercept: https://blog.kitware.com/DICOM-rescale-intercept-rescale-slope-and-itk/
            """
        
        dcm = dicom.read_file(filename)
        dcm_image = dcm.pixel_array

        try:
            intercept = dcm.RescaleIntercept
        except AttributeError:
            intercept = 0.0
        try:
            slope = dcm.RescaleSlope
        except AttributeError:
            slope = 0.0

        if intercept != 0.0 and slope != 0.0:
            dcm_image = dcm_image * slope + intercept

        return dcm_image

    def parse_contour_file(self, filename, width, height):
        """Parse contour file and return associated mask and polygons

        :param filename: filepath to the contourfile to parse
        :param width: scalar image width
        :param height: scalar image height
        :return: Boolean mask of shape (height, width) and polygons
        """
        polygon = []

        with open(filename, 'r') as infile:
            for line in infile:
                coords = line.strip().split()

                if(len(coords) != 2):
                    raise ValueError("Unexpected content in contour file {}".format(filename))

                x_coord = float(coords[0])
                y_coord = float(coords[1])
                polygon.append((x_coord, y_coord))

        img = Image.new(mode='L', size=(width, height), color=0)
        ImageDraw.Draw(img).polygon(xy=polygon, outline=1, fill=1)
        mask = np.array(img).astype(bool)
        return mask, polygon

    def get_dicom_and_mask(self, ocfile):
        """
        Parses contour files and the associated DICOM file and returns numpy arrays for DICOM and masks,
        along with the polygons
        Note: Leverages contour filename to get the associated DICOM file 
        :param cfile: Contour filename
        :return: Numpy arrays for DICOM and masks, also polygons representing the contour
        """
        icfile = ocfile.replace('o-contours', 'i-contours').replace('ocontour','icontour')
  
        dicom_path = self.get_dicom_path(icfile)
        dicom_arr = self.parse_dicom_file(dicom_path)

        shape = dicom_arr.shape
        imask, ipolygon = self.parse_contour_file(icfile, shape[0], shape[1])
        omask, opolygon = self.parse_contour_file(ocfile, shape[0], shape[1])
       
        return dicom_arr, imask, ipolygon, omask, opolygon

    def get_dicom_path(self, cfile):
        """
        Gets DICOM filepath given contour filepath
        :param cfile: Contour filepath
        :return: Associated DICOM filepath
        """
        pattern = self.contour_path + CFILE_REGEX_PATTERN
        
        match = re.search(pattern, cfile)
        error_msg = "Unexpected contour filename pattern, please update CFILE_REGEX_PATTERN. Expected:{}\n Found:{}".format(pattern, cfile)
        if(match == None):
            raise ValueError(error_msg)
        try:
            orig_id = match.group(1)
            dicom_num = int(match.group(2))
        except:
            print(error_msg)
            raise

        patient_id = self.linkdict[orig_id]
        return self.dicom_path + patient_id + '/' + str(dicom_num) + '.dcm'

    def visualize_sidebyside(self, dicom_arr, mask):
        """
        Visualization utility to compare DICOM and mask side by side
        :param DICOM_arr: numpy DICOM
        :param mask: numpy mask
        :return: displays side by side plot
        """
        plt.clf()
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(dicom_arr)
        axarr[1].imshow(mask)
        plt.show()

    def visualize_overlay(self, filename, dicom_arr, polygons):
        """
        Visualization utility to overlay polygon over the DICOM
        :param DICOM_arr: numpy DICOM
        :param polygons: contour polygons
        :return: displays overlayed DICOM
        """
        plt.clf()
        plt.imshow(dicom_arr)
        colors=['r','g']
        for index, polygon in enumerate(polygons):
            x = [point[0] for point in polygon]
            y = [point[1] for point in polygon]
            plt.plot(x, y, alpha=1, color=colors[index%2])
        plt.title(filename)
        plt.show()

    def data_generator(self, batch_size=8, cfiles = None):
        """
        A generator which serves batches of DICOM and associated masks

        :param batch_size: Number of samples per iteration
        :param cfiles: This parameter is only for unit test purposes. Please do not use it in production
        :return: Batched DICOM and mask numpy arrays
        """
        if(cfiles == None):#cfiles is set only in unit tests
            cregex = self.contour_path + OCFILE_GLOB_PATTERN
            cfiles = glob.glob(cregex)

        if(len(cfiles) == 0):
            raise ValueError('Could not find contour files in the format {}.\n Please update OCFILE_GLOB_PATTERN '
                             .format(cregex))

        random.shuffle(cfiles)
        print("Starting data generator")

        steps = math.floor(len(cfiles) / batch_size)
        #TODO: If total samples is not a multiple of batch size, we are currently not utilizing the incomplete last batch
        for i in range(steps):
            #TODO: Handle SIZE better
            yield self.get_batch(cfiles, i * batch_size, i * batch_size + batch_size, SIZE)

    def get_batch(self, cfiles, start, end, size):
        """
        Processes contour files indexed from start to end within cfiles list and batches them
        :param cfiles: list of o-contour files
        :param start: start index (inclusive)
        :param end: end index (exclusive)
        :return: Batched DICOM and mask numpy arrays
        """
        # Need to reshape the empty np holder to be able to concatenate inside for loop below
        dicom_final = np.array([]).reshape(0, size)
        imask_final = np.array([]).reshape(0, size)
        omask_final = np.array([]).reshape(0, size)
        
        for i in range(start, end):
            ocfile = cfiles[i]
            dicom_arr, imask, ipolygon, omask, opolygon = self.get_dicom_and_mask(ocfile)
                
            dicom_final = np.concatenate([dicom_final, dicom_arr])
            imask_final = np.concatenate([imask_final, imask])
            omask_final = np.concatenate([omask_final, omask])
            
        return dicom_final, imask_final, omask_final
   
    def get_ocfiles(self, num=0):
        cfiles = []

        cregex =  self.contour_path + OCFILE_GLOB_PATTERN
        cfiles = glob.glob(cregex)
        #random.shuffle(cfiles)
        if num==0:
            return cfiles
        else:
            return cfiles[:num]