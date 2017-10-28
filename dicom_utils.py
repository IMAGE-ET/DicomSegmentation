"""This class contains utilities for handling DICOM files and associated contour files for segmentation purposes

Note: Code follows the following directory structure - please make modifications to section DATA STRUCTURE below
if data directory structure conventions changes.

* DICOM_PATH contains sub folders one for each patient.
    * Directory named after patient_id (SCD0000101). See link.scv for details
    * Each patient's folder contains ~200 DICOM files (1.dcm, 200.dcm)
* CONTOUR_PATH contains sub folders one for each patient.
    * Directory named after original_id (SC-HF-I-1). See link.csv for details
    * Each folder contains i-contours and o-contours sub folders
    * Each i-contour folder contains ~20 contour files (IM-0001-0048-icontour-manual.txt)
    * Assumption: 0048 refers to the dcm file name. TODO:Check assumption

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
CFILE_REGEX_PATTERN = '(SC-HF-I-\d+)/i-contours/IM-\d+-(\d+)-icontour-manual.txt'
CFILE_GLOB_PATTERN = '*/i-*/*'

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
            raise ValueError('Path does not exist: %s'.format(data_path))
        if (not os.path.exists(data_path + DICOM_SUFFIX)):
            raise ValueError('Path does not exist: %s \n Please update DICOM_SUFFIX'.format(data_path + DICOM_SUFFIX))
        if (not os.path.exists(data_path + CONTOUR_SUFFIX)):
            raise ValueError('Path does not exist: %s \n Please update CONTOUR_SUFFIX'.format(data_path + CONTOUR_SUFFIX))
        if (not os.path.exists(data_path + LINK_FILE)):
            raise ValueError('Path does not exist: %s \n Please update LINK_FILE'.format(data_path + LINK_FILE))

    def parse_dicom_file(self, filename):
        """Parse the given DICOM filename

            :param filename: filepath to the DICOM file to parse
            :return: DICOM image data

            To understand slope and intercept: https://blog.kitware.com/DICOM-rescale-intercept-rescale-slope-and-itk/
            """
        try:
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
        except InvalidDicomError:
            return None

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
                    raise ValueError("Unexpected content in contour file %s".format(filename))

                x_coord = float(coords[0])
                y_coord = float(coords[1])
                polygon.append((x_coord, y_coord))

        img = Image.new(mode='L', size=(width, height), color=0)
        ImageDraw.Draw(img).polygon(xy=polygon, outline=1, fill=1)
        mask = np.array(img).astype(bool)
        return mask, polygon

    def get_dicom_and_mask(self, cfile):
        """
        Parses contour file and the associated DICOM file and returns numpy arrays for DICOM and mask,
        along with the polygon
        Note: Leverages contour filename to get the associated DICOM file 
        :param cfile: Contour filename
        :return: Numpy arrays for DICOM and mask, polygon representing the contour
        """
        dicom_path = self.get_dicom_path(cfile)
        dicom_arr = self.parse_dicom_file(dicom_path)

        shape = dicom_arr.shape
        mask, polygon = self.parse_contour_file(cfile, shape[0], shape[1])
        return dicom_arr, mask, polygon

    def get_dicom_path(self, cfile):
        """
        Gets DICOM filepath given contour filepath
        :param cfile: Contour filepath
        :return: Associated DICOM filepath
        """
        pattern = self.contour_path + CFILE_REGEX_PATTERN
        match = re.search(pattern, cfile)
        error_msg = "Unexpected contour filename pattern, please update CFILE_REGEX_PATTERN as needed"
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

    def visualize_overlay(self, dicom_arr, polygon):
        """
        Visualization utility to overlay polygon over the DICOM
        :param DICOM_arr: numpy DICOM
        :param polygon: contour polygon
        :return: displays overlayed DICOM
        """
        plt.clf()
        plt.imshow(dicom_arr)
        x = [point[0] for point in polygon]
        y = [point[1] for point in polygon]
        plt.plot(x, y, 'r', alpha=1)
        plt.show()

    def data_generator(self, batch_size=8, cfiles = None):
        """
        A generator which serves batches of DICOM and associated masks

        :param batch_size: Number of samples per iteration
        :param cfiles: This parameter is only for unit test purposes. Please do not use it in production
        :return: Batched DICOM and mask numpy arrays
        """
        if(cfiles == None):#cfiles is set only in unit tests
            cregex = self.contour_path + CFILE_GLOB_PATTERN
            cfiles = glob.glob(cregex)

        if(len(cfiles) == 0):
            raise ValueError('Could not find contour files in the format %s.\n Please update CFILE_GLOB_PATTERN '
                             .format(cregex))

        random.shuffle(cfiles)
        print("Starting data generator")

        steps = math.floor(len(cfiles) / batch_size)
        #TODO: If total samples is not a multiple of batch size, we are currently not utilizing the incomplete last batch
        for i in range(steps):
            #TODO: Handle SIZE better
            yield self.get_data(cfiles, i * batch_size, i * batch_size + batch_size, SIZE)

    def get_data(self, cfiles, start, end, size):
        """
        Processes contour files indexed from start to end within cfiles list and batches them
        :param cfiles: list of contour files
        :param start: start index (inclusive)
        :param end: end index (exclusive)
        :return: Batched DICOM and mask numpy arrays
        """
        # Need to reshape the empty np holder to be able to concatenate inside for loop below
        dicom_final = np.array([]).reshape(0, size)
        mask_final = np.array([]).reshape(0, size)

        for i in range(start, end):
            dicom_arr, mask, polygon = self.get_dicom_and_mask(cfiles[i])
            dicom_final = np.concatenate([dicom_final, dicom_arr])
            mask_final = np.concatenate([mask_final, mask])
        return dicom_final, mask_final