import pytest
import glob
import random

DATA_PATH='/home/sravya/data/DicomSegmentation/final_data/'
SIZE = 256

@pytest.fixture
def du():
    from dicom_utils import DicomUtils
    return DicomUtils(DATA_PATH)

def test_parse_dicom_file(du):
    path = du.dicom_path + '*/*'
    dicom_file = random.choice(glob.glob(path))
    dicom_arr = du.parse_dicom_file(dicom_file)
    assert dicom_arr.shape == (SIZE, SIZE), "Dicom shape is not ({},{})".format(SIZE, SIZE)

def test_parse_contour_file(du):
    # Create a dummy contour file
    x = []
    y = []
    x.append(50)
    y.append(50)
    x.append(x[0] + 50)
    y.append(y[0])
    x.append(x[0] + 50)
    y.append(y[0] + 50)
    x.append(x[0])
    y.append(y[0] + 50)
    with open('workfile', 'w') as f:
        for i in range(4):
            entry = str(x[i]) + ' ' + str(y[i]) + '\n'
            f.write(entry)
    f.close()

    height = SIZE
    width = SIZE
    mask, polygon = du.parse_contour_file('workfile', width, height)
    assert mask.shape == (width, height), "Mask shape is not ({},{})".format(width, height)
    assert len(polygon) == 4, "Unexpected polygon size {}".format(len(polygon))

    for i in range(SIZE):
        for j in range(SIZE):
            if (50 <= i <= 100 and 50 <= j <= 100):
                assert mask[i][j] == True, "i = {}, j = {}".format(i, j)
            else:
                assert mask[i][j] == False, "i = {}, j = {}".format(i, j)

    #TODO: Use temporary directory/file so that we clean up workfile elegeantly

def test_get_dicom_and_mask(du):
    path = du.contour_path + '*/i-*/*'
    contour_file = random.choice(glob.glob(path))
    height = SIZE
    width = SIZE
    dicom, mask, polygon = du.get_dicom_and_mask(contour_file)
    assert dicom.shape == (width, height), "Dicom shape is not ({},{})".format(width, height)
    assert mask.shape == (width, height), "Mask shape is not ({},{})".format(width, height)
    assert len(polygon) != 0, "Polygon is empty"

def test_data_generator(du):
    path = du.contour_path + '*/i-*/*'
    cfiles = glob.glob(path)
    batch_size = 2
    gen = du.data_generator(batch_size, cfiles[:8])
    i = 0
    try:
        while (1):
            dicoms, masks = next(gen)
            i += 1
            assert dicoms.shape == (SIZE * batch_size, SIZE), "Unexpected batched DICOM shape"
            assert masks.shape == (SIZE * batch_size, SIZE), "Unexpected batched mask shape"
    except StopIteration:
        pass
    assert i == 8 / batch_size, "Unexpected number of iterations when using data_generator"
    gen.close()

"""
Todo:
Error test cases:
- Data folder structure:
  - data_path does not exist
  - dicom path does not exist
  - contour path does not exist
  - link file does not exist
  - I contour file name pattern is different
  - I countour folder name is different (used in data_generator
- DicomError: How to generate illformatted dicom for test?
- i-contour/ocountor polygons are outside the image
"""