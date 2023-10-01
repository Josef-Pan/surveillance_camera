import glob, piexif
from PIL import Image
import numpy as np
from .utils import print_exception


def glob_image_files(source: str, formats=['.jpg', '.png', '.jpeg'], filters=None, reverse: bool = False):
    """
    Search <source> for image files recursively, both lower case and upper case files
    will be included in result
    :param filters: A fileter string to be used at the front of the file name
    :param source:
    :param formats:
    :param reverse, whether to return images in reversed order
    :return: a sorted list of images
    """
    image_files = []
    if filters:
        for format_ in formats:
            image_files_of_format_1 = glob.glob(source.rstrip('/') + f"/{filters}*.{format_.lower()}")
            image_files_of_format_2 = glob.glob(source.rstrip('/') + f"/{filters}*.{format_.upper()}")
            image_files = image_files + image_files_of_format_1 + image_files_of_format_2
    else:
        for format_ in formats:
            image_files_of_format_1 = glob.glob(source.rstrip('/') + f"/*{format_.lower()}")
            image_files_of_format_2 = glob.glob(source.rstrip('/') + f"/*{format_.upper()}")
            image_files = image_files + image_files_of_format_1 + image_files_of_format_2
    return sorted(image_files, reverse=reverse)


def get_exif_with_lens_make(file):
    try:
        exif_dict = piexif.load(file)
    except:
        return None
    try:
        lens_make = exif_dict['Exif'][piexif.ExifIFD.LensMake].decode("utf-8")
        return lens_make
    except:
        return None


def mark_exif_with_lens_make(file, lens_make_string=''):
    try:
        exif_dict = piexif.load(file)
    except:
        exif_dict = {}
    try:
        exif_dict['Exif'][piexif.ExifIFD.LensMake] = lens_make_string.encode("utf-8")
        exif_bytes = piexif.dump(exif_dict)
        im = Image.open(file)
        im.save(file, exif=exif_bytes)
    except Exception as instance:
        print(f'\033[31mFailed to set exif data\033[0m\n{str(instance.args)}')


def save_img_np(file: str, image_np: np.ndarray) -> bool:
    try:
        img_new = np.asarray(image_np).astype(np.uint8)
        img_new = Image.fromarray(img_new)
        img_new.save(file)
        return True
    except Exception as e:
        print_exception(e)
        return False


def save_img_np_grayscale(file: str, image_np: np.ndarray) -> bool:
    try:
        img_new = np.asarray(image_np).astype(np.uint8)
        img_new = Image.fromarray(img_new)
        img_new = img_new.convert("L")
        img_new.save(file)
        return True
    except Exception as e:
        print_exception(e)
        return False
