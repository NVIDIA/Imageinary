#!/usr/bin/env python
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import re
import numpy
from argparse import ArgumentParser, Namespace
from PIL import Image
from multiprocessing.pool import Pool
try:
    from mxnet.recordio import IRHeader, MXIndexedRecordIO, pack
except ImportError:
    IRHeader = None
from time import perf_counter
from typing import Generator, List, NoReturn, Optional, Tuple
from math import ceil
try:
    from tensorflow.io import TFRecordWriter
    from tensorflow.train import (BytesList,
                                  Example,
                                  Feature,
                                  Features,
                                  Int64List)
except ImportError:
    TFRecordWriter = None


STANDARD_IMAGE = 'create-images'
TFRECORD = 'create-tfrecord'
RECORDIO = 'create-recordio'
SUPPORTED_IMAGE_FORMATS = {"jpg": "jpg", "jpeg": "jpg", "bmp": "bmp",
                           "bitmap": "bmp", "png": "png"}


def parse_args() -> Namespace:
    """
    Parse arguments passed to the application.

    A custom argument parser handles multiple commands and options to launch
    the desired function.

    Returns
    -------
    Namespace
        Returns a ``Namespace`` of all of the arguments that were parsed from
        the application during runtime.
    """
    message = """
    CLI for generating a fake dataset of various quantities at different
    resolutions.

    Supported file types: .bmp, .png, and .jpg.
    Supported record types: TFRecords, and RecordIO.
    TFRecords requires an external index file creation step.
    """
    parser = ArgumentParser(message)
    # Required positional command subparser which should be specified first
    commands = parser.add_subparsers(dest='command', metavar='command')
    commands_parent = ArgumentParser(add_help=False)

    # Options specific to record types
    commands_parent.add_argument('source_path', metavar='source-path',
                                 help='Path containing valid input images to '
                                 'convert to records')
    commands_parent.add_argument('dest_path', metavar='dest-path',
                                 help='Path to save record files to')
    commands_parent.add_argument('name', help='Name to prepend files with, '
                                 'such as "sample_record_"')
    commands_parent.add_argument('--img-per-file', type=int, default=1000)
    commands.add_parser(TFRECORD, help='Create TFRecords from input images',
                        parents=[commands_parent])
    commands.add_parser(RECORDIO, help='Create RecordIO from input images',
                        parents=[commands_parent])

    # Options specific to generating standard images
    standard = commands.add_parser(STANDARD_IMAGE, help='Generate random '
                                   'images')
    standard.add_argument('path', help='Path to save images to')
    standard.add_argument('name', help='Name to prepend files with, such as '
                          '"sample_image_"')
    standard.add_argument('image_format', metavar='image-format', help='The '
                          'image format to generate',
                          choices=SUPPORTED_IMAGE_FORMATS.keys())
    standard.add_argument('--width', help='The image width in pixels',
                          type=int, default=1920)
    standard.add_argument('--height', help='The image height in pixels',
                          type=int, default=1080)
    standard.add_argument('--count', help='The number of images to generate',
                          type=int, default=1)
    standard.add_argument('--seed', help='The seed to use while generating '
                          'random image data', type=int, default=0)
    standard.add_argument('--size', help='Display the first image size and '
                          'the directory size for the images')
    return parser.parse_args()


def try_create_directory(directory: str) -> NoReturn:
    """
    Create a directory if it doesn't exist.

    Given a name of a directory as a ``string``, a directory should be created
    with the requested name if and only if it doesn't exist already. If the
    directory exists, the function will return without any changes.

    Parameters
    ----------
    directory : string
        A ``string`` of a path pointing to a directory to attempt to create.
    """
    if not os.path.exists(directory):
        os.mkdir(directory)


def check_directory_exists(directory: str) -> NoReturn:
    """
    Check if a directory exists.

    Check if a requested directory exists and raise an error if not.

    Parameters
    ----------
    directory : string
        A ``string`` of the requested directory to check.

    Raises
    ------
    RuntimeError
        Raises a ``RuntimeError`` if the requested directory does not exist.
    """
    if not os.path.exists(directory):
        raise RuntimeError('Error: Please specify an input directory which '
                           'contains valid images.')


def create_images(
    path: str,
    name: str,
    width: int,
    height: int,
    count: int,
    image_format: str,
    seed: Optional[int] = 0,
    size: Optional[bool] = False,
    chunksize: Optional[int] = 64
) -> NoReturn:
    """
    Randomly generate standard images.

    Generate random images of standard formats, such as JPG, PNG, and BMP of
    variable height and width. Images are generated by creating a random numpy
    array of the requested dimensions and converting the image to the desired
    format. All images will be saved in the specified directory with each name
    beginning with the passed ``name`` variable and ending with a counter
    starting at zero.

    Parameters
    ----------
    path : string
        The path to the directory to save images to. The directory will be
        created if it doesn't exist.
    name : string
        A ``string`` to prepend to all filenames, such as `random_image_`.
        Filenames will end with a counter starting at zero, followed by the
        file format's extension.
    width : int
        The width of the image to generate in pixels.
    height : int
        The height of the image to generate in pixels.
    count : int
        The number of images to generate.
    image_format : str
        The format the images should be saved as. Choices are: {}
    seed : int (optional)
        A seed to use for numpy for creating the random image data. Defaults
        to 0.
    size : bool (optional)
        If `True`, will print image size information including the size of the
        first image and the final directory size.
    chunksize : int (optional)
        Specify the number of chunks to divide the requested amount of images
        into. Higher chunksizes reduce the amount of memory consumed with minor
        additional overhead.
    """.format(SUPPORTED_IMAGE_FORMATS.keys())
    print('Creating {} {} files located at {} of {}x{} resolution with a base '
          'base filename of {}'.format(count, image_format, path, width,
                                       height, name))
    try_create_directory(path)
    combined_path = os.path.join(path, name)

    # Expected to yield a thread pool equivalent to the number of CPU cores in
    # the system.
    pool = Pool()
    try:
        start_time = perf_counter()
        # NOTE: For very large image counts on memory-constrained systems, this
        # can stall-out. Either reduce the image count request, or increase the
        # chunk size.
        pool.starmap(image_creation,
                     ((combined_path, width, height, seed, image_format, n)
                      for n in range(count)),
                     chunksize=chunksize)
    finally:
        pool.close()
        pool.join()

    stop_time = perf_counter()

    if size:
        print_image_information(path)

    print('Created {} files in {} seconds'.format(count, stop_time-start_time))


def record_slice(
    source_path: str,
    dest_path: str,
    name: str,
    image_files: List[str],
    images_per_file: int,
    num_of_records: int
) -> Generator[Tuple[str, str, str, List[str], int], None, None]:
    """
    Generate subcomponents for a thread.

    While creating RecordIO files, a tuple needs to be generated to pass to
    every thread in a multiprocessing pool. Each tuple corresponds with a
    unique record file with a new path, name, and subset of images. The subset
    of images is calculated by taking the first N-images where
    N = (total images) / (number of records). The next subset begins at N + 1
    and so on.

    Parameters
    ----------
    source_path : string
        Path to the directory where the input images are stored.
    dest_path : string
        Path to the directory where the record files should be saved. Will be
        created if it does not exist.
    name : string
        A ``string`` to prepend to all filenames, such as `random_record_`.
        Filenames will end with a counter starting at zero, followed by the
        file format's extension.
    image_files : list
        A ``list`` of ``strings`` of the image filenames to use for the record
        files.
    images_per_file : int
        The number of images to include per record file.
    num_of_records : int
        The total number of record files to create. Note that one record
        assumes a record file plus a corresponding index file.

    Returns
    -------
    Generator
        Yields a ``tuple`` of objects specific to each record file. The tuple
        includes the `source_path` as a ``string``, `dest_path` as a
        ``string``, `name` as a ``string``, a subset of image names from
        `image_files` as a ``list`` of ``strings``, and a counter for the
        record file starting at 0 as an ``int``.
    """
    for num in range(num_of_records):
        subset = num * images_per_file
        yield (source_path,
               dest_path,
               name,
               image_files[subset:(subset + images_per_file)],
               num)


def create_recordio(
    source_path: str,
    dest_path: str,
    name: str,
    img_per_file: int
) -> NoReturn:
    """
    Create RecordIO files based on standard images.

    Generate one or multiple RecordIO records based on standard input images.
    Records are created by specifying an input path containing standard image
    files in JPG, PNG, or BMP format, an output directory to save the images
    to, a name to prepend the records with, and the number of record files to
    generate. Each record file contains N images where N is the total number of
    images in the input directory divided by the number of images per record
    file. Images are pulled sequentially from the input directory and placed
    into each record.

    Parameters
    ----------
    source_path : string
        Path to the directory where the input images are stored.
    dest_path : string
        Path to the directory where the record files should be saved. Will be
        created if it does not exist.
    name : string
        A ``string`` to prepend to all filenames, such as `random_record_`.
        Filenames will end with a counter starting at zero, followed by the
        file format's extension.
    images_per_file : int
        The number of images to include per record file.
    """
    print('Creating RecordIO files at {} from {} targeting {} files per '
          'record with a base filename of {}'.format(dest_path,
                                                     source_path,
                                                     img_per_file,
                                                     name))
    if not IRHeader:
        raise ImportError('MXNet not found! Please install MXNet dependency '
                          'using "pip install nvidia-imageinary[\'mxnet\']".')
    image_files = []
    source_path = os.path.abspath(source_path)
    dest_path = os.path.abspath(dest_path)
    check_directory_exists(source_path)
    try_create_directory(dest_path)

    print_image_information(source_path)

    for image_name in os.listdir(source_path):
        if not os.path.isdir(os.path.join(source_path, image_name)):
            image_files.append(image_name)

    num_of_records = ceil(len(image_files) / img_per_file)
    pool = Pool()
    try:
        start_time = perf_counter()
        pool.starmap(recordio_creation,
                     record_slice(source_path,
                                  dest_path,
                                  name,
                                  image_files,
                                  img_per_file,
                                  num_of_records))
    finally:
        pool.close()
        pool.join()

    stop_time = perf_counter()
    print('Completed in {} seconds'.format(stop_time-start_time))


def create_tfrecords(
    source_path: str,
    dest_path: str,
    name: str,
    img_per_file: int
) -> NoReturn:
    """
    Create TFRecords based on standard images.

    Generate one or multiple TFRecords based on standard input images. Records
    are created by specifying an input path containing standard image files in
    JPG, PNG, or BMP format, an output directory to save the images to, a name
    to prepend the records with, and the number of record files to generate.
    Each record file contains N images where N is the total number of images in
    the input directory divided by the number of images per record file. Images
    are pulled sequentially from the input directory and placed into each
    record.

    Parameters
    ----------
    source_path : string
        Path to the directory where the input images are stored.
    dest_path : string
        Path to the directory where the record files should be saved. Will be
        created if it does not exist.
    name : string
        A ``string`` to prepend to all filenames, such as `random_record_`.
        Filenames will end with a counter starting at zero.
    images_per_file : int
        The number of images to include per record file.
    """
    print('Creating TFRecord files at {} from {} targeting {} files per '
          'TFRecord with a base filename of {}'.format(dest_path,
                                                       source_path,
                                                       img_per_file,
                                                       name))
    if not TFRecordWriter:
        raise ImportError('TensorFlow not found! Please install TensorFlow '
                          'dependency using "pip install '
                          'nvidia-imageinary[\'tfrecord\']".')
    check_directory_exists(source_path)
    try_create_directory(dest_path)
    combined_path = os.path.join(dest_path, name)

    print_image_information(source_path)

    image_count = 0
    record = 0

    start_time = perf_counter()
    writer = TFRecordWriter(combined_path + str(record))
    for image_name in os.listdir(source_path):
        image_path = os.path.join(source_path, image_name)
        if os.path.isdir(image_path):
            continue
        image_count += 1
        if image_count > img_per_file:
            image_count = 1
            writer.close()
            record += 1
            writer = TFRecordWriter(combined_path + str(record))

        with open(image_path, 'rb') as image_file:
            image = image_file.read()
        feature = {
            'image/encoded': Feature(bytes_list=BytesList(value=[image])),
            'image/class/label': Feature(int64_list=Int64List(value=[0]))
        }

        tfrecord_entry = Example(features=Features(feature=feature))
        writer.write(tfrecord_entry.SerializeToString())

    writer.close()
    stop_time = perf_counter()

    print('Completed in {} seconds'.format(stop_time-start_time))


def print_image_information(path: str) -> NoReturn:
    """
    Print the image and directory size.

    Print the size of the first image in the directory, which is assumed to be
    a good approximator for the average image size of all images in the
    directory, as well as the total size of the directory, in bytes.

    Parameters
    ----------
    path : string
        The path to the directory where generated images are stored.
    """
    is_first_image = True
    first_image_size = 0
    directory_size = 0
    for image_name in os.listdir(path):
        image_path = os.path.join(path, image_name)
        if os.path.isdir(image_path):
            continue
        directory_size += os.path.getsize(image_path)
        if is_first_image:
            first_image_size = directory_size
            is_first_image = False
    print('First image size from {}, in bytes: {}'.format(path,
                                                          first_image_size))
    print('Directory {} size, in bytes: {}'.format(path, directory_size))


def recordio_creation(
    source_path: str,
    dest_path: str,
    name: str,
    image_files: List[str],
    n: int
) -> NoReturn:
    """
    Create a RecordIO file based on input images.

    Given a subset of images, a RecordIO file should be created with a
    corresponding index file with the given name and counter.

    Parameters
    ----------
    source_path : string
        Path to the directory where the input images are stored.
    dest_path : string
        Path to the directory where the record files should be saved. Will be
        created if it does not exist.
    name : string
        A ``string`` to prepend the record filename with.
    image_files : list
        A ``list`` of ``strings`` of image filenames to be used for the record
        creation.
    n : int
        An ``integer`` of the current count the record file points to, starting
        at zero.
    """
    combined_path = os.path.join(dest_path, name)
    regex = re.compile(r'\d+')
    dataset_rec = combined_path + str(n) + '.rec'
    dataset_idx = combined_path + str(n) + '.idx'
    recordio_ds = MXIndexedRecordIO(os.path.join(dest_path, dataset_idx),
                                    os.path.join(dest_path, dataset_rec),
                                    'w')

    for image_name in image_files:
        image_path = os.path.join(source_path, image_name)
        image_index = int(regex.findall(image_name)[0])
        header = IRHeader(0, 0, image_index, 0)
        image = open(image_path, "rb").read()
        packed_image = pack(header, image)
        recordio_ds.write_idx(image_index, packed_image)

    recordio_ds.close()


def image_creation(
    combined_path: str,
    width: int,
    height: int,
    seed: int,
    image_format: str,
    n: int
) -> NoReturn:
    """
    Generate a random image.

    Given a name, dimensions, a seed, and an image format, a random image is
    generated by creating a numpy array of random data for the specified
    dimensions and three color channels, then converting the array to an image
    of the specified format and saving the result to the output directory with
    the requested name postfixed with with the zero-based image counter and the
    file extension.

    Parameters
    ----------
    combined_path : string
        The full path to the output image file including the requested name as
        a prefix for the filename.
    width : int
        The width of the image to generate in pixels.
    height : int
        The height of the image to generate in pixels.
    image_format : str
        The format the images should be saved as.
    n : int
        The zero-based counter for the image.
    """
    numpy.random.seed(seed + n)
    a = numpy.random.rand(height, width, 3) * 255
    file_ext = SUPPORTED_IMAGE_FORMATS.get(image_format.lower(), 'png')
    if file_ext == "jpg":
        im_out = Image.fromarray(a.astype('uint8')).convert('RGB')
    else:
        im_out = Image.fromarray(a.astype('uint8')).convert('RGBA')

    im_out.save('%s%d.%s' % (combined_path, n, file_ext))


def main() -> NoReturn:
    """
    Randomly generate images or record files.

    Create standard images or record files using randomized data to be ingested
    into a deep learning application.
    """
    args = parse_args()
    if args.command == STANDARD_IMAGE:
        create_images(args.path, args.name, args.width, args.height,
                      args.count, args.image_format, args.seed, args.size)
    elif args.command == TFRECORD:
        create_tfrecords(args.source_path, args.dest_path, args.name,
                         args.img_per_file)
    elif args.command == RECORDIO:
        create_recordio(args.source_path, args.dest_path, args.name,
                        args.img_per_file)


if __name__ == "__main__":
    main()
