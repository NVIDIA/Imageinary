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
    if not os.path.exists(directory):
        os.mkdir(directory)


def check_directory_exists(directory: str) -> NoReturn:
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
    numpy.random.seed(seed + n)
    a = numpy.random.rand(height, width, 3) * 255
    file_ext = SUPPORTED_IMAGE_FORMATS.get(image_format.lower(), 'png')
    if file_ext == "jpg":
        im_out = Image.fromarray(a.astype('uint8')).convert('RGB')
    else:
        im_out = Image.fromarray(a.astype('uint8')).convert('RGBA')

    im_out.save('%s%d.%s' % (combined_path, n, file_ext))


def main() -> NoReturn:
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
