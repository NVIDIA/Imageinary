# Imageinary
Imageinary is a reproducible mechanism which is used to generate large image
datasets at various resolutions. The tool supports multiple image types,
including JPEGs, PNGs, BMPs, RecordIO, and TFRecord files.

## Use Cases
While benchmarking deep learning applications involving images, there are
typically only a handful of public datasets that can be used and they tend to
have small and limited image sizes. In an effort to run DL tests against various
input sizes, we designed a tool to quickly and easily generate images of
variable dimensions and types which can be fed to convolutional neural networks
and deep learning pipelines.

The images are generated using random numpy arrays and are then converted to the
requested output format and saved to the specified location.

## Requirements
  * Python 3.6 or greater
  * TensorFlow 2 or TensorFlow 1.14
  * MXNet
  * Pillow
  * Numpy
  * Click

## Dependency Installation
It is recommended to run this program in a Python virtual environment to avoid
dependency interference. The virtual environment can be installed and activated
with:

```bash
pip install virtualenv
virtualenv --python python3 env
source env/bin/activate
```

Next, to install dependencies targetting TensorFlow 2, run:

```bash
pip install -r requirements.txt
```

Or, to install dependencies targetting TensorFlow 1.14, run:

```bash
pip install -r requirements.txt.tf1
```

Once finished running the code, you can leave the virtual environment with:

```bash
deactivate
```

## Running
Imageinary supports many different image types which can be specified while
running the application.

### JPEGs
A basic run to create 1000 4K JPEGs, and display the size of the first file and
all files in the target directory path (not including subdirectories):

```bash
python3 imagine.py create-images \
    --path /mnt/nvme/test_dir \
    --name random_image_ \
    --width 3840 \
    --height 2160 \
    --count 1000 \
    --image_format jpg \
    --size
```

The above command will generate 1,000 unique JPEG images in the
`/mnt/nvme/test_dir` directory. Each filename will begin with `random_image_`
and end with an image number starting from 0, such as `random_image_0.jpg`,
`random_image_1.jpg`, etc. The images will have dimensions of 3840x2160. The
`--size` flag displays information on the images, such as the size of the first
image and the size of the overall directory.

### TFRecords
TFRecords can also be easily generated using the application. This command
expects images to be pre-loaded to be used as the basis for the TFRecord files.

```bash
python3 imagine.py create-tfrecords \
    --source_path /mnt/nvme/test_dir \
    --dest_path /mnt/nvme/tf_record_dir \
    --name random_tfrecord_ \
    --img_per_file 100
```

This command uses the JPEGs created during the previous step and creates
TFRecords based on those images. The TFRecords will be saved to
`/mnt/nvme/tf_record_dir` where each file will be comprised of 100 JPEGs.

### RecordIO
Similarly, RecordIO files can be generated with a single command:

```bash
python3 imagine.py create-recordio \
    --source_path /mnt/nvme/test_dir \
    --dest_path /mnt/nvme/record_files \
    --name random_recordio_ \
    --img_per_file 100
```

This command uses the JPEGs created during the previous step and creates
TFRecords based on those images. The TFRecords will be saved to
`/mnt/nvme/record_files` where each file will be comprised of 100 JPEGs.
