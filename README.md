This is an early prototype of a reproducible mechanism to generate large image datasets quickly at various resolutions.

Python3 is required, and this leverages numpy, click, PIL, and multiprocessing.pool.

A basic run to create 1000 4K JPEGs, and echo out the size of the first file and all files in the target directory path (not including subdirectories):

python3 imagine.py create_images --path /mnt/nvme/test_dir --name bobber_file_ --width 3840 --height 2160 --count 1000 --image_format jpg --size

## Dependency Installation
It is recommended to run this program in a Python virtual environment to avoid dependency interference. The virtual environment can be installed and activated with:

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
