This is an early prototype of a reproducible mechanism to generate large image datasets quickly at various resolutions.

Python3 is required, and this leverages numpy, click, PIL, and multiprocessing.pool.

A basic run to create 10000 4K images:

python3 imagine.py create_jpegs --path /mnt/nvme/ --name big_image --width 3840 --height 2160 --count 10000

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
