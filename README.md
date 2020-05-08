This is an early prototype of a reproducible mechanism to generate large image datasets quickly at various resolutions.

Python3 is required, and this leverages numpy, click, PIL, and multiprocessing.pool.

A basic run to create 10000 4K images:

python3 imagine.py create_jpegs --path /mnt/nvme/ --name big_image --width 3840 --height 2160 --count 10000