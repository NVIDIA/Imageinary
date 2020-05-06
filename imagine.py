import os
import numpy
import click
from PIL import Image
from multiprocessing.pool import Pool
from time import perf_counter

@click.group()
def main():
    """
    CLI for generating a fake dataset of various quantities at different resolutions. Supports .jpg for now.
    """
    pass

@main.command()
@click.option('--path', required=True)
@click.option('--name', required=True)
@click.option('--width', default=1920, required=True)
@click.option('--height', default=1080, required=True)
@click.option('--count', default=1, required=True)
@click.option('--seed', default=0)
def create(path, name, width, height, count, seed):
    click.echo("Creating {} JPEG files located at {} of {}x{} resolution with a base filename of {}".format(count, path, width, height, name))

    combined_path = os.path.join(path, name)

    #Expected to yield a thread pool equivalent to the number of CPU cores in the system
    pool = Pool()
    start_time = perf_counter()
    result = pool.starmap(imageCreation, ((combined_path, width, height, seed, n) for n in range(count)))

    stop_time = perf_counter()
    
    click.echo("Created {} files in {} seconds".format(count, stop_time-start_time))

def imageCreation(combined_path, width, height, seed, n):
    numpy.random.seed(seed + n)
    a = numpy.random.rand(height,width,3) * 255
    im_out = Image.fromarray(a.astype('uint8')).convert('RGB')
    im_out.save('%s%d.jpg' % (combined_path, n))
    return

if __name__ == "__main__":
    main()