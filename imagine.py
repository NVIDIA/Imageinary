import numpy
import click
from PIL import Image

@click.group()
def main():
    """
    CLI for generating a fake dataset of various total capacities at different resolutions
    """
    pass

@main.command()
@click.option('--path', required=True)
@click.option('--name', required=True)
@click.option('--width', required=True)
@click.option('--height', required=True)
@click.option('--count', required=True)
@click.option('--seed')
def create(path, name, width, height, count, seed):
    click.echo("Creating {} files located at {} of {}x{} resolution with a base filename of {}".format(count, path, width, height, name))
   
    numpy.random.RandomState(seed=0)

    for n in xrange(int(count)):
        a = numpy.random.rand(int(height),int(width),3) * 255
        im_out = Image.fromarray(a.astype('uint8')).convert('RGB')
        im_out.save('%s%s%d.jpg' % (path, name, n))

if __name__ == "__main__":
    main()