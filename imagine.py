import os
import numpy
import click
import tensorflow as tf
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
def create_jpegs(path, name, width, height, count, seed):
    click.echo("Creating {} JPEG files located at {} of {}x{} resolution with a base filename of {}".format(count, path, width, height, name))

    combined_path = os.path.join(path, name)

    #Expected to yield a thread pool equivalent to the number of CPU cores in the system
    with Pool() as pool:
        start_time = perf_counter()
        pool.starmap(image_creation, ((combined_path, width, height, seed, n) for n in range(count)))

    stop_time = perf_counter()
    
    click.echo("Created {} files in {} seconds".format(count, stop_time-start_time))

@main.command()
@click.option('--source_path', required=True)
@click.option('--dest_path', required=True)
@click.option('--name', required=True)
@click.option('--img_per_file', default=1000)
def create_tfrecords(source_path, dest_path, name, img_per_file):
    click.echo("Creating TFRecord files at {} from {} targeting {} files per TFRecord with a base filename of {}".format(dest_path, source_path, img_per_file, name))

    combined_path = os.path.join(dest_path, name)

    image_count = 0
    record = 0

    start_time = perf_counter()
    writer = tf.io.TFRecordWriter(combined_path + str(record))
    for image_name in os.listdir(source_path):
        if os.path.isdir(os.path.join(source_path, image_name)):
            continue
        image_count += 1
        if image_count > img_per_file:
            image_count = 1
            writer.close()
            record += 1
            writer = tf.io.TFRecordWriter(combined_path + str(record))
        image_path = os.path.join(source_path, image_name)
        image = open(image_path, "rb").read()
        tfrecord_entry = tf.train.Example(features=tf.train.Features(feature={"image/encoded": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                                            "image/class/label": tf.train.Feature(int64_list=tf.train.Int64List(value=[0]))}))
        writer.write(tfrecord_entry.SerializeToString())
    
    writer.close()
    stop_time = perf_counter()

    click.echo("Completed in {} seconds".format(stop_time-start_time))

def image_creation(combined_path, width, height, seed, n):
    numpy.random.seed(seed + n)
    a = numpy.random.rand(height,width,3) * 255
    im_out = Image.fromarray(a.astype('uint8')).convert('RGB')
    im_out.save('%s%d.jpg' % (combined_path, n))
    return

if __name__ == "__main__":
    main()