import pytest
import re
import os
from click.testing import CliRunner
from glob import glob
from imagine import main
from PIL import Image


class TestPNGCreation:
    @pytest.fixture(autouse=True)
    def setup(self, tmpdir):
        self.tmpdir = tmpdir.mkdir('png_files')
        self.runner = CliRunner()

    def teardown_method(self):
        for image in glob(f'{str(self.tmpdir)}/*'):
            os.remove(image)
        os.rmdir(str(self.tmpdir))

    def test_creating_one_hundred_images(self):
        self.runner.invoke(main, ['create-images',
                                  '--path', str(self.tmpdir),
                                  '--name', 'tmp_',
                                  '--width', 1920,
                                  '--height', 1080,
                                  '--count', 100,
                                  '--image_format', 'png'])

        images = glob(f'{str(self.tmpdir)}/*')

        assert len(images) == 100
        for image in images:
            assert re.search(r'tmp_\d+.png', image)
            im = Image.open(image)
            assert im.size == (1920, 1080)

    def test_creating_one_hundred_4K_images(self):
        self.runner.invoke(main, ['create-images',
                                  '--path', str(self.tmpdir),
                                  '--name', 'tmp_',
                                  '--width', 3840,
                                  '--height', 2160,
                                  '--count', 100,
                                  '--image_format', 'png'])

        images = glob(f'{str(self.tmpdir)}/*')

        assert len(images) == 100
        for image in images:
            assert re.search(r'tmp_\d+.png', image)
            im = Image.open(image)
            assert im.size == (3840, 2160)
