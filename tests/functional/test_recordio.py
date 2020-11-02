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
import pytest
import re
import os
from click.testing import CliRunner
from glob import glob
from imagine import main
from PIL import Image


class TestRecordIOCreation:
    @pytest.fixture(autouse=True)
    def setup(self, tmpdir):
        self.tmpdir = tmpdir.mkdir('input_files')
        self.outdir = tmpdir.mkdir('output_files')
        self.runner = CliRunner()

    def teardown_method(self):
        for image in glob(f'{str(self.tmpdir)}/*'):
            os.remove(image)
        for record in glob(f'{str(self.outdir)}/*'):
            os.remove(record)
        os.rmdir(str(self.tmpdir))
        os.rmdir(str(self.outdir))

    def test_creating_recordio_from_100_jpgs(self):
        # Create sample images which will be used as a basis.
        self.runner.invoke(main, ['create-images',
                                  '--path', str(self.tmpdir),
                                  '--name', 'tmp_',
                                  '--width', 1920,
                                  '--height', 1080,
                                  '--count', 100,
                                  '--image_format', 'jpg'])
        self.runner.invoke(main, ['create-recordio',
                                  '--source_path', str(self.tmpdir),
                                  '--dest_path', str(self.outdir),
                                  '--name', 'tmprecord_',
                                  '--img_per_file', 100])

        records = glob(f'{str(self.outdir)}/*')

        assert len(records) == 2
        for record in records:
            assert 'tmprecord_0.idx' in record or \
                'tmprecord_0.rec' in record

    def test_creating_recordio_from_100_pngs(self):
        # Create sample images which will be used as a basis.
        self.runner.invoke(main, ['create-images',
                                  '--path', str(self.tmpdir),
                                  '--name', 'tmp_',
                                  '--width', 1920,
                                  '--height', 1080,
                                  '--count', 100,
                                  '--image_format', 'png'])
        self.runner.invoke(main, ['create-recordio',
                                  '--source_path', str(self.tmpdir),
                                  '--dest_path', str(self.outdir),
                                  '--name', 'tmprecord_',
                                  '--img_per_file', 100])

        records = glob(f'{str(self.outdir)}/*')

        assert len(records) == 2
        for record in records:
            assert 'tmprecord_0.idx' in record or \
                'tmprecord_0.rec' in record

    def test_creating_recordio_from_100_jpg_multiple_files(self):
        # Create sample images which will be used as a basis.
        self.runner.invoke(main, ['create-images',
                                  '--path', str(self.tmpdir),
                                  '--name', 'tmp_',
                                  '--width', 1920,
                                  '--height', 1080,
                                  '--count', 100,
                                  '--image_format', 'jpg'])
        self.runner.invoke(main, ['create-recordio',
                                  '--source_path', str(self.tmpdir),
                                  '--dest_path', str(self.outdir),
                                  '--name', 'tmprecord_',
                                  '--img_per_file', 10])

        records = glob(f'{str(self.outdir)}/*')

        assert len(records) == 20
        for record in records:
            assert re.search(r'tmprecord_\d+.idx', record) or \
                re.search(r'tmprecord_\d+.rec', record)

    def test_creating_recordio_from_100_pngs_multiple_files(self):
        # Create sample images which will be used as a basis.
        self.runner.invoke(main, ['create-images',
                                  '--path', str(self.tmpdir),
                                  '--name', 'tmp_',
                                  '--width', 1920,
                                  '--height', 1080,
                                  '--count', 100,
                                  '--image_format', 'png'])
        self.runner.invoke(main, ['create-recordio',
                                  '--source_path', str(self.tmpdir),
                                  '--dest_path', str(self.outdir),
                                  '--name', 'tmprecord_',
                                  '--img_per_file', 10])

        records = glob(f'{str(self.outdir)}/*')

        assert len(records) == 20
        for record in records:
            assert re.search(r'tmprecord_\d+.idx', record) or \
                re.search(r'tmprecord_\d+.rec', record)
