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
from glob import glob
from imagine.imagine import create_images, create_recordio
from PIL import Image


class TestRecordIOCreation:
    @pytest.fixture(autouse=True)
    def setup(self, tmpdir):
        self.tmpdir = tmpdir.mkdir('input_files')
        self.outdir = tmpdir.mkdir('output_files')

    def teardown_method(self):
        for image in glob(f'{str(self.tmpdir)}/*'):
            os.remove(image)
        for record in glob(f'{str(self.outdir)}/*'):
            os.remove(record)
        os.rmdir(str(self.tmpdir))
        os.rmdir(str(self.outdir))

    def test_creating_recordio_from_100_jpgs(self):
        # Create sample images which will be used as a basis.
        create_images(
            str(self.tmpdir),
            'tmp_',
            1920,
            1080,
            100,
            'jpg',
            0,
            False
        )
        create_recordio(
            str(self.tmpdir),
            str(self.outdir),
            'tmprecord_',
            100
        )

        records = glob(f'{str(self.outdir)}/*')

        assert len(records) == 2
        for record in records:
            assert 'tmprecord_0.idx' in record or \
                'tmprecord_0.rec' in record

    def test_creating_recordio_from_100_pngs(self):
        # Create sample images which will be used as a basis.
        create_images(
            str(self.tmpdir),
            'tmp_',
            1920,
            1080,
            100,
            'png',
            0,
            False
        )
        create_recordio(
            str(self.tmpdir),
            str(self.outdir),
            'tmprecord_',
            100
        )

        records = glob(f'{str(self.outdir)}/*')

        assert len(records) == 2
        for record in records:
            assert 'tmprecord_0.idx' in record or \
                'tmprecord_0.rec' in record

    def test_creating_recordio_from_100_jpg_multiple_files(self):
        # Create sample images which will be used as a basis.
        create_images(
            str(self.tmpdir),
            'tmp_',
            1920,
            1080,
            100,
            'jpg',
            0,
            False
        )
        create_recordio(
            str(self.tmpdir),
            str(self.outdir),
            'tmprecord_',
            10
        )

        records = glob(f'{str(self.outdir)}/*')

        assert len(records) == 20
        for record in records:
            assert re.search(r'tmprecord_\d+.idx', record) or \
                re.search(r'tmprecord_\d+.rec', record)

    def test_creating_recordio_from_100_pngs_multiple_files(self):
        # Create sample images which will be used as a basis.
        create_images(
            str(self.tmpdir),
            'tmp_',
            1920,
            1080,
            100,
            'png',
            0,
            False
        )
        create_recordio(
            str(self.tmpdir),
            str(self.outdir),
            'tmprecord_',
            10
        )

        records = glob(f'{str(self.outdir)}/*')

        assert len(records) == 20
        for record in records:
            assert re.search(r'tmprecord_\d+.idx', record) or \
                re.search(r'tmprecord_\d+.rec', record)
