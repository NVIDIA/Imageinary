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
import imagine
import pytest
import os


class TestUnits:
    def setup_method(self, tmpdir):
        self.tmpdir = tmpdir

    def teardown_method(self):
        try:
            os.rmdir(str(self.tmpdir))
        except OSError:
            # The directory wasn't created, as expected
            pass

    def test_directory_creation_if_not_exist(self):
        imagine.try_create_directory(str(self.tmpdir))

    def test_error_input_directory_doesnt_exist(self):
        with pytest.raises(RuntimeError):
            imagine.check_directory_exists(str(self.tmpdir))

    def test_record_slice_yields_expected_results(self):
        slices = [range(x, x + 100) for x in range(0, 1000, 100)]
        results = imagine.record_slice(self.tmpdir,
                                       self.tmpdir,
                                       'test_record_',
                                       range(0, 1000),
                                       100,
                                       10)

        for count, result in enumerate(results):
            source, dest, name, images, num = result
            assert source == self.tmpdir
            assert dest == self.tmpdir
            assert name == 'test_record_'
            assert images == slices[count]
            assert num == count
        # Enumerate is 0-based, so the final number will be 9 for 10 records
        assert count == 10 - 1
