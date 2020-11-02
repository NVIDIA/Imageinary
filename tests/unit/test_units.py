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
