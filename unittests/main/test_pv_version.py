import unittest
from yobx.ext_test_case import ExtTestCase
from yobx.pv_version import PvVersion


class TestPvVersion(ExtTestCase):
    def test_pv_version(self):
        self.assertLess(PvVersion("3.4.5"), PvVersion("3.4.6"))

    def test_pv_raise(self):
        self.assertRaise(lambda: PvVersion("5.0.07"), ValueError)

    def test_pv_version2(self):
        self.assertEqual(PvVersion("5.0"), PvVersion("5.0"))
        self.assertNotEqual(PvVersion("5.0"), PvVersion("5.1"))
        self.assertTrue(PvVersion("5.0") < PvVersion("5.1"))
        self.assertTrue(PvVersion("5.0") <= PvVersion("5.1"))
        self.assertFalse(PvVersion("5.0") >= PvVersion("5.1"))
        self.assertFalse(PvVersion("5.0") > PvVersion("5.1"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
