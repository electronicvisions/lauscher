import unittest

from lauscher.helpers import CommandLineArguments


class TestCommandLineArguments(unittest.TestCase):
    def test_singleton(self):
        instance_a = CommandLineArguments()
        instance_a.num_concurrent_jobs = 42
        instance_b = CommandLineArguments()
        self.assertEqual(42, instance_b.num_concurrent_jobs)
        self.assertIs(instance_a, instance_b)
