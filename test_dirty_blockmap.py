#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# --- BEGIN_HEADER ---
#
# test_dirty_blockmap - lustre-dirty-blockmap
# Copyright (C) 2026  The Science HPC Center at UCPH
#
# This file is part of lustre-dirty-blockmap.
#
# lustre-dirty-blockmap is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# lustre-dirty-blockmap is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
#
# -- END_HEADER ---
#
# -- END_HEADER ---
#
# =============================================================================
#
# Functional tests for the user.dirty_blockmap Lustre llite patch.
#
# These tests must be run on a node with the patched Lustre client module
# loaded and a Lustre filesystem mounted with the user_xattr option.
#
# Usage:
#   python3 test_dirty_blockmap.py --lustre-mount /lustre
#
# All tests create and clean up their own files under a test directory
# on the Lustre mount. The tests require root or a user with write access
# to the mount point.
#
# Requirements:
#   - Patched kmod-lustre-client-2.15.8_dirty installed and loaded
#   - Lustre filesystem mounted at --lustre-mount (default: /lustre/debug)
#   - mount option user_xattr enabled
#
# =============================================================================

import argparse
import os
import struct
import sys
import tempfile
import unittest

# ---------------------------------------------------------------------------
# Constants — must match dirty_blockmap.c
# ---------------------------------------------------------------------------
BLOCK_SIZE      = 2 * 1024 ** 3        # 2 GB
MIN_FILE_SIZE   = 2 * 1024 ** 3        # 2 GB
XATTR_NAME      = 'user.dirty_blockmap'

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_bitmap(path):
    """Return (words: list[int], raw: bytes) or raise OSError."""
    data = os.getxattr(path, XATTR_NAME)
    nwords = len(data) // 8
    words  = list(struct.unpack(f'<{nwords}Q', data))
    return words, data


def get_bit(words, block):
    """Return the value (0 or 1) of block bit in the bitmap word list."""
    word = block // 64
    bit  = block % 64
    if word >= len(words):
        return 0
    return (words[word] >> bit) & 1


def has_xattr(path):
    """Return True if user.dirty_blockmap xattr exists on path."""
    try:
        os.getxattr(path, XATTR_NAME)
        return True
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Test base class — manages per-test temp files on the Lustre mount
# ---------------------------------------------------------------------------

class DirtyBlockmapTestCase(unittest.TestCase):
    """Base class providing self.lustre_dir and per-test file creation."""

    lustre_dir = None   # set by setUpClass from CLI arg

    @classmethod
    def setUpClass(cls):
        if cls.lustre_dir is None:
            raise unittest.SkipTest(
                'No Lustre mount specified — pass --lustre-mount <path>')
        if not os.path.isdir(cls.lustre_dir):
            raise unittest.SkipTest(
                f'Lustre mount directory not found: {cls.lustre_dir}')

    def setUp(self):
        # Each test gets its own uniquely named file so tests are independent
        fd, self.path = tempfile.mkstemp(
            prefix='dbm_test_', dir=self.lustre_dir)
        os.close(fd)
        os.unlink(self.path)    # mkstemp creates the file; we want a clean path

    def tearDown(self):
        try:
            os.unlink(self.path)
        except FileNotFoundError:
            pass

    def make_sparse(self, size):
        """Create a sparse file of given size at self.path."""
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
        os.ftruncate(fd, size)
        os.close(fd)

    def pwrite_and_close(self, offset, data=b'X' * 1024):
        """Open self.path, pwrite data at offset, close."""
        fd = os.open(self.path, os.O_RDWR)
        os.pwrite(fd, data, offset)
        os.close(fd)


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------

class TestSmallFile(DirtyBlockmapTestCase):
    """Files below 2 GB must not get a dirty_blockmap xattr."""

    def test_no_xattr_for_1gb_file(self):
        """A 1 GB write must not produce a dirty_blockmap xattr."""
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
        os.ftruncate(fd, 1 * 1024 ** 3)
        os.write(fd, b'A' * 1024)
        os.close(fd)
        self.assertFalse(has_xattr(self.path),
                         'xattr must not exist for file < 2 GB')

    def test_no_xattr_for_empty_file(self):
        """An empty file must not produce a dirty_blockmap xattr."""
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
        os.close(fd)
        self.assertFalse(has_xattr(self.path),
                         'xattr must not exist for empty file')

    def test_no_xattr_small_write_only(self):
        """A small write that never crosses 2 GB must not produce an xattr."""
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
        os.write(fd, b'B' * 4096)
        os.close(fd)
        self.assertFalse(has_xattr(self.path),
                         'xattr must not exist for small write-only file')


class TestDdStyleWrite(DirtyBlockmapTestCase):
    """Simulate dd: O_TRUNC open, sequential write past 2 GB boundary."""

    def test_dd_3gb_marks_both_blocks(self):
        """Sequential write through 3 GB must mark blocks 0 and 1 dirty."""
        # Write 3 GB sequentially in 1 MB chunks (simulate dd bs=1M count=3000)
        chunk   = b'\x00' * (1024 * 1024)
        written = 0
        target  = 3 * 1024 ** 3
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
        while written < target:
            n = os.write(fd, chunk)
            written += n
        os.close(fd)

        self.assertTrue(has_xattr(self.path), 'xattr must exist after 3 GB write')
        words, _ = get_bitmap(self.path)
        self.assertEqual(get_bit(words, 0), 1, 'block 0 must be dirty')
        self.assertEqual(get_bit(words, 1), 1, 'block 1 must be dirty')

    def test_dd_2gb_exact_marks_block_0_only(self):
        """A write of exactly 2 GB must mark only block 0."""
        chunk   = b'\x00' * (1024 * 1024)
        written = 0
        target  = 2 * 1024 ** 3
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
        while written < target:
            n = os.write(fd, chunk)
            written += n
        os.close(fd)

        self.assertTrue(has_xattr(self.path), 'xattr must exist after 2 GB write')
        words, _ = get_bitmap(self.path)
        self.assertEqual(get_bit(words, 0), 1, 'block 0 must be dirty')
        self.assertEqual(get_bit(words, 1), 0, 'block 1 must be clean')


class TestSparseFileWrites(DirtyBlockmapTestCase):
    """Writes into pre-allocated sparse files (ftruncate + pwrite)."""

    def test_pwrite_block1_only(self):
        """pwrite at 2.5 GB into 3 GB sparse file must mark only block 1."""
        self.make_sparse(3 * 1024 ** 3)
        self.pwrite_and_close(int(2.5 * 1024 ** 3))

        self.assertTrue(has_xattr(self.path), 'xattr must exist')
        words, _ = get_bitmap(self.path)
        self.assertEqual(get_bit(words, 0), 0, 'block 0 must be clean')
        self.assertEqual(get_bit(words, 1), 1, 'block 1 must be dirty')

    def test_pwrite_block0_in_sparse_file(self):
        """pwrite at offset 0 into 3 GB sparse file must mark block 0."""
        self.make_sparse(3 * 1024 ** 3)
        self.pwrite_and_close(0)

        self.assertTrue(has_xattr(self.path), 'xattr must exist')
        words, _ = get_bitmap(self.path)
        self.assertEqual(get_bit(words, 0), 1, 'block 0 must be dirty')
        self.assertEqual(get_bit(words, 1), 0, 'block 1 must be clean')

    def test_pwrite_both_blocks(self):
        """pwrite into both blocks of a 3 GB sparse file must mark both."""
        self.make_sparse(3 * 1024 ** 3)
        self.pwrite_and_close(0)
        self.pwrite_and_close(int(2.5 * 1024 ** 3))

        words, _ = get_bitmap(self.path)
        self.assertEqual(get_bit(words, 0), 1, 'block 0 must be dirty')
        self.assertEqual(get_bit(words, 1), 1, 'block 1 must be dirty')


class TestOrMerge(DirtyBlockmapTestCase):
    """The OR merge: bits written in one open must survive subsequent opens."""

    def test_block0_survives_second_open_writing_block1(self):
        """Block 0 bit set in first open must survive second open writing block 1."""
        self.make_sparse(3 * 1024 ** 3)

        # First open: write only to block 0
        self.pwrite_and_close(0)
        words, _ = get_bitmap(self.path)
        self.assertEqual(get_bit(words, 0), 1, 'block 0 must be dirty after first write')
        self.assertEqual(get_bit(words, 1), 0, 'block 1 must be clean after first write')

        # Second open: write only to block 1
        self.pwrite_and_close(int(2.5 * 1024 ** 3))
        words, _ = get_bitmap(self.path)
        self.assertEqual(get_bit(words, 0), 1,
                         'block 0 bit must survive OR merge')
        self.assertEqual(get_bit(words, 1), 1,
                         'block 1 must be dirty after second write')

    def test_block1_survives_second_open_writing_block0(self):
        """Block 1 bit set in first open must survive second open writing block 0."""
        self.make_sparse(3 * 1024 ** 3)

        # First open: write only to block 1
        self.pwrite_and_close(int(2.5 * 1024 ** 3))
        words, _ = get_bitmap(self.path)
        self.assertEqual(get_bit(words, 1), 1, 'block 1 must be dirty after first write')

        # Second open: write only to block 0
        self.pwrite_and_close(0)
        words, _ = get_bitmap(self.path)
        self.assertEqual(get_bit(words, 0), 1,
                         'block 0 must be dirty after second write')
        self.assertEqual(get_bit(words, 1), 1,
                         'block 1 bit must survive OR merge')


class TestReadOnlyOpen(DirtyBlockmapTestCase):
    """Read-only open+close must not modify the xattr."""

    def test_readonly_open_does_not_create_xattr(self):
        """Read-only open of a small file must not create an xattr."""
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
        os.write(fd, b'hello')
        os.close(fd)

        fd = os.open(self.path, os.O_RDONLY)
        os.close(fd)
        self.assertFalse(has_xattr(self.path),
                         'read-only open must not create xattr on small file')

    def test_readonly_open_does_not_change_existing_xattr(self):
        """Read-only open of a large file must not alter an existing xattr."""
        self.make_sparse(3 * 1024 ** 3)
        self.pwrite_and_close(0)                    # set block 0
        words_before, data_before = get_bitmap(self.path)

        fd = os.open(self.path, os.O_RDONLY)
        os.close(fd)

        words_after, data_after = get_bitmap(self.path)
        self.assertEqual(data_before, data_after,
                         'xattr must be unchanged after read-only open')


class TestXattrEncoding(DirtyBlockmapTestCase):
    """Verify xattr encoding: little-endian, no header, correct word layout."""

    def test_xattr_size_is_multiple_of_8(self):
        """xattr payload must be a multiple of 8 bytes (array of uint64_t)."""
        self.make_sparse(3 * 1024 ** 3)
        self.pwrite_and_close(0)
        _, data = get_bitmap(self.path)
        self.assertEqual(len(data) % 8, 0,
                         'xattr size must be a multiple of 8 bytes')

    def test_block0_is_lsb_of_first_word(self):
        """Block 0 must be the LSB of the first uint64_t word."""
        self.make_sparse(3 * 1024 ** 3)
        self.pwrite_and_close(0)                    # write only block 0
        _, data = get_bitmap(self.path)
        first_word = struct.unpack_from('<Q', data, 0)[0]
        self.assertEqual(first_word & 1, 1,
                         'bit 0 of first word must be set for block 0')
        self.assertEqual(first_word & 2, 0,
                         'bit 1 of first word must be clear (block 1 not written)')

    def test_block1_is_second_bit_of_first_word(self):
        """Block 1 must be bit 1 of the first uint64_t word."""
        self.make_sparse(3 * 1024 ** 3)
        self.pwrite_and_close(int(2.5 * 1024 ** 3))  # write only block 1
        _, data = get_bitmap(self.path)
        first_word = struct.unpack_from('<Q', data, 0)[0]
        self.assertEqual(first_word & 2, 2,
                         'bit 1 of first word must be set for block 1')
        self.assertEqual(first_word & 1, 0,
                         'bit 0 of first word must be clear (block 0 not written)')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Functional tests for user.dirty_blockmap Lustre patch')
    parser.add_argument(
        '--lustre-mount',
        default='/lustre/debug',
        metavar='PATH',
        help='Path to a writable directory on the patched Lustre mount '
             '(default: /lustre/debug)')
    # Pass remaining args to unittest
    args, unittest_args = parser.parse_known_args()
    return args, unittest_args


if __name__ == '__main__':
    args, unittest_args = parse_args()

    # Inject the mount path into the test case base class
    DirtyBlockmapTestCase.lustre_dir = args.lustre_mount

    # Check that the mount looks like a Lustre filesystem
    try:
        mounts = open('/proc/mounts').read()
        if args.lustre_mount not in mounts or 'lustre' not in mounts:
            print(f'WARNING: {args.lustre_mount} does not appear to be a '
                  f'Lustre mount — tests may fail or produce misleading results',
                  file=sys.stderr)
    except OSError:
        pass

    # Run tests
    sys.argv = [sys.argv[0]] + unittest_args
    unittest.main(verbosity=2)
