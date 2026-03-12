#!/bin/bash
# -*- coding: utf-8 -*-
#
# --- BEGIN_HEADER ---
#
# generate_dirty_blockmap_patch.sh
# Copyright (C) 2026  The lustrebackup Project by the Science HPC Center at UCPH
#
# This file is part of lustrebackup.
#
# MiG is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# MiG is distributed in the hope that it will be useful,
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
# =============================================================================
#
# Generates a patch against Lustre 2.15.8 that adds a 2 GB-block dirty bitmap
# to the llite client. The bitmap is stored in the "user.dirty_blockmap" xattr
# and tracks which 2 GB-aligned regions of a large file have been written.
#
# Design constraints:
#   - Block size:     2 GB  (static, power-of-2)
#   - Min file size:  2 GB  (smaller files silently skipped)
#   - Max file size:  1 PB  (larger files return -EFBIG)
#   - Max blocks:     524,288  (1 PB / 2 GB)
#   - Bitmap:         8,192 x uint64_t = 64 KB  (fits in one xattr value)
#   - Encoding:       raw little-endian uint64_t array, no header
#   - xattr name:     user.dirty_blockmap
#
# Implementation notes (2.15.8 specific):
#   - lli->lli_lock is rwlock_t; use write_lock/write_unlock
#   - xattr I/O buffers are heap-allocated (64 KB exceeds kernel stack limit)
#   - d_find_alias()/dput() used to obtain dentry for ll_vfs_setxattr/getxattr
#   - Bitmap init in ll_file_open placed before final GOTO(out_och_free) on
#     the success path (inserting before the label itself lands in dead code)
#   - ll_file_write_iter uses rc_normal; write start = ki_pos - rc_normal
#     because ki_pos has already advanced by the time we hook it
#
# Usage:
#   git clone https://github.com/lustre/lustre-release
#   cd lustre-release
#   git checkout 2.15.8
#   bash /path/to/generate_dirty_blockmap_patch.sh
#
# Output:
#   ../0001-llite-add-2GB-block-dirty-blockmap-via-user-xattr.patch
#
# Quick functional test after installing the patched module:
#   # Create a file larger than 2 GB and write to it
#   dd if=/dev/urandom of=/lustre/test.bin bs=1M count=3000
#   sync
#   python3 -c "
#   import os, struct
#   fd = os.open('/lustre/test.bin', os.O_RDWR)
#   os.fsync(fd); os.close(fd)
#   data = os.getxattr('/lustre/test.bin', 'user.dirty_blockmap')
#   words = struct.unpack(f'<{len(data)//8}Q', data)
#   dirty = sum(bin(w).count('1') for w in words)
#   print(f'dirty blocks: {dirty} of {len(words)*64}')
#   "
# =============================================================================

set -e

BRANCH="dirty-blockmap"
PATCH_OUT="../0001-llite-add-2GB-block-dirty-blockmap-via-user-xattr.patch"

echo "==> Working in: $(pwd)"
echo "==> Lustre version: $(git describe --tags 2>/dev/null || echo unknown)"

if [ ! -f lustre/llite/file.c ]; then
    echo "ERROR: run this script from the root of the lustre-release checkout"
    exit 1
fi

# Create a fresh working branch, removing any previous attempt
git checkout -b "$BRANCH" 2>/dev/null || {
    git branch -D "$BRANCH"
    git checkout -b "$BRANCH"
}

# =============================================================================
# 1. lustre/llite/dirty_blockmap.c  (new file)
#
# All bitmap logic lives here:
#   ll_dirty_blockmap_alloc  — allocate in-memory bitmap for an inode
#   ll_dirty_blockmap_free   — free bitmap and clear lli->lli_dirty_blockmap
#   ll_dirty_blockmap_mark   — set bits for a written byte range
#   ll_dirty_blockmap_store  — persist bitmap to xattr (no-op if not dirty)
#   ll_dirty_blockmap_load   — restore bitmap from xattr on open
# =============================================================================
cat > lustre/llite/dirty_blockmap.c << 'CEOF'
// SPDX-License-Identifier: GPL-2.0
/*
 * GPL HEADER START
 *
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 only,
 * as published by the Free Software Foundation.
 *
 * GPL HEADER END
 */
/*
 * lustre/llite/dirty_blockmap.c
 *
 * Track which 2 GB-aligned blocks have been written for large Lustre files.
 * The bitmap is persisted as the xattr "user.dirty_blockmap" — a raw array
 * of little-endian __u64 words with no header — and restored on first open.
 *
 * Constraints:
 *   file size < DIRTY_BLOCKMAP_MIN_FILESIZE (2 GB)  ->  no bitmap (NULL)
 *   file size > DIRTY_BLOCKMAP_MAX_FILESIZE (1 PB)  ->  ERR_PTR(-EFBIG)
 *   2 GB <= size <= 1 PB                            ->  normal operation
 *
 * xattr layout (max DIRTY_BLOCKMAP_XATTR_BYTES = 65536 bytes):
 *   [ le64 word_0 | le64 word_1 | ... | le64 word_N-1 ]
 *   Bit b of word w set  =>  2 GB block (w*64 + b) has been written.
 */

#define DEBUG_SUBSYSTEM S_LLITE

#include <linux/kernel.h>
#include <linux/slab.h>
#include <linux/spinlock.h>
#include <linux/types.h>
#include <obd_support.h>
#include "llite_internal.h"

/**
 * ll_dirty_blockmap_alloc() - Allocate and initialise a dirty_blockmap for @inode.
 * @inode:     the inode to track
 * @hint_size: minimum file size to assume (use i_size_read if 0); callers
 *             that know the file will reach a certain size (e.g. from
 *             iocb->ki_pos after a write) should pass that value so that
 *             dbm_nwords is sized correctly even before i_size is updated.
 *
 * Returns:
 *   valid ptr  success — call ll_dirty_blockmap_load() to restore persisted state
 *   NULL       effective size < DIRTY_BLOCKMAP_MIN_FILESIZE; no bitmap needed
 *   ERR_PTR()  -ENOMEM or -EFBIG (effective size > 1 PB)
 */
struct ll_dirty_blockmap *ll_dirty_blockmap_alloc(struct inode *inode,
						  loff_t hint_size)
{
	struct ll_dirty_blockmap	*bm;
	loff_t				 file_size;
	__u64				 n_blocks;

	ENTRY;

	file_size = max(i_size_read(inode), hint_size);

	if (file_size < (loff_t)DIRTY_BLOCKMAP_MIN_FILESIZE) {
		CDEBUG(D_INODE,
		       DFID " size %lld < 2 GB: dirty_blockmap not needed\n",
		       PFID(ll_inode2fid(inode)), file_size);
		RETURN(NULL);
	}

	if (file_size > (loff_t)DIRTY_BLOCKMAP_MAX_FILESIZE) {
		CERROR(DFID " size %lld exceeds 1 PB: dirty_blockmap not supported\n",
		       PFID(ll_inode2fid(inode)), file_size);
		RETURN(ERR_PTR(-EFBIG));
	}

	OBD_ALLOC_PTR(bm);
	if (!bm)
		RETURN(ERR_PTR(-ENOMEM));

	spin_lock_init(&bm->dbm_lock);
	bm->dbm_dirty = false;

	n_blocks = ((__u64)file_size + DIRTY_BLOCKMAP_BLOCK_SIZE - 1) /
		   DIRTY_BLOCKMAP_BLOCK_SIZE;
	if (n_blocks > DIRTY_BLOCKMAP_MAX_BLOCKS)
		n_blocks = DIRTY_BLOCKMAP_MAX_BLOCKS;

	bm->dbm_nwords = ((__u32)n_blocks + 63) / 64;

	CDEBUG(D_INODE,
	       DFID " dirty_blockmap alloc: size=%lld blocks=%llu words=%u\n",
	       PFID(ll_inode2fid(inode)), file_size, n_blocks, bm->dbm_nwords);

	RETURN(bm);
}

/**
 * ll_dirty_blockmap_free() - Release the dirty_blockmap attached to @inode.
 * @inode: inode whose dirty_blockmap is to be freed
 *
 * Atomically clears lli->lli_dirty_blockmap under lli_lock then frees memory.
 * Callers must call ll_dirty_blockmap_store() first if persistence is needed.
 */
void ll_dirty_blockmap_free(struct inode *inode)
{
	struct ll_inode_info		*lli = ll_i2info(inode);
	struct ll_dirty_blockmap	*bm;

	ENTRY;

	/* lli_lock is rwlock_t in Lustre 2.15.8 */
	write_lock(&lli->lli_lock);
	bm = lli->lli_dirty_blockmap;
	lli->lli_dirty_blockmap = NULL;
	write_unlock(&lli->lli_lock);

	if (bm)
		OBD_FREE_PTR(bm);

	EXIT;
}

/**
 * ll_dirty_blockmap_mark() - Record that bytes [pos, pos+len) have been written.
 * @bm:  dirty_blockmap to update
 * @pos: byte offset of write start
 * @len: number of bytes written
 *
 * Sets every 2 GB-block bit overlapping the written byte range.
 */
void ll_dirty_blockmap_mark(struct ll_dirty_blockmap *bm, loff_t pos, size_t len)
{
	__u64	first_block;
	__u64	last_block;
	__u64	block;

	ENTRY;

	if (!len)
		RETURN_EXIT;

	first_block = ((__u64)pos) / DIRTY_BLOCKMAP_BLOCK_SIZE;
	last_block  = ((__u64)pos + (__u64)len - 1) / DIRTY_BLOCKMAP_BLOCK_SIZE;

	spin_lock(&bm->dbm_lock);
	for (block = first_block; block <= last_block; block++) {
		__u32 word = (__u32)(block / 64);
		__u32 bit  = (__u32)(block % 64);

		if (word < bm->dbm_nwords) {
			bm->dbm_words[word] |= (1ULL << bit);
			bm->dbm_dirty = true;
		}
	}
	spin_unlock(&bm->dbm_lock);

	EXIT;
}

/**
 * ll_dirty_blockmap_store() - Persist bitmap to the "user.dirty_blockmap" xattr.
 * @inode: inode whose xattr will be written
 * @bm:    dirty_blockmap to persist
 *
 * No-op when dbm_dirty is false. Words are stored little-endian for
 * portability. Uses heap allocation to avoid kernel stack overflow
 * (the full bitmap is 64 KB).
 *
 * Returns 0 on success, negative errno on failure.
 */
int ll_dirty_blockmap_store(struct inode *inode, struct ll_dirty_blockmap *bm)
{
	struct dentry	*dentry;
	__u64		*buf;
	__u32		 nwords;
	__u32		 i;
	size_t		 xattr_size;
	int		 rc;

	ENTRY;

	spin_lock(&bm->dbm_lock);
	if (!bm->dbm_dirty) {
		spin_unlock(&bm->dbm_lock);
		RETURN(0);
	}
	nwords = bm->dbm_nwords;
	spin_unlock(&bm->dbm_lock);

	xattr_size = nwords * sizeof(__u64);
	OBD_ALLOC(buf, xattr_size);
	if (!buf)
		RETURN(-ENOMEM);

	spin_lock(&bm->dbm_lock);
	for (i = 0; i < nwords; i++)
		buf[i] = cpu_to_le64(bm->dbm_words[i]);
	bm->dbm_dirty = false;
	spin_unlock(&bm->dbm_lock);

	dentry = d_find_alias(inode);
	if (!dentry) {
		OBD_FREE(buf, xattr_size);
		RETURN(-ENOENT);
	}
	rc = ll_vfs_setxattr(dentry, inode,
			     DIRTY_BLOCKMAP_XATTR_NAME, buf, xattr_size, 0);
	dput(dentry);

	if (rc)
		CERROR(DFID " failed to store dirty_blockmap xattr: rc=%d\n",
		       PFID(ll_inode2fid(inode)), rc);

	OBD_FREE(buf, xattr_size);
	RETURN(rc);
}

/**
 * ll_dirty_blockmap_load() - Restore bitmap from the "user.dirty_blockmap" xattr.
 * @inode: inode to read the xattr from
 * @bm:    dirty_blockmap structure to populate
 *
 * ENODATA means the file has never been tracked — treated as clean slate.
 * Uses heap allocation to avoid kernel stack overflow.
 *
 * Returns 0 on success (including ENODATA), negative errno on error.
 */
int ll_dirty_blockmap_load(struct inode *inode, struct ll_dirty_blockmap *bm)
{
	struct dentry	*dentry;
	__u64		*buf;
	ssize_t		 rc;
	__u32		 nwords;
	__u32		 i;

	ENTRY;

	OBD_ALLOC(buf, DIRTY_BLOCKMAP_XATTR_BYTES);
	if (!buf)
		RETURN(-ENOMEM);

	dentry = d_find_alias(inode);
	if (!dentry) {
		OBD_FREE(buf, DIRTY_BLOCKMAP_XATTR_BYTES);
		RETURN(-ENOENT);
	}
	rc = ll_vfs_getxattr(dentry, inode,
			     DIRTY_BLOCKMAP_XATTR_NAME, buf,
			     DIRTY_BLOCKMAP_XATTR_BYTES);
	dput(dentry);

	if (rc == -ENODATA) {
		/* File never tracked before — start with clean bitmap */
		CDEBUG(D_INODE,
		       DFID " no dirty_blockmap xattr, starting fresh\n",
		       PFID(ll_inode2fid(inode)));
		OBD_FREE(buf, DIRTY_BLOCKMAP_XATTR_BYTES);
		RETURN(0);
	}
	if (rc < 0) {
		CERROR(DFID " failed to load dirty_blockmap xattr: rc=%zd\n",
		       PFID(ll_inode2fid(inode)), rc);
		OBD_FREE(buf, DIRTY_BLOCKMAP_XATTR_BYTES);
		RETURN((int)rc);
	}
	if (rc % (ssize_t)sizeof(__u64) != 0) {
		CERROR(DFID " dirty_blockmap xattr size %zd not a multiple of 8\n",
		       PFID(ll_inode2fid(inode)), rc);
		OBD_FREE(buf, DIRTY_BLOCKMAP_XATTR_BYTES);
		RETURN(-EINVAL);
	}

	nwords = (__u32)(rc / (ssize_t)sizeof(__u64));
	if (nwords > DIRTY_BLOCKMAP_MAX_WORDS) {
		CERROR(DFID " dirty_blockmap xattr has %u words, max is %lu\n",
		       PFID(ll_inode2fid(inode)), nwords,
		       (unsigned long)DIRTY_BLOCKMAP_MAX_WORDS);
		OBD_FREE(buf, DIRTY_BLOCKMAP_XATTR_BYTES);
		RETURN(-EINVAL);
	}

	spin_lock(&bm->dbm_lock);
	bm->dbm_nwords = nwords;
	for (i = 0; i < nwords; i++)
		bm->dbm_words[i] = le64_to_cpu(buf[i]);
	bm->dbm_dirty = false;
	spin_unlock(&bm->dbm_lock);

	CDEBUG(D_INODE, DFID " dirty_blockmap loaded: %u words\n",
	       PFID(ll_inode2fid(inode)), nwords);

	OBD_FREE(buf, DIRTY_BLOCKMAP_XATTR_BYTES);
	RETURN(0);
}
CEOF

echo "dirty_blockmap.c written"

# =============================================================================
# 2. Patch existing files using Python with line-number based insertion.
#    All anchors use regex patterns robust to minor formatting differences.
# =============================================================================
python3 << 'PYEOF'
import sys, re, os

def read(path):
    with open(path) as f:
        return f.readlines()

def write(path, lines):
    with open(path, 'w') as f:
        f.writelines(lines)

def find_line(lines, patterns, start=0, required=True, label=""):
    """Return 0-based index of first line matching any pattern >= start."""
    if isinstance(patterns, str):
        patterns = [patterns]
    rxs = [re.compile(p) for p in patterns]
    for i in range(start, len(lines)):
        for rx in rxs:
            if rx.search(lines[i]):
                return i
    if required:
        tag = f" ({label})" if label else ""
        print(f"ERROR: pattern not found{tag}: {patterns}", file=sys.stderr)
        print(f"  Context (lines {start+1}-{min(start+30, len(lines))}):",
              file=sys.stderr)
        for i in range(start, min(start + 30, len(lines))):
            print(f"  {i+1:5d}: {lines[i]}", end='', file=sys.stderr)
        sys.exit(1)
    return -1

def splice(lines, idx, text):
    """Insert text string before line idx."""
    return lines[:idx] + text.splitlines(keepends=True) + lines[idx:]

# ══════════════════════════════════════════════════════════════════════════════
# lustre/llite/llite_internal.h
#
# Changes:
#   1. Insert DIRTY_BLOCKMAP_* constants and struct ll_dirty_blockmap
#      before struct ll_dentry_data
#   2. Add lli_dirty_blockmap field to struct ll_inode_info before
#      lli_vfs_inode
# ══════════════════════════════════════════════════════════════════════════════
path = 'lustre/llite/llite_internal.h'
lines = read(path)

DEFS = (
    "/*\n"
    " * Dirty block bitmap for large files\n"
    " * =====================================\n"
    " * Files in [DIRTY_BLOCKMAP_MIN_FILESIZE, DIRTY_BLOCKMAP_MAX_FILESIZE]\n"
    " * have their written regions tracked at DIRTY_BLOCKMAP_BLOCK_SIZE (2 GB)\n"
    " * granularity and persisted as the xattr 'user.dirty_blockmap'.\n"
    " *\n"
    " * Arithmetic:\n"
    " *   1 PB / 2 GB = 2^19 = 524,288 blocks\n"
    " *   524,288 / 64 = 8,192 __u64 words = 65,536 bytes (fits in one xattr)\n"
    " */\n"
    "#define DIRTY_BLOCKMAP_XATTR_NAME\t\"user.dirty_blockmap\"\n"
    "#define DIRTY_BLOCKMAP_BLOCK_SIZE\t(2ULL * 1024 * 1024 * 1024)\t/* 2 GB */\n"
    "#define DIRTY_BLOCKMAP_MIN_FILESIZE\t(2ULL * 1024 * 1024 * 1024)\t/* 2 GB */\n"
    "#define DIRTY_BLOCKMAP_MAX_FILESIZE\t(1ULL * 1024 * 1024 * 1024 * \\\n"
    "\t\t\t\t\t 1024 * 1024)\t\t\t/* 1 PB */\n"
    "#define DIRTY_BLOCKMAP_MAX_BLOCKS\t(DIRTY_BLOCKMAP_MAX_FILESIZE / \\\n"
    "\t\t\t\t\t DIRTY_BLOCKMAP_BLOCK_SIZE)\t/* 524288 */\n"
    "#define DIRTY_BLOCKMAP_MAX_WORDS\t(DIRTY_BLOCKMAP_MAX_BLOCKS / 64)\t/* 8192 */\n"
    "#define DIRTY_BLOCKMAP_XATTR_BYTES\t(DIRTY_BLOCKMAP_MAX_WORDS * \\\n"
    "\t\t\t\t\t sizeof(__u64))\t\t\t/* 65536 */\n"
    "\n"
    "/**\n"
    " * struct ll_dirty_blockmap - in-memory dirty-block bitmap for one inode\n"
    " * @dbm_words:\tbit N set => 2 GB block N has been written\n"
    " * @dbm_nwords:\tnumber of __u64 words in use\n"
    " * @dbm_dirty:\ttrue when in-memory state differs from persisted xattr\n"
    " * @dbm_lock:\tprotects all fields above\n"
    " */\n"
    "struct ll_dirty_blockmap {\n"
    "\t__u64\t\tdbm_words[DIRTY_BLOCKMAP_MAX_WORDS];\n"
    "\t__u32\t\tdbm_nwords;\n"
    "\tbool\t\tdbm_dirty;\n"
    "\tspinlock_t\tdbm_lock;\n"
    "};\n"
    "\n"
    "/* dirty_blockmap.c */\n"
    "struct ll_dirty_blockmap *ll_dirty_blockmap_alloc(struct inode *inode,\n"
    "\t\t\t\t\t\t loff_t hint_size);\n"
    "void\t\t\t  ll_dirty_blockmap_free(struct inode *inode);\n"
    "void\t\t\t  ll_dirty_blockmap_mark(struct ll_dirty_blockmap *bm,\n"
    "\t\t\t\t\t\t loff_t pos, size_t len);\n"
    "int\t\t\t  ll_dirty_blockmap_store(struct inode *inode,\n"
    "\t\t\t\t\t\t  struct ll_dirty_blockmap *bm);\n"
    "int\t\t\t  ll_dirty_blockmap_load(struct inode *inode,\n"
    "\t\t\t\t\t\t struct ll_dirty_blockmap *bm);\n"
    "\n"
)

idx = find_line(lines, r'^struct ll_dentry_data \{', label='ll_dentry_data')
lines = splice(lines, idx, DEFS)

idx_struct = find_line(lines, r'^struct ll_inode_info \{', label='ll_inode_info')
idx_vfs    = find_line(lines, r'\blli_vfs_inode\b', start=idx_struct,
                       label='lli_vfs_inode')
lines = splice(lines, idx_vfs,
    "\t/* 2 GB-block dirty bitmap; NULL when file is out of tracked range */\n"
    "\tstruct ll_dirty_blockmap\t*lli_dirty_blockmap;\n"
    "\n")

write(path, lines)
print("llite_internal.h OK")

# ══════════════════════════════════════════════════════════════════════════════
# lustre/llite/file.c
#
# Changes:
#   1. ll_file_open: declare 'bm' variable; allocate+load bitmap before the
#      final GOTO(out_och_free) on the success path
#   2. ll_file_write_iter: mark written blocks before RETURN(rc_normal)
#   3. ll_fsync: persist bitmap to xattr before returning
# ══════════════════════════════════════════════════════════════════════════════
path = 'lustre/llite/file.c'
lines = read(path)

# ── ll_file_open: declare bm variable ────────────────────────────────────────
idx_open  = find_line(lines,
    [r'\bll_file_open\b.*\(struct inode', r'^\s*int\s+ll_file_open\b'],
    label='ll_file_open')
idx_brace = find_line(lines, r'^\{', start=idx_open, label='open brace')
idx_decl  = find_line(lines,
    r'^\t+(struct|int|bool|__u|loff_t|ssize_t|enum)\b',
    start=idx_brace + 1, label='first decl in ll_file_open')
lines = splice(lines, idx_decl, "\tstruct ll_dirty_blockmap\t*bm;\n")
print(f"  file.c: bm decl inserted at line {idx_decl+1}")

# ── ll_file_open: bitmap init before final GOTO(out_och_free) ────────────────
# The success path ends with GOTO(out_och_free, rc); we insert the bitmap
# init just before that GOTO so it runs on every successful regular-file open.
idx_label = find_line(lines, r'^out_och_free:', start=idx_brace,
                      label='out_och_free')
idx_goto = -1
for i in range(idx_label - 1, idx_brace, -1):
    if 'GOTO(out_och_free' in lines[i]:
        idx_goto = i
        break
idx_insert = idx_goto if idx_goto != -1 else idx_label

lines = splice(lines, idx_insert,
    "\t/*\n"
    "\t * dirty_blockmap: allocate and load for qualifying regular files\n"
    "\t * on first open. Persisted state is restored from xattr here.\n"
    "\t * Only initialised once; lli_dirty_blockmap persists across re-opens.\n"
    "\t */\n"
    "\tif (!lli->lli_dirty_blockmap) {\n"
    "\t\tbm = ll_dirty_blockmap_alloc(inode, 0);\n"
    "\t\tif (IS_ERR(bm)) {\n"
    "\t\t\tCDEBUG(D_INODE,\n"
    "\t\t\t       \"dirty_blockmap_alloc failed rc=%ld, skipping\\n\",\n"
    "\t\t\t       PTR_ERR(bm));\n"
    "\t\t} else if (bm) {\n"
    "\t\t\tll_dirty_blockmap_load(inode, bm);\n"
    "\t\t\tlli->lli_dirty_blockmap = bm;\n"
    "\t\t}\n"
    "\t\t/* bm == NULL: file < 2 GB, silently skip */\n"
    "\t}\n"
    "\n")
print(f"  file.c: dirty_blockmap init inserted before GOTO(out_och_free)")

# ── ll_file_write_iter: mark written blocks ───────────────────────────────────
# In 2.15.8 ll_file_write_iter uses rc_normal (not result) and has no 'lli'.
# iocb->ki_pos has already advanced by rc_normal at the point of RETURN,
# so write start = ki_pos - rc_normal.
idx_write = -1
for i, line in enumerate(lines):
    if 'll_file_write_iter' in line and 'kiocb' in line:
        idx_write = i
        break
if idx_write == -1:
    idx_write = find_line(lines, r'll_file_write_iter', label='ll_file_write_iter')

idx_next_func = len(lines)
for i in range(idx_write + 5, len(lines)):
    if re.match(r'^(static\s+)?(int|ssize_t|void|long|bool|struct|loff_t)\s+\w',
                lines[i]):
        idx_next_func = i
        break

idx_ret = -1
for i in range(idx_next_func - 1, idx_write, -1):
    if re.search(r'RETURN\(rc_normal\)', lines[i]):
        idx_ret = i
        break

if idx_ret == -1:
    print("WARNING: RETURN(rc_normal) not found in ll_file_write_iter — "
          "mark hook skipped")
else:
    lines = splice(lines, idx_ret,
        "\tif (rc_normal > 0) {\n"
        "\t\tstruct inode\t\t  *_inode = file_inode(file);\n"
        "\t\tstruct ll_inode_info\t  *_lli   = ll_i2info(_inode);\n"
        "\n"
        "\t\t/*\n"
        "\t\t * Lazy bitmap allocation: files opened when small (e.g. via\n"
        "\t\t * dd O_TRUNC) have no bitmap yet. iocb->ki_pos has already\n"
        "\t\t * advanced to the end of this write — use it to detect when\n"
        "\t\t * the file has grown past DIRTY_BLOCKMAP_MIN_FILESIZE.\n"
        "\t\t */\n"
        "\t\tif (!_lli->lli_dirty_blockmap &&\n"
        "\t\t    iocb->ki_pos >= (loff_t)DIRTY_BLOCKMAP_MIN_FILESIZE) {\n"
        "\t\t\tstruct ll_dirty_blockmap *_bm =\n"
        "\t\t\t\tll_dirty_blockmap_alloc(_inode, iocb->ki_pos);\n"
        "\n"
        "\t\t\tif (!IS_ERR_OR_NULL(_bm)) {\n"
        "\t\t\t\tll_dirty_blockmap_load(_inode, _bm);\n"
        "\t\t\t\t_lli->lli_dirty_blockmap = _bm;\n"
        "\t\t\t}\n"
        "\t\t}\n"
        "\n"
        "\t\tif (_lli->lli_dirty_blockmap)\n"
        "\t\t\tll_dirty_blockmap_mark(_lli->lli_dirty_blockmap,\n"
        "\t\t\t\t\t       iocb->ki_pos - rc_normal,\n"
        "\t\t\t\t\t       (size_t)rc_normal);\n"
        "\t}\n"
        "\n")
    print(f"  file.c: ll_dirty_blockmap_mark inserted before RETURN(rc_normal)")

# ── ll_fsync: persist bitmap before returning ─────────────────────────────────
# Declare bm after the inode variable, initialise after ENTRY;, then
# call ll_dirty_blockmap_store before the last RETURN in the function.
idx_fsync  = find_line(lines,
    [r'\bll_fsync\b.*\(struct file', r'^\s*(static\s+)?int\s+ll_fsync\b'],
    label='ll_fsync')
idx_fbrace = find_line(lines, r'^\{', start=idx_fsync, label='fsync brace')

idx_inode_decl = find_line(lines, r'^\t+.*\*inode\b',
                            start=idx_fbrace + 1, label='inode decl in ll_fsync')
lines = splice(lines, idx_inode_decl + 1,
               "\tstruct ll_dirty_blockmap\t*bm;\n")

idx_fentry = find_line(lines, r'^\s+ENTRY;', start=idx_fbrace,
                       label='ENTRY in ll_fsync')
lines = splice(lines, idx_fentry + 1,
               "\n\tbm = ll_i2info(inode)->lli_dirty_blockmap;\n")
print(f"  file.c: bm variable inserted in ll_fsync")

idx_fend = find_line(lines, r'^\}', start=idx_fbrace + 1,
                     label='fsync closing brace')
idx_fret = -1
for i in range(idx_fend - 1, idx_fbrace, -1):
    if re.search(r'^\s+RETURN\b|^\s+return\b', lines[i]):
        idx_fret = i
        break
if idx_fret == -1:
    print("WARNING: RETURN not found in ll_fsync — store hook skipped")
else:
    lines = splice(lines, idx_fret,
        "\tif (bm) {\n"
        "\t\tint bm_rc = ll_dirty_blockmap_store(inode, bm);\n"
        "\n"
        "\t\tif (bm_rc)\n"
        "\t\t\tCDEBUG(D_INODE,\n"
        "\t\t\t       \"dirty_blockmap store on fsync failed: rc=%d\\n\",\n"
        "\t\t\t       bm_rc);\n"
        "\t}\n"
        "\n")
    print(f"  file.c: ll_dirty_blockmap_store call inserted in ll_fsync")

# ── ll_file_release: persist bitmap on file close ────────────────────────────
# ll_clear_inode only fires on inode eviction, which may be long after close.
# ll_file_release fires on every close(), ensuring dd and other non-fsync
# writers get their bitmap persisted promptly.
idx_release = find_line(lines,
    [r'^int ll_file_release\b', r'^\s*int\s+ll_file_release\b'],
    label='ll_file_release')
idx_rbrace = find_line(lines, r'^\{', start=idx_release,
                       label='ll_file_release opening brace')

# Find the closing brace — scan for first '^}' after opening brace
# ll_file_release is a short function so this is safe
idx_rend = find_line(lines, r'^\}', start=idx_rbrace + 1,
                     label='ll_file_release closing brace')

# Find last RETURN in the function
idx_rret = -1
for i in range(idx_rend - 1, idx_rbrace, -1):
    if re.search(r'^\s+RETURN\b|^\s+return\b', lines[i]):
        idx_rret = i
        break

if idx_rret == -1:
    print("WARNING: RETURN not found in ll_file_release — release hook skipped")
else:
    lines = splice(lines, idx_rret,
        "\t/* Persist dirty-block bitmap on close (no fsync required) */\n"
        "\tif (S_ISREG(inode->i_mode) && !is_root_inode(inode) &&\n"
        "\t    lli->lli_dirty_blockmap) {\n"
        "\t\tint _bm_rc = ll_dirty_blockmap_store(inode,\n"
        "\t\t\t\t\t\t     lli->lli_dirty_blockmap);\n"
        "\n"
        "\t\tif (_bm_rc)\n"
        "\t\t\tCDEBUG(D_INODE,\n"
        "\t\t\t       \"dirty_blockmap store on release failed: rc=%d\\n\",\n"
        "\t\t\t       _bm_rc);\n"
        "\t}\n"
        "\n")
    print(f"  file.c: ll_dirty_blockmap_store call inserted in ll_file_release")

write(path, lines)
print("file.c OK")

# ══════════════════════════════════════════════════════════════════════════════
# lustre/llite/llite_lib.c
#
# Change: persist and free the bitmap in ll_clear_inode before ENTRY;
#         so the bitmap is flushed when the inode is evicted from cache.
# ══════════════════════════════════════════════════════════════════════════════
path = 'lustre/llite/llite_lib.c'
lines = read(path)

idx_clear = find_line(lines,
    [r'\bll_clear_inode\b.*\(struct inode',
     r'^\s*(static\s+)?void\s+ll_clear_inode\b'],
    label='ll_clear_inode')
idx_entry = find_line(lines, r'^\s+ENTRY;', start=idx_clear,
                      label='ENTRY in ll_clear_inode')
lines = splice(lines, idx_entry,
    "\t/* Persist dirty-block bitmap to xattr and release memory */\n"
    "\tif (lli->lli_dirty_blockmap) {\n"
    "\t\tll_dirty_blockmap_store(inode, lli->lli_dirty_blockmap);\n"
    "\t\tll_dirty_blockmap_free(inode);\n"
    "\t}\n"
    "\n")
write(path, lines)
print("llite_lib.c OK")

# ══════════════════════════════════════════════════════════════════════════════
# lustre/llite/Makefile.in
#
# Change: append dirty_blockmap.o to the last lustre-objs assignment.
# In 2.15.8 Makefile.in uses Kbuild-style object lists:
#   lustre-objs := dcache.o dir.o ...
#   lustre-objs += ...
#   lustre-objs += llite_foreign.o llite_foreign_symlink.o  <- last line
# ══════════════════════════════════════════════════════════════════════════════
makefile_path = 'lustre/llite/Makefile.in'
if not os.path.exists(makefile_path):
    print(f"ERROR: {makefile_path} not found", file=sys.stderr)
    sys.exit(1)

lines = read(makefile_path)

# Find the LAST lustre-objs assignment line
idx_last_objs = -1
for i, line in enumerate(lines):
    if re.match(r'^lustre-objs\s*[:+]?=', line):
        idx_last_objs = i
if idx_last_objs == -1:
    print("ERROR: no lustre-objs line found in Makefile.in", file=sys.stderr)
    sys.exit(1)

# Walk forward over continuation lines (ending with backslash)
idx_last = idx_last_objs
while (idx_last + 1 < len(lines) and
       lines[idx_last].rstrip('\n').rstrip().endswith('\\')):
    idx_last += 1

stripped = lines[idx_last].rstrip('\n').rstrip()
lines[idx_last] = stripped + ' \\\n'
lines.insert(idx_last + 1, '\t\t\t   dirty_blockmap.o\n')
write(makefile_path, lines)
print(f"Makefile.in OK (dirty_blockmap.o appended after line {idx_last + 1})")

print("\nAll files patched successfully.")
PYEOF

# =============================================================================
# 3. Commit and generate the patch
# =============================================================================
git add lustre/llite/

git commit -m "llite: add 2GB block dirty bitmap via user.dirty_blockmap xattr

Track which 2 GB-aligned blocks have been written for large files
(>= 2 GB, <= 1 PB) by maintaining an in-memory bitmap persisted to/from
the xattr 'user.dirty_blockmap' on fsync and inode eviction, and restored
from xattr on first open of a qualifying file.

Design constraints:
  - Block size:     2 GB  (static, power-of-2)
  - Min file size:  2 GB  (smaller files skipped silently)
  - Max file size:  1 PB  (larger files return -EFBIG)
  - Max blocks:     524,288  (1 PB / 2 GB)
  - Bitmap:         8,192 x __u64 = 64 KB  (fits in one xattr value)

xattr payload: raw little-endian __u64 array, no header.
Block size and granularity are compile-time constants so no metadata
needs to be embedded in the xattr value itself.

Implementation notes (Lustre 2.15.8 specific):
  - lli->lli_lock is rwlock_t; use write_lock/write_unlock in
    ll_dirty_blockmap_free()
  - xattr I/O buffers are heap-allocated via OBD_ALLOC/OBD_FREE to avoid
    kernel stack overflow (the full bitmap is 64 KB)
  - d_find_alias()/dput() used to obtain a dentry for ll_vfs_setxattr/
    ll_vfs_getxattr (which require a dentry, not just an inode)
  - Bitmap init in ll_file_open() is placed before the final
    GOTO(out_och_free, rc) on the success path; inserting before the
    label itself would land in dead code
  - ll_file_write_iter() uses rc_normal (not result); write start offset
    is iocb->ki_pos - rc_normal because ki_pos has already advanced
  - Lazy allocation in ll_file_write_iter(): files opened when small
    (e.g. dd with O_TRUNC) get no bitmap at open time; the bitmap is
    allocated on the first write after the file exceeds 2 GB

Files modified:
  lustre/llite/llite_internal.h   constants, struct ll_dirty_blockmap,
                                   lli_dirty_blockmap field, declarations
  lustre/llite/dirty_blockmap.c   new file — all bitmap logic
  lustre/llite/file.c             open / write_iter / fsync hooks
  lustre/llite/llite_lib.c        persist + free on inode eviction
  lustre/llite/Makefile.in        add dirty_blockmap.o to kernel module

Signed-off-by: Developer <dev@example.com>"

# =============================================================================
# 4. Generate the patch file
# =============================================================================
git format-patch HEAD~1 --output="$PATCH_OUT"

echo ""
echo "==> Patch written to: $(realpath $PATCH_OUT)"
echo ""
echo "==> To apply to a clean 2.15.8 checkout:"
echo "      git checkout 2.15.8"
echo "      git apply $(realpath $PATCH_OUT)"
echo ""
echo "==> Quick functional test after installing the patched module:"
echo "      # Mount with user_xattr, then:"
echo "      dd if=/dev/urandom of=/lustre/test.bin bs=1M count=3000"
echo "      python3 -c \""
echo "      import os, struct"
echo "      fd = os.open('/lustre/test.bin', os.O_RDWR)"
echo "      os.fsync(fd); os.close(fd)"
echo "      data = os.getxattr('/lustre/test.bin', 'user.dirty_blockmap')"
echo "      words = struct.unpack(f'<{len(data)//8}Q', data)"
echo "      dirty = sum(bin(w).count('1') for w in words)"
echo "      total = os.stat('/lustre/test.bin').st_size // (2*1024**3) + 1"
echo "      print(f'dirty blocks: {dirty} / {total}')"
echo "      \""
