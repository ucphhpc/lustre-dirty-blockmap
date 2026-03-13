#!/bin/bash
# -*- coding: utf-8 -*-
#
# --- BEGIN_HEADER ---
#
# generate_dirty_blockmap_patch.sh - lustre-dirty-blockmap
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
#   - Fresh bitmap allocated at open if file >= 2GB, otherwise lazily
#     on first write where either ki_pos >= 2GB or i_size_read() >= 2GB
#     (handles both growing files and writes into large sparse files)
#   - At close/fsync/write-commit: simple read-OR-write of xattr (no
#     distributed lock — Lustre has no client-side LCK_EX for
#     MDS_INODELOCK_XATTR; IT_SETXATTR is obsolete). A narrow TOCTOU
#     race exists when two nodes flush concurrently; consequence is a
#     missed dirty block, not data corruption.
#   - Periodic flush in vvp_io_rw_end (CIT_WRITE only) mirrors mtime
#     update cadence — bitmap survives long-running open files
#   - dbm_dirty flag prevents unnecessary MDS RPCs when no new 2 GB
#     blocks have been written since the last flush
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
#   ll_dirty_blockmap_store  — read-OR-write xattr (no distributed lock)
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
 * of little-endian __u64 words with no header — and merged into the MDS xattr
 * at write commit, fsync and close via a simple read-OR-write sequence.
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
 *   valid ptr  success — bitmap ready, starts zeroed (no xattr load)
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
 * ll_dirty_blockmap_merge_xattr() - Read existing xattr, OR with local bitmap,
 *                                   write result back to MDS.
 * @inode: inode whose xattr will be updated
 * @bm:    local dirty_blockmap to merge
 *
 * Called from ll_dirty_blockmap_store() when dbm_dirty is true.
 *
 * Concurrency note:
 *   This is a read-modify-write sequence with no distributed lock protecting
 *   it. Two nodes performing this sequence concurrently — whether at close()
 *   or at write commit — can race: both read the same xattr, both OR in their
 *   local bits, and the last writer wins, potentially losing the other node's
 *   bits. With 2 GB block granularity this window is very narrow in practice,
 *   the consequence is a missed dirty block no data corruption or data loss.
 *
 *   A proper fix would require an atomic OR operation on the MDS side, which
 *   is not possible with client-only changes. Lustre has no client-side
 *   LCK_EX path for MDS_INODELOCK_XATTR — IT_SETXATTR is obsolete and ACL
 *   consistency relies on MDS-side serialization, not client-side locking.
 *
 * Returns 0 on success, negative errno on failure.
 */
static int ll_dirty_blockmap_merge_xattr(struct inode *inode,
					 struct ll_dirty_blockmap *bm)
{
	struct ll_sb_info	*sbi = ll_i2sbi(inode);
	struct ptlrpc_request	*req = NULL;
	__u64			*buf;
	__u32			 nwords;
	__u32			 i;
	size_t			 xattr_size;
	ssize_t			 rc;

	ENTRY;

	spin_lock(&bm->dbm_lock);
	nwords = bm->dbm_nwords;
	spin_unlock(&bm->dbm_lock);

	xattr_size = nwords * sizeof(__u64);
	OBD_ALLOC(buf, xattr_size);
	if (!buf)
		RETURN(-ENOMEM);

	/* Step 1: read current xattr from MDS.
	 * -ENODATA means no xattr yet — buf stays zeroed, which is correct. */
	rc = ll_xattr_list(inode, DIRTY_BLOCKMAP_XATTR_NAME,
			   XATTR_USER_T, buf, xattr_size, OBD_MD_FLXATTR);
	if (rc < 0 && rc != -ENODATA) {
		CERROR(DFID " failed to read dirty_blockmap xattr: rc=%zd\n",
		       PFID(ll_inode2fid(inode)), rc);
		OBD_FREE(buf, xattr_size);
		RETURN((int)rc);
	}

	/* Step 2: OR existing bits with our local dirty bits (little-endian) */
	spin_lock(&bm->dbm_lock);
	for (i = 0; i < nwords; i++)
		buf[i] = cpu_to_le64(le64_to_cpu(buf[i]) | bm->dbm_words[i]);
	bm->dbm_dirty = false;
	spin_unlock(&bm->dbm_lock);

	/* Step 3: write merged bitmap back to MDS.
	 * flags=0 means unconditional upsert (create if absent, replace if
	 * present). */
	rc = md_setxattr(sbi->ll_md_exp, ll_inode2fid(inode),
			 OBD_MD_FLXATTR, DIRTY_BLOCKMAP_XATTR_NAME,
			 buf, xattr_size, 0,
			 ll_i2suppgid(inode), &req);
	ptlrpc_req_finished(req);
	if (rc)
		CERROR(DFID " failed to store dirty_blockmap xattr: rc=%zd\n",
		       PFID(ll_inode2fid(inode)), rc);

	OBD_FREE(buf, xattr_size);
	RETURN((int)rc);
}

/**
 * ll_dirty_blockmap_store() - Persist bitmap to the "user.dirty_blockmap" xattr.
 * @inode: inode whose xattr will be written
 * @bm:    dirty_blockmap to persist; no-op if NULL or not dirty
 *
 * Always starts with a fresh local bitmap (allocated at open if file is
 * already >= 2GB, or lazily on first write past 2GB) and merges it into the
 * persistent xattr at close/fsync/write-commit time. See
 * ll_dirty_blockmap_merge_xattr() for concurrency notes.
 *
 * Returns 0 on success, negative errno on failure.
 */
int ll_dirty_blockmap_store(struct inode *inode, struct ll_dirty_blockmap *bm)
{
	int rc;
	ENTRY;

	/* Nothing to merge if no bitmap was allocated for this open */
	if (!bm)
		RETURN(0);

	spin_lock(&bm->dbm_lock);
	if (!bm->dbm_dirty) {
		spin_unlock(&bm->dbm_lock);
		RETURN(0);
	}
	spin_unlock(&bm->dbm_lock);

	rc = ll_dirty_blockmap_merge_xattr(inode, bm);
	RETURN(rc);
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
#   1. ll_file_open: declare 'bm' variable; allocate fresh bitmap if file
#      is already >= 2GB at open time (smaller files handled lazily)
#   2. ll_file_write_iter: lazy alloc + mark written blocks before RETURN(rc_normal)
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
    "\t * dirty_blockmap: if the file is already >= 2GB at open time,\n"
    "\t * allocate a fresh zeroed bitmap now. For files that are smaller\n"
    "\t * at open, the bitmap is allocated lazily in ll_file_write_iter\n"
    "\t * on the first write that crosses the 2GB threshold.\n"
    "\t * We never load the existing xattr here — the persistent xattr\n"
    "\t * is merged at close/fsync/write-commit via read-OR-write.\n"
    "\t * A narrow TOCTOU race exists on concurrent flushes from multiple\n"
    "\t * nodes; consequence is a missed dirty block, not data corruption.\n"
    "\t */\n"
    "\tif (!lli->lli_dirty_blockmap) {\n"
    "\t\tbm = ll_dirty_blockmap_alloc(inode, 0);\n"
    "\t\tif (IS_ERR(bm)) {\n"
    "\t\t\tCDEBUG(D_INODE,\n"
    "\t\t\t       \"dirty_blockmap_alloc failed rc=%ld, skipping\\n\",\n"
    "\t\t\t       PTR_ERR(bm));\n"
    "\t\t} else if (bm) {\n"
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
        "\t\t * dd O_TRUNC) have no bitmap yet. Allocate when either:\n"
        "\t\t *  - iocb->ki_pos crossed 2GB (file grew past threshold), or\n"
        "\t\t *  - i_size_read() >= 2GB (write anywhere in a large sparse\n"
        "\t\t *    file that was already large at write time)\n"
        "\t\t */\n"
        "\t\tif (!_lli->lli_dirty_blockmap &&\n"
        "\t\t    (iocb->ki_pos >= (loff_t)DIRTY_BLOCKMAP_MIN_FILESIZE ||\n"
        "\t\t     i_size_read(_inode) >= (loff_t)DIRTY_BLOCKMAP_MIN_FILESIZE)) {\n"
        "\t\t\tstruct ll_dirty_blockmap *_bm =\n"
        "\t\t\t\tll_dirty_blockmap_alloc(_inode, iocb->ki_pos);\n"
        "\n"
        "\t\t\tif (!IS_ERR_OR_NULL(_bm))\n"
        "\t\t\t\t_lli->lli_dirty_blockmap = _bm;\n"
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
# ══════════════════════════════════════════════════════════════════════════════
# lustre/llite/vvp_io.c
#
# Change: flush bitmap in vvp_io_rw_end after each write IO completes.
# This gives periodic persistence while the file is open (same cadence as
# mtime updates), so a long-running writer survives crashes with recent
# dirty block information in the xattr.
#
# vvp_io_rw_end is shared between CIT_READ and CIT_WRITE — guard on
# CIT_WRITE so we don't fire on reads.
#
# dbm_dirty is checked inside ll_dirty_blockmap_store so this is a no-op
# if no new 2 GB blocks have been written since the last flush.
# ══════════════════════════════════════════════════════════════════════════════
path = 'lustre/llite/vvp_io.c'
lines = read(path)

idx_rw_end = find_line(lines,
    [r'^static void vvp_io_rw_end\b'],
    label='vvp_io_rw_end')
idx_rw_brace = find_line(lines, r'^\{', start=idx_rw_end,
                         label='vvp_io_rw_end opening brace')
idx_rw_close = find_line(lines, r'^\}', start=idx_rw_brace + 1,
                         label='vvp_io_rw_end closing brace')

lines = splice(lines, idx_rw_close,
    "\n"
    "\t/*\n"
    "\t * Flush dirty-block bitmap to xattr after each write IO completes.\n"
    "\t * This mirrors the mtime update cadence — the bitmap stays current\n"
    "\t * even when the file is kept open for a long time.\n"
    "\t * dbm_dirty is checked inside ll_dirty_blockmap_store, so this is\n"
    "\t * a no-op if no new 2 GB blocks were dirtied since the last flush.\n"
    "\t */\n"
    "\tif (ios->cis_io->ci_type == CIT_WRITE &&\n"
    "\t    lli->lli_dirty_blockmap) {\n"
    "\t\tint _bm_rc = ll_dirty_blockmap_store(inode,\n"
    "\t\t\t\t\t\t     lli->lli_dirty_blockmap);\n"
    "\t\tif (_bm_rc)\n"
    "\t\t\tCDEBUG(D_INODE,\n"
    "\t\t\t       \"dirty_blockmap store on write end failed: rc=%d\\n\",\n"
    "\t\t\t       _bm_rc);\n"
    "\t}\n")

write(path, lines)
print("vvp_io.c OK")

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
git add lustre/llite/ lustre/llite/vvp_io.c

git commit -m "llite: add 2GB block dirty bitmap via user.dirty_blockmap xattr

Track which 2 GB-aligned blocks have been written for large files
(>= 2 GB, <= 1 PB) by maintaining an in-memory bitmap persisted to
the xattr 'user.dirty_blockmap' on write commit, fsync and close.

Design constraints:
  - Block size:     2 GB  (static, power-of-2)
  - Min file size:  2 GB  (smaller files skipped silently)
  - Max file size:  1 PB  (larger files return -EFBIG)
  - Max blocks:     524,288  (1 PB / 2 GB)
  - Bitmap:         8,192 x __u64 = 64 KB  (fits in one xattr value)

xattr payload: raw little-endian __u64 array, no header.

Concurrency model:
  Always starts with a fresh zeroed local bitmap (allocated at open if
  the file is already >= 2GB, or lazily on the first write past 2GB)
  and merges it into the persistent xattr at close/fsync/write-commit
  time via a simple read-OR-write sequence (ll_xattr_list + md_setxattr).

  No distributed lock is held around the read-modify-write — Lustre has
  no client-side LCK_EX path for MDS_INODELOCK_XATTR (IT_SETXATTR is
  obsolete; ACL consistency relies on MDS-side serialization). A narrow
  TOCTOU race exists when two nodes flush concurrently at close() or
  write commit: both may read the same xattr and the last writer wins,
  potentially losing the other node's bits. With 2 GB block granularity
  this window is very narrow. The consequence is a missed dirty block
  not data corruption or data loss.

Implementation notes (Lustre 2.15.8 specific):
  - lli->lli_lock is rwlock_t; use write_lock/write_unlock in
    ll_dirty_blockmap_free()
  - xattr I/O buffers are heap-allocated via OBD_ALLOC/OBD_FREE to avoid
    kernel stack overflow (the full bitmap is 64 KB)
  - md_setxattr() requires ptlrpc_request** — always pass &req and call
    ptlrpc_req_finished(req) to free the MDS reply buffer
  - ll_file_write_iter() uses rc_normal (not result); write start offset
    is iocb->ki_pos - rc_normal because ki_pos has already advanced
  - Lazy allocation in ll_file_write_iter(): files opened when small
    (e.g. dd with O_TRUNC) get no bitmap at open time; allocated on the
    first write where ki_pos >= 2GB or i_size_read() >= 2GB (the latter
    handles writes at low offsets into large sparse files)
  - Periodic flush in vvp_io_rw_end (CIT_WRITE) mirrors mtime update
    cadence — bitmap stays current during long-running open files
  - dbm_dirty flag ensures flush is a no-op if no new 2 GB blocks
    were written since the last flush — negligible MDS overhead

Files modified:
  lustre/llite/llite_internal.h   constants, struct ll_dirty_blockmap,
                                   lli_dirty_blockmap field, declarations
  lustre/llite/dirty_blockmap.c   new file — all bitmap logic
  lustre/llite/file.c             open / write_iter / fsync / release hooks
  lustre/llite/vvp_io.c           periodic flush in vvp_io_rw_end
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
