"""A passthrough FUSE filesystem that makes every read cost what it would on a slower disk.

    python3 bench/slowfs.py BACKING_DIR MOUNT_DIR --latency-ms 1 --mb-per-s 125

Each read request that reaches this process sleeps `latency + bytes / bandwidth` before it
returns. Reads the kernel serves from its page cache never get here, so a cached page costs what
it costs on any disk; only cache misses pay the emulated device. Writes pass through unchanged.

It emulates a disk's latency and bandwidth, not its queueing or IOPS limits, and the FUSE round
trip adds its own cost: measure it with `--latency-ms 0 --mb-per-s 0` and report it beside the
emulated profiles. Needs libfuse 2 and the fusepy package; unmount with `fusermount -u MOUNT_DIR`.
"""
import argparse
import errno
import os
import time

from fuse import FUSE, FuseOSError, Operations


class SlowFs(Operations):
    def __init__(self, root, latency, bytes_per_second):
        self.root = os.path.realpath(root)
        self.latency = latency
        self.bytes_per_second = bytes_per_second

    def __call__(self, op, path, *args):
        return super().__call__(op, self.root + path, *args)

    def access(self, path, mode):
        if not os.access(path, mode):
            raise FuseOSError(errno.EACCES)

    def getattr(self, path, fh=None):
        st = os.lstat(path)
        keys = ("st_atime", "st_ctime", "st_gid", "st_mode", "st_mtime", "st_nlink", "st_size", "st_uid")
        return {key: getattr(st, key) for key in keys}

    def readdir(self, path, fh):
        return [".", ".."] + os.listdir(path)

    def statfs(self, path):
        st = os.statvfs(path)
        keys = ("f_bavail", "f_bfree", "f_blocks", "f_bsize", "f_favail", "f_ffree", "f_files", "f_flag",
                "f_frsize", "f_namemax")
        return {key: getattr(st, key) for key in keys}

    def open(self, path, flags):
        return os.open(path, flags)

    def create(self, path, mode, fi=None):
        return os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)

    def read(self, path, size, offset, fh):
        data = os.pread(fh, size, offset)
        delay = self.latency + (len(data) / self.bytes_per_second if self.bytes_per_second else 0)
        if delay > 0:
            time.sleep(delay)
        return data

    def write(self, path, data, offset, fh):
        return os.pwrite(fh, data, offset)

    def truncate(self, path, length, fh=None):
        os.truncate(path, length)

    def flush(self, path, fh):
        return 0

    def fsync(self, path, datasync, fh):
        os.fsync(fh)

    def release(self, path, fh):
        os.close(fh)

    def rename(self, old, new):
        os.rename(old, self.root + new)

    def unlink(self, path):
        os.unlink(path)

    def mkdir(self, path, mode):
        os.mkdir(path, mode)

    def rmdir(self, path):
        os.rmdir(path)

    def chmod(self, path, mode):
        os.chmod(path, mode)

    def utimens(self, path, times=None):
        os.utime(path, times)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("backing")
    parser.add_argument("mount")
    parser.add_argument("--latency-ms", type=float, required=True, help="added to every read request")
    parser.add_argument("--mb-per-s", type=float, required=True, help="bandwidth per read request; 0 is unlimited")
    args = parser.parse_args()
    fs = SlowFs(args.backing, args.latency_ms / 1000, args.mb_per_s * 1e6)
    FUSE(fs, args.mount, foreground=True, nothreads=False)


if __name__ == "__main__":
    main()
