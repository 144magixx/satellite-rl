"""Owned-process resource checks reuse the fixed Linux RSS provider."""
import os,shutil
from .r5_stage_guard import rss_tree

def check(directory,*,gpu=False):
 if shutil.disk_usage(directory).free<8589934592:raise OSError('8GiB reserve reached')
 if rss_tree(os.getpid())>4294967296:raise MemoryError('model parent/children RSS4GiB')
 if gpu:
  import torch
  if torch.cuda.is_initialized() and torch.cuda.max_memory_reserved()>12884901888:raise MemoryError('GPU allocation12GiB')
