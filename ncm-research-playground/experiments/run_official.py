"""Run an official research-ncm script unmodified on a non-CUDA device.

The official scripts hard-code `.cuda()` and `device="cuda"`. This launcher
remaps those requests to --device (default cpu) and then executes the script
with runpy, so the evaluation logic itself is exactly the official code.

    python experiments/run_official.py [--device cpu] scripts/eval_shard.py <its args...>
"""
import functools
import os
import runpy
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "research-ncm"))

args = sys.argv[1:]
DEV = "cpu"
if args[:1] == ["--device"]:
    DEV, args = args[1], args[2:]
script, script_args = os.path.join(ROOT, args[0]), args[1:]


def remap(d):
    if isinstance(d, str) and d.startswith("cuda"):
        return DEV
    if isinstance(d, torch.device) and d.type == "cuda":
        return torch.device(DEV)
    return d


def wrap_device_kw(fn, kw="device"):
    @functools.wraps(fn)
    def inner(*a, **k):
        if kw in k:
            k[kw] = remap(k[kw])
        return fn(*a, **k)
    return inner


for name in ("full", "tensor", "zeros", "ones", "empty", "arange", "randn", "rand", "randint"):
    setattr(torch, name, wrap_device_kw(getattr(torch, name)))
torch.load = wrap_device_kw(torch.load, "map_location")

_tensor_to, _module_to = torch.Tensor.to, torch.nn.Module.to


def tensor_to(self, *a, **k):
    a = tuple(remap(x) for x in a)
    if "device" in k:
        k["device"] = remap(k["device"])
    return _tensor_to(self, *a, **k)


def module_to(self, *a, **k):
    a = tuple(remap(x) for x in a)
    if "device" in k:
        k["device"] = remap(k["device"])
    return _module_to(self, *a, **k)


torch.Tensor.to = tensor_to
torch.nn.Module.to = module_to
torch.Tensor.cuda = lambda self, *a, **k: _tensor_to(self, DEV)
torch.nn.Module.cuda = lambda self, *a, **k: _module_to(self, DEV)

os.chdir(ROOT)
os.environ.setdefault("NCM_ROOT", ROOT)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, ROOT)
sys.argv = [script, *script_args]
print(f"[run_official] {os.path.relpath(script, ROOT)} on {DEV} (cuda requests remapped)", flush=True)
runpy.run_path(script, run_name="__main__")
