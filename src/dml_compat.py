"""Compatibility shims so ultralytics can train on AMD DirectML (iGPU).

Install BEFORE constructing the trainer:

    import dml_compat
    dml = dml_compat.install()   # -> torch.device('privateuseone:0')

Covers three gaps found in practice:
  1. select_device() rejects the 'directml:0' string (CUDA-only parsing).
  2. torch.unique(..., return_counts=True) is not implemented by DirectML
     (hit in v8DetectionLoss.preprocess). The tensor is tiny, so we run the
     op on CPU and move the results back.
  3. torchvision NMS kernels have no DirectML kernel (used during validation
     when torchvision is present). Same CPU round-trip strategy.
"""

from __future__ import annotations

import sys

import torch
import torch_directml

_installed = False


def install() -> torch.device:
    global _installed
    if _installed:
        return torch_directml.device()
    _installed = True

    dml = torch_directml.device()
    print(f"[dml_compat] device={dml} torch={torch.__version__}", flush=True)

    # Must precede any ultralytics import: smart_inference_mode binds its
    # decorator at class-definition time.
    _patch_inference_mode()
    _patch_select_device(dml)
    _patch_unique()
    _patch_scatter_add()
    _patch_nms()
    _patch_tal_dtype()
    _patch_memory_report()
    _patch_indexing_ops()
    _patch_python_nms()
    return dml


def _patch_python_nms() -> None:
    """ultralytics' own non_max_suppression does torch.cat on DML tensors and
    raises 'The parameter is incorrect'. Predictions are small next to the conv
    backbone, so run the whole NMS on CPU and move the results back.

    Validation passes a (inference, training_dict) tuple that nms unwraps
    internally, so the device probe has to look inside the structure.
    """
    from ultralytics.utils import nms as nms_mod

    orig = nms_mod.non_max_suppression

    def first_tensor(x):
        if isinstance(x, torch.Tensor):
            return x
        if isinstance(x, (list, tuple)):
            for t in x:
                r = first_tensor(t)
                if r is not None:
                    return r
        if isinstance(x, dict):
            for v in x.values():
                r = first_tensor(v)
                if r is not None:
                    return r
        return None

    def to_cpu(x):
        if isinstance(x, torch.Tensor):
            return x.cpu()
        if isinstance(x, tuple):
            return tuple(to_cpu(t) for t in x)
        if isinstance(x, list):
            return [to_cpu(t) for t in x]
        if isinstance(x, dict):
            return {k: to_cpu(v) for k, v in x.items()}
        return x

    def non_max_suppression(prediction, *args, **kwargs):
        t = first_tensor(prediction)
        if t is not None and t.device.type == "privateuseone":
            dev = t.device
            out = orig(to_cpu(prediction), *to_cpu(args), **to_cpu(kwargs))
            return [o.to(dev) for o in out]
        return orig(prediction, *args, **kwargs)

    nms_mod.non_max_suppression = non_max_suppression
    print("[dml_compat] ultralytics non_max_suppression CPU-fallback installed", flush=True)


# DirectML raises "The parameter is incorrect" across its indexing/view kernels
# (split, chunk, gather, ...). All are hit inside v8DetectionLoss on small
# boolean-masked tensors, so a CPU round-trip is cheap and keeps the conv
# backbone - the actual expensive part - on the GPU.
_INDEXING_OPS = (
    "split", "chunk", "tensor_split", "unbind",
    "gather", "index_select", "masked_select", "take",
)


def _patch_indexing_ops() -> None:
    """Grad-safe CPU fallback for DirectML's broken indexing/view kernels.

    Must NOT detach: pred_bboxes/pred_dist carry gradient, and .to() is
    differentiable across devices, so autograd routes grads back correctly.
    """
    def make(orig):
        def fn(self, *args, **kwargs):
            if self.device.type != "privateuseone":
                return orig(self, *args, **kwargs)
            try:
                return orig(self.contiguous(), *args, **kwargs)
            except RuntimeError:
                dev = self.device
                a = [x.cpu() if isinstance(x, torch.Tensor) else x for x in args]
                k = {kk: (vv.cpu() if isinstance(vv, torch.Tensor) else vv)
                     for kk, vv in kwargs.items()}
                out = orig(self.cpu().contiguous(), *a, **k)
                if isinstance(out, (tuple, list)):
                    return tuple(o.to(dev) if isinstance(o, torch.Tensor) else o for o in out)
                return out.to(dev) if isinstance(out, torch.Tensor) else out
        return fn

    for name in _INDEXING_OPS:
        orig = getattr(torch.Tensor, name, None)
        if orig is not None:
            setattr(torch.Tensor, name, make(orig))
    print(f"[dml_compat] grad-safe CPU fallback installed for {', '.join(_INDEXING_OPS)}",
          flush=True)


def _patch_inference_mode() -> None:
    """DirectML's batch_norm bumps the tensor version counter, which inference
    tensors forbid ("Cannot set version_counter for inference tensor") and
    validation dies. no_grad gives identical no-autograd semantics without that
    restriction, at a small bookkeeping cost.
    """
    torch.inference_mode = lambda mode=True: torch.no_grad()
    print("[dml_compat] inference_mode -> no_grad (DirectML version_counter fix)", flush=True)


def _patch_select_device(dml: torch.device) -> None:
    import ultralytics.utils.torch_utils as tu

    orig = tu.select_device

    def select_device(device="", batch=0, newline=False, verbose=True):
        if isinstance(device, str) and device.strip().lower().startswith("directml"):
            print(f"[dml_compat] select_device({device!r}) -> {dml}", flush=True)
            return dml
        return orig(device, batch, newline, verbose)

    tu.select_device = select_device
    import ultralytics.engine.trainer as tr  # noqa: F401  (binds patched name on import)

    if hasattr(tr, "select_device"):
        tr.select_device = select_device


def _patch_unique() -> None:
    orig = torch.unique

    def unique(input, sorted=True, return_inverse=False, return_counts=False, dim=None):
        try:
            return orig(input, sorted=sorted, return_inverse=return_inverse,
                        return_counts=return_counts, dim=dim)
        except RuntimeError as e:
            if "DirectML" not in str(e) and "not implemented" not in str(e):
                raise
            dev = input.device
            out = orig(input.detach().cpu(), sorted=sorted, return_inverse=return_inverse,
                       return_counts=return_counts, dim=dim)
            if isinstance(out, tuple):
                return tuple(t.to(dev) for t in out)
            return out.to(dev)

    torch.unique = unique
    print("[dml_compat] torch.unique CPU-fallback installed", flush=True)


def _patch_scatter_add() -> None:
    """DirectML int64 scatter_add_ raises 'The parameter is incorrect' (or corrupts
    offsets in-place) - v8DetectionLoss.preprocess builds per-image offsets with it.
    Tiny tensors: run on CPU and copy the result back into the original storage."""

    orig = torch.Tensor.scatter_add_

    def scatter_add_(self, dim, index, src, **kw):
        if self.device.type == "privateuseone":
            ref = src if isinstance(src, torch.Tensor) else None
            res = self.detach().cpu()
            res.scatter_add_(
                dim,
                index.detach().cpu(),
                ref.detach().cpu() if ref is not None else src,
                **kw,
            )
            self.copy_(res.to(self.device))
            return self
        return orig(self, dim, index, src, **kw)

    torch.Tensor.scatter_add_ = scatter_add_
    print("[dml_compat] scatter_add_ CPU-fallback installed", flush=True)


def _patch_tal_dtype() -> None:
    """DirectML evaluates int8.mul_(bool) as Bool, so TaskAlignedAssigner's
    mask_pos silently becomes Bool; `align_metric *= mask_pos` then adopts that
    dtype and the next in-place mul raises "result type Float can't be cast to
    the desired output type Bool". Float32 masks are equivalent for every
    downstream consumer (sum/where/max/bool), so keep them float end-to-end."""
    from ultralytics.utils.tal import TaskAlignedAssigner

    orig = TaskAlignedAssigner.get_pos_mask

    def get_pos_mask(self, *args, **kwargs):
        mask_pos, align_metric, overlaps = orig(self, *args, **kwargs)
        return mask_pos.float(), align_metric.float(), overlaps.float()

    TaskAlignedAssigner.get_pos_mask = get_pos_mask
    print("[dml_compat] TaskAlignedAssigner float32-mask patch installed", flush=True)


def _patch_memory_report() -> None:
    """The progress bar calls accelerator.memory_reserved(), which the DirectML
    PrivateUse1 backend does not implement. It is cosmetic logging only."""
    import ultralytics.engine.trainer as tr

    tr.BaseTrainer._get_memory = lambda self, fraction=False: 0.0
    print("[dml_compat] GPU-memory reporting stubbed (DirectML has no memory_reserved)", flush=True)


def _patch_nms() -> None:
    try:
        import torchvision.ops as tvo
    except Exception as e:  # torchvision absent: ultralytics uses its own path
        print(f"[dml_compat] torchvision not importable ({e}); NMS patch skipped", flush=True)
        return

    orig = tvo.nms

    def nms(boxes, scores, iou_threshold):
        if boxes.is_cuda or boxes.device.type == "privateuseone":
            return orig(boxes.cpu(), scores.cpu(), iou_threshold).to(boxes.device)
        return orig(boxes, scores, iou_threshold)

    tvo.nms = nms
    # also catch modules that did `from torchvision.ops import nms`
    bound = 0
    for name, mod in list(sys.modules.items()):
        if mod is not None and getattr(mod, "nms", None) is orig:
            try:
                setattr(mod, "nms", nms)
                bound += 1
            except Exception:
                pass
    print(f"[dml_compat] torchvision nms CPU-fallback installed ({bound} by-name imports)", flush=True)
