"""Find which TAL assigner tensor turns bool on DirectML."""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch
import dml_compat

dml = dml_compat.install()

try:
    from ultralytics.utils.metrics import bbox_iou  # noqa: E402
except ImportError:
    from ultralytics.utils.ops import bbox_iou  # noqa: E402
from ultralytics.utils.tal import TaskAlignedAssigner  # noqa: E402


def main():
    # 1) raw IoU dtype on dml
    a = torch.rand(64, 4, device=dml)
    b = a * 0.7 + 0.05
    iou = bbox_iou(a, b, xywh=False, CIoU=True)
    print("bbox_iou     dtype:", iou.dtype, "| sample:", [round(v, 4) for v in iou.flatten()[:4].cpu().tolist()])
    cl = iou.squeeze(-1).clamp_(0)
    print("clamp        dtype:", cl.dtype)
    print("pow(6.0)     dtype:", cl.pow(6.0).dtype)

    # 2) full get_pos_mask
    bs, na, nc, nbox = 2, 8400, 4, 8
    pd_scores = torch.rand(bs, na, nc, device=dml)
    pd_bboxes = torch.rand(bs, na, 4, device=dml)
    anc_points = torch.rand(na, 2, device=dml)
    gt_labels = torch.randint(0, nc, (bs, nbox, 1), device=dml).float()
    gt_bboxes = torch.rand(bs, nbox, 4, device=dml)
    mask_gt = torch.ones(bs, nbox, 1, device=dml, dtype=torch.bool)

    tal = TaskAlignedAssigner(
        topk=10, num_classes=nc, alpha=0.5, beta=6.0,
        stride=[8.0, 16.0, 32.0], topk2=10,
    )
    tal.bs = bs
    tal.n_max_boxes = nbox
    try:
        mask_pos, align_metric, overlaps = tal.get_pos_mask(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt,
        )
        print("get_pos_mask mask_pos  :", mask_pos.dtype)
        print("get_pos_mask align     :", align_metric.dtype)
        print("get_pos_mask overlaps  :", overlaps.dtype)

        # full _forward (exactly what training calls)
        try:
            out = tal._forward(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)
            print("_forward OK, dtypes:", [tuple(x.dtype.__str__().split(".")) for x in out])
        except Exception:
            traceback.print_exc(limit=6)

        # intermediate dtypes around select_highest_overlaps
        try:
            m2, a2, o2 = tal.get_pos_mask(
                pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt,
            )
            print("pre-select:", m2.dtype, a2.dtype, o2.dtype)
            tgi, fg, m3 = tal.select_highest_overlaps(m2, o2, nbox, a2)
            print("post-select: align", a2.dtype, "mask", m3.dtype, "fg", fg.dtype)
            a2 *= m3
            print("after *= mask_pos: align", a2.dtype)
            pam = a2.amax(dim=-1, keepdim=True)
            o2 *= m3
            pom = o2.amax(dim=-1, keepdim=True)
            print("pos_align", pam.dtype, "pos_overlaps", pom.dtype)
            a2.mul_(pom).div_(pam + tal.eps)
            print("normalize OK:", a2.dtype)
        except Exception:
            traceback.print_exc(limit=6)
        if align_metric.dtype != torch.float32:
            print("ALIGN IS NOT FLOAT -> probing get_box_metrics alone")
            mi = tal.select_candidates_in_gts(anc_points, gt_bboxes, mask_gt)
            print("mask_in_gts dtype:", mi.dtype)
            av, ov = tal.get_box_metrics(
                pd_scores, pd_bboxes, gt_labels, gt_bboxes, mi * mask_gt,
            )
            print("get_box_metrics align:", av.dtype, "overlaps:", ov.dtype)
    except Exception:
        traceback.print_exc()

    print("PROBE2_DONE")


if __name__ == "__main__":
    main()
