

# upper_keypoint_model.py
import numpy as np
import cv2


class KeypointModel:
    """
    Geometry-aware CV Keypoint Model

    Input:
      - pred_mask: (H, W) int segmentation output
        (class id per pixel)

    Output:
      - dict[tooth_number] = {
            "centroid": (x, y),
            "M": (x, y),
            "D": (x, y)
        }
    """

    def __init__(self,
                 class_to_tooth: dict,
                 target_size=512,
                 min_area=80,
                 max_assign_dist=140.0,
                 scan_range=25,
                 scan_step=1,
                 outer_quantile=0.90,
                 sample_step=1.0):

        # 🔑 핵심 변경
        self.class_to_tooth = class_to_tooth          # {class_id: tooth_num}
        self.class_ids = list(class_to_tooth.keys()) # segmentation class ids

        self.target_size = target_size
        self.min_area = min_area
        self.max_assign_dist = max_assign_dist
        self.scan_range = scan_range
        self.scan_step = scan_step
        self.outer_quantile = outer_quantile
        self.sample_step = sample_step

    # =====================================================
    # public API
    # =====================================================
    def __call__(self, pred_mask):
        arch_center = self._get_arch_center(pred_mask)

        pred_refined, seed_masks = self._refine_masks(pred_mask)
        final_masks = self._build_final_masks(pred_refined, seed_masks)

        results = {}
        for cls_id, mask in final_masks.items():
            centroid = self._dt_centroid(mask)
            if centroid is None:
                continue

            md = self._md_scanline(mask, centroid, arch_center)
            if md is None:
                continue

            M, D = md
            tooth_num = self.class_to_tooth[cls_id]  # 🔑 여기

            results[tooth_num] = {
                "centroid": tuple(centroid),
                "M": tuple(M),
                "D": tuple(D),
            }

        return results

    # =====================================================
    # internal utils
    # =====================================================
    def _get_arch_center(self, pred_mask):
        ys, xs = np.where(pred_mask > 0)
        if len(xs) == 0:
            return np.array([self.target_size / 2,
                             self.target_size / 2], dtype=np.float32)
        return np.array([xs.mean(), ys.mean()], dtype=np.float32)

    def _unit(self, v):
        n = float(np.linalg.norm(v))
        if n < 1e-6:
            return v
        return v / n

    # -----------------------------------------------------
    # 1) mask refinement (seed + reassign)
    # -----------------------------------------------------
    def _keep_largest_cc(self, binary_mask):
        bm = (binary_mask > 0).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(bm, 8)
        if n <= 1:
            return bm

        areas = stats[1:, cv2.CC_STAT_AREA]
        if areas.size == 0:
            return np.zeros_like(bm)

        idx = int(np.argmax(areas)) + 1
        if stats[idx, cv2.CC_STAT_AREA] < self.min_area:
            return np.zeros_like(bm)

        return (labels == idx).astype(np.uint8)

    def _refine_masks(self, pred_mask):
        H, W = pred_mask.shape

        seed_masks = {}
        present = []

        for cls_id in self.class_ids:
            bm = (pred_mask == cls_id).astype(np.uint8)
            if bm.sum() < self.min_area:
                continue
            seed = self._keep_largest_cc(bm)
            if seed.sum() < self.min_area:
                continue
            seed_masks[cls_id] = seed
            present.append(cls_id)

        if len(present) == 0:
            return pred_mask.copy(), seed_masks

        union_seed = np.zeros((H, W), dtype=np.uint8)
        for s in seed_masks.values():
            union_seed = np.maximum(union_seed, s)

        stray_mask = ((pred_mask > 0) & (union_seed == 0)).astype(np.uint8)

        dist_maps = []
        for cls_id in present:
            seed = seed_masks[cls_id].astype(np.uint8)
            inv = (1 - seed).astype(np.uint8)
            dist = cv2.distanceTransform((inv * 255).astype(np.uint8),
                                         cv2.DIST_L2, 5)
            dist_maps.append(dist)
        dist_stack = np.stack(dist_maps, axis=0)

        argmin_idx = np.argmin(dist_stack, axis=0)
        min_dist = np.min(dist_stack, axis=0)

        new_mask = np.zeros_like(pred_mask, dtype=np.int32)

        for cls_id in present:
            new_mask[seed_masks[cls_id] > 0] = cls_id

        ys, xs = np.where(stray_mask > 0)
        for y, x in zip(ys, xs):
            if min_dist[y, x] > self.max_assign_dist:
                continue
            new_cls = present[int(argmin_idx[y, x])]
            new_mask[y, x] = new_cls

        return new_mask, seed_masks

    def _build_final_masks(self, pred_refined, seed_masks):
        final_masks = {}
        for cls_id in self.class_ids:
            rr = (pred_refined == cls_id).astype(np.uint8)
            seed = seed_masks.get(cls_id, np.zeros_like(rr))
            final = np.maximum(rr, seed)
            final = self._keep_largest_cc(final)
            if final.sum() >= self.min_area:
                final_masks[cls_id] = final
        return final_masks

    # -----------------------------------------------------
    # 2) centroid (DT peak)
    # -----------------------------------------------------
    def _dt_centroid(self, binary_mask):
        if binary_mask.sum() < self.min_area:
            return None
        dist = cv2.distanceTransform((binary_mask * 255).astype(np.uint8),
                                     cv2.DIST_L2, 5)
        y, x = np.unravel_index(np.argmax(dist), dist.shape)
        return np.array([float(x), float(y)], dtype=np.float32)

    # -----------------------------------------------------
    # 3) M / D detection (scanline max chord)
    # -----------------------------------------------------
    def _md_scanline(self, binary_mask, centroid, arch_center):
        H, W = binary_mask.shape

        radial = self._unit(centroid - arch_center)
        tangent = np.array([-radial[1], radial[0]], dtype=np.float32)
        normal = radial

        ys, xs = np.where(binary_mask > 0)
        pts = np.column_stack([xs, ys]).astype(np.float32)
        if len(pts) < 50:
            return None

        u = pts @ normal
        u0 = np.quantile(u, self.outer_quantile)

        best_len = -1
        best_seg = None

        for du in np.arange(0, self.scan_range + 1e-6, self.scan_step):
            base = centroid + (u0 - (centroid @ normal) + du) * normal
            ts = np.arange(-220, 220, self.sample_step)
            coords = base[None, :] + ts[:, None] * tangent[None, :]

            xs_s = coords[:, 0].round().astype(int)
            ys_s = coords[:, 1].round().astype(int)

            valid = (xs_s >= 0) & (xs_s < W) & (ys_s >= 0) & (ys_s < H)
            inside = np.zeros_like(xs_s, dtype=np.uint8)
            inside[valid] = binary_mask[ys_s[valid], xs_s[valid]]

            i = 0
            while i < len(inside):
                if inside[i] == 1:
                    j = i
                    while j < len(inside) and inside[j] == 1:
                        j += 1
                    length = j - i
                    if length > best_len:
                        best_len = length
                        best_seg = (coords[i], coords[j - 1])
                    i = j
                else:
                    i += 1

        if best_seg is None:
            return None

        p1 = best_seg[0].round().astype(int)
        p2 = best_seg[1].round().astype(int)

        img_mid_x = self.target_size / 2
        if abs(p1[0] - img_mid_x) < abs(p2[0] - img_mid_x):
            return p1, p2
        else:
            return p2, p1