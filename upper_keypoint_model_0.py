import os
import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

# =========================================================
# [1] Keypoint Model (Hybrid Logic) - 변경 없음
# =========================================================
class KeypointModel:
    def __init__(self,
                 class_to_tooth: dict,
                 target_size=512,
                 min_area=80,
                 max_assign_dist=140.0,
                 scan_range=25,
                 scan_step=1,
                 outer_quantile=0.90,
                 sample_step=1.0):

        self.class_to_tooth = class_to_tooth
        self.class_ids = list(class_to_tooth.keys())
        self.target_size = target_size
        self.min_area = min_area
        self.max_assign_dist = max_assign_dist
        self.scan_range = scan_range
        self.scan_step = scan_step
        self.outer_quantile = outer_quantile
        self.sample_step = sample_step
        self.FRONT_TEETH = [11, 12, 13, 21, 22, 23]

    def __call__(self, pred_mask):
        arch_center = self._get_arch_center(pred_mask)
        pred_refined, seed_masks = self._refine_masks(pred_mask)
        final_masks = self._build_final_masks(pred_refined, seed_masks)

        results = {}
        for cls_id, mask in final_masks.items():
            tooth_num = self.class_to_tooth[cls_id]
            centroid = self._dt_centroid(mask)
            if centroid is None: continue

            if tooth_num in self.FRONT_TEETH:
                md = self._md_scanline(mask, centroid, arch_center)
            else:
                md = self._md_posterior_rect(mask)

            if md is None: continue
            M, D = md
            results[tooth_num] = {
                "centroid": tuple(centroid),
                "M": tuple(M),
                "D": tuple(D),
                # 🌟 Gap 계산을 위해 Mask 자체를 저장하지는 않고, 
                # Scorer에서 원본 pred_mask를 쓰도록 함 (메모리 절약)
            }
        return results

    def _md_posterior_rect(self, binary_mask):
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None
        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) < self.min_area: return None

        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int64(box) # numpy 최신 버전 호환

        dists = []
        for i in range(4):
            p1, p2 = box[i], box[(i+1)%4]
            d = np.linalg.norm(p1 - p2)
            dists.append((d, p1, p2))
        
        dists.sort(key=lambda x: x[0]) 
        short_edge_1 = dists[0]
        short_edge_2 = dists[1]
        
        m1 = (short_edge_1[1] + short_edge_1[2]) / 2.0
        m2 = (short_edge_2[1] + short_edge_2[2]) / 2.0
        
        img_mid_x = self.target_size / 2
        if abs(m1[0] - img_mid_x) < abs(m2[0] - img_mid_x):
            return (m1, m2)
        else:
            return (m2, m1)

    def _get_arch_center(self, pred_mask):
        ys, xs = np.where(pred_mask > 0)
        if len(xs) == 0: return np.array([256.0, 256.0])
        return np.array([xs.mean(), ys.mean()], dtype=np.float32)

    def _unit(self, v):
        n = float(np.linalg.norm(v))
        if n < 1e-6: return v
        return v / n

    def _keep_largest_cc(self, binary_mask):
        bm = (binary_mask > 0).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(bm, 8)
        if n <= 1: return bm
        areas = stats[1:, cv2.CC_STAT_AREA]
        if areas.size == 0: return np.zeros_like(bm)
        idx = int(np.argmax(areas)) + 1
        if stats[idx, cv2.CC_STAT_AREA] < self.min_area: return np.zeros_like(bm)
        return (labels == idx).astype(np.uint8)

    def _refine_masks(self, pred_mask):
        H, W = pred_mask.shape
        seed_masks = {}
        present = []
        for cls_id in self.class_ids:
            bm = (pred_mask == cls_id).astype(np.uint8)
            if bm.sum() < self.min_area: continue
            seed = self._keep_largest_cc(bm)
            if seed.sum() < self.min_area: continue
            seed_masks[cls_id] = seed
            present.append(cls_id)
        if len(present) == 0: return pred_mask.copy(), seed_masks
        union_seed = np.zeros((H, W), dtype=np.uint8)
        for s in seed_masks.values(): union_seed = np.maximum(union_seed, s)
        stray_mask = ((pred_mask > 0) & (union_seed == 0)).astype(np.uint8)
        dist_maps = []
        for cls_id in present:
            seed = seed_masks[cls_id].astype(np.uint8)
            inv = (1 - seed).astype(np.uint8)
            dist = cv2.distanceTransform((inv * 255).astype(np.uint8), cv2.DIST_L2, 5)
            dist_maps.append(dist)
        if not dist_maps: return pred_mask, seed_masks
        dist_stack = np.stack(dist_maps, axis=0)
        argmin_idx = np.argmin(dist_stack, axis=0)
        min_dist = np.min(dist_stack, axis=0)
        new_mask = np.zeros_like(pred_mask, dtype=np.int32)
        for cls_id in present:
            new_mask[seed_masks[cls_id] > 0] = cls_id
        ys, xs = np.where(stray_mask > 0)
        for y, x in zip(ys, xs):
            if min_dist[y, x] > self.max_assign_dist: continue
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

    def _dt_centroid(self, binary_mask):
        if binary_mask.sum() < self.min_area: return None
        dist = cv2.distanceTransform((binary_mask * 255).astype(np.uint8), cv2.DIST_L2, 5)
        y, x = np.unravel_index(np.argmax(dist), dist.shape)
        return np.array([float(x), float(y)], dtype=np.float32)

    def _md_scanline(self, binary_mask, centroid, arch_center):
        H, W = binary_mask.shape
        radial = self._unit(centroid - arch_center)
        tangent = np.array([-radial[1], radial[0]], dtype=np.float32)
        normal = radial
        ys, xs = np.where(binary_mask > 0)
        pts = np.column_stack([xs, ys]).astype(np.float32)
        if len(pts) < 50: return None
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
                    while j < len(inside) and inside[j] == 1: j += 1
                    if (j - i) > best_len:
                        best_len = j - i
                        best_seg = (coords[i], coords[j - 1])
                    i = j
                else: i += 1
        if best_seg is None: return None
        p1, p2 = best_seg[0], best_seg[1]
        img_mid_x = self.target_size / 2
        return (p1, p2) if abs(p1[0] - img_mid_x) < abs(p2[0] - img_mid_x) else (p2, p1)


# =========================================================
# [2] Scorer (Contour Min Dist 적용)
# =========================================================
class UpperArchScorer:
    ARCH_ORDER = [16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26]
    SCORING_CFG = {
        "LII":  {"good": 0.1819, "bad": 0.6606, "weight": 0.5},
        "Pont": {"good": 0.0197, "bad": 0.4448, "weight": 0.3},
        "U":    {"good": 0.0310, "bad": 0.1796, "weight": 0.2},
    }
    
    def __init__(self, model_path, device=None, gap_threshold=20.0, gap_penalty=5.0):
        self.model_path = model_path
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.target_size = 512
        self.gap_threshold = gap_threshold
        self.gap_penalty = gap_penalty

        self._load_models()
        # 치아 번호 -> Class ID 매핑 역추적용
        self.tooth_to_class = {t: i+1 for i, t in enumerate(sorted(self.ARCH_ORDER))}
        print(f"[INFO] UpperArchScorer Initialized on {self.device}")

    def _load_models(self):
        self.seg_model = smp.Unet(encoder_name="efficientnet-b0", in_channels=3, classes=13).to(self.device)
        if os.path.exists(self.model_path):
            self.seg_model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        else:
            print(f"[WARN] Model file not found at {self.model_path}")
        self.seg_model.eval()
        
        class_map = {i+1: t for i, t in enumerate(sorted(self.ARCH_ORDER))} 
        self.kp_model = KeypointModel(class_to_tooth=class_map, target_size=self.target_size)
        
        self.transform = A.Compose([A.Normalize(), ToTensorV2()])

    def _analyze_geometry(self, img_bgr):
        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(img_rgb, (self.target_size, self.target_size))
        
        x = self.transform(image=resized)["image"].unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.seg_model(x)
            pred_mask = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.int32)
            
        kps = self.kp_model(pred_mask)
        
        # 원본 크기 변환 비율
        rw, rh = w / self.target_size, h / self.target_size
        
        teeth_data = {}
        for t, data in kps.items():
            teeth_data[t] = {
                "centroid_512": data["centroid"],
                "M_orig": (data["M"][0]*rw, data["M"][1]*rh),
                "D_orig": (data["D"][0]*rw, data["D"][1]*rh),
            }
        
        # Gap 계산을 위해 scale ratio도 반환
        return teeth_data, pred_mask, resized, (rw, rh)

    # 🌟 [New] Contour 간 최단 거리 계산 함수
    def _get_contour_min_dist(self, mask, cls_a, cls_b):
        """
        두 Class ID(치아)의 윤곽선(Contour) 사이의 최단 거리를 구함.
        """
        # 1. 각 치아의 Contour 포인트 추출
        def get_pts(cls_id):
            bm = (mask == cls_id).astype(np.uint8)
            contours, _ = cv2.findContours(bm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours: return None
            # 가장 큰 덩어리만 사용
            cnt = max(contours, key=cv2.contourArea)
            # (N, 1, 2) -> (N, 2)
            return cnt.reshape(-1, 2)

        pts_a = get_pts(cls_a)
        pts_b = get_pts(cls_b)

        if pts_a is None or pts_b is None:
            return None

        # 2. 최단 거리 계산 (NumPy Broadcasting)
        # pts_a: (N, 2), pts_b: (M, 2)
        # dist matrix: (N, M)
        # 메모리 효율을 위해 float32 사용
        pa = pts_a.astype(np.float32)
        pb = pts_b.astype(np.float32)
        
        # (N, 1, 2) - (1, M, 2) -> (N, M, 2)
        diff = pa[:, np.newaxis, :] - pb[np.newaxis, :, :]
        
        # Euclidean Distance
        dists = np.sqrt(np.sum(diff**2, axis=2))
        
        # 3. 전체 중 최소값 반환
        return np.min(dists)

    def _calculate_gaps(self, teeth_data, pred_mask, scale_ratio):
        gaps = {}
        detected = set(teeth_data.keys())
        check_list = [t for t in self.ARCH_ORDER if t not in [16, 26]] # 양 끝단 제외
        
        rw, rh = scale_ratio # (width_ratio, height_ratio)
        # Contour 거리 계산은 이미지 픽셀 단위이므로 대략적인 평균 ratio 사용
        scale_avg = (rw + rh) / 2.0 

        for t_missing in check_list:
            if t_missing in detected: continue # 이미 치아가 있으면 Gap 아님
            idx = self.ARCH_ORDER.index(t_missing)
            
            # 왼쪽, 오른쪽 치아 찾기
            left, right = None, None
            for i in range(idx-1, -1, -1):
                if self.ARCH_ORDER[i] in detected: left = self.ARCH_ORDER[i]; break
            for i in range(idx+1, len(self.ARCH_ORDER)):
                if self.ARCH_ORDER[i] in detected: right = self.ARCH_ORDER[i]; break
            
            if left and right:
                # 🌟 [수정됨] M/D 포인트 대신 Contour 최단 거리 사용
                cls_l = self.tooth_to_class[left]
                cls_r = self.tooth_to_class[right]
                
                dist_512 = self._get_contour_min_dist(pred_mask, cls_l, cls_r)
                
                if dist_512 is not None:
                    # 512x512 기준 거리를 원본 해상도로 변환
                    gap_real = dist_512 * scale_avg
                    gaps[f"Gap_{t_missing}"] = round(gap_real, 2)
                else:
                    gaps[f"Gap_{t_missing}"] = None
            else:
                gaps[f"Gap_{t_missing}"] = None
        return gaps

    def _convert_score(self, val, name):
        if val is None: return 0.0
        cfg = self.SCORING_CFG[name]
        if val <= cfg["good"]: return 100.0
        if val >= cfg["bad"]: return 0.0
        return (1.0 - (val - cfg["good"])/(cfg["bad"] - cfg["good"])) * 100.0

    def process(self, img_path):
        img = cv2.imread(img_path)
        fname = os.path.basename(img_path)
        if img is None: return {"filename": fname, "status": "Read Error", "final_score": 0}, None, None

        # analyze_geometry에서 scale_ratio도 받아옴
        teeth_data, mask, resized_img, scale_ratio = self._analyze_geometry(img)
        
        if not teeth_data:
             return {"filename": fname, "status": "No Teeth", "final_score": 0}, mask, resized_img

        # Metrics
        dists = []
        pairs = [(11,21), (12,11), (21,22)] 
        for a, b in pairs:
            if a in teeth_data and b in teeth_data:
                 dists.append(np.linalg.norm(np.array(teeth_data[a]["M_orig"]) - np.array(teeth_data[b]["M_orig"])))
        lii_raw = float(np.mean(dists))/100.0 if dists else 0.5
        
        base_score = self._convert_score(lii_raw, "LII") 
        
        # 🌟 Gap 계산 시 mask와 scale_ratio 전달
        gaps = self._calculate_gaps(teeth_data, mask, scale_ratio)
        
        penalty = 0.0
        penalty_list = []
        for k, v in gaps.items():
            if v and v > self.gap_threshold:
                penalty += self.gap_penalty
                penalty_list.append(k)
        
        final_score = max(0, base_score - penalty)
        
        result = {
            "filename": fname,
            "status": "Success",
            "final_score": round(final_score, 1),
            "base_score": round(base_score, 1),
            "penalty_score": penalty,
            "penalty_causes": str(penalty_list),
            "detected_count": len(teeth_data)
        }
        result.update(gaps)
        return result, mask, resized_img
