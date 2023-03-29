#! /usr/bin/env python3

__all__ = ["track_and_calc_colors"]

from typing import List, Optional, Tuple

import numpy as np
import cv2
from collections import defaultdict
from corners import CornerStorage
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _camtrack import (
    PointCloudBuilder,
    create_cli,
    calc_point_cloud_colors,
    pose_to_view_mat3x4,
    to_opencv_camera_mat3x3,
    view_mat3x4_to_pose,
    build_correspondences,
    triangulate_correspondences,
    rodrigues_and_translation_to_view_mat3x4,
    _remove_correspondences_with_ids,
    eye3x4,
    Correspondences,
    compute_reprojection_errors,
    TriangulationParameters,
)


class CamTracker:
    def __init__(
        self,
        length,
        intrinsic_mat,
        corner_storage,
        v1,
        v2,
        triang_params,
        retriang_params,
    ):

        self.REPR_ERR = 1.6
        self.RETR_TOP_BORDER = 10
        self.RETR_ITERS = 4

        self.intr_mat = intrinsic_mat
        self.corner_storage = corner_storage
        self.length = length
        self.triang_params = triang_params
        self.retriang_params = retriang_params
        self.view_mats = [None] * self.length
        self._inliers = [None] * self.length
        self.empty_view = self.length
        self.cloud = {}
        self.corners_to_frames = defaultdict(list)
        self.retriangulated = {}

        if v1 is None or v2 is None:
            print("Initialize tracking")
            v1, v2 = self.find_start_poses()
            print("Initial frames: [{}, {}]".format(v1[0], v2[0]))

        self.add_to_view_mats(v1[0], pose_to_view_mat3x4(v1[1]), -1)
        self.add_to_view_mats(v2[0], pose_to_view_mat3x4(v2[1]), -1)

        for fn in range(self.length):
            for ind, cn in enumerate(self.corner_storage[fn].ids.flatten()):
                self.corners_to_frames[cn] += [(fn, ind)]

        corr = build_correspondences(
            self.corner_storage[v1[0]], self.corner_storage[v2[0]]
        )
        p3d, ids, median_cos = triangulate_correspondences(
            corr,
            self.view_mats[v1[0]],
            self.view_mats[v2[0]],
            self.intr_mat,
            self.triang_params,
        )
        for p, i, l in zip(p3d, ids, 2 * np.ones(ids.shape)):
            self.add_to_cloud(p, i, l)
        print("Cloud initialised with {} points".format(len(self.cloud)))

    def find_start_poses(self):
        best_frames = (0, 0)
        best_points = 0
        best_second_pose = None
        indent = 5 if self.length > 30 else 1
        for i in range(self.length):
            print("Check frame {}".format(i))
            for j in range(i + indent, self.length):
                pose, points = self.find_poses_by_2_frames(i, j)
                if points > best_points:
                    best_frames = (i, j)
                    best_second_pose = pose
                    best_points = points
        return (
            (best_frames[0], view_mat3x4_to_pose(eye3x4())),
            (best_frames[1], best_second_pose),
        )

    def find_poses_by_2_frames(self, frame1, frame2):
        corr = build_correspondences(
            self.corner_storage[frame1], self.corner_storage[frame2]
        )

        H, mask_h = cv2.findHomography(corr[1], corr[2], method=cv2.RANSAC)

        E, mask_e = cv2.findEssentialMat(
            corr[1], corr[2], self.intr_mat, method=cv2.RANSAC, threshold=1.0
        )

        best_pose = None
        best_points = 0

        if np.sum(mask_h.flatten()) / np.sum(mask_e.flatten()) <= 0.5:
            corr = _remove_correspondences_with_ids(
                corr, np.argwhere(mask_e == 0)
            )
            R1, R2, t = cv2.decomposeEssentialMat(E)

            poses = [
                Pose(R1.T, R1.T @ t),
                Pose(R2.T, R2.T @ t),
                Pose(R1.T, R1.T @ (-t)),
                Pose(R2.T, R2.T @ (-t)),
            ]

            for pose in poses:
                points, ids, median_cos = triangulate_correspondences(
                    corr,
                    eye3x4(),
                    pose_to_view_mat3x4(pose),
                    self.intr_mat,
                    self.triang_params,
                )
                if len(points) > best_points:
                    best_pose = pose
                    best_points = len(points)

        return best_pose, best_points

    def add_to_cloud(self, point, ind, isin):
        if ind not in self.cloud or self.cloud[ind]["inl"] < isin:
            self.cloud[ind] = {"inl": 1, "coord": point}

    def add_to_view_mats(self, ind, view_mat3x4, inliers):
        if self.view_mats[ind] is None:
            self.empty_view -= 1
        self.view_mats[ind] = view_mat3x4
        self._inliers[ind] = inliers
        print("Frame tracked\n{} remains".format(self.empty_view))

    def find_pose(self, fn):
        points3d, points2d, status = self.map_3d_2d_on_frame(fn)
        if not status:
            return None

        status, R, t, inl = cv2.solvePnPRansac(
            points3d,
            points2d,
            self.intr_mat,
            None,
            confidence=0.999,
            flags=cv2.SOLVEPNP_EPNP,
            reprojectionError=self.REPR_ERR,
        )

        if not status:
            return None

        inlier_ind = np.array(inl).flatten()

        g_points3d = points3d[inlier_ind]
        g_points2d = points2d[inlier_ind]

        status, R, t = cv2.solvePnP(
            g_points3d,
            g_points2d,
            self.intr_mat,
            None,
            flags=cv2.SOLVEPNP_ITERATIVE,
            useExtrinsicGuess=True,
            rvec=R,
            tvec=t,
        )

        return R, t, len(inlier_ind)

    def retriangulate(self, ind_c):
        frames, points2d, mats = [], [], []
        for frame, _id in self.corners_to_frames[ind_c]:
            if self.view_mats[frame] is not None:
                frames += [frame]
                mats += [self.view_mats[frame]]
                points2d += [self.corner_storage[frame].points[_id]]
        fr_leng = len(frames)
        if fr_leng > 2:
            if fr_leng > self.RETR_TOP_BORDER:
                mask = np.random.choice(
                    np.arange(fr_leng), size=self.RETR_TOP_BORDER, replace=False
                )
                frames = np.array(frames)[mask]
                points2d = np.array(points2d)[mask]
                mats = np.array(mats)[mask]
                fr_leng = self.RETR_TOP_BORDER
            else:
                frames = np.array(frames)
                points2d = np.array(points2d)
                mats = np.array(mats)

            _pos = None
            _metric = None
            for i in range(self.RETR_ITERS):
                x, y = np.random.choice(fr_leng, 2, replace=False)
                point3d, ids, median_cos = triangulate_correspondences(
                    Correspondences(
                        np.zeros(1),
                        np.array([points2d[x]]),
                        np.array([points2d[y]]),
                    ),
                    mats[x],
                    mats[y],
                    self.intr_mat,
                    self.retriang_params,
                )
                if point3d.size == 0:
                    continue
                metric = self.compute_inliers_repr(frames, points2d, point3d)
                if _pos is None or _metric < metric:
                    _pos = point3d[0]
                    _metric = metric
            return (_pos, _metric) if _pos is not None else None
        else:
            return None

    def compute_inliers_repr(self, frames, points2d, point3d):
        errors = []
        for frame, point in zip(frames, points2d):
            val = compute_reprojection_errors(
                point3d,
                np.array([point]),
                self.intr_mat @ self.view_mats[frame],
            )
            errors += [val.flatten()[0]]
        return np.count_nonzero(np.array(errors) <= self.REPR_ERR)

    def map_3d_2d_on_frame(self, fn):
        p3d = []
        p2d = []
        status = True
        frame_corn = self.corner_storage[fn]
        for ind, p in zip(frame_corn.ids.flatten(), frame_corn.points):
            if ind in self.cloud:
                p3d += [self.cloud[ind]["coord"]]
                p2d += [p]
        if len(p3d) < 4:
            status = False

        p3d = np.array(p3d, dtype="float")
        p2d = np.array(p2d, dtype="float")
        return p3d, p2d, status

    def retriangulatible_corners(self, frame, it):
        retr_candidates = []
        for ind in self.corner_storage[frame].ids.flatten():
            if (
                ind not in self.retriangulated
                or self.retriangulated[ind] < it - 5
            ):
                retr_candidates += [ind]
        if len(retr_candidates) > 1000:
            retr_candidates = np.random.choice(
                retr_candidates, 1000, replace=False
            )
        else:
            retr_candidates = np.array(retr_candidates)

        return retr_candidates

    def run_find_pose(self, refind=False):
        if refind:
            frames = [
                i for i in range(self.length) if self.view_mats[i] is not None
            ]
        else:
            frames = [
                i for i in range(self.length) if self.view_mats[i] is None
            ]
        best_fr = None
        best_Rt = None
        metric = None
        for fn in frames:
            res = self.find_pose(fn)
            if res is not None:
                R, t, met = res
                if refind:
                    if met > self._inliers[fn] and self._inliers[fn] != -1:
                        self.add_to_view_mats(
                            fn,
                            rodrigues_and_translation_to_view_mat3x4(R, t),
                            met,
                        )
                else:
                    if best_fr is None or metric > met:
                        metric = met
                        best_fr = fn
                        best_Rt = (R, t)
        return best_fr, best_Rt, metric

    def track(self):
        it = 0
        while self.empty_view > 0:
            it += 1

            new_fr, new_Rt, new_inl = self.run_find_pose()

            self.add_to_view_mats(
                new_fr,
                rodrigues_and_translation_to_view_mat3x4(*new_Rt),
                new_inl,
            )

            retr_c = self.retriangulatible_corners(new_fr, it)
            print("Trying to retriangulate {} points".format(len(retr_c)))
            update_points = []
            for ind in retr_c:
                res = self.retriangulate(ind)
                self.retriangulated[ind] = it
                if res is not None:
                    point, inl = res
                    update_points += [(ind, point, inl)]

            for ind, point, inl in update_points:
                self.add_to_cloud(point, ind, inl)

            if it % 5:
                print("Trying to update pose")
                new_fr, new_Rt, new_inl = self.run_find_pose(True)
                if (
                    self._inliers is not None
                    and new_fr is not None
                    and self._inliers[new_fr] < new_inl
                ):
                    self.add_to_view_mats(new_fr, new_Rt, new_inl)


def track_and_calc_colors(
    camera_parameters: CameraParameters,
    corner_storage: CornerStorage,
    frame_sequence_path: str,
    known_view_1: Optional[Tuple[int, Pose]] = None,
    known_view_2: Optional[Tuple[int, Pose]] = None,
) -> Tuple[List[Pose], PointCloud]:
    # if known_view_1 is None or known_view_2 is None:
    # raise NotImplementedError()

    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters, rgb_sequence[0].shape[0]
    )

    frame_length = len(corner_storage)
    retriang_params = TriangulationParameters(1.5, 2.5, 0.1)
    triang_params = TriangulationParameters(2, 1e-3, 1e-4)

    camtracker = CamTracker(
        frame_length,
        intrinsic_mat,
        corner_storage,
        known_view_1,
        known_view_2,
        triang_params,
        retriang_params,
    )

    camtracker.track()

    ids = []
    points = []
    for k, v in camtracker.cloud.items():
        ids += [k]
        points += [v["coord"]]
    point_cloud_builder = PointCloudBuilder(np.array(ids), np.array(points))
    view_mats = camtracker.view_mats

    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        corner_storage,
        1.6,
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats))
    return poses, point_cloud


if __name__ == "__main__":
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()
