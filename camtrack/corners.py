#! /usr/bin/env python3

__all__ = [
    "FrameCorners",
    "CornerStorage",
    "build",
    "dump",
    "load",
    "draw",
    "calc_track_interval_mappings",
    "calc_track_len_array_mapping",
    "without_short_tracks",
]

import click
import cv2
import numpy as np
import pims
from scipy.interpolate import griddata

from _corners import (
    FrameCorners,
    CornerStorage,
    StorageImpl,
    dump,
    load,
    draw,
    calc_track_interval_mappings,
    calc_track_len_array_mapping,
    without_short_tracks,
    create_cli,
    _to_int_tuple,
)


class _CornerStorageBuilder:
    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def get_corners_at_frame(self, frame):
        return self._corners[frame]

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))


class Tracker:
    def __init__(self, frame_0, treshold=200):
        self.frame_0 = frame_0
        self.treshold = treshold
        self.block_s = round(max(self.frame_0.shape) * 0.01)
        self.win_s = self.block_s * 3
        self.levels = 4
        self.dist = self.block_s
        self.quality = 0.03
        self.max_corners = max(
            500,
            min(
                2000,
                round(
                    self.frame_0.shape[0]
                    / self.block_s
                    * self.frame_0.shape[1]
                    / self.block_s
                ),
            ),
        )
        self.corners, self.corner_sizes = self.find_corners(frame_0)
        self.ids = np.arange(self.corners.shape[0])
        self.track_length = np.ones((self.corners.shape[0]), dtype="int")
        self.max_length = 0

    def find_flow(self, frame_1):
        tracked, mask, mask1 = self.use_optflow(
            self.frame_0, frame_1, self.corners
        )

        m = (mask1 == 1) & (mask == 1)
        new_m = m.reshape((-1,))

        new_corners, new_sizes = self.find_corners(
            frame_1,
            mask=self.get_corner_cover(tracked[m], self.corner_sizes[new_m]),
        )

        self.frame_0 = frame_1
        self.corners = np.concatenate((tracked[m], new_corners), axis=0)

        self.corner_sizes = np.concatenate(
            (self.corner_sizes[new_m], new_sizes), axis=0
        )

        last_id = self.ids[-1]
        new_ids = np.arange(last_id + 1, last_id + new_corners.shape[0] + 1)

        self.ids = np.concatenate((self.ids[new_m], new_ids), axis=0)

        self.track_length[new_m] += 1
        self.track_length = np.concatenate(
            (
                self.track_length[new_m],
                np.ones(
                    new_corners.shape[0],
                ),
            ),
            axis=0,
        )

        cur_max_lenght = self.track_length.max()
        if cur_max_lenght > self.max_length:
            self.max_length = cur_max_lenght

    def find_layer_corners(self, frame, mask=None):
        corners = cv2.goodFeaturesToTrack(
            frame,
            self.max_corners,
            self.quality,
            self.dist,
            mask=mask,
            blockSize=self.block_s,
        )
        if corners is None:
            return np.empty((0, 2), dtype=float)
        corners = corners[:, 0, :]
        return corners

    def find_corners(self, frame, mask=None):
        layer = frame.copy()
        if mask is not None:
            mask_l = mask.copy()
        else:
            mask_l = np.full(frame.shape, 255, dtype="uint8")
        k = 1
        corners = np.empty((0, 2), dtype=float)
        sizes = np.empty((0,), dtype=float)

        for _ in range(self.levels):
            layer_corners = self.find_layer_corners(layer, mask=mask_l)

            filt_mask = np.zeros((len(layer_corners),), dtype="bool")
            if layer_corners is not None:
                for i, point in enumerate(layer_corners):
                    rx, ry = np.round(point).astype(int)
                    if mask_l[ry, rx] != 0:
                        filt_mask[i] = True

            layer_corners = layer_corners[filt_mask]
            if np.sum(~filt_mask) > 0:
                print("oh shit")

            corners = np.concatenate((corners, layer_corners * k), axis=0)
            sizes = np.concatenate(
                (sizes, np.array([k * self.block_s] * layer_corners.shape[0])),
                axis=0,
            )
            mask_l = self.enlarge_corner_cover(
                mask_l, layer_corners, k * self.block_s
            )
            layer = cv2.pyrDown(layer)
            mask_l = cv2.pyrDown(mask_l).astype(np.uint8)
            mask_l = np.where(mask_l < 200, 0, 255).astype("uint8")
            k *= 2

        return corners, sizes

    def get_corner_cover(self, corners=None, corner_sizes=None):
        if corners is None:
            corners = self.corners
            corner_sizes = self.corner_sizes
        cover = np.full(self.frame_0.shape, 255, dtype="uint8")
        for corner, rad in zip(corners, corner_sizes):
            coord = _to_int_tuple(corner)
            cover = cv2.circle(cover, coord, int(rad), color=0, thickness=-1)
        return cover

    def enlarge_corner_cover(self, cover, corners, rad):
        for corner in corners:
            coord = _to_int_tuple(corner)
            cover = cv2.circle(cover, coord, int(rad), color=0, thickness=-1)
        return cover

    def use_optflow(self, frame_0, frame_1, points, eps=1e-2, steps=5):
        levels, frame_0_pyr = cv2.buildOpticalFlowPyramid(
            (frame_0 * 255).astype(np.uint8),
            (self.win_s, self.win_s),
            self.levels,
            None,
            False,
        )

        levels, frame_1_pyr = cv2.buildOpticalFlowPyramid(
            (frame_1 * 255).astype(np.uint8),
            (self.win_s, self.win_s),
            self.levels,
            None,
            False,
        )

        new_points, mask, _ = cv2.calcOpticalFlowPyrLK(
            frame_0_pyr[0],
            frame_1_pyr[0],
            points.astype("float32").reshape((-1, 1, 2)),
            None,
            winSize=(self.win_s, self.win_s),
            maxLevel=self.levels,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                10,
                0.03,
            ),
        )

        new_points1, mask1, _ = cv2.calcOpticalFlowPyrLK(
            frame_1_pyr[0],
            frame_0_pyr[0],
            new_points.astype("float32").reshape((-1, 1, 2)),
            None,
            winSize=(self.win_s, self.win_s),
            maxLevel=self.levels,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                10,
                0.03,
            ),
        )

        deltas = points.reshape((-1, 1, 2)) - new_points1
        deltas = np.linalg.norm(deltas, axis=2)
        mask = (mask == 1) & (deltas < 0.7)

        return new_points, mask, mask1

    def filter_corners_mask(self, frame):
        eignvals = cv2.cornerMinEigenVal(frame, 10)
        xx, yy = np.meshgrid(
            np.arange(frame.shape[1]), np.arange(frame.shape[0])
        )
        coords = np.array((xx.ravel(), yy.ravel())).T

        corn_eig = griddata(
            coords, eignvals.flatten(), self.corners, fill_value=0
        )
        mask = corn_eig > np.quantile(corn_eig, 0.5)
        return mask

    def get_corners(self):
        return FrameCorners(self.ids[:], self.corners[:], self.corner_sizes[:])


def _build_impl(
    frame_sequence: pims.FramesSequence, builder: _CornerStorageBuilder
) -> None:
    tracker = Tracker(frame_sequence[0])
    builder.set_corners_at_frame(0, tracker.get_corners())
    for frame, image_1 in enumerate(frame_sequence[1:], 1):
        tracker.find_flow(image_1)  # find new corners and track them
        builder.set_corners_at_frame(frame, tracker.get_corners())
    print("\n", tracker.max_length)


def build(
    frame_sequence: pims.FramesSequence, progress: bool = True
) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.

    :param frame_sequence: grayscale float32 frame sequence.
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(
            length=len(frame_sequence), label="Calculating corners"
        ) as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder)
    return builder.build_corner_storage()


if __name__ == "__main__":
    create_cli(build)()  # pylint:disable=no-value-for-parameter
