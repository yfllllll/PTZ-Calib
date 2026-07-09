"""Track building from pairwise matches.

Python port of src/core/tracks.h and src/core/tracks.cc.

Data structures:
    Track = Dict[image_id, feature_id]  # a single track
    Tracks = Dict[track_id, Track]      # all tracks
"""

from __future__ import annotations

import sys
from typing import Dict, List, Set, Tuple

from .types import MatchesInfo
from .union_find import UnionFind

# {ImageId, FeatureId}
IndexedFeaturePair = Tuple[int, int]

# {ImageId -> FeatureId} for a single track
Track = Dict[int, int]

# {TrackId -> Track}
Tracks = Dict[int, Track]


class TracksBuilder:
    """Incremental track builder using Union-Find.

    Reference: openMVG's tracks.hpp.
    """

    def __init__(self):
        # (image_id, feature_id) -> node index in UF tree
        self.map_node_to_index_: Dict[IndexedFeaturePair, int] = {}
        # Ordered list of (pair, index) for stable iteration matching C++ ordering.
        self._nodes: List[IndexedFeaturePair] = []
        self.uf_tree_ = UnionFind()

    def build(self, matches_info: List[MatchesInfo]) -> None:
        # 1. Collect all unique (image_id, feature_id) tuples.
        all_features: Set[IndexedFeaturePair] = set()
        for mi in matches_info:
            i = mi.src_img_idx
            j = mi.dst_img_idx
            if i < 0 or j < 0:
                continue
            for m in mi.matches:
                all_features.add((i, m.queryIdx))
                all_features.add((j, m.trainIdx))

        # 2. Assign each unique node a stable index (sorted to match C++ std::set order).
        self._nodes = sorted(all_features)
        self.map_node_to_index_ = {feat: idx for idx, feat in enumerate(self._nodes)}

        # 3. Init the UF tree.
        self.uf_tree_.init_sets(len(self._nodes))

        # 4. Union matched features.
        for mi in matches_info:
            i = mi.src_img_idx
            j = mi.dst_img_idx
            if i < 0 or j < 0:
                continue
            for m in mi.matches:
                index_i = self.map_node_to_index_[(i, m.queryIdx)]
                index_j = self.map_node_to_index_[(j, m.trainIdx)]
                self.uf_tree_.union(index_i, index_j)

    def filter(self, min_track_length: int = 2) -> None:
        """Filter out invalid or too-short tracks."""
        tracks_map: Dict[int, Set[int]] = {}  # track_id -> {image_id, ...}
        problematic_track_id: Set[int] = set()

        n = len(self._nodes)
        for k in range(n):
            track_id = self.uf_tree_.find(k)
            img_id = self._nodes[k][0]
            s = tracks_map.setdefault(track_id, set())
            if img_id in s:
                # image already listed => invalid track (id collision)
                problematic_track_id.add(track_id)
            else:
                s.add(img_id)

        # Too-few observations
        for tid, imgs in tracks_map.items():
            if len(imgs) < min_track_length:
                problematic_track_id.add(tid)

        # Mark parents of nodes belonging to problematic tracks as invalid.
        INVALID = sys.maxsize
        cc_parent = self.uf_tree_.m_cc_parent
        cc_size = self.uf_tree_.m_cc_size
        for idx in range(len(cc_parent)):
            root = cc_parent[idx]
            if root in problematic_track_id:
                cc_size[root] = 1
                cc_parent[idx] = INVALID

    def export_to_stl(self) -> Tracks:
        """Return a dict {track_id: {image_id: feature_id}}."""
        INVALID = sys.maxsize
        tracks: Tracks = {}
        cc_parent = self.uf_tree_.m_cc_parent
        cc_size = self.uf_tree_.m_cc_size

        for k in range(len(self._nodes)):
            track_id = cc_parent[k]
            if track_id == INVALID:
                continue
            # Only keep tracks with 2+ observations.
            if track_id < len(cc_size) and cc_size[track_id] <= 1:
                continue
            img_id, feat_id = self._nodes[k]
            tracks.setdefault(track_id, {})[img_id] = feat_id
        return tracks

    def nb_tracks(self) -> int:
        INVALID = sys.maxsize
        parents = set(self.uf_tree_.m_cc_parent)
        parents.discard(INVALID)
        return len(parents)


def length(tracks: Tracks) -> Tuple[int, int, int]:
    """Return (total_length, max_length, min_length).

    Length of a track is the number of image observations.
    """
    total_length = 0
    max_length = 0
    min_length = sys.maxsize
    for _, track in tracks.items():
        n = len(track)
        total_length += n
        if n > max_length:
            max_length = n
        if n < min_length:
            min_length = n
    if not tracks:
        min_length = 0
    return total_length, max_length, min_length


def find_max_co_visible(tracks: Tracks, num_images: int) -> Set[int]:
    """Return the largest co-visible image set based on tracks.

    Two images are considered connected if they share at least one track.
    Returns the set of image indices in the largest connected component.
    """
    connected_img_sets: List[Set[int]] = []

    for _, track in tracks.items():
        # Which existing sets does this track connect to?
        matched_idx: List[int] = []
        for i, s in enumerate(connected_img_sets):
            for img_id in track.keys():
                if img_id in s:
                    matched_idx.append(i)
                    break

        new_set: Set[int] = set(track.keys())

        if not matched_idx:
            connected_img_sets.append(new_set)
        else:
            # Merge all matched sets into new_set, then remove them (from highest index).
            for i in sorted(set(matched_idx), reverse=True):
                new_set |= connected_img_sets[i]
                del connected_img_sets[i]
            connected_img_sets.append(new_set)

    max_set: Set[int] = set()
    for s in connected_img_sets:
        if len(s) > len(max_set):
            max_set = s
    return max_set


def save_tracks(tracks: Tracks, img_names: List[str], outpath: str) -> None:
    """Save tracks to a text file. Each line contains a single track:
        "img_name feature_id img_name feature_id ..."
    """
    try:
        with open(outpath, "w") as f:
            for _, track in tracks.items():
                parts: List[str] = []
                for img_id, feat_id in track.items():
                    parts.append(f"{img_names[img_id]} {feat_id}")
                f.write(" ".join(parts) + "\n")
    except OSError:
        print(f"Cannot write file to {outpath}", file=sys.stderr)
