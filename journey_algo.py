#!/usr/bin/env python3
"""
Stellar Branching Algorithm - Biased Random Walk on a Real Star Graph
=====================================================================

This script grows a breadth-first branching tree over real 3D stellar
positions from the athyg-33 catalog. A single entity departs Sol, claims the
nearest star at or above Sol's spectral rank, then branches outward until
exactly 268 leaf entities occupy unique destination stars.

Key rules:
  - each destination star can be claimed only once
  - a child may target stars at its inherited spectral floor or one step higher
  - candidate stars are ranked by a blend of proximity and directional alignment
  - the first fork creates two opposing branches; later forks spread children in
    a cone around the parent's incoming direction

Default inputs and outputs:
  - catalog: athyg-33.csv located next to this script
  - output:  stellar_tree_versionN.toml in the current working directory

Usage:
  python journey_algo.py
  python journey_algo.py --seed 42
  python journey_algo.py --target 268
  python journey_algo.py --output .\\stellar_tree_custom.toml
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tomllib
from scipy.spatial import KDTree

MK_CLASSES = "OBAFGKM"
SPECTRAL_RE = re.compile(r"^\s*([OBAFGKM])\s*([0-9])", re.IGNORECASE)
OUTPUT_PATTERN = re.compile(r"^stellar_tree_version(\d+)\.toml$")

DEFAULT_TARGET_LEAVES = 268
DEFAULT_SEED = 42
COORDINATE_COLUMN_OPTIONS = (("x0", "y0", "z0"), ("x", "y", "z"))
INTERNAL_COORD_COLUMNS = ("_coord_x_pc", "_coord_y_pc", "_coord_z_pc")


@dataclass(frozen=True)
class SearchAttempt:
    min_rank: int
    k_candidates: int
    direction_weight: float


@dataclass(frozen=True)
class SpatialIndex:
    tree: KDTree
    coords: np.ndarray
    cat_indices: np.ndarray
    spectral_ranks: np.ndarray
    index_lookup: dict[int, int]


@dataclass
class TreeNode:
    """Represents one entity's arrival at a star."""

    cat_idx: int
    parent_id: int | None
    direction: np.ndarray
    min_rank: int
    node_id: int = field(init=False)
    children_ids: list[int] = field(default_factory=list)
    is_leaf: bool = True

    _next_id = 0

    def __post_init__(self) -> None:
        self.node_id = TreeNode._next_id
        TreeNode._next_id += 1


def resolve_coordinate_columns(df: pd.DataFrame) -> tuple[str, str, str]:
    """Return the first supported parsec-coordinate column set in the catalog."""
    for columns in COORDINATE_COLUMN_OPTIONS:
        if all(column in df.columns for column in columns):
            return columns
    supported = " or ".join("/".join(columns) for columns in COORDINATE_COLUMN_OPTIONS)
    raise ValueError(f"Catalog must include parsec coordinate columns: {supported}.")


def optional_int(row: pd.Series, *columns: str) -> int | None:
    """Read the first present numeric identifier from a row."""
    for column in columns:
        if column not in row.index or pd.isna(row[column]):
            continue
        value = pd.to_numeric(row[column], errors="coerce")
        if pd.notna(value):
            return int(value)
    return None


def optional_float(row: pd.Series, column: str) -> float | None:
    """Read an optional numeric field from a row."""
    if column not in row.index or pd.isna(row[column]):
        return None
    value = pd.to_numeric(row[column], errors="coerce")
    if pd.isna(value):
        return None
    return float(value)


def parse_spectral_rank(spect_str: str) -> int | None:
    """
    Convert a spectral type string such as 'G2V' or 'F5III' to a numeric rank.

    Returns None when the input does not start with a valid MK class letter
    followed by a numeric subclass.
    """
    if not isinstance(spect_str, str):
        return None

    match = SPECTRAL_RE.match(spect_str)
    if not match:
        return None

    primary = match.group(1).upper()
    subclass = int(match.group(2))
    class_index = MK_CLASSES.index(primary)
    return (6 - class_index) * 10 + (9 - subclass)


def load_catalog(csv_path: Path) -> pd.DataFrame:
    """
    Load the athyg CSV and prepare it for spatial queries.

    Filters out stars that lack valid 3D coordinates or a parseable spectral
    type. Adds a 'spectral_rank' column.
    """
    print(f"Loading catalog from {csv_path} ...")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"  Raw rows: {len(df)}")

    source_coord_columns = resolve_coordinate_columns(df)
    for source_column, internal_column in zip(
        source_coord_columns, INTERNAL_COORD_COLUMNS
    ):
        df[internal_column] = pd.to_numeric(df[source_column], errors="coerce")

    df["spectral_rank"] = df["spect"].apply(parse_spectral_rank)
    df = df.dropna(subset=["spectral_rank", *INTERNAL_COORD_COLUMNS]).copy()
    df["spectral_rank"] = df["spectral_rank"].astype(int)
    df["proper"] = df["proper"].fillna("")

    print(f"  Usable stars after filtering: {len(df)}")
    print(f"  Coordinate columns: {', '.join(source_coord_columns)}")
    print(
        f"  Spectral rank range: {df['spectral_rank'].min()} - "
        f"{df['spectral_rank'].max()}"
    )
    return df


def build_spatial_index(df: pd.DataFrame) -> SpatialIndex:
    """Build a KD-tree and aligned NumPy arrays for fast nearest-neighbor search."""
    coords = df[list(INTERNAL_COORD_COLUMNS)].to_numpy(dtype=np.float64)
    cat_indices = df.index.to_numpy()
    spectral_ranks = df["spectral_rank"].to_numpy(dtype=np.int16)
    index_lookup = {int(cat_idx): pos for pos, cat_idx in enumerate(cat_indices)}
    tree = KDTree(coords)

    print(f"  KD-tree built over {len(coords)} stars.")
    return SpatialIndex(
        tree=tree,
        coords=coords,
        cat_indices=cat_indices,
        spectral_ranks=spectral_ranks,
        index_lookup=index_lookup,
    )


def get_position(spatial_index: SpatialIndex, cat_idx: int) -> np.ndarray:
    """Return the 3D position for a catalog row label."""
    return spatial_index.coords[spatial_index.index_lookup[int(cat_idx)]]


def random_unit_vector(rng: random.Random) -> np.ndarray:
    """Generate a random unit vector uniformly on the sphere."""
    raw = np.array([rng.gauss(0, 1) for _ in range(3)], dtype=np.float64)
    norm = np.linalg.norm(raw)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return raw / norm


def select_destination(
    origin_pos: np.ndarray,
    direction: np.ndarray,
    min_rank: int,
    spatial_index: SpatialIndex,
    claimed: set[int],
    k_candidates: int,
    direction_weight: float,
) -> int | None:
    """
    Find the best destination star for an entity departing from `origin_pos`.

    Candidate stars are filtered by claim status and spectral rank floor, then
    scored by a weighted blend of inverse distance and directional alignment.
    """
    k = min(k_candidates, len(spatial_index.coords))
    dists, tree_idxs = spatial_index.tree.query(origin_pos, k=k)
    dists = np.atleast_1d(dists)
    tree_idxs = np.atleast_1d(tree_idxs)

    best_score = -1.0
    best_cat_idx: int | None = None

    dir_norm = np.linalg.norm(direction)
    has_direction = dir_norm > 1e-12
    dir_unit = direction / dir_norm if has_direction else None

    for dist, tree_idx in zip(dists, tree_idxs):
        tree_idx = int(tree_idx)
        cat_idx = int(spatial_index.cat_indices[tree_idx])

        if cat_idx in claimed:
            continue
        if dist < 1e-9:
            continue
        if int(spatial_index.spectral_ranks[tree_idx]) < min_rank:
            continue

        proximity = 1.0 / (1.0 + float(dist))

        if has_direction:
            to_candidate = spatial_index.coords[tree_idx] - origin_pos
            to_candidate_norm = np.linalg.norm(to_candidate)
            if to_candidate_norm > 1e-12:
                cos_sim = float(np.dot(dir_unit, to_candidate / to_candidate_norm))
                direction_score = (cos_sim + 1.0) / 2.0
            else:
                direction_score = 0.5
        else:
            direction_score = 0.5

        combined = (1.0 - direction_weight) * proximity + (
            direction_weight * direction_score
        )
        if combined > best_score:
            best_score = combined
            best_cat_idx = cat_idx

    return best_cat_idx


def choose_destination(
    origin_pos: np.ndarray,
    direction: np.ndarray,
    spatial_index: SpatialIndex,
    claimed: set[int],
    attempts: list[SearchAttempt],
) -> tuple[int | None, int]:
    """Try a series of progressively relaxed destination-search attempts."""
    last_rank = attempts[-1].min_rank
    for attempt in attempts:
        destination = select_destination(
            origin_pos=origin_pos,
            direction=direction,
            min_rank=attempt.min_rank,
            spatial_index=spatial_index,
            claimed=claimed,
            k_candidates=attempt.k_candidates,
            direction_weight=attempt.direction_weight,
        )
        if destination is not None:
            return destination, attempt.min_rank
        last_rank = attempt.min_rank
    return None, last_rank


def opposing_directions(rng: random.Random) -> list[np.ndarray]:
    """Generate exactly two opposing unit direction vectors."""
    unit = random_unit_vector(rng)
    return [unit, -unit]


def spread_directions(
    incoming_direction: np.ndarray,
    n_children: int,
    rng: random.Random,
    cone_half_angle_deg: float = 60.0,
) -> list[np.ndarray]:
    """Generate child directions spread through a cone around the incoming vector."""
    dir_norm = np.linalg.norm(incoming_direction)
    if dir_norm < 1e-12:
        return [random_unit_vector(rng) for _ in range(n_children)]

    w = incoming_direction / dir_norm

    if abs(w[0]) < 0.9:
        perp = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        perp = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    u = np.cross(w, perp)
    u /= np.linalg.norm(u)
    v = np.cross(w, u)

    cone_half_rad = math.radians(cone_half_angle_deg)
    directions: list[np.ndarray] = []

    for idx in range(n_children):
        azimuth = (2.0 * math.pi * idx / n_children) + rng.uniform(-0.3, 0.3)
        polar = rng.uniform(0.3 * cone_half_rad, cone_half_rad)

        dx = math.sin(polar) * math.cos(azimuth)
        dy = math.sin(polar) * math.sin(azimuth)
        dz = math.cos(polar)

        direction = dx * u + dy * v + dz * w
        direction /= np.linalg.norm(direction)
        directions.append(direction)

    return directions


def build_star_payload(
    catalog: pd.DataFrame,
    cat_idx: int,
    star_cache: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    """Serialize one star payload once and reuse it across the output tree."""
    cat_idx = int(cat_idx)
    if cat_idx in star_cache:
        return star_cache[cat_idx]

    row = catalog.loc[cat_idx]
    payload: dict[str, Any] = {
        "catalog_id": cat_idx,
        "spectral_type": row["spect"],
        "spectral_rank": int(row["spectral_rank"]),
        "position_parsecs": {
            "x": round(float(row["_coord_x_pc"]), 6),
            "y": round(float(row["_coord_y_pc"]), 6),
            "z": round(float(row["_coord_z_pc"]), 6),
        },
    }

    if row["proper"]:
        payload["proper_name"] = row["proper"]

    hyg_id = optional_int(row, "hyg") if "hyg" in row.index else optional_int(row, "id")
    if hyg_id is not None:
        payload["hyg_id"] = hyg_id

    if "hyg" in row.index:
        athyg_id = optional_int(row, "id")
        if athyg_id is not None:
            payload["athyg_id"] = athyg_id

    hip = optional_int(row, "hip")
    if hip is not None:
        payload["hip"] = hip

    luminosity = optional_float(row, "lum")
    if luminosity is not None:
        payload["luminosity"] = luminosity

    absolute_magnitude = optional_float(row, "absmag")
    if absolute_magnitude is not None:
        payload["absolute_magnitude"] = absolute_magnitude

    distance_from_sol = optional_float(row, "dist")
    if distance_from_sol is not None:
        payload["distance_from_sol_pc"] = round(distance_from_sol, 4)

    star_cache[cat_idx] = payload
    return payload


def build_node_payload(
    node: TreeNode,
    catalog: pd.DataFrame,
    star_cache: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    """Serialize a tree node to a TOML-friendly dictionary."""
    return {
        "node_id": node.node_id,
        "parent_id": node.parent_id,
        "children_ids": node.children_ids,
        "is_leaf": node.is_leaf,
        "star": build_star_payload(catalog, node.cat_idx, star_cache),
        "min_spectral_rank_required": node.min_rank,
    }


def compute_depths(all_nodes: dict[int, TreeNode]) -> dict[int, int]:
    """Compute the depth for every node in the tree."""
    depths: dict[int, int] = {}

    def compute_depth(node_id: int) -> int:
        if node_id in depths:
            return depths[node_id]

        node = all_nodes[node_id]
        if node.parent_id is None:
            depths[node_id] = 0
        else:
            depths[node_id] = compute_depth(node.parent_id) + 1
        return depths[node_id]

    for node_id in all_nodes:
        compute_depth(node_id)

    return depths


def run_branching_walk(
    catalog: pd.DataFrame,
    spatial_index: SpatialIndex,
    target_leaves: int = DEFAULT_TARGET_LEAVES,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Execute the stellar branching algorithm and return a serializable result."""
    rng = random.Random(seed)
    TreeNode._next_id = 0

    sol_mask = catalog["proper"] == "Sol"
    if not sol_mask.any():
        raise ValueError("Sol not found in catalog.")

    sol_idx = int(catalog.index[sol_mask][0])
    sol_pos = get_position(spatial_index, sol_idx)
    sol_rank = int(catalog.at[sol_idx, "spectral_rank"])

    print(
        f"\nSol found: catalog index {sol_idx}, spectral rank {sol_rank} "
        f"({catalog.at[sol_idx, 'spect']})"
    )

    claimed: set[int] = {sol_idx}
    all_nodes: dict[int, TreeNode] = {}

    first_dest, first_min_rank = choose_destination(
        origin_pos=sol_pos,
        direction=np.zeros(3, dtype=np.float64),
        spatial_index=spatial_index,
        claimed=claimed,
        attempts=[
            SearchAttempt(min_rank=sol_rank, k_candidates=128, direction_weight=0.0)
        ],
    )
    if first_dest is None:
        raise RuntimeError("No eligible star found for the initial departure from Sol.")

    claimed.add(first_dest)
    first_pos = get_position(spatial_index, first_dest)
    travel_dir = first_pos - sol_pos

    root = TreeNode(
        cat_idx=first_dest,
        parent_id=None,
        direction=travel_dir,
        min_rank=first_min_rank,
    )
    all_nodes[root.node_id] = root

    first_star_name = (
        catalog.at[first_dest, "proper"] or catalog.at[first_dest, "spect"]
    )
    first_star_dist = np.linalg.norm(travel_dir)
    print(
        f"  First destination: {first_star_name} "
        f"(rank {catalog.at[first_dest, 'spectral_rank']}, "
        f"{first_star_dist:.2f} pc from Sol)"
    )

    leaf_queue: deque[int] = deque()
    current_leaves: set[int] = set()

    root.is_leaf = False
    for direction in opposing_directions(rng):
        requested_rank = root.min_rank + rng.choice([0, 1])
        child_dest, child_min_rank = choose_destination(
            origin_pos=first_pos,
            direction=direction,
            spatial_index=spatial_index,
            claimed=claimed,
            attempts=[
                SearchAttempt(
                    min_rank=requested_rank,
                    k_candidates=128,
                    direction_weight=0.6,
                ),
                SearchAttempt(
                    min_rank=root.min_rank,
                    k_candidates=256,
                    direction_weight=0.4,
                ),
            ],
        )
        if child_dest is None:
            raise RuntimeError(
                "Unable to find a destination for one of the first two branches."
            )

        claimed.add(child_dest)
        child_pos = get_position(spatial_index, child_dest)
        child_dir = child_pos - first_pos

        child_node = TreeNode(
            cat_idx=child_dest,
            parent_id=root.node_id,
            direction=child_dir,
            min_rank=child_min_rank,
        )
        all_nodes[child_node.node_id] = child_node
        root.children_ids.append(child_node.node_id)
        leaf_queue.append(child_node.node_id)
        current_leaves.add(child_node.node_id)

    print(f"  After first fork: {len(current_leaves)} leaves")

    iteration = 0
    while len(current_leaves) < target_leaves and leaf_queue:
        iteration += 1

        node_id = leaf_queue.popleft()
        node = all_nodes[node_id]
        node_pos = get_position(spatial_index, node.cat_idx)

        leaves_needed = target_leaves - len(current_leaves)
        max_children = min(5, leaves_needed + 1)
        child_count = rng.randint(1, max_children)

        if child_count == 1:
            jitter = np.array([rng.gauss(0, 0.3) for _ in range(3)], dtype=np.float64)
            direction = node.direction / (np.linalg.norm(node.direction) + 1e-12)
            child_dirs = [direction + jitter]
            child_dirs[0] /= np.linalg.norm(child_dirs[0]) + 1e-12
        else:
            child_dirs = spread_directions(node.direction, child_count, rng)

        children_created = 0
        for direction in child_dirs:
            requested_rank = node.min_rank + rng.choice([0, 0, 1])
            attempts = [
                SearchAttempt(
                    min_rank=requested_rank,
                    k_candidates=128,
                    direction_weight=0.55,
                ),
                SearchAttempt(
                    min_rank=node.min_rank,
                    k_candidates=256,
                    direction_weight=0.3,
                ),
                SearchAttempt(
                    min_rank=node.min_rank,
                    k_candidates=1024,
                    direction_weight=0.1,
                ),
            ]

            child_dest, child_min_rank = choose_destination(
                origin_pos=node_pos,
                direction=direction,
                spatial_index=spatial_index,
                claimed=claimed,
                attempts=attempts,
            )
            if child_dest is None:
                continue

            claimed.add(child_dest)
            child_pos = get_position(spatial_index, child_dest)
            child_dir = child_pos - node_pos

            child_node = TreeNode(
                cat_idx=child_dest,
                parent_id=node.node_id,
                direction=child_dir,
                min_rank=child_min_rank,
            )
            all_nodes[child_node.node_id] = child_node
            node.children_ids.append(child_node.node_id)
            leaf_queue.append(child_node.node_id)
            current_leaves.add(child_node.node_id)
            children_created += 1

        if children_created == 0:
            raise RuntimeError(
                f"Unable to extend branching walk from node {node_id}; "
                "the exact leaf target can no longer be guaranteed."
            )

        node.is_leaf = False
        current_leaves.discard(node_id)

        if iteration % 25 == 0:
            print(
                f"  Iteration {iteration}: {len(current_leaves)} leaves, "
                f"{len(all_nodes)} total nodes, {len(claimed)} stars claimed"
            )

    if len(current_leaves) != target_leaves:
        raise RuntimeError(
            f"Branching walk finished with {len(current_leaves)} leaves, "
            f"expected {target_leaves}."
        )

    print(
        f"\n  DONE: {len(current_leaves)} leaf entities across "
        f"{len(all_nodes)} total nodes ({iteration} branching events)"
    )

    star_cache: dict[int, dict[str, Any]] = {}
    sol_row = catalog.loc[sol_idx]
    origin: dict[str, Any] = {
        "catalog_id": sol_idx,
        "proper_name": "Sol",
        "spectral_type": sol_row["spect"],
        "spectral_rank": sol_rank,
        "position_parsecs": {"x": 0.0, "y": 0.0, "z": 0.0},
    }

    sol_hyg_id = (
        optional_int(sol_row, "hyg")
        if "hyg" in sol_row.index
        else optional_int(sol_row, "id")
    )
    if sol_hyg_id is not None:
        origin["hyg_id"] = sol_hyg_id

    if "hyg" in sol_row.index:
        sol_athyg_id = optional_int(sol_row, "id")
        if sol_athyg_id is not None:
            origin["athyg_id"] = sol_athyg_id

    sol_hip = optional_int(sol_row, "hip")
    if sol_hip is not None:
        origin["hip"] = sol_hip

    sol_luminosity = optional_float(sol_row, "lum")
    if sol_luminosity is not None:
        origin["luminosity"] = sol_luminosity

    tree_nodes = {
        str(node_id): build_node_payload(all_nodes[node_id], catalog, star_cache)
        for node_id in sorted(all_nodes)
    }
    leaf_destinations = [
        {
            "node_id": all_nodes[node_id].node_id,
            "star": build_star_payload(catalog, all_nodes[node_id].cat_idx, star_cache),
        }
        for node_id in sorted(current_leaves)
    ]

    depths = compute_depths(all_nodes)
    leaf_depths = [depths[node_id] for node_id in current_leaves]

    return {
        "metadata": {
            "algorithm": "Stellar Branching - Biased Random Walk on HYG Star Graph",
            "description": (
                "A tree of entity paths starting from Sol, branching at each star, "
                "with entities biased toward stars of equal or greater spectral rank. "
                "Growth halts when 268 leaf entities occupy unique destination stars."
            ),
            "output_format": "toml",
            "target_leaves": target_leaves,
            "actual_leaves": len(current_leaves),
            "total_nodes": len(all_nodes),
            "total_stars_claimed": len(claimed),
            "branching_events": iteration,
            "seed": seed,
            "tree_depth_min": min(leaf_depths),
            "tree_depth_max": max(leaf_depths),
            "tree_depth_mean": round(sum(leaf_depths) / len(leaf_depths), 2),
        },
        "origin": origin,
        "tree": tree_nodes,
        "leaf_destinations": leaf_destinations,
    }


def verify_and_summarize(result: dict[str, Any]) -> bool:
    """Run integrity checks on the output tree and print a summary."""
    tree = result["tree"]
    leaves = result["leaf_destinations"]
    meta = result["metadata"]
    errors: list[str] = []

    print("\n" + "=" * 70)
    print("VERIFICATION & SUMMARY")
    print("=" * 70)

    actual_leaves = sum(1 for node in tree.values() if node["is_leaf"])
    print(f"  Leaf count in tree: {actual_leaves}  (target: {meta['target_leaves']})")
    if actual_leaves != meta["target_leaves"]:
        errors.append(
            f"Leaf count mismatch: {actual_leaves} != {meta['target_leaves']}"
        )

    star_ids = [node["star"]["catalog_id"] for node in tree.values()]
    unique_stars = set(star_ids)
    print(f"  Unique stars visited: {len(unique_stars)}  (nodes: {len(star_ids)})")
    if len(unique_stars) != len(star_ids):
        errors.append("Duplicate star detected in serialized tree.")

    violations = sum(
        1
        for node in tree.values()
        if node["star"]["spectral_rank"] < node["min_spectral_rank_required"]
    )
    print(f"  Spectral rank violations (star rank < required minimum): {violations}")
    if violations:
        errors.append(f"{violations} spectral rank floor violations found.")

    internal_childless = sum(
        1 for node in tree.values() if not node["is_leaf"] and not node["children_ids"]
    )
    leaf_with_children = sum(
        1 for node in tree.values() if node["is_leaf"] and node["children_ids"]
    )
    print(f"  Internal nodes without children: {internal_childless}")
    print(f"  Leaf nodes with children: {leaf_with_children}")
    if internal_childless:
        errors.append(f"{internal_childless} internal nodes are missing children.")
    if leaf_with_children:
        errors.append(f"{leaf_with_children} leaf nodes incorrectly list children.")

    leaf_dists = [
        leaf["star"]["distance_from_sol_pc"]
        for leaf in leaves
        if "distance_from_sol_pc" in leaf["star"]
    ]
    leaf_ranks = [leaf["star"]["spectral_rank"] for leaf in leaves]
    leaf_lums = [
        leaf["star"]["luminosity"] for leaf in leaves if "luminosity" in leaf["star"]
    ]

    print("\n  TREE STRUCTURE:")
    print(f"    Total nodes: {meta['total_nodes']}")
    print(f"    Branching events: {meta['branching_events']}")
    print(
        f"    Tree depth: min={meta['tree_depth_min']}, "
        f"max={meta['tree_depth_max']}, mean={meta['tree_depth_mean']}"
    )

    print("\n  LEAF DESTINATIONS (the 268 final stars):")
    print(
        f"    Distance from Sol: min={min(leaf_dists):.1f} pc, "
        f"max={max(leaf_dists):.1f} pc, "
        f"median={sorted(leaf_dists)[len(leaf_dists) // 2]:.1f} pc"
    )
    print(
        f"    Spectral rank: min={min(leaf_ranks)}, max={max(leaf_ranks)}, "
        f"mean={sum(leaf_ranks) / len(leaf_ranks):.1f}"
    )
    if leaf_lums:
        print(
            f"    Luminosity (solar): min={min(leaf_lums):.3f}, "
            f"max={max(leaf_lums):.1f}, "
            f"median={sorted(leaf_lums)[len(leaf_lums) // 2]:.3f}"
        )
    else:
        print("    Luminosity (solar): unavailable in catalog")

    named_leaves = [leaf for leaf in leaves if "proper_name" in leaf["star"]]
    if named_leaves:
        print("\n  NAMED STARS among the final destinations:")
        for leaf in named_leaves[:15]:
            star = leaf["star"]
            lum_text = (
                f"{star['luminosity']:.2f} L_sun" if "luminosity" in star else "n/a"
            )
            dist_text = (
                f"{star['distance_from_sol_pc']:.1f} pc"
                if "distance_from_sol_pc" in star
                else "n/a"
            )
            print(
                f"    {star['proper_name']:20s}  {star['spectral_type']:8s}  "
                f"rank={star['spectral_rank']:2d}  dist={dist_text:8s}  "
                f"lum={lum_text}"
            )

    if errors:
        print("\n  *** SOME CHECKS FAILED ***")
        for error in errors:
            print(f"    - {error}")
        return False

    print("\n  ALL CHECKS PASSED.")
    return True


def quote_toml_key(key: str) -> str:
    """Quote a TOML key segment when needed."""
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", key):
        return key
    return json.dumps(key)


def format_toml_scalar(value: Any) -> str:
    """Format a Python scalar as TOML."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("TOML output does not support NaN or infinity.")
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    raise TypeError(f"Unsupported TOML scalar type: {type(value)!r}")


def is_array_of_tables(value: Any) -> bool:
    """Return True when a value should be emitted as an array-of-tables."""
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, dict) for item in value)
    )


def emit_toml_key_value(lines: list[str], key: str, value: Any) -> None:
    """Write a scalar or scalar-array TOML assignment."""
    if value is None:
        return

    if isinstance(value, list):
        if not value:
            lines.append(f"{quote_toml_key(key)} = []")
            return
        if any(isinstance(item, dict) for item in value):
            raise TypeError("Nested table arrays must be emitted separately.")
        rendered = ", ".join(format_toml_scalar(item) for item in value)
        lines.append(f"{quote_toml_key(key)} = [{rendered}]")
        return

    lines.append(f"{quote_toml_key(key)} = {format_toml_scalar(value)}")


def split_mapping_items(
    mapping: dict[str, Any],
) -> tuple[
    list[tuple[str, Any]],
    list[tuple[str, dict[str, Any]]],
    list[tuple[str, list[dict[str, Any]]]],
]:
    """Split a mapping into scalar values, child tables, and arrays-of-tables."""
    scalars: list[tuple[str, Any]] = []
    tables: list[tuple[str, dict[str, Any]]] = []
    array_tables: list[tuple[str, list[dict[str, Any]]]] = []

    for key, value in mapping.items():
        if value is None:
            continue
        if isinstance(value, dict):
            tables.append((key, value))
        elif is_array_of_tables(value):
            array_tables.append((key, value))
        else:
            scalars.append((key, value))

    return scalars, tables, array_tables


def render_table_path(path: list[str]) -> str:
    """Render a TOML table path."""
    return ".".join(quote_toml_key(segment) for segment in path)


def emit_toml_mapping(
    lines: list[str],
    path: list[str],
    mapping: dict[str, Any],
    *,
    emit_header: bool,
) -> None:
    """Recursively emit a TOML table."""
    scalars, tables, array_tables = split_mapping_items(mapping)

    if emit_header:
        lines.append(f"[{render_table_path(path)}]")

    for key, value in scalars:
        emit_toml_key_value(lines, key, value)

    for key, child in tables:
        if lines and lines[-1] != "":
            lines.append("")
        emit_toml_mapping(lines, path + [key], child, emit_header=True)

    for key, items in array_tables:
        for item in items:
            if lines and lines[-1] != "":
                lines.append("")
            emit_toml_array_item(lines, path + [key], item)


def emit_toml_array_item(
    lines: list[str], path: list[str], mapping: dict[str, Any]
) -> None:
    """Emit one item in a TOML array-of-tables."""
    scalars, tables, array_tables = split_mapping_items(mapping)
    lines.append(f"[[{render_table_path(path)}]]")

    for key, value in scalars:
        emit_toml_key_value(lines, key, value)

    for key, child in tables:
        if lines and lines[-1] != "":
            lines.append("")
        emit_toml_mapping(lines, path + [key], child, emit_header=True)

    for key, items in array_tables:
        for item in items:
            if lines and lines[-1] != "":
                lines.append("")
            emit_toml_array_item(lines, path + [key], item)


def dumps_toml(data: dict[str, Any]) -> str:
    """Serialize the result object to TOML and validate it with tomllib."""
    lines: list[str] = []
    emit_toml_mapping(lines, [], data, emit_header=False)
    text = "\n".join(lines).rstrip() + "\n"
    tomllib.loads(text)
    return text


def resolve_default_catalog_path() -> Path:
    """Return the repo-local AT-HYG catalog path."""
    return Path(__file__).resolve().parent / "athyg-33.csv"


def next_versioned_output_path(directory: Path) -> Path:
    """Return the next available stellar_tree_versionN.toml path."""
    highest_version = 0
    for path in directory.glob("stellar_tree_version*.toml"):
        match = OUTPUT_PATTERN.fullmatch(path.name)
        if match:
            highest_version = max(highest_version, int(match.group(1)))
    return directory / f"stellar_tree_version{highest_version + 1}.toml"


def main() -> None:
    default_catalog = resolve_default_catalog_path()

    parser = argparse.ArgumentParser(
        description="Stellar branching algorithm over the AT-HYG star graph."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for reproducibility (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=default_catalog,
        help=f"Path to the AT-HYG CSV catalog (default: {default_catalog.name})",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=DEFAULT_TARGET_LEAVES,
        help=f"Target number of leaf entities (default: {DEFAULT_TARGET_LEAVES})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output TOML path. Default: next stellar_tree_versionN.toml in the "
            "current working directory."
        ),
    )
    args = parser.parse_args()

    catalog_path = args.catalog.resolve()
    if not catalog_path.exists():
        raise FileNotFoundError(f"Catalog not found: {catalog_path}")

    catalog = load_catalog(catalog_path)
    spatial_index = build_spatial_index(catalog)
    result = run_branching_walk(
        catalog=catalog,
        spatial_index=spatial_index,
        target_leaves=args.target,
        seed=args.seed,
    )

    verification_ok = verify_and_summarize(result)
    toml_text = dumps_toml(result)

    output_path = (
        args.output.resolve() if args.output else next_versioned_output_path(Path.cwd())
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(toml_text, encoding="utf-8")

    print(f"\n  Output written to: {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")

    if not verification_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
