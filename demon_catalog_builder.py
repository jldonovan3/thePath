#!/usr/bin/env python3
"""
Build a merged demon catalog from a stellar tree and demon list.

Default inputs and outputs:
  - stellar tree: latest stellar_tree_versionN.toml in the current directory
  - demon list:   demonlist.toml in the current directory
  - output:       next demon_catalog_vN.toml in the current directory
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import tomllib

STELLAR_TREE_PATTERN = re.compile(r"^stellar_tree_version(\d+)\.toml$")
DEMON_CATALOG_PATTERN = re.compile(r"^demon_catalog_v(\d+)\.toml$")

ORIGIN_FIELDS = ("athyg_id", "proper_name")
TREE_FIELDS = ("children_ids", "is_leaf")
STAR_FIELDS = ("hip", "spectral_type", "distance_from_sol_pc")
DEMON_FIELDS = (
    "hip",
    "demon_name",
    "demon_type",
    "poetic_utterance",
    "concordance",
    "demon_form",
    "known",
    "first_revelation",
)


class CatalogValidationError(ValueError):
    """Raised when source TOML files cannot produce a valid demon catalog."""


class ValidationContext:
    def __init__(self) -> None:
        self.errors: list[str] = []

    def add(self, message: str) -> None:
        self.errors.append(message)

    def raise_if_any(self) -> None:
        if not self.errors:
            return

        details = "\n".join(f"  - {error}" for error in self.errors)
        raise CatalogValidationError(
            f"Cannot build demon catalog; found {len(self.errors)} validation "
            f"error(s):\n{details}"
        )


def load_toml(path: Path, label: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"{label} file not found: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{label} path is not a file: {path}")

    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise CatalogValidationError(
            f"{label} file is not valid TOML: {path}: {exc}"
        ) from exc


def describe_value(value: Any) -> str:
    return f"{value!r} ({type(value).__name__})"


def is_exact_type(value: Any, expected_type: type) -> bool:
    return type(value) is expected_type


def require_table(
    ctx: ValidationContext,
    mapping: dict[str, Any],
    key: str,
    location: str,
) -> dict[str, Any] | None:
    value = mapping.get(key)
    if isinstance(value, dict):
        return value

    if key not in mapping:
        ctx.add(f"{location} is missing required table {key!r}.")
    else:
        ctx.add(f"{location}.{key} must be a table, got {describe_value(value)}.")
    return None


def require_field(
    ctx: ValidationContext,
    mapping: dict[str, Any],
    key: str,
    expected_type: type,
    location: str,
) -> Any:
    if key not in mapping:
        ctx.add(f"{location} is missing required field {key!r}.")
        return None

    value = mapping[key]
    if not is_exact_type(value, expected_type):
        ctx.add(
            f"{location}.{key} must be {expected_type.__name__}, "
            f"got {describe_value(value)}."
        )
        return None

    return value


def require_int_array(
    ctx: ValidationContext,
    mapping: dict[str, Any],
    key: str,
    location: str,
) -> list[int] | None:
    value = require_field(ctx, mapping, key, list, location)
    if value is None:
        return None

    bad_items = [
        (index, item)
        for index, item in enumerate(value)
        if not is_exact_type(item, int)
    ]
    if bad_items:
        rendered = ", ".join(
            f"index {index}: {describe_value(item)}" for index, item in bad_items[:5]
        )
        suffix = "" if len(bad_items) <= 5 else f", plus {len(bad_items) - 5} more"
        ctx.add(
            f"{location}.{key} must contain only int values; got {rendered}{suffix}."
        )
        return None

    return value


def sorted_numeric_keys(
    mapping: dict[str, Any], location: str, ctx: ValidationContext
) -> list[str]:
    numeric_keys: list[str] = []
    for key, value in mapping.items():
        if not isinstance(value, dict):
            ctx.add(f"{location}.{key} must be a table, got {describe_value(value)}.")
            continue
        if not key.isdecimal():
            ctx.add(f"{location} contains non-numeric tree id {key!r}.")
            continue
        numeric_keys.append(key)

    return sorted(numeric_keys, key=int)


def parse_stellar_tree(
    data: dict[str, Any], path: Path, ctx: ValidationContext
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    origin = require_table(ctx, data, "origin", str(path))
    tree = require_table(ctx, data, "tree", str(path))

    parsed_origin: dict[str, Any] = {}
    if origin is not None:
        athyg_id = require_field(ctx, origin, "athyg_id", int, f"{path} [origin]")
        proper_name = require_field(ctx, origin, "proper_name", str, f"{path} [origin]")
        if athyg_id is not None:
            parsed_origin["athyg_id"] = athyg_id
        if proper_name is not None:
            parsed_origin["proper_name"] = proper_name

    nodes: dict[str, dict[str, Any]] = {}
    leaves: dict[str, dict[str, Any]] = {}

    if tree is None:
        return parsed_origin, nodes, leaves

    for tree_id in sorted_numeric_keys(tree, f"{path} [tree]", ctx):
        entry = tree[tree_id]
        location = f'{path} [tree."{tree_id}"]'

        parsed_entry: dict[str, Any] = {}
        if tree_id != "0":
            parent_id = require_field(ctx, entry, "parent_id", int, location)
            if parent_id is not None:
                parsed_entry["parent_id"] = parent_id
        elif "parent_id" in entry:
            ctx.add(f'{location}.parent_id must be omitted for root tree id "0".')

        children_ids = require_int_array(ctx, entry, "children_ids", location)
        is_leaf = require_field(ctx, entry, "is_leaf", bool, location)
        star = require_table(ctx, entry, "star", location)

        if children_ids is not None:
            parsed_entry["children_ids"] = children_ids
        if is_leaf is not None:
            parsed_entry["is_leaf"] = is_leaf

        if star is not None:
            star_location = f'{path} [tree."{tree_id}".star]'
            hip = require_field(ctx, star, "hip", int, star_location)
            spectral_type = require_field(
                ctx, star, "spectral_type", str, star_location
            )
            distance = require_field(
                ctx,
                star,
                "distance_from_sol_pc",
                float,
                star_location,
            )
            if hip is not None:
                parsed_entry["hip"] = hip
            if spectral_type is not None:
                parsed_entry["spectral_type"] = spectral_type
            if distance is not None:
                parsed_entry["distance_from_sol_pc"] = distance

        if parsed_entry.get("is_leaf") is True:
            leaves[tree_id] = parsed_entry
        elif parsed_entry.get("is_leaf") is False:
            nodes[tree_id] = parsed_entry

    return parsed_origin, nodes, leaves


def parse_demonlist(
    data: dict[str, Any], path: Path, ctx: ValidationContext
) -> dict[int, dict[str, Any]]:
    by_hip: dict[int, dict[str, Any]] = {}
    first_table_by_hip: dict[int, str] = {}

    for table_name in sorted(data):
        table = data[table_name]
        location = f'{path} ["{table_name}"]'
        if not isinstance(table, dict):
            ctx.add(f"{location} must be a table, got {describe_value(table)}.")
            continue

        parsed: dict[str, Any] = {}
        field_types = {
            "hip": int,
            "demon_name": str,
            "demon_type": int,
            "poetic_utterance": str,
            "concordance": str,
            "demon_form": str,
            "known": bool,
            "first_revelation": int,
        }
        for field_name in DEMON_FIELDS:
            value = require_field(
                ctx, table, field_name, field_types[field_name], location
            )
            if value is not None:
                parsed[field_name] = value

        hip = parsed.get("hip")
        if hip is None:
            continue
        if hip in by_hip:
            ctx.add(
                f"{path} has duplicate demon hip {hip}: "
                f"{first_table_by_hip[hip]!r} and {table_name!r}."
            )
            continue

        by_hip[hip] = parsed
        first_table_by_hip[hip] = table_name

    return by_hip


def validate_leaf_hips(
    leaves: dict[str, dict[str, Any]],
    demons_by_hip: dict[int, dict[str, Any]],
    ctx: ValidationContext,
) -> None:
    leaf_hip_to_id: dict[int, str] = {}
    for tree_id, leaf in leaves.items():
        hip = leaf.get("hip")
        if hip is None:
            continue
        if hip in leaf_hip_to_id:
            ctx.add(
                f"stellar tree has duplicate leaf hip {hip}: "
                f'tree ids "{leaf_hip_to_id[hip]}" and "{tree_id}".'
            )
            continue
        leaf_hip_to_id[hip] = tree_id

    leaf_hips = set(leaf_hip_to_id)
    demon_hips = set(demons_by_hip)
    missing_demons = sorted(leaf_hips - demon_hips)
    extra_demons = sorted(demon_hips - leaf_hips)

    if missing_demons:
        ctx.add(
            "demonlist.toml is missing demon rows for leaf hip values: "
            f"{missing_demons}."
        )
    if extra_demons:
        ctx.add(
            "demonlist.toml contains hip values that are not leaf destinations: "
            f"{extra_demons}."
        )


def ordered_tree_row(entry: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    if "parent_id" in entry:
        row["parent_id"] = entry["parent_id"]
    row["children_ids"] = entry["children_ids"]
    row["hip"] = entry["hip"]
    row["is_leaf"] = entry["is_leaf"]
    row["spectral_type"] = entry["spectral_type"]
    row["distance_from_sol_pc"] = entry["distance_from_sol_pc"]
    return row


def build_catalog(
    origin: dict[str, Any],
    nodes: dict[str, dict[str, Any]],
    leaves: dict[str, dict[str, Any]],
    demons_by_hip: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    node_tables = {
        tree_id: ordered_tree_row(nodes[tree_id]) for tree_id in sorted(nodes, key=int)
    }

    leaf_tables: dict[str, dict[str, Any]] = {}
    for tree_id in sorted(leaves, key=int):
        leaf = leaves[tree_id]
        row = ordered_tree_row(leaf)
        demon = demons_by_hip[leaf["hip"]]
        for field_name in DEMON_FIELDS:
            row[field_name] = demon[field_name]
        leaf_tables[tree_id] = row

    return {
        "demon_tree": {
            "origin": {
                "athyg_id": origin["athyg_id"],
                "proper_name": origin["proper_name"],
            },
            "node": node_tables,
            "leaf": leaf_tables,
        }
    }


def quote_toml_key(key: str) -> str:
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", key):
        return key
    return json.dumps(key)


def format_toml_scalar(value: Any) -> str:
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


def emit_key_value(lines: list[str], key: str, value: Any) -> None:
    if isinstance(value, list):
        if any(isinstance(item, dict) for item in value):
            raise TypeError("Nested table arrays are not supported.")
        rendered = ", ".join(format_toml_scalar(item) for item in value)
        lines.append(f"{quote_toml_key(key)} = [{rendered}]")
        return

    lines.append(f"{quote_toml_key(key)} = {format_toml_scalar(value)}")


def split_mapping_items(
    mapping: dict[str, Any],
) -> tuple[list[tuple[str, Any]], list[tuple[str, dict[str, Any]]]]:
    scalars: list[tuple[str, Any]] = []
    tables: list[tuple[str, dict[str, Any]]] = []

    for key, value in mapping.items():
        if isinstance(value, dict):
            tables.append((key, value))
        else:
            scalars.append((key, value))

    return scalars, tables


def render_table_path(path: list[str]) -> str:
    return ".".join(quote_toml_key(segment) for segment in path)


def emit_mapping(
    lines: list[str],
    path: list[str],
    mapping: dict[str, Any],
    *,
    emit_header: bool,
) -> None:
    scalars, tables = split_mapping_items(mapping)

    if emit_header:
        lines.append(f"[{render_table_path(path)}]")

    for key, value in scalars:
        emit_key_value(lines, key, value)

    for key, child in tables:
        if lines and lines[-1] != "":
            lines.append("")
        emit_mapping(lines, path + [key], child, emit_header=True)


def dumps_toml(data: dict[str, Any]) -> str:
    lines: list[str] = []
    emit_mapping(lines, [], data, emit_header=False)
    text = "\n".join(lines).rstrip() + "\n"
    tomllib.loads(text)
    return text


def latest_stellar_tree_path(directory: Path) -> Path:
    highest_version = -1
    highest_path: Path | None = None

    for path in directory.glob("stellar_tree_version*.toml"):
        match = STELLAR_TREE_PATTERN.fullmatch(path.name)
        if not match:
            continue
        version = int(match.group(1))
        if version > highest_version:
            highest_version = version
            highest_path = path

    if highest_path is None:
        raise FileNotFoundError(
            f"No stellar_tree_versionN.toml files found in {directory}."
        )
    return highest_path


def next_catalog_output_path(directory: Path) -> Path:
    highest_version = 0
    for path in directory.glob("demon_catalog_v*.toml"):
        match = DEMON_CATALOG_PATTERN.fullmatch(path.name)
        if match:
            highest_version = max(highest_version, int(match.group(1)))
    return directory / f"demon_catalog_v{highest_version + 1}.toml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build demon_catalog_vN.toml from stellar and demon TOML inputs."
    )
    parser.add_argument(
        "--stellar-tree",
        type=Path,
        default=None,
        help=(
            "Path to stellar_tree_versionN.toml. Default: latest version in the "
            "current working directory."
        ),
    )
    parser.add_argument(
        "--demonlist",
        type=Path,
        default=Path("demonlist.toml"),
        help="Path to demonlist.toml (default: ./demonlist.toml).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output TOML path. Default: next demon_catalog_vN.toml in the "
            "current working directory."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print validation and merge summaries before writing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    stellar_tree_path = (
        args.stellar_tree.resolve()
        if args.stellar_tree
        else latest_stellar_tree_path(Path.cwd()).resolve()
    )
    demonlist_path = args.demonlist.resolve()
    output_path = (
        args.output.resolve()
        if args.output
        else next_catalog_output_path(Path.cwd()).resolve()
    )

    ctx = ValidationContext()
    stellar_data = load_toml(stellar_tree_path, "stellar tree")
    demon_data = load_toml(demonlist_path, "demon list")

    origin, nodes, leaves = parse_stellar_tree(stellar_data, stellar_tree_path, ctx)
    demons_by_hip = parse_demonlist(demon_data, demonlist_path, ctx)
    validate_leaf_hips(leaves, demons_by_hip, ctx)
    ctx.raise_if_any()

    catalog = build_catalog(origin, nodes, leaves, demons_by_hip)
    toml_text = dumps_toml(catalog)

    if args.verbose:
        print(f"Stellar tree: {stellar_tree_path}")
        print(f"Demon list:   {demonlist_path}")
        print(f"Nodes:        {len(nodes)}")
        print(f"Leaves:       {len(leaves)}")
        print(f"Demon rows:   {len(demons_by_hip)}")
        print(f"HIP matches:  {len(leaves)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(toml_text, encoding="utf-8")
    print(f"Output written to: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except (
        CatalogValidationError,
        FileNotFoundError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
