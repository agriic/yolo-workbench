"""Rebalance: reassign images to train/val/test by rules, preview the result, and apply it as one undoable move.

Rules (`parse_rules` fills in the defaults; the last ones applied are kept in `.yolo-workbench/rebalance.json`):

- targets: a ratio per split, counted in images or objects;
- what stays put: frozen splits (test by default: its metrics stay comparable across runs; nothing leaves or
  joins a frozen split), frozen split folders, and images matching a gallery filter;
- keep together: images sharing a key never end up in different splits. Keys: the subfolder below the split
  folder (at a depth), a filename pattern (a regex whose first group is the key), the batch (the folder an image
  was added in, splitstats.batch_key), near-duplicates and capture bursts (imagekeys.py);
- balance (soft goals, each weighted off / low / normal / high): images containing each class (multi-label
  stratification), objects per class, background images, each batch spread over the splits in proportion, object
  sizes, and a subfolder or pattern key spread over the splits;
- hard limits: a minimum number of images per class in val and test, and which images may go to val/test (a
  gallery filter; the others stay in or go to train);
- stability: "minimal" starts from the current splits and moves an image only when that improves the balance by
  more than a move costs; "reassign" places every movable image anew. A seed decides ties.

The engine (`plan`) works on units (union-find over the keep-together keys) with sparse feature counts and
minimizes a chi-square-like error against the targets by greedy placement and local search (single moves, then
seeded swaps of similar units), with an iteration budget, so the same records, rules and seed give the same plan.
A moved image goes to its batch's folder in the target split (splitstats.twin_label), created when missing; the
whole move is one transaction (Dataset.relocate_images), taken after an automatic snapshot (`apply`).
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path

from . import snapshots
from .dataset import WORKBENCH_DIR, Dataset, DatasetError, WriteConflict
from .jobs import Job
from .splitstats import SPLITS, batch_key, below, relative_path, split_table, twin_label

WEIGHTS = {"off": 0.0, "low": 0.3, "normal": 1.0, "high": 3.0}
RULES_FILE = "rebalance.json"
MAX_PASSES = 20
SWAP_SAMPLES = 6  # units of another split tried per unit in a swap pass
SWAP_UNITS = 20000  # units given swap attempts per round (sampled)
SWAP_ROUNDS = 3  # swap rounds, each once single moves stop helping
MOVE_COST = 0.01  # error a move must remove ("minimal" mode), per image moved
MINIMUM_WEIGHT = 50.0
SIZE_EDGES = (32 * 32, 96 * 96)  # COCO's small / medium / large object areas, px²
EPSILON = 1e-9

DEFAULT_BALANCE = {"classes": "normal", "objects": "low", "background": "normal", "batches": "normal", "sizes": "off", "spread": "off"}


# ---------- rules ----------


def default_rules(dataset: Dataset) -> dict:
    with dataset._lock:
        counts = {split: len(ids) for split, ids in dataset._split_image_ids.items()}
    return {
        "targets": {"train": 0.8, "val": 0.1, "test": 0.1} if counts.get("test") or "test" in dataset._split_folders else {"train": 0.85, "val": 0.15, "test": 0.0},
        "count": "images",
        "frozen_splits": ["test"] if counts.get("test") else [],
        "frozen_folders": [],
        "pinned": None,  # a gallery filter: matching images stay where they are
        "eligible": None,  # a gallery filter: only matching images may be in val/test
        "together": {"depth": 0, "pattern": "", "batch": False, "near_duplicates": None, "bursts": None},
        "balance": dict(DEFAULT_BALANCE),
        "spread": {"depth": 0, "pattern": ""},
        "minimum": 0,
        "mode": "minimal",
        "max_moves": 0,
        "seed": 0,
        "folders": {},  # "<source folder>|<split>" -> target folder label (else the batch's twin)
    }


FILTER_KEYS = {"split", "class_id", "search", "min_labels", "max_labels", "review", "issue", "reviewer", "folder"}


def _filter(raw) -> dict | None:
    if not raw:
        return None
    if not isinstance(raw, dict) or set(raw) - FILTER_KEYS:
        raise DatasetError(f"A rule filter takes {', '.join(sorted(FILTER_KEYS))}")
    return {key: value for key, value in raw.items() if value not in (None, "")} or None


def _pattern(text: str, what: str) -> re.Pattern | None:
    if not text:
        return None
    try:
        return re.compile(text)
    except re.error as exc:
        raise DatasetError(f"The {what} pattern is not a valid regular expression: {exc}") from exc


def parse_rules(dataset: Dataset, raw: dict | None) -> dict:
    """The rules with defaults filled in, checked (DatasetError says what is wrong)."""
    rules = default_rules(dataset)
    raw = raw or {}
    unknown = set(raw) - set(rules)
    if unknown:
        raise DatasetError(f"Unknown rebalance rule: {', '.join(sorted(unknown))}")
    for key in ("together", "balance", "spread"):
        if key in raw:
            if not isinstance(raw[key], dict) or set(raw[key]) - set(rules[key]):
                raise DatasetError(f"Unknown {key} rule: {', '.join(sorted(set(raw[key]) - set(rules[key]))) if isinstance(raw[key], dict) else raw[key]}")
            rules[key] = {**rules[key], **raw[key]}
    for key in ("targets", "count", "frozen_splits", "frozen_folders", "minimum", "mode", "max_moves", "seed", "folders"):
        if key in raw:
            rules[key] = raw[key]
    rules["pinned"] = _filter(raw.get("pinned"))
    rules["eligible"] = _filter(raw.get("eligible"))
    try:
        rules["targets"] = {split: max(0.0, float(rules["targets"].get(split) or 0)) for split in SPLITS}
        rules["minimum"] = max(0, int(rules["minimum"] or 0))
        rules["max_moves"] = max(0, int(rules["max_moves"] or 0))
        rules["seed"] = int(rules["seed"] or 0)
        rules["together"]["depth"] = max(0, int(rules["together"]["depth"] or 0))
        rules["spread"]["depth"] = max(0, int(rules["spread"]["depth"] or 0))
    except (TypeError, ValueError, AttributeError) as exc:
        raise DatasetError(f"Rebalance rules: {exc}") from exc
    if rules["count"] not in ("images", "objects"):
        raise DatasetError("Count must be images or objects")
    if rules["mode"] not in ("minimal", "reassign"):
        raise DatasetError("Mode must be minimal or reassign")
    if not isinstance(rules["frozen_splits"], list) or set(rules["frozen_splits"]) - set(SPLITS):
        raise DatasetError("Frozen splits must be train, val or test")
    if not isinstance(rules["frozen_folders"], list) or not isinstance(rules["folders"], dict):
        raise DatasetError("Frozen folders must be a list and folders a mapping")
    for key, value in rules["balance"].items():
        if value not in WEIGHTS:
            raise DatasetError(f"Balance {key} must be one of {', '.join(WEIGHTS)}")
    for key in ("near_duplicates", "bursts"):
        value = rules["together"][key]
        if value is not None and (not isinstance(value, (int, float)) or value < 0):
            raise DatasetError(f"Keep together: {key.replace('_', ' ')} takes a number or null")
    movable = [split for split in SPLITS if split not in rules["frozen_splits"]]
    if not any(rules["targets"][split] > 0 for split in movable):
        raise DatasetError("Give at least one split that is not frozen a share")
    _pattern(rules["together"]["pattern"], "keep-together")
    _pattern(rules["spread"]["pattern"], "spread")
    return rules


def saved_rules(dataset: Dataset) -> dict | None:
    try:
        return json.loads((dataset.root / WORKBENCH_DIR / RULES_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def save_rules(dataset: Dataset, rules: dict) -> None:
    path = dataset.root / WORKBENCH_DIR / RULES_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset._atomic_write(path, json.dumps(rules, indent=1))


# ---------- the dataset as the engine sees it ----------


class _Records:
    """What the engine needs of every image, read under the dataset lock."""

    def __init__(self, dataset: Dataset, rules: dict, imagekeys: dict | None):
        root = dataset.root
        classification = dataset.category == "classification"
        self.ids, self.splits, self.folders, self.paths, self.classes, self.areas = [], [], [], [], [], []
        self.below: list[str] = []  # the path below the split folder (POSIX), or the file name
        self.stems: list[str] = []
        self.keys: list[dict | None] = []  # near-duplicate hash / capture time and camera (imagekeys.py)
        self.stamps: list = []
        for record in sorted(dataset.images.values(), key=lambda item: str(item.path)):
            folder = dataset._folder_of(record)
            label = dataset.folder_label(folder) if folder is not None else None
            self.ids.append(record.id)
            self.splits.append(record.split)
            self.folders.append(label)
            self.paths.append(relative_path(root, record.path))
            inside = below(folder, record.path) if folder is not None else record.path.name
            if classification and "/" in inside:
                inside = inside.split("/", 1)[1]  # below the class folder
            self.below.append(inside)
            self.stems.append(record.path.stem)
            self.classes.append([annotation.class_id for annotation in record.annotations])
            if rules["balance"]["sizes"] != "off" and record.width and not classification:
                self.areas.append([dataset._annotation_size(annotation)[0] * record.width * dataset._annotation_size(annotation)[1] * record.height for annotation in record.annotations])
            else:
                self.areas.append(None)
            self.keys.append((imagekeys or {}).get(str(record.path)))
            self.stamps.append((record.image_mtime, record.image_size))
        self.index = {image_id: index for index, image_id in enumerate(self.ids)}

    def __len__(self) -> int:
        return len(self.ids)


def _filtered(dataset: Dataset, query: dict | None) -> set[str] | None:
    """The image ids a gallery filter matches. Caller holds the lock."""
    if not query:
        return None
    records = dataset._filter_images(query.get("split", "all"), query.get("class_id"), query.get("search", ""), None, query.get("min_labels"), query.get("max_labels"), query.get("review"), query.get("issue"), query.get("reviewer"), query.get("folder"))
    return {record.id for record in records}


def _component_key(below: str, depth: int) -> str | None:
    parts = below.split("/")[:-1]
    return "/".join(parts[:depth]) if depth and parts else None


def _pattern_key(pattern: re.Pattern | None, stem: str) -> str | None:
    if pattern is None:
        return None
    match = pattern.search(stem)
    if match is None:
        return None
    return match.group(1) if match.groups() and match.group(1) is not None else match.group(0)


class _UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != root:
            self.parent[item], item = root, self.parent[item]
        return root

    def union(self, a: int, b: int) -> None:
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def _near_duplicates(records: _Records, distance: int, union: _UnionFind) -> int:
    """Join images whose 64-bit dHash differ in at most `distance` bits (pigeonhole: they share one of
    distance + 1 chunks exactly). Returns how many images have no hash yet."""
    hashes = [(index, int(keys["dhash"], 16)) for index, keys in enumerate(records.keys) if keys and keys.get("dhash")]
    missing = len(records) - len(hashes)
    distance = min(int(distance), 16)
    chunks = distance + 1
    width = 64 // chunks
    for chunk in range(chunks):
        shift = chunk * width
        mask = (1 << (width if chunk < chunks - 1 else 64 - shift)) - 1
        buckets: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for index, value in hashes:
            buckets[(value >> shift) & mask].append((index, value))
        for bucket in buckets.values():
            if len(bucket) < 2:
                continue
            if len(bucket) > 400:  # a crowd of one look (blank frames): compare neighbours by value only
                bucket.sort(key=lambda item: item[1])
                pairs = zip(bucket, bucket[1:])
            else:
                pairs = ((a, b) for i, a in enumerate(bucket) for b in bucket[i + 1 :])
            for (a, first), (b, second) in pairs:
                if (first ^ second).bit_count() <= distance:
                    union.union(a, b)
    return missing


def _bursts(records: _Records, seconds: float, union: _UnionFind) -> None:
    """Join images from the same camera taken within `seconds` of the one before."""
    by_camera: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for index, keys in enumerate(records.keys):
        if keys and keys.get("time") is not None:
            by_camera[keys.get("camera") or ""].append((keys["time"], index))
    for shots in by_camera.values():
        shots.sort()
        for (before, a), (after, b) in zip(shots, shots[1:]):
            if after - before <= seconds:
                union.union(a, b)


# ---------- the engine ----------


class _Engine:
    def __init__(self, splits: list[str], ratios: dict[str, float], mode: str, seed: int, max_moves: int):
        self.splits = splits  # the splits images move between (not frozen)
        self.k = len(splits)
        total = sum(ratios[split] for split in splits)
        self.share = [ratios[split] / total for split in splits]
        self.move_cost = MOVE_COST if mode == "minimal" else 0.0
        self.rng = random.Random(seed)
        self.max_moves = max_moves
        self.features: dict[str, int] = {}
        self.weights: list[float] = []
        self.size_feature = self.feature("size", 1.0)  # images (or objects) per split: the ratio targets

    def feature(self, name: str, weight: float) -> int:
        index = self.features.get(name)
        if index is None:
            index = self.features[name] = len(self.weights)
            self.weights.append(weight)
        return index

    def setup(self, units: list[dict], minimum: int, minimum_splits: set[int], class_features: dict[int, int]) -> None:
        count = len(self.weights)
        totals = [0.0] * count
        for unit in units:
            for index, value in unit["feats"]:
                totals[index] += value
        self.totals = totals
        self.target = [[totals[index] * share for index in range(count)] for share in self.share]
        self.scale = [[1.0 / max(value, 1.0) for value in row] for row in self.target]
        self.actual = [[0.0] * count for _ in range(self.k)]
        # minimums: images of a class in val/test, at most an even share of what the class has
        self.minimum = [[0.0] * count for _ in range(self.k)]
        self.unmet: list[tuple[int, int, int]] = []  # (class id, asked, possible)
        for class_id, index in class_features.items():
            possible = int(totals[index] // max(1, self.k))
            for split in minimum_splits:
                self.minimum[split][index] = min(minimum, possible)
            if minimum and possible < minimum and minimum_splits:
                self.unmet.append((class_id, minimum, possible))
        self.minimum_features = sorted(class_features.values()) if minimum and minimum_splits else []

    def _penalty(self, split: int, index: int, value: float) -> float:
        need = self.minimum[split][index]
        return MINIMUM_WEIGHT * (need - value) ** 2 if value < need else 0.0

    def delta(self, unit: dict, a: int | None, b: int) -> float:
        """The error's change when `unit` goes from split `a` (None: unplaced) to `b`."""
        change = 0.0
        weights, actual_b, target_b, scale_b = self.weights, self.actual[b], self.target[b], self.scale[b]
        if a is None:
            for index, value in unit["feats"]:
                change += weights[index] * value * (value + 2 * (actual_b[index] - target_b[index])) * scale_b[index]
        else:
            actual_a, target_a, scale_a = self.actual[a], self.target[a], self.scale[a]
            for index, value in unit["feats"]:
                change += weights[index] * value * ((value - 2 * (actual_a[index] - target_a[index])) * scale_a[index] + (value + 2 * (actual_b[index] - target_b[index])) * scale_b[index])
        if self.minimum_features:
            for index, value in unit["min_feats"]:
                if a is not None:
                    change += self._penalty(a, index, self.actual[a][index] - value) - self._penalty(a, index, self.actual[a][index])
                change += self._penalty(b, index, self.actual[b][index] + value) - self._penalty(b, index, self.actual[b][index])
        if self.move_cost:
            change += self.move_cost * ((unit["orig"].get(a, 0) if a is not None else unit["size"]) - unit["orig"].get(b, 0))
        return change

    def place(self, unit: dict, b: int) -> None:
        a = unit["at"]
        for index, value in unit["feats"]:
            if a is not None:
                self.actual[a][index] -= value
            self.actual[b][index] += value
        unit["at"] = b

    def moves(self, units: list[dict]) -> int:
        return sum(unit["size"] - unit["orig"].get(unit["at"], 0) for unit in units if unit["at"] is not None)

    def run(self, fixed: list[dict], movable: list[dict], mode: str) -> None:
        for unit in fixed:  # counted where they are
            split, unit["at"] = unit["at"], None
            self.place(unit, split)
        start = [unit for unit in movable if unit["start"] is not None and mode == "minimal"]
        for unit in start:
            unit["at"] = None
            self.place(unit, unit["start"])
        unplaced = [unit for unit in movable if unit["at"] is None]
        # rarest feature first, bigger units first: what is scarce gets spread before the rest fills in
        rarity = lambda unit: min((self.totals[index] for index, _ in unit["feats"] if index != self.size_feature), default=float("inf"))  # noqa: E731
        order = list(unplaced)
        self.rng.shuffle(order)
        order.sort(key=lambda unit: (rarity(unit), -unit["size"]))
        for unit in order:
            best = min(unit["eligible"], key=lambda split: (self.delta(unit, None, split), split))
            self.place(unit, best)
        moved = self.moves(movable)
        swap_rounds = 0
        for _ in range(MAX_PASSES):
            changed = 0
            order = list(movable)
            self.rng.shuffle(order)
            for unit in order:
                a = unit["at"]
                best, best_change = None, -EPSILON
                for b in unit["eligible"]:
                    if b == a:
                        continue
                    if self.max_moves and moved + unit["orig"].get(a, 0) - unit["orig"].get(b, 0) > self.max_moves:
                        continue
                    change = self.delta(unit, a, b)
                    if change < best_change:
                        best, best_change = b, change
                if best is not None:
                    moved += unit["orig"].get(a, 0) - unit["orig"].get(best, 0)
                    self.place(unit, best)
                    changed += 1
            if changed:
                continue
            # at a local optimum of single moves: swaps of similar units may get further (a few rounds)
            if swap_rounds >= SWAP_ROUNDS or not self._swaps(movable, moved):
                break
            swap_rounds += 1
            moved = self.moves(movable)

    def _swaps(self, movable: list[dict], moved: int) -> int:
        """Swap similar units between splits where that helps: class balance at the same split sizes."""
        by_split: dict[int, list[dict]] = defaultdict(list)
        for unit in movable:
            by_split[unit["at"]].append(unit)
        swapped = 0
        order = list(movable)
        self.rng.shuffle(order)
        for unit in order[:SWAP_UNITS]:
            a = unit["at"]
            for b in unit["eligible"]:
                if b == a or not by_split[b]:
                    continue
                for other in self.rng.sample(by_split[b], min(SWAP_SAMPLES, len(by_split[b]))):
                    if other["at"] != b or a not in other["eligible"] or abs(other["size"] - unit["size"]) > max(1, unit["size"] // 4):
                        continue
                    step = unit["orig"].get(a, 0) - unit["orig"].get(b, 0) + other["orig"].get(b, 0) - other["orig"].get(a, 0)
                    if self.max_moves and moved + step > self.max_moves:
                        continue
                    first = self.delta(unit, a, b)
                    self.place(unit, b)
                    second = self.delta(other, b, a)
                    if first + second < -EPSILON:
                        self.place(other, a)
                        by_split[a].append(other)
                        by_split[b].append(unit)
                        moved += step
                        swapped += 1
                        break
                    self.place(unit, a)
                if unit["at"] != a:
                    break
        return swapped


def plan(dataset: Dataset, raw_rules: dict | None, imagekeys: dict | None = None, extra_pins: set[str] | None = None) -> dict:
    """What a rebalance by these rules would do: the moves, split tables before/after/target, leakage, warnings
    and a digest naming the plan (apply refuses another one). `extra_pins`: image ids that stay put too."""
    rules = parse_rules(dataset, raw_rules)
    if imagekeys is None and (rules["together"]["near_duplicates"] is not None or rules["together"]["bursts"] is not None):
        from . import imagekeys as keys_module

        imagekeys = keys_module.load(dataset)
    with dataset._lock:
        if not dataset.capabilities()["image_ops"]:
            raise DatasetError("This dataset's splits are file lists, not plain folders, so images cannot be moved between them")
        records = _Records(dataset, rules, imagekeys)
        pinned_ids = (_filtered(dataset, rules["pinned"]) or set()) | (extra_pins or set())
        eligible_ids = _filtered(dataset, rules["eligible"])
        split_folders = {split: [dataset.folder_label(folder) for folder in folders] for split, folders in dataset._split_folders.items()}
        fingerprint = snapshots.fingerprint(dataset)
        names = dict(dataset.names)
        category = dataset.category
        yaml_editable = dataset.yaml_path.is_file()
    frozen = [split for split in SPLITS if split in rules["frozen_splits"]]
    movable_splits = [split for split in SPLITS if split not in frozen and rules["targets"][split] > 0]
    # a split that is neither frozen nor given a share is emptied into the others
    emptied = [split for split in SPLITS if split not in frozen and rules["targets"][split] <= 0]
    engine = _Engine(movable_splits, rules["targets"], rules["mode"], rules["seed"], rules["max_moves"])
    split_index = {split: index for index, split in enumerate(movable_splits)}
    weights = {key: WEIGHTS[value] for key, value in rules["balance"].items()}
    warnings: list[str] = []

    # keep-together keys → units
    union = _UnionFind(len(records))
    together = rules["together"]
    together_pattern = _pattern(together["pattern"], "keep-together")
    spread_pattern = _pattern(rules["spread"]["pattern"], "spread")
    first_of: dict[str, int] = {}
    keyed = 0
    for index in range(len(records)):
        keys = []
        folder = records.folders[index]
        if together["depth"]:
            key = _component_key(records.below[index], together["depth"])
            if key is not None:
                keys.append(f"sub|{batch_key(folder) if folder else ''}|{key}")
        key = _pattern_key(together_pattern, records.stems[index])
        if key is not None:
            keys.append(f"pat|{key}")
        if together["batch"] and folder:
            keys.append(f"batch|{batch_key(folder)}")
        keyed += bool(keys)
        for key in keys:
            if key in first_of:
                union.union(first_of[key], index)
            else:
                first_of[key] = index
    if together["near_duplicates"] is not None:
        missing = _near_duplicates(records, together["near_duplicates"], union)
        if missing:
            warnings.append(f"{missing} image(s) have no near-duplicate fingerprint yet; scan the images first (Keep together → Scan)")
    if together["bursts"] is not None:
        _bursts(records, float(together["bursts"]), union)
    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(records)):
        groups[union.find(index)].append(index)

    leakage_before = sum(1 for members in groups.values() if len({records.splits[index] for index in members}) > 1)

    # features
    size_feature = engine.size_feature
    batches = bool(weights["batches"]) and any(len(listed) > 1 for listed in split_folders.values())
    class_features: dict[int, int] = {}
    units: list[dict] = []
    fixed: list[dict] = []
    frozen_folders = set(rules["frozen_folders"])
    pinned_conflicts = 0
    frozen_leaks = 0
    ineligible = 0
    for members in groups.values():
        in_frozen = [index for index in members if records.splits[index] in frozen]
        rest = [index for index in members if records.splits[index] not in frozen]
        if in_frozen and rest:
            frozen_leaks += 1  # a group with images in a frozen split: those stay; the others are a unit of their own
        if not rest:
            continue
        feats: dict[int, float] = defaultdict(float)
        min_feats: dict[int, float] = defaultdict(float)
        for index in rest:
            classes = records.classes[index]
            feats[size_feature] += len(classes) if rules["count"] == "objects" else 1
            if weights["classes"] or rules["minimum"]:
                for class_id in set(classes):
                    feature = class_features.setdefault(class_id, engine.feature(f"class|{class_id}", weights["classes"]))
                    feats[feature] += 1
                    min_feats[feature] += 1
            if weights["objects"]:
                for class_id in classes:
                    feats[engine.feature(f"objects|{class_id}", weights["objects"] * 0.5)] += 1
            if weights["background"] and not classes:
                feats[engine.feature("background", weights["background"])] += 1
            folder = records.folders[index]
            if batches and folder:
                feats[engine.feature(f"batch|{batch_key(folder)}", weights["batches"])] += 1
            if weights["sizes"] and records.areas[index]:
                for area in records.areas[index]:
                    bucket = "small" if area < SIZE_EDGES[0] else "medium" if area < SIZE_EDGES[1] else "large"
                    feats[engine.feature(f"size|{bucket}", weights["sizes"])] += 1
            if weights["spread"]:
                key = _component_key(records.below[index], rules["spread"]["depth"]) if rules["spread"]["depth"] else _pattern_key(spread_pattern, records.stems[index])
                if key is not None:
                    feats[engine.feature(f"spread|{key}", weights["spread"])] += 1
        orig: dict[int | None, int] = defaultdict(int)
        for index in rest:
            orig[split_index.get(records.splits[index])] += 1
        pins = {records.splits[index] for index in rest if records.ids[index] in pinned_ids or records.folders[index] in frozen_folders}
        eligible = set(range(engine.k))
        bad = [index for index in rest if eligible_ids is not None and records.ids[index] not in eligible_ids]
        if bad:
            ineligible += len(bad)
            eligible = {split_index[split] for split in movable_splits if split not in ("val", "test")}
        unit = {"members": rest, "feats": list(feats.items()), "min_feats": list(min_feats.items()), "size": len(rest), "orig": dict(orig), "at": None, "start": None}
        if len(pins) > 1:
            pinned_conflicts += 1
            unit["fixed"] = True
        elif pins:
            split = next(iter(pins))
            if split in split_index:
                unit["eligible"] = [split_index[split]]
            else:
                unit["fixed"] = True  # pinned in a split that gets no share: stays there
        if unit.get("fixed"):
            # counted where its images are; a unit split over several splits counts in the first
            unit["at"] = split_index.get(records.splits[rest[0]])
            if unit["at"] is None:
                continue
            fixed.append(unit)
            units.append(unit)
            continue
        if "eligible" not in unit:
            unit["eligible"] = sorted(eligible)
        if not unit["eligible"]:
            unit["fixed"] = True
            unit["at"] = split_index.get(records.splits[rest[0]])
            warnings.append("Some images may not go to val/test and train is frozen or gets no share: they stay where they are")
            if unit["at"] is not None:
                fixed.append(unit)
                units.append(unit)
            continue
        # where it starts in "minimal" mode: the allowed split holding most of its images
        present = sorted(((count, -split) for split, count in orig.items() if split is not None and split in unit["eligible"]), reverse=True)
        unit["start"] = -present[0][1] if present else None
        units.append(unit)
    movable = [unit for unit in units if not unit.get("fixed")]
    minimum_splits = {split_index[split] for split in ("val", "test") if split in split_index}
    engine.setup(units, rules["minimum"], minimum_splits, class_features)
    engine.run(fixed, movable, rules["mode"])

    # the result
    final = list(records.splits)
    for unit in movable:
        for index in unit["members"]:
            final[index] = movable_splits[unit["at"]]
    final_groups = sum(1 for members in groups.values() if len({final[index] for index in members}) > 1)
    moves, folder_moves = _targets(records, final, split_folders, rules["folders"], warnings)
    create = sorted({(item["split"], item["to"]) for item in folder_moves if item["create"]})
    if create and not yaml_editable and category != "classification":
        raise DatasetError("The rebalance needs new split folders, and this dataset has no YAML file to list them in")
    for class_id, asked, possible in engine.unmet:
        warnings.append(f"{names.get(class_id, f'class {class_id}')} has too few images for {asked} in each of val/test: at most {possible} each")
    if pinned_conflicts:
        warnings.append(f"{pinned_conflicts} group(s) have pinned images in different splits and stay as they are")
    if frozen_leaks:
        warnings.append(f"{frozen_leaks} group(s) also have images in a frozen split; those stay there, so the group stays split")
    if ineligible:
        warnings.append(f"{ineligible} image(s) may not be in val/test by the rules; they go to (or stay in) train")
    if emptied and any(records.splits[index] in emptied for index in range(len(records))):
        warnings.append(f"{', '.join(emptied)} get no share: their images move to the other splits")
    target = {}
    for split, position in split_index.items():
        row = {"images" if rules["count"] == "images" else "objects": round(engine.target[position][size_feature], 1), "classes": {}}
        for class_id, feature in class_features.items():
            row["classes"][str(class_id)] = {"images": round(engine.target[position][feature], 1)}
        target[split] = row
    digest = hashlib.sha1(json.dumps([fingerprint, rules, [(move["id"], move["to"], move["to_folder"]) for move in moves]], sort_keys=True).encode()).hexdigest()[:16]
    return {
        "digest": digest,
        "rules": rules,
        "moves": moves,
        "create": [{"split": split, "folder": folder} for split, folder in create],
        "folders": folder_moves,
        "counts": {"images": len(records), "moves": len(moves), "units": len(groups), "groups": sum(1 for members in groups.values() if len(members) > 1), "grouped_images": keyed, "pinned": len(pinned_ids), "frozen": sum(1 for split in records.splits if split in frozen)},
        "table": {
            "before": split_table(zip(records.splits, records.classes)),
            "after": split_table(zip(final, records.classes)),
            "target": target,
        },
        "count": rules["count"],
        "leakage": {"before": leakage_before, "after": final_groups},
        "warnings": warnings,
        "splits": movable_splits,
        "frozen": frozen,
        "names": [names[class_id] for class_id in sorted(names)],
    }


def _targets(records: _Records, final: list[str], split_folders: dict[str, list[str]], overrides: dict, warnings: list[str]) -> tuple[list[dict], list[dict]]:
    """Each moved image's target folder: its batch's folder in the target split (made when missing, named like its
    source folder), or the one the rules name."""
    keys = {split: {batch_key(label): label for label in reversed(labels)} for split, labels in split_folders.items()}
    every = [label for labels in split_folders.values() for label in labels]
    overlaps = lambda label: any(label == other or label.startswith(f"{other}/") or other.startswith(f"{label}/") for other in every)  # noqa: E731
    moves, summary = [], {}
    for index, split in enumerate(final):
        if split == records.splits[index]:
            continue
        source = records.folders[index]
        override = overrides.get(f"{source}|{split}")
        if override:
            target, create = override, override not in split_folders.get(split, [])
        else:
            target = keys.get(split, {}).get(batch_key(source)) if source else None
            create = False
            if target is None:
                twin = twin_label(source, split) if source and not Path(source).is_absolute() else None
                if twin and not overlaps(twin):
                    target, create = twin, True
                elif split_folders.get(split):
                    target = split_folders[split][0]
                else:
                    warnings.append(f"An image of {source} cannot go to {split}: it has no folder there")
                    continue
        moves.append({"id": records.ids[index], "path": records.paths[index], "from": records.splits[index], "to": split, "from_folder": source, "to_folder": target})
        entry = summary.setdefault((source, split), {"from": source, "split": split, "to": target, "create": create, "images": 0})
        entry["images"] += 1
    return moves, sorted(summary.values(), key=lambda item: (item["from"] or "", item["split"]))


# ---------- applying ----------


def _relocate(dataset: Dataset, result: dict) -> dict:
    with dataset._lock:
        create = [(item["split"], dataset.folder_path(item["folder"])) for item in result["create"]]
        known = {dataset.folder_label(folder): folder for folder in dataset._all_folders()}
        created = {dataset.folder_label(folder): folder for _, folder in create}
        moves = []
        for move in result["moves"]:
            folder = known.get(move["to_folder"]) or created.get(move["to_folder"]) or dataset.folder_path(move["to_folder"])
            moves.append((move["id"], move["to"], folder))
        return dataset.relocate_images(moves, create)


def apply(dataset: Dataset, raw_rules: dict | None, digest: str, job: Job | None = None, label: str = "rebalance splits") -> dict:
    """Apply the plan the preview showed (`digest`): a snapshot of the dataset first, then every move as one
    undoable transaction. The dataset changed since the preview: WriteConflict (preview again)."""
    from . import identity

    dataset.check_writable(structure=True)
    result = plan(dataset, raw_rules)
    if result["digest"] != digest:
        raise WriteConflict("The dataset changed since the preview; preview the rebalance again")
    if not result["moves"]:
        return {"moved": 0, "snapshot": None, "created": []}
    snapshot = snapshots.take(dataset, time.strftime("Before rebalance %Y-%m-%d %H:%M"), kind="auto", job=job, extra={"before_rebalance": True})
    if job:
        job.progress(0, len(result["moves"]), "Moving the images…")
    # image moves waited for the snapshot (its structure hold): the planned images are where the plan saw them
    with identity.action(label):
        moved = _relocate(dataset, result)
    save_rules(dataset, result["rules"])
    return {"moved": moved["moved"], "created": moved["created"], "snapshot": {"id": snapshot["id"], "name": snapshot["name"]}, "can_undo": moved["can_undo"]}


def proposed_snapshot(dataset: Dataset, raw_rules: dict | None, digest: str, job: Job | None = None) -> dict:
    """A snapshot laid out by the plan (a proposed split to train on); nothing in the dataset moves."""
    result = plan(dataset, raw_rules)
    if result["digest"] != digest:
        raise WriteConflict("The dataset changed since the preview; preview the rebalance again")
    assignment = {move["id"]: move["to"] for move in result["moves"]}
    meta = snapshots.take(dataset, time.strftime("Proposed split %Y-%m-%d %H:%M"), kind="auto", job=job, assignment=assignment, extra={"rebalance": {"rules": result["rules"], "digest": digest, "moves": len(assignment)}})
    save_rules(dataset, result["rules"])
    return meta


def place_new(dataset: Dataset, raw_rules: dict | None, new_ids: set[str]) -> dict:
    """Place images just imported by the rules: every other image stays where it is (the import's split mode
    "by the rebalance rules"); one undoable move."""
    from . import identity

    rules = {**parse_rules(dataset, raw_rules), "mode": "minimal"}
    with dataset._lock:
        others = {record.id for record in dataset.images.values() if record.id not in new_ids}
    result = plan(dataset, rules, extra_pins=others)
    if not result["moves"]:
        return {"moved": 0, "created": []}
    with identity.action("place imported images"):
        moved = _relocate(dataset, result)
    return {"moved": moved["moved"], "created": moved["created"]}


def leakage(dataset: Dataset, raw_rules: dict | None) -> dict:
    """Groups (by the keep-together rules) whose images are in more than one split, and their images."""
    rules = parse_rules(dataset, raw_rules)
    with dataset._lock:
        records = _Records(dataset, rules, None)
    union = _UnionFind(len(records))
    together = rules["together"]
    pattern = _pattern(together["pattern"], "keep-together")
    first_of: dict[str, int] = {}
    for index in range(len(records)):
        keys = []
        if together["depth"]:
            key = _component_key(records.below[index], together["depth"])
            if key is not None:
                keys.append(f"sub|{batch_key(records.folders[index]) if records.folders[index] else ''}|{key}")
        key = _pattern_key(pattern, records.stems[index])
        if key is not None:
            keys.append(f"pat|{key}")
        for key in keys:
            if key in first_of:
                union.union(first_of[key], index)
            else:
                first_of[key] = index
    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(records)):
        groups[union.find(index)].append(index)
    split_groups = [members for members in groups.values() if len({records.splits[index] for index in members}) > 1]
    return {"groups": len(split_groups), "images": [records.ids[index] for members in split_groups for index in members], "configured": bool(together["depth"] or together["pattern"])}
