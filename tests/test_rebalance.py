import asyncio
import json
import random

import pytest
import yaml
from httpx import ASGITransport, AsyncClient
from PIL import Image

from fscount import counting
from test_training import ARGS, FakeLauncher
from yolo_workbench import imagekeys, rebalance, snapshots
from yolo_workbench.dataset import Dataset, DatasetError, DatasetLocked, WriteConflict
from yolo_workbench.jobs import JobManager
from yolo_workbench.media_cache import MediaCache
from yolo_workbench.splitstats import batch_key, twin_label
from yolo_workbench.training import TrainingManager
from yolo_workbench.web import create_app

FOLDERS = {"train": ["images/train_2024-01", "images/train_2024-02"], "val": ["images/val_2024-01"]}


def make_dataset(tmp_path, count=120, seed=1, folders=FOLDERS, clips=4, rare=0.05):
    """`count` images in clips of `clips` frames (clip<N>_<i>.jpg) spread over the folders, 10% background, class 2
    rare. Most images start in train."""
    rng = random.Random(seed)
    root = tmp_path / "ds"
    root.mkdir(parents=True)
    Image.new("RGB", (40, 30), "white").save(root / "base.jpg")
    data = (root / "base.jpg").read_bytes()
    (root / "base.jpg").unlink()
    for labels in folders.values():
        for label in labels:
            (root / label).mkdir(parents=True)
            (root / label.replace("images", "labels", 1)).mkdir(parents=True)
    every = [label for labels in folders.values() for label in labels]
    clip_folder = {}
    for index in range(count):
        clip = index // clips
        if clip not in clip_folder:
            clip_folder[clip] = rng.choice(folders["train"]) if rng.random() < 0.9 else rng.choice(folders.get("val") or folders["train"])
        folder = clip_folder[clip]
        if clips > 1 and clip % 7 == 0 and index % clips == clips - 1:  # every 7th clip's last frame elsewhere: a leak
            folder = every[(every.index(folder) + 1) % len(every)]
        name = f"clip{clip}_{index:04d}"
        (root / folder / f"{name}.jpg").write_bytes(data)
        draw = rng.random()
        line = "" if draw < 0.1 else f"{0 if draw < 0.7 else 1 if draw < 1 - rare else 2} 0.5 0.5 0.2 0.2\n"
        (root / folder.replace("images", "labels", 1) / f"{name}.txt").write_text(line)
    config = {split: labels if len(labels) > 1 else labels[0] for split, labels in folders.items()}
    config["names"] = {0: "car", 1: "bus", 2: "tram"}
    (root / "data.yaml").write_text(yaml.safe_dump(config))
    return Dataset(root / "data.yaml", "detection")


def splits_of(dataset):
    return {split: len(ids) for split, ids in dataset._split_image_ids.items() if ids}


RULES = {"targets": {"train": 0.7, "val": 0.15, "test": 0.15}, "frozen_splits": [], "together": {"pattern": r"^(clip\d+)_"}}


# ---------- names ----------


def test_batches_and_twin_folders():
    assert batch_key("images/train_2024-03") == batch_key("images/val_2024-03") == "images/*_2024-03"
    assert batch_key("2024-03/images/train") == batch_key("2024-03/images/valid") == "2024-03/images/*"
    assert batch_key("batch7") == batch_key("batch7_val") == "batch7_*"
    assert twin_label("images/train_2024-03", "val") == "images/val_2024-03"
    assert twin_label("2024-03/images/train", "test") == "2024-03/images/test"
    assert twin_label("data/Training-01", "val") == "data/Validation-01"
    assert twin_label("export/valid", "train") == "export/train"
    assert twin_label("batch7", "val") == "batch7_val"


# ---------- the engine ----------


def test_the_plan_meets_the_targets_keeps_clips_together_and_spreads_the_classes(tmp_path):
    dataset = make_dataset(tmp_path)
    result = rebalance.plan(dataset, RULES)
    after = result["table"]["after"]
    sizes = {split: after[split]["images"] for split in after}
    assert abs(sizes["train"] - 84) <= 4 and abs(sizes["val"] - 18) <= 4 and abs(sizes["test"] - 18) <= 4  # clips of 4
    assert result["leakage"]["after"] == 0 and result["counts"]["groups"] == 30
    clips = {}
    finals = {move["id"]: move["to"] for move in result["moves"]}
    for record in dataset.images.values():
        clips.setdefault(record.path.stem.split("_")[0], set()).add(finals.get(record.id, record.split))
    assert all(len(splits) == 1 for splits in clips.values())
    total = sum(row["classes"].get("1", {}).get("images", 0) for row in after.values())
    assert abs(after["val"]["classes"]["1"]["images"] - total * 0.15) <= 4
    assert rebalance.plan(dataset, RULES)["digest"] == result["digest"]  # deterministic
    assert rebalance.plan(dataset, {**RULES, "seed": 5})["counts"]["moves"] >= 0


def test_moving_as_few_images_as_possible_and_nothing_once_balanced(tmp_path):
    dataset = make_dataset(tmp_path)
    minimal = rebalance.plan(dataset, RULES)
    reassign = rebalance.plan(dataset, {**RULES, "mode": "reassign"})
    assert minimal["counts"]["moves"] <= reassign["counts"]["moves"]
    rebalance.apply(dataset, RULES, minimal["digest"])
    assert splits_of(dataset) == {split: row["images"] for split, row in minimal["table"]["after"].items()}
    assert rebalance.plan(dataset, RULES)["counts"]["moves"] == 0
    capped = rebalance.plan(make_dataset(tmp_path / "other"), {**RULES, "max_moves": 10})
    assert capped["counts"]["moves"] <= 10


def test_frozen_splits_pins_and_val_test_rules(tmp_path):
    dataset = make_dataset(tmp_path, folders={**FOLDERS, "test": ["images/test_2024-01"]})
    test_ids = set(dataset._split_image_ids.get("test", set()))
    # frozen test: nothing leaves or joins it
    result = rebalance.plan(dataset, {**RULES, "frozen_splits": ["test"], "targets": {"train": 0.8, "val": 0.2, "test": 0}})
    assert not [move for move in result["moves"] if move["from"] == "test" or move["to"] == "test"]
    assert result["frozen"] == ["test"] and result["counts"]["frozen"] == len(test_ids)
    # reviewed images stay put; only images without issues may be in val/test
    reviewed = [record.id for record in dataset.images.values() if record.split == "train"][:20]
    dataset.set_review(reviewed, "done")
    result = rebalance.plan(dataset, {**RULES, "together": {}, "pinned": {"review": "done"}, "eligible": {"review": "done"}})
    moved = {move["id"]: move for move in result["moves"]}
    assert not set(reviewed) & set(moved)
    assert all(move["to"] == "train" for move in moved.values())  # nothing else may go to val/test
    assert any("may not be in val/test" in warning for warning in result["warnings"])


def test_a_minimum_per_class_reaches_val_and_test(tmp_path):
    dataset = make_dataset(tmp_path, count=200, rare=0.06)
    rules = {**RULES, "together": {}, "minimum": 3}
    after = rebalance.plan(dataset, rules)["table"]["after"]
    assert after["val"]["classes"]["2"]["images"] >= 3 and after["test"]["classes"]["2"]["images"] >= 3
    too_many = rebalance.plan(dataset, {**rules, "minimum": 500})
    assert any("too few images" in warning for warning in too_many["warnings"])


def test_rules_are_checked(tmp_path):
    dataset = make_dataset(tmp_path, count=12)
    for bad in ({"targets": {"train": 0, "val": 0, "test": 0}}, {"mode": "random"}, {"balance": {"classes": "max"}}, {"together": {"pattern": "("}}, {"nope": 1}, {"frozen_splits": ["all"]}, {"eligible": {"colour": "red"}}):
        with pytest.raises(DatasetError):
            rebalance.plan(dataset, bad)


def test_near_duplicates_and_bursts_keep_together(tmp_path):
    dataset = make_dataset(tmp_path, count=40)
    records = sorted(dataset.images.values(), key=lambda record: str(record.path))
    keys = {str(record.path): {"dhash": f"{index * 0x0101010101010101 & 0xFFFFFFFFFFFFFFFF:016x}", "time": None, "camera": None} for index, record in enumerate(records)}
    twins = [records[0], records[-1]]
    keys[str(twins[1].path)]["dhash"] = f"{int(keys[str(twins[0].path)]['dhash'], 16) ^ 0b111:016x}"  # 3 bits apart
    keys[str(records[5].path)].update(time=1000.0, camera="X")
    keys[str(records[30].path)].update(time=1001.5, camera="X")
    rules = {**RULES, "together": {"near_duplicates": 3, "bursts": 2}, "mode": "reassign"}
    result = rebalance.plan(dataset, rules, imagekeys=keys)
    final = {record.id: record.split for record in records} | {move["id"]: move["to"] for move in result["moves"]}
    assert final[twins[0].id] == final[twins[1].id]
    assert final[records[5].id] == final[records[30].id]
    assert result["counts"]["groups"] == 2


def test_image_keys_fingerprint_look_alikes(tmp_path):
    root = tmp_path / "keys"
    (root / "images" / "train").mkdir(parents=True)
    (root / "labels" / "train").mkdir(parents=True)
    base = Image.effect_noise((16, 16), 80).resize((64, 64)).convert("RGB")
    base.save(root / "images" / "train" / "a.jpg", quality=95)
    base.save(root / "images" / "train" / "b.jpg", quality=60)  # the same picture, compressed harder
    Image.effect_noise((16, 16), 80).resize((64, 64)).convert("RGB").save(root / "images" / "train" / "c.jpg")
    (root / "data.yaml").write_text(yaml.safe_dump({"train": "images/train", "names": {0: "x"}}))
    dataset = Dataset(root / "data.yaml", "detection")
    assert imagekeys.status(dataset) == {"scanned": 0, "images": 3}
    assert imagekeys.scan(dataset)["read"] == 3
    keys = imagekeys.load(dataset)
    value = {path.rsplit("/", 1)[-1]: int(entry["dhash"], 16) for path, entry in keys.items()}
    assert (value["a.jpg"] ^ value["b.jpg"]).bit_count() <= 4 < (value["a.jpg"] ^ value["c.jpg"]).bit_count()
    assert imagekeys.scan(dataset) == {"read": 0, "failed": 0, "scanned": 3, "images": 3}  # cached


# ---------- applying ----------


def test_apply_moves_into_batch_folders_snapshots_first_and_undoes_in_one_step(tmp_path):
    dataset = make_dataset(tmp_path)
    root = dataset.root
    yaml_before = (root / "data.yaml").read_text()
    before = {record.id: record.path for record in dataset.images.values()}
    result = rebalance.plan(dataset, RULES)
    assert {item["folder"] for item in result["create"]} == {"images/test_2024-01", "images/test_2024-02", "images/val_2024-02"}
    applied = rebalance.apply(dataset, RULES, result["digest"])
    assert applied["moved"] == len(result["moves"]) and applied["snapshot"]["name"].startswith("Before rebalance")
    snapshot = snapshots.get(root, applied["snapshot"]["id"])
    assert snapshot["kind"] == "auto" and snapshot["splits"] == {split: row["images"] for split, row in result["table"]["before"].items()}
    config = yaml.safe_load((root / "data.yaml").read_text())
    assert sorted(config["test"]) == ["images/test_2024-01", "images/test_2024-02"]
    for move in result["moves"]:
        record = dataset.images[move["id"]]
        assert record.split == move["to"] and dataset.folder_label(dataset._folder_of(record)) == move["to_folder"]
        assert record.path.parent.name == move["to_folder"].rsplit("/", 1)[-1]
        assert record.label_path.parent.name == move["to_folder"].rsplit("/", 1)[-1] and record.label_path.is_file()
    assert json.loads((root / ".yolo-workbench" / "rebalance.json").read_text())["targets"] == {"train": 0.7, "val": 0.15, "test": 0.15}

    dataset.history("undo")
    assert {record.id: record.path for record in dataset.images.values()} == before
    assert (root / "data.yaml").read_text() == yaml_before
    assert not (root / "images" / "test_2024-01").exists() and not (root / "labels" / "test_2024-01").exists()
    assert not dataset.issues() or all(issue["split"] in ("train", "val") for issue in dataset.issues())


def test_apply_refuses_a_stale_preview_and_a_locked_dataset(tmp_path):
    dataset = make_dataset(tmp_path)
    result = rebalance.plan(dataset, RULES)
    record = next(iter(dataset.images.values()))
    dataset.replace_annotations(record.id, [{"class_id": 2, "points": [0.5, 0.5, 0.1, 0.1]}])
    with pytest.raises(WriteConflict, match="changed since the preview"):
        rebalance.apply(dataset, RULES, result["digest"])
    dataset.write_guard = lambda: "a training run is in progress"
    with pytest.raises(DatasetLocked):
        rebalance.apply(dataset, RULES, rebalance.plan(dataset, RULES)["digest"])
    assert snapshots.list_snapshots(dataset.root) == []


def test_big_moves_ask_the_disk_nothing_per_image(tmp_path):
    dataset = make_dataset(tmp_path, count=400, clips=1)
    with dataset._lock:
        target = dataset.resolve_folder("val")
        moves = [(record.id, "val", target) for record in dataset.images.values() if record.split == "train"][:120]
    assert len(moves) == 120
    with counting(dataset.root) as calls:
        dataset.relocate_images(moves)
    # listings per folder and two renames per image, no stat or exists per image
    assert calls["stat"] + calls["lstat"] < 30
    assert calls["scandir"] < 10
    with counting(dataset.root) as calls:
        dataset.history("undo")
    assert calls["stat"] + calls["lstat"] < 30


def test_a_proposed_split_trains_without_moving_anything(tmp_path):
    dataset = make_dataset(tmp_path)
    result = rebalance.plan(dataset, RULES)
    before = {record.id: record.path for record in dataset.images.values()}
    meta = rebalance.proposed_snapshot(dataset, RULES, result["digest"])
    planned = {split: row["images"] for split, row in result["table"]["after"].items()}
    assert meta["splits"] == planned and meta["rebalance"]["digest"] == result["digest"]
    assert {record.id: record.path for record in dataset.images.values()} == before
    copy = snapshots.open_dataset(dataset.root, meta["id"], "detection")
    assert splits_of(copy) == planned
    launcher = FakeLauncher()
    manager = TrainingManager.for_dataset(dataset, jobs=JobManager(synchronous=True), launcher=launcher, weights_dir=tmp_path / "weights")
    manager.attach(dataset)
    assert manager.enqueue(None, dataset, "", "", ARGS, confirm=True, snapshot=meta["id"])["queued"]
    data = (launcher.launches[-1][0] / "workbench" / "test.txt").read_text().split()
    assert len(data) == planned["test"]


def test_new_images_are_placed_by_the_rules_and_the_rest_stays(tmp_path):
    dataset = make_dataset(tmp_path, count=60)
    old = {record.id: record.path for record in dataset.images.values()}
    for index in range(30):
        path = dataset.root / "images" / "train_2024-01" / f"new_{index:03d}.jpg"
        Image.new("RGB", (40, 30), "black").save(path)
        (dataset.root / "labels" / "train_2024-01" / f"new_{index:03d}.txt").write_text("2 0.5 0.5 0.3 0.3\n")
    dataset.rescan()
    new = set(dataset.images) - set(old)
    placed = rebalance.place_new(dataset, {**RULES, "together": {}}, new)
    assert placed["moved"] > 0
    assert all(dataset.images[image_id].path == path for image_id, path in old.items())
    assert {dataset.images[image_id].split for image_id in new} == {"train", "val", "test"}


def test_leakage_by_the_saved_rules(tmp_path):
    dataset = make_dataset(tmp_path)
    assert rebalance.leakage(dataset, None)["configured"] is False
    leak = rebalance.leakage(dataset, RULES)
    assert leak["groups"] > 0 and len(leak["images"]) >= 2 * leak["groups"]


# ---------- restore ----------


def test_restoring_a_snapshot_in_place(tmp_path):
    dataset = make_dataset(tmp_path, count=30)
    root = dataset.root
    taken = snapshots.take(dataset, "Good state")
    wanted = sorted((record.split, str(record.path), (record.label_path.read_text() if record.label_path.exists() else None)) for record in dataset.images.values())
    first, second = sorted(dataset.images.values(), key=lambda record: str(record.path))[:2]
    dataset.replace_annotations(first.id, [{"class_id": 2, "points": [0.1, 0.1, 0.1, 0.1]}])
    dataset.move_images([second.id], "val")
    dataset.delete_images([next(record.id for record in dataset.images.values() if record.id not in (first.id, second.id))])
    Image.new("RGB", (8, 8)).save(root / "images" / "train_2024-01" / "extra.jpg")
    dataset.rescan()
    with open(root / "data.yaml", "a") as stream:
        stream.write("# edited\n")
    result = snapshots.restore(dataset, taken["id"])
    assert result["before"]["name"].startswith("Before restoring") and result["trashed"] >= 2 and result["restored"] >= 2
    now = sorted((record.split, str(record.path), (record.label_path.read_text() if record.label_path.exists() else None)) for record in dataset.images.values())
    assert now == wanted
    assert "# edited" not in (root / "data.yaml").read_text()
    assert dataset.history_state() == {"can_undo": False, "can_redo": False}
    assert (root / ".yolo-workbench" / "trash").is_dir()  # what was added since went to the trash, not away


# ---------- API ----------


def client_for(app):
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://127.0.0.1", headers={"X-Workbench": "1"})


def test_rebalance_api(tmp_path):
    dataset = make_dataset(tmp_path)
    jobs = JobManager(synchronous=True)
    app = create_app(dataset, media_cache=MediaCache(tmp_path / "media"), jobs=jobs, training=TrainingManager.for_dataset(dataset, jobs=jobs, launcher=FakeLauncher(), weights_dir=tmp_path / "weights"))

    async def scenario():
        async with client_for(app) as client:
            state = (await client.get("/api/v1/rebalance")).json()
            assert state["saved"] is False and state["rules"]["frozen_splits"] == [] and state["scan"] == {"scanned": 0, "images": 120}
            preview = (await client.post("/api/v1/rebalance/preview", json={"rules": RULES})).json()
            assert preview["counts"]["moves"] > 0 and "moves" not in preview and len(preview["sample"]) <= 20
            moved = (await client.get("/api/v1/images", params={"subset": preview["subset"], "limit": 500})).json()
            assert moved["total"] == preview["counts"]["moves"]
            bad = await client.post("/api/v1/rebalance/preview", json={"rules": {"mode": "x"}})
            assert bad.status_code == 400
            stale = await client.post("/api/v1/rebalance/apply", json={"rules": RULES, "digest": "0" * 16})
            assert stale.json()["state"] == "failed" and "preview" in stale.json()["error"]
            proposed = (await client.post("/api/v1/rebalance/snapshot", json={"rules": RULES, "digest": preview["digest"]})).json()
            assert proposed["state"] == "done" and proposed["result"]["rebalance"]["digest"] == preview["digest"]
            job = (await client.post("/api/v1/rebalance/apply", json={"rules": RULES, "digest": preview["digest"]})).json()
            assert job["state"] == "done" and job["result"]["moved"] == preview["counts"]["moves"]
            assert (await client.get("/api/v1/rebalance")).json()["saved"] is True
            assert (await client.get("/api/v1/rebalance/leakage")).json()["groups"] == 0
            undo = (await client.post("/api/v1/history/undo")).json()
            assert undo["can_redo"]
            scan = (await client.post("/api/v1/rebalance/scan")).json()
            assert scan["state"] == "done" and scan["result"]["scanned"] == 120
            restored = (await client.post(f"/api/v1/snapshots/{job['result']['snapshot']['id']}/restore")).json()
            assert restored["state"] == "done", restored

    asyncio.run(scenario())


def test_a_project_from_a_snapshot(tmp_path):
    from test_workspace import make_app, make_workspace

    workspace = make_workspace(tmp_path)
    workspace.create("Source", "detection", ["car"])
    session = workspace.session("source")
    image = session.dataset.root / "images" / "train" / "a.jpg"
    Image.new("RGB", (16, 16), "red").save(image)
    (session.dataset.root / "labels" / "train" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    session.dataset.rescan()
    meta = snapshots.take(session.dataset, "v1")
    app = make_app(workspace, tmp_path)

    async def scenario():
        async with client_for(app) as client:
            created = (await client.post(f"/api/v1/projects/source/snapshots/{meta['id']}/project", json={"name": "From v1"})).json()
            assert created["slug"] == "from-v1" and created["description"].startswith("From snapshot 'v1'")
            copy = (await client.get("/api/v1/projects/from-v1/images")).json()
            assert [item["name"] for item in copy["items"]] == ["a.jpg"]

    asyncio.run(scenario())
    assert (tmp_path / "projects" / "from-v1" / "images" / "train" / "a.jpg").stat().st_ino == image.stat().st_ino


def test_an_import_placed_by_the_rules(tmp_path):
    dataset = make_dataset(tmp_path, count=40)
    rebalance.save_rules(dataset, rebalance.parse_rules(dataset, {**RULES, "together": {}}))
    source = tmp_path / "export"
    (source / "images").mkdir(parents=True)
    (source / "labels").mkdir()
    for index in range(20):
        Image.new("RGB", (40, 30), (index * 10, 0, 0)).save(source / "images" / f"fresh_{index:02d}.png")
        (source / "labels" / f"fresh_{index:02d}.txt").write_text("1 0.5 0.5 0.2 0.2\n")
    (source / "classes.txt").write_text("car\nbus\ntram\n")
    jobs = JobManager(synchronous=True)
    app = create_app(dataset, media_cache=MediaCache(tmp_path / "media"), jobs=jobs, training=TrainingManager.for_dataset(dataset, jobs=jobs, launcher=FakeLauncher(), weights_dir=tmp_path / "weights"))
    before = {record.id: record.path for record in dataset.images.values()}

    async def scenario():
        async with client_for(app) as client:
            job = (await client.post("/api/v1/import/commit", json={"path": str(source), "split_mode": "rules"})).json()
            assert job["state"] == "done", job
            return job["result"]

    result = asyncio.run(scenario())
    assert result["imported"] == 20 and result["placed"]["moved"] > 0
    assert all(dataset.images[image_id].path == path for image_id, path in before.items())
    fresh = [record for record in dataset.images.values() if record.path.name.startswith("fresh_")]
    assert {record.split for record in fresh} == {"train", "val", "test"}
