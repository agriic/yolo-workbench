"""Several people in one project: per-user undo, the change feed, presence, class-list revisions, attribution."""

import asyncio
import json
import time
from contextlib import asynccontextmanager
from dataclasses import fields, replace as dc_replace

import pytest
import yaml
from httpx import ASGITransport, AsyncClient
from PIL import Image

from yolo_workbench import auth as auth_module, classes, session as session_module
from yolo_workbench.config import load_config
from yolo_workbench.dataset import Dataset, DatasetError, SchemaChanged, WriteConflict
from yolo_workbench.identity import User, acting_as
from yolo_workbench.runtime import RUNTIME
from yolo_workbench.server import build_app

ALICE, BOB = User("alice", "Alice"), User("bob", "Bob")
PASSWORD = "correct horse"


@pytest.fixture(autouse=True)
def restore_runtime(monkeypatch):
    saved = dc_replace(RUNTIME)
    monkeypatch.setattr(auth_module, "SCRYPT", {"n": 2**10, "r": 8, "p": 1})
    yield
    for item in fields(RUNTIME):
        setattr(RUNTIME, item.name, getattr(saved, item.name))


def write_images(root, count=3):
    (root / "images" / "train").mkdir(parents=True)
    (root / "labels" / "train").mkdir(parents=True)
    for index in range(count):
        Image.new("RGB", (64, 48), "white").save(root / "images" / "train" / f"img{index}.jpg")
        (root / "labels" / "train" / f"img{index}.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    (root / "data.yaml").write_text(yaml.safe_dump({"train": "images/train", "names": {0: "car", 1: "bus", 2: "van"}}))


def make(tmp_path, count=3):
    root = tmp_path / "data"
    write_images(root, count)
    dataset = Dataset(root / "data.yaml", "detection")
    ids = {record.path.stem: record.id for record in dataset.images.values()}
    return dataset, ids


def box(dataset, image_id, class_id=0, x=0.5):
    return [*[a.as_dict() for a in dataset.images[image_id].annotations], {"class_id": class_id, "points": [x, 0.5, 0.1, 0.1]}]


# ---------- per-user undo ----------


def test_each_user_undoes_only_their_own_changes(tmp_path):
    dataset, ids = make(tmp_path)
    with acting_as(ALICE, "tab-a"):
        dataset.replace_annotations(ids["img0"], box(dataset, ids["img0"]))
    with acting_as(BOB, "tab-b"):
        dataset.replace_annotations(ids["img1"], box(dataset, ids["img1"]))
        assert dataset.history_state() == {"can_undo": True, "can_redo": False}
    with acting_as(ALICE, "tab-a"):
        result = dataset.history("undo")
        assert result["image_ids"] == [ids["img0"]] and result["action"] == "edit"
        assert dataset.history_state() == {"can_undo": False, "can_redo": True}
        with pytest.raises(DatasetError, match="Nothing to undo"):
            dataset.history("undo")
    assert len(dataset.images[ids["img0"]].annotations) == 1 and len(dataset.images[ids["img1"]].annotations) == 2
    with acting_as(BOB, "tab-b"):
        dataset.replace_annotations(ids["img2"], box(dataset, ids["img2"]))  # Bob's new change keeps Alice's redo
    with acting_as(ALICE, "tab-a"):
        assert dataset.history("redo")["image_ids"] == [ids["img0"]]
        dataset.replace_annotations(ids["img0"], box(dataset, ids["img0"], 1))
        dataset.history("undo")
        dataset.replace_annotations(ids["img0"], box(dataset, ids["img0"], 2))  # her own new change drops her redo
        assert dataset.history_state()["can_redo"] is False


def test_undo_is_refused_when_someone_else_changed_the_same_image_since(tmp_path):
    dataset, ids = make(tmp_path)
    label = dataset.images[ids["img0"]].label_path
    with acting_as(ALICE, "tab-a"):
        dataset.replace_annotations(ids["img0"], box(dataset, ids["img0"], 1))
    with acting_as(BOB, "tab-b"):
        dataset.replace_annotations(ids["img0"], box(dataset, ids["img0"], 2))
    bobs = label.read_text()
    with acting_as(ALICE, "tab-a"):
        with pytest.raises(WriteConflict, match="Bob changed the same images since"):
            dataset.history("undo")
    assert label.read_text() == bobs  # nothing was overwritten
    with acting_as(BOB, "tab-b"):
        dataset.history("undo")
    with acting_as(ALICE, "tab-a"):
        dataset.history("undo")  # Bob's change is gone, so hers can be undone now
    assert label.read_text() == "0 0.5 0.5 0.2 0.2\n"


def test_one_users_history_stays_a_plain_stack(tmp_path):
    # without sign-in everything belongs to nobody: undo/redo walk one stack, also over the same file
    dataset, ids = make(tmp_path)
    label = dataset.images[ids["img0"]].label_path
    states = [label.read_text()]
    for class_id in (1, 2):
        dataset.replace_annotations(ids["img0"], box(dataset, ids["img0"], class_id))
        states.append(label.read_text())
    dataset.history("undo")
    dataset.history("undo")
    assert label.read_text() == states[0]
    dataset.history("redo")
    dataset.history("redo")
    assert label.read_text() == states[2]
    assert not (dataset.root / ".yolo-workbench" / "activity.jsonl").exists()  # no sign-in, no activity log


def test_class_changes_by_others_block_undo_and_outdated_saves(tmp_path):
    dataset, ids = make(tmp_path)
    generation = dataset.schema_generation
    with acting_as(ALICE, "tab-a"):
        dataset.replace_annotations(ids["img0"], box(dataset, ids["img0"], 1))
    with acting_as(BOB, "tab-b"):
        classes.reorder_classes(dataset, [2, 0, 1])  # van, car, bus
    assert dataset.names == {0: "van", 1: "car", 2: "bus"}
    with acting_as(ALICE, "tab-a"):
        with pytest.raises(WriteConflict, match="Bob changed the class list since"):
            dataset.history("undo")
        with pytest.raises(SchemaChanged) as refused:
            dataset.replace_annotations(ids["img1"], box(dataset, ids["img1"], 1), schema_generation=generation)
    assert refused.value.mapping == {0: 1, 1: 2, 2: 0} and refused.value.generation == dataset.schema_generation
    assert "by Bob" in str(refused.value)
    with acting_as(BOB, "tab-b"):
        classes.remove_class(dataset, 0, "delete")  # van goes: car 1 -> 0, bus 2 -> 1
    assert dataset.schema_mapping(generation) == {0: 0, 1: 1, 2: None}
    with acting_as(ALICE, "tab-a"):
        dataset.replace_annotations(ids["img1"], box(dataset, ids["img1"], 1), schema_generation=dataset.schema_generation)  # current list: fine


# ---------- the change feed and attribution ----------


def test_changes_are_reported_to_other_tabs_with_their_author(tmp_path):
    dataset, ids = make(tmp_path)
    start = dataset.revision
    with acting_as(ALICE, "tab-a"):
        dataset.replace_annotations(ids["img0"], box(dataset, ids["img0"]))
        assert dataset.changes(start)["images"] == {}  # her own tab already shows it
    with acting_as(BOB, "tab-b"):
        feed = dataset.changes(start)
        assert feed["images"][ids["img0"]]["by"] == {"name": "alice", "display_name": "Alice"} and not feed["structure"]
        assert dataset.changes(feed["revision"])["images"] == {}
        dataset.set_review([ids["img1"]], "done")
    with acting_as(ALICE, "tab-a"):
        feed = dataset.changes(start)
        assert feed["images"][ids["img1"]]["kind"] == "review"
        dataset.delete_images([ids["img2"]])
    with acting_as(BOB, "tab-b"):
        feed = dataset.changes(start)
        assert feed["structure"] and ids["img2"] in feed["images"]
        assert dataset.changes(-1)["reset"]

    detail = dataset.detail(ids["img0"])
    assert detail["edited"]["by"]["name"] == "alice" and detail["reviewed"] is None
    reviewed = dataset.detail(ids["img1"])
    assert reviewed["reviewed"]["by"]["name"] == "bob" and reviewed["review"] == "done"
    assert [item["id"] for item in dataset.list_images(reviewer="bob")["items"]] == [ids["img1"]]
    dataset.close()  # saves the debounced "last edited by" bookkeeping
    again = Dataset(dataset.yaml_path, "detection")
    assert again.detail(ids["img0"])["edited"]["by"]["name"] == "alice"
    assert again.images[ids["img1"]].reviewed_by == "bob"
    actions = [(entry["user"], entry["action"]) for entry in again.recent_activity()]
    assert actions == [("alice", "delete images"), ("bob", "review: done"), ("alice", "edit")]


def test_the_change_log_is_bounded(tmp_path, monkeypatch):
    dataset, ids = make(tmp_path)
    dataset._change_log = type(dataset._change_log)(maxlen=2)
    for x in (0.2, 0.3, 0.4):
        dataset.replace_annotations(ids["img0"], box(dataset, ids["img0"], x=x))
    assert dataset.changes(0)["reset"] and not dataset.changes(dataset.revision - 1)["reset"]


# ---------- over HTTP, two signed-in browsers ----------


@asynccontextmanager
async def browser(app, name, tab):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://gpu-box", headers={"X-Workbench": "1", "X-Workbench-Client": tab}) as client:
        response = await client.post("/api/v1/auth/login", json={"name": name, "password": PASSWORD})
        assert response.status_code == 200, response.text
        yield client


def server(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    root = tmp_path / "projects"
    root.mkdir()
    write_images(root / "cars")
    (root / "cars" / "project.json").write_text(json.dumps({"version": 1, "name": "Cars", "category": "detection", "yaml": "data.yaml"}))
    config = load_config(None, {"projects_root": root, "allowed_hosts": ["gpu-box"], "idle_unload_minutes": 0}, environ={})
    app = build_app(config, root)
    app.state.auth.users.add("alice", PASSWORD, "Alice")
    app.state.auth.users.add("bob", PASSWORD, "Bob")
    return app


def test_two_browsers_share_changes_presence_and_jobs(tmp_path, monkeypatch):
    app = server(tmp_path, monkeypatch)
    base = "/api/v1/projects/cars"

    async def run():
        async with browser(app, "alice", "tab-a") as alice, browser(app, "bob", "tab-b") as bob:
            meta = (await alice.get(f"{base}/dataset")).json()
            image = (await alice.get(f"{base}/images")).json()["items"][0]
            detail = (await alice.get(f"{base}/images/{image['id']}")).json()
            start = meta["revision"]

            # Alice opens the image in her editor; Bob's poll shows it, hers does not show herself
            assert (await alice.post(f"{base}/presence", json={"image_id": image["id"]})).status_code == 200
            feed = (await bob.get(f"{base}/changes", params={"since": start})).json()
            assert feed["presence"] == {image["id"]: [{"name": "alice", "display_name": "Alice"}]} and feed["images"] == {}
            assert (await alice.get(f"{base}/changes", params={"since": start})).json()["presence"] == {}

            saved = await alice.put(f"{base}/images/{image['id']}/annotations", json={"revision": detail["revision"], "schema_generation": meta["schema_generation"], "annotations": [*detail["annotations"], {"class_id": 1, "points": [0.2, 0.2, 0.1, 0.1]}]})
            assert saved.status_code == 200 and saved.json()["edited"]["by"]["name"] == "alice"
            feed = (await bob.get(f"{base}/changes", params={"since": start})).json()
            assert feed["images"][image["id"]]["by"]["display_name"] == "Alice"

            # Bob saves the same image from an outdated copy: 409; his undo has nothing of his
            stale = await bob.put(f"{base}/images/{image['id']}/annotations", json={"revision": detail["revision"], "annotations": detail["annotations"]})
            assert stale.status_code == 409
            assert (await bob.post(f"{base}/history/undo")).status_code == 400
            assert (await bob.get(f"{base}/dataset")).json()["can_undo"] is False
            assert (await alice.get(f"{base}/dataset")).json()["can_undo"] is True

            # Bob renames a class; Alice's next save from the old class list gets the mapping
            assert (await bob.post(f"{base}/classes/reorder", json={"order": [1, 0, 2]})).status_code == 200
            current = (await alice.get(f"{base}/images/{image['id']}")).json()
            refused = await alice.put(f"{base}/images/{image['id']}/annotations", json={"revision": current["revision"], "schema_generation": meta["schema_generation"], "annotations": current["annotations"]})
            assert refused.status_code == 409
            body = refused.json()
            assert body["mapping"] == {"0": 1, "1": 0, "2": 2} and body["schema_generation"] == meta["schema_generation"] + 1
            undo = await alice.post(f"{base}/history/undo")
            assert undo.status_code == 409 and "Bob changed the class list" in undo.json()["detail"]

            # jobs and activity name who started them
            activity = (await bob.get(f"{base}/activity")).json()["items"]
            assert [(item["user"], item["action"]) for item in activity[:2]] == [("bob", "reorder classes"), ("alice", "edit")]
            assert (await alice.post(f"{base}/presence", json={"image_id": None})).status_code == 200
            assert (await bob.get(f"{base}/changes", params={"since": start})).json()["presence"] == {}

    asyncio.run(run())


def test_presence_expires(monkeypatch, tmp_path):
    dataset, ids = make(tmp_path)
    session = session_module.DatasetSession.open(dataset)
    with acting_as(ALICE, "tab-a"):
        session.set_presence(ids["img0"])
    with acting_as(BOB, "tab-b"):
        assert list(session.others_present()) == [ids["img0"]]
        later = time.monotonic() + session_module.PRESENCE_TTL + 1
        monkeypatch.setattr(session_module.time, "monotonic", lambda: later)
        assert session.others_present() == {}
    session.close()


def test_jobs_and_runs_remember_who_started_them(tmp_path):
    from yolo_workbench.jobs import JobManager

    jobs = JobManager(synchronous=True)
    seen = []
    with acting_as(ALICE, "tab-a"):
        job = jobs.start("import", "cars", "Import", lambda job: seen.append((auth_module.current_user_name())) or {})
    assert job["started_by"] == {"name": "alice", "display_name": "Alice"} and seen == ["alice"]


def test_polling_neither_opens_nor_keeps_a_project_open_and_a_reopened_project_resets_pages(tmp_path, monkeypatch):
    app = server(tmp_path, monkeypatch)
    workspace = app.state.workspace
    base = "/api/v1/projects/cars"

    async def run():
        async with browser(app, "alice", "tab-a") as alice:
            meta = (await alice.get(f"{base}/dataset")).json()
            assert meta["feed"] and "cars" in workspace.sessions
            workspace.close("cars")
            feed = (await alice.get(f"{base}/changes", params={"since": meta["revision"], "feed": meta["feed"]})).json()
            assert feed["closed"] and "cars" not in workspace.sessions  # the poll did not open it again
            again = (await alice.get(f"{base}/dataset")).json()  # a real request opens it: a new feed
            assert again["feed"] != meta["feed"]
            feed = (await alice.get(f"{base}/changes", params={"since": meta["revision"], "feed": meta["feed"]})).json()
            assert feed["reset"] and feed["feed"] == again["feed"]

    asyncio.run(run())


# ---------- found in review ----------


def test_a_class_change_is_not_undone_over_labels_others_wrote_for_the_new_list(tmp_path):
    dataset, ids = make(tmp_path)
    label = dataset.images[ids["img2"]].label_path
    with acting_as(ALICE, "tab-a"):
        classes.reorder_classes(dataset, [2, 1, 0])  # van, bus, car
    with acting_as(BOB, "tab-b"):
        dataset.replace_annotations(ids["img2"], [{"class_id": 0, "points": [0.5, 0.5, 0.1, 0.1]}])  # a van
    with acting_as(ALICE, "tab-a"):
        with pytest.raises(WriteConflict, match="Bob changed labels since"):
            dataset.history("undo")
    assert dataset.names[0] == "van" and label.read_text().startswith("0 ")
    with acting_as(BOB, "tab-b"):
        dataset.set_review([ids["img1"]], "done")  # reviews do not involve class ids
        dataset.history("undo")
    with acting_as(ALICE, "tab-a"):
        dataset.history("undo")  # Bob's label is gone again: the class change can be undone
    assert dataset.names == {0: "car", 1: "bus", 2: "van"}


def test_undoing_a_move_keeps_a_label_made_since_with_its_image(tmp_path):
    dataset, ids = make(tmp_path)
    record = dataset.images[ids["img2"]]
    record.label_path.unlink()
    dataset.rescan()
    (dataset.root / "images" / "val").mkdir()
    dataset.add_folder("val", "images/val")
    with acting_as(ALICE, "tab-a"):
        dataset.move_images([ids["img2"]], "val")
    moved_id = next(record.id for record in dataset.images.values() if record.path.stem == "img2")
    with acting_as(BOB, "tab-b"):
        dataset.replace_annotations(moved_id, [{"class_id": 1, "points": [0.5, 0.5, 0.1, 0.1]}])
    with acting_as(ALICE, "tab-a"):
        with pytest.raises(WriteConflict, match="created since"):
            dataset.history("undo")
    assert not dataset.orphan_labels and len(dataset.images[moved_id].annotations) == 1


def test_adding_a_folder_does_not_invalidate_saves(tmp_path):
    dataset, ids = make(tmp_path)
    generation = dataset.schema_generation
    (dataset.root / "images" / "extra").mkdir()
    with acting_as(BOB, "tab-b"):
        dataset.add_folder("train", "images/extra")
    assert dataset.schema_generation == generation
    with acting_as(ALICE, "tab-a"):
        dataset.replace_annotations(ids["img0"], box(dataset, ids["img0"]), schema_generation=generation)


def test_rescan_reports_removed_and_changed_images(tmp_path):
    dataset, ids = make(tmp_path)
    start = dataset.revision
    dataset.images[ids["img0"]].label_path.write_text("1 0.5 0.5 0.3 0.3\n")
    dataset.images[ids["img1"]].path.unlink()
    dataset.rescan()
    feed = dataset.changes(start)
    assert {ids["img0"], ids["img1"]} <= set(feed["images"]) and feed["structure"]
