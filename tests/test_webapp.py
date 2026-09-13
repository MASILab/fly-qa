from pathlib import Path

from fastapi.testclient import TestClient

from fly_qa.events import EventBus, ResultEvent
from fly_qa.viz import VizNode, VizSubset
from fly_qa.webapp.server import create_app


def _make_root(tmp_path: Path) -> Path:
    root = tmp_path / "root"
    folder = root / "sub"
    folder.mkdir(parents=True)
    (folder / "a.png").write_bytes(b"\x89PNG\r\n\x1a\nfakepngbytes")
    return root


def test_index_endpoint():
    bus = EventBus()
    app = create_app(bus)
    with TestClient(app) as client:
        r = client.get("/")
        assert r.status_code == 200
        assert "The Fly Who Does QA" in r.text
        assert "WHO DOES QA" in r.text


def test_viz_subset_endpoint_empty_by_default():
    bus = EventBus()
    app = create_app(bus)
    with TestClient(app) as client:
        r = client.get("/api/viz-subset")
        assert r.json() == {"nodes": [], "edges": []}


def test_viz_subset_endpoint_returns_real_subset():
    bus = EventBus()
    subset = VizSubset(
        nodes=[VizNode(1, "r1r6", "R1-R6"), VizNode(2, "dnp20", "DNp20")],
        edges=[(1, 2, 0.5)],
    )
    app = create_app(bus, viz_subset=subset)
    with TestClient(app) as client:
        r = client.get("/api/viz-subset")
        data = r.json()
        assert len(data["nodes"]) == 2
        assert data["edges"][0]["source"] == 1


def test_image_endpoint_serves_valid_file_by_relative_path(tmp_path: Path):
    root = _make_root(tmp_path)
    bus = EventBus()
    app = create_app(bus, root=root)
    with TestClient(app) as client:
        r = client.get("/api/image", params={"path": "sub/a.png"})
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/png"


def test_image_endpoint_serves_valid_file_by_absolute_path(tmp_path: Path):
    root = _make_root(tmp_path)
    bus = EventBus()
    app = create_app(bus, root=root)
    with TestClient(app) as client:
        abs_path = str((root / "sub" / "a.png").resolve())
        r = client.get("/api/image", params={"path": abs_path})
        assert r.status_code == 200


def test_image_endpoint_allows_root_itself_as_single_file(tmp_path: Path):
    # fly-qa can be pointed at a single PNG file (root == the file itself).
    single_file = tmp_path / "only.png"
    single_file.write_bytes(b"\x89PNG\r\n\x1a\nfakepngbytes")
    bus = EventBus()
    app = create_app(bus, root=single_file)
    with TestClient(app) as client:
        r = client.get("/api/image", params={"path": str(single_file.resolve())})
        assert r.status_code == 200


def test_image_endpoint_rejects_path_traversal(tmp_path: Path):
    root = _make_root(tmp_path)
    bus = EventBus()
    app = create_app(bus, root=root)
    with TestClient(app) as client:
        r = client.get("/api/image", params={"path": "../../../../etc/passwd"})
        assert r.status_code == 400


def test_image_endpoint_rejects_absolute_path_outside_root(tmp_path: Path):
    root = _make_root(tmp_path)
    other = tmp_path / "elsewhere.png"
    other.write_bytes(b"\x89PNG\r\n\x1a\nfakepngbytes")
    bus = EventBus()
    app = create_app(bus, root=root)
    with TestClient(app) as client:
        r = client.get("/api/image", params={"path": str(other.resolve())})
        assert r.status_code == 400


def test_image_endpoint_missing_file_404(tmp_path: Path):
    root = _make_root(tmp_path)
    bus = EventBus()
    app = create_app(bus, root=root)
    with TestClient(app) as client:
        r = client.get("/api/image", params={"path": "sub/nope.png"})
        assert r.status_code == 404


def test_image_endpoint_no_root_configured():
    bus = EventBus()
    app = create_app(bus)
    with TestClient(app) as client:
        r = client.get("/api/image", params={"path": "a.png"})
        assert r.status_code == 404


def test_websocket_delivers_result_event():
    bus = EventBus()
    app = create_app(bus)
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            bus.publish(ResultEvent(type="started", total_images=5))
            data = ws.receive_json()
            assert data["type"] == "started"
            assert data["total_images"] == 5
