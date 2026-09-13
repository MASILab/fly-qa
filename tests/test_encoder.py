from PIL import Image

from fly_qa.encoder import encode_image, letterbox


def test_letterbox_preserves_aspect_and_target_size():
    img = Image.new("RGB", (600, 1200), (10, 20, 30))
    out = letterbox(img, (256, 256))
    assert out.size == (256, 256)


def test_letterbox_wide_image():
    img = Image.new("RGB", (1200, 300), (10, 20, 30))
    out = letterbox(img, (256, 256))
    assert out.size == (256, 256)


def test_encode_image_covers_all_body_ids():
    img = Image.new("RGB", (600, 1200), (100, 150, 200))
    r1_r6_ids = list(range(100))
    r8_ids = list(range(1000, 1050))

    encoded = encode_image(img, r1_r6_ids, r8_ids, grid_size=(64, 64))

    assert set(encoded.r1_r6_current.keys()) == set(r1_r6_ids)
    assert set(encoded.r8_current.keys()) == set(r8_ids)
    for v in encoded.r1_r6_current.values():
        assert 0.0 <= v <= 1.0


def test_encode_image_differs_for_different_images():
    r1_r6_ids = list(range(50))
    r8_ids = list(range(1000, 1020))

    dark = Image.new("RGB", (600, 1200), (5, 5, 5))
    bright = Image.new("RGB", (600, 1200), (250, 250, 250))

    dark_enc = encode_image(dark, r1_r6_ids, r8_ids, grid_size=(32, 32))
    bright_enc = encode_image(bright, r1_r6_ids, r8_ids, grid_size=(32, 32))

    dark_mean = sum(dark_enc.r1_r6_current.values()) / len(dark_enc.r1_r6_current)
    bright_mean = sum(bright_enc.r1_r6_current.values()) / len(bright_enc.r1_r6_current)
    assert bright_mean > dark_mean
