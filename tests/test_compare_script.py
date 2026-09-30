"""`scripts/compare_tiling.py`의 계산 부분 — IoU, 정답 짝짓기, 기대 문자열 대조. (모델 호출은 스크립트를 직접 돌려 확인한다)"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "compare_tiling.py"


@pytest.fixture(scope="module")
def compare():
    spec = importlib.util.spec_from_file_location("compare_tiling", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def box(x0, y0, x1, y1, label=""):
    return {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0, "label": label, "type": "object"}


def test_iou(compare):
    assert compare.iou((0, 0, 1, 1), (0, 0, 1, 1)) == 1.0
    assert compare.iou((0, 0, 0.2, 0.2), (0.1, 0, 0.3, 0.2)) == pytest.approx(1 / 3)
    assert compare.iou((0, 0, 0.2, 0.2), (0.2, 0, 0.4, 0.2)) == 0.0          # 맞닿기만 한 박스
    assert compare.iou((0, 0, 0.2, 0.2), (0.5, 0.5, 0.6, 0.6)) == 0.0


def test_truth_boxes_are_scaled_by_the_image_size(compare, tmp_path):
    path = tmp_path / "truth.json"
    path.write_text(json.dumps({"image": {"width": 2000, "height": 1000},
                                "targets": [{"label": "a", "bbox": [200, 100, 400, 300]}],
                                "distractors": [{"label": "b", "bbox": [1000, 500, 1200, 700]}]}), encoding="utf-8")
    truth = compare.load_truth(path)
    assert truth["targets"] == [{"label": "a", "box": (0.1, 0.1, 0.2, 0.3)}]
    assert truth["distractors"][0]["box"] == (0.5, 0.5, 0.6, 0.7)


def test_each_target_is_paired_with_one_prediction(compare):
    truth = {"targets": [{"label": "one", "box": (0.10, 0.10, 0.20, 0.20)}, {"label": "two", "box": (0.60, 0.60, 0.70, 0.70)},
                         {"label": "three", "box": (0.10, 0.80, 0.20, 0.90)}],
             "distractors": [{"label": "blue", "box": (0.40, 0.10, 0.50, 0.20)}]}
    predicted = [
        box(0.10, 0.10, 0.20, 0.20),            # one을 정확히
        box(0.11, 0.10, 0.21, 0.20),            # one을 한 번 더 — 정답 하나에 예측 하나만 짝을 맺는다
        box(0.62, 0.60, 0.72, 0.70),            # two를 조금 어긋나게 (IoU 0.667)
        box(0.40, 0.10, 0.50, 0.20),            # 음성 대조 위
    ]
    score = compare.score_boxes(predicted, truth)
    assert [item["iou"] for item in score["targets"]] == [1.0, 0.667, 0.0]      # three는 못 찾았다
    assert score["hits"] == 2 and score["meanIou"] == pytest.approx(0.556, abs=0.001)
    assert score["unmatched"] == 2 and score["onDistractor"] == 1
    empty = compare.score_boxes([], truth)
    assert empty["hits"] == 0 and empty["meanIou"] == 0.0 and empty["unmatched"] == 0


def test_expected_strings_are_matched_ignoring_case_and_spacing(compare):
    text = "\n".join([
        "[VISUAL SOURCE: plan.pdf · page 1]", "[CLASSIFICATION: scanned-raster]",
        "[TILED TRANSCRIPTION: this page was read as 4 rows x 5 columns of overlapping tiles.]",
        "[TILE r1c1]", "ROOM 101   OFFICE", "area 16.0 m2", "[TILE r4c5]", "DWG NO: AR-2O44-C", "[UNCLEAR]",
        "The image shows a floor plan with several rooms.",
    ])
    report = compare.expectation_report(text, ["ROOM 101 OFFICE", "AREA 16.0 m2", "DWG NO: AR-2044-C", "3600"])
    assert (report["found"], report["expected"]) == (2, 4)
    assert report["missing"] == ["DWG NO: AR-2044-C", "3600"]                   # 0을 O로 잘못 읽은 것은 읽은 것으로 치지 않는다
    assert report["lines"] == 4                                                # 표식 줄은 세지 않는다
    assert report["unrelated"] == ["The image shows a floor plan with several rooms."]   # 오독은 무관한 줄이 아니다
    assert report["exact"] == 2                                                # 오독한 줄은 "글자까지 맞는 줄"이 아니다


def test_near_miss_inventions_are_not_unrelated_but_are_not_exact_either(compare):
    """gemma3 실측: 타일 가장자리에서 잘린 날짜를 `DATE: 2023`으로 지어냈다. 무관한 줄 지표는 이것을 놓친다."""
    observed = "[TILE r4c4]\nREV: B SCALE 1:50 DATE: 2023\nDWG NO: AR-2044-C\n[TILE r1c2]\nROOM 101 STORAGE"
    report = compare.expectation_report(observed, ["REV: B   SCALE: 1:50   DATE: 2026-05-08", "DWG NO: AR-2044-C",
                                                   "ROOM 101 OFFICE", "ROOM 102 STORAGE"])
    assert report["unrelated"] == [] and report["lines"] == 3
    assert report["exact"] == 1 and report["found"] == 1


def test_synthetic_image_matches_its_truth_file(compare, tmp_path):
    image_path, truth_path = compare.synthetic_image(tmp_path)
    raw = json.loads(truth_path.read_text(encoding="utf-8"))
    assert (len(raw["targets"]), len(raw["distractors"])) == (4, 3)
    with Image.open(image_path) as image:
        assert image.size == (2400, 1600)
        for target in raw["targets"]:
            x0, y0, x1, y1 = target["bbox"]
            red, green, blue = image.getpixel(((x0 + x1) // 2, (y0 + y1) // 2))
            assert red > 180 and green < 80 and blue < 80
        for item in raw["distractors"]:
            x0, y0, x1, y1 = item["bbox"]
            red, green, blue = image.getpixel(((x0 + x1) // 2, (y0 + y1) // 2))
            assert blue > 180 and red < 80
