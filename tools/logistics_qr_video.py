from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    import cv2
    import numpy as np
    from PIL import ImageFont
    from qreader import QReader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

cv2 = None
np = None
QReader = None
Image = None
ImageDraw = None
ImageFont = None
WINDOW_NAME = "Logistics QR Reader"

FONT_CANDIDATES = (
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/PingFang.ttc",
)


@dataclass
class QRResult:
    text: str | None
    confidence: float | None
    bbox_xyxy: tuple[int, int, int, int] | None
    quad_xy: np.ndarray | None
    frame_index: int
    timestamp_sec: float


def parse_source(value: str) -> int | str:
    if value.isdigit():
        return int(value)
    return value


def parse_roi(value: str | None) -> tuple[int, int, int, int] | None:
    if value is None:
        return None
    parts = [int(part.strip()) for part in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("--roi must be formatted as x,y,w,h")
    x, y, w, h = parts
    if min(x, y, w, h) < 0 or w == 0 or h == 0:
        raise argparse.ArgumentTypeError("--roi values must be non-negative and width/height must be positive")
    return x, y, w, h


def import_runtime_dependencies() -> None:
    global Image, ImageDraw, ImageFont, QReader, cv2, np

    try:
        import cv2 as cv2_module
        import numpy as np_module
        from PIL import Image as ImageModule
        from PIL import ImageDraw as ImageDrawModule
        from PIL import ImageFont as ImageFontModule
        from qreader import QReader as QReaderClass
    except (ModuleNotFoundError, ImportError) as exc:
        missing = getattr(exc, "name", None) or "a required package"
        raise RuntimeError(
            f"Missing dependency '{missing}'. Install project dependencies with: "
            "python -m pip install --editable ."
        ) from exc

    cv2 = cv2_module
    np = np_module
    Image = ImageModule
    ImageDraw = ImageDrawModule
    ImageFont = ImageFontModule
    QReader = QReaderClass


def resolve_device(device: str) -> str:
    if device != "auto":
        return device

    try:
        import torch
    except (ModuleNotFoundError, ImportError):
        return "cpu"

    return "cuda:0" if torch.cuda.is_available() else "cpu"


def configure_qreader_device(qreader: QReader, device: str, half: bool) -> None:
    detector = getattr(qreader, "detector", None)
    model = getattr(detector, "model", None)
    if detector is None or model is None:
        raise RuntimeError("QReader detector model is not available; cannot configure YOLO device.")

    try:
        model.to(device)
    except Exception as exc:
        raise RuntimeError(f"Could not move YOLO model to device '{device}': {exc}") from exc

    # qrdet 2.x currently calls YOLO.predict(..., device=None), so force the selected device here.
    try:
        from qrdet import _prepare_input, _yolo_v8_results_to_dict
    except (ModuleNotFoundError, ImportError) as exc:
        raise RuntimeError(f"Could not load qrdet helpers needed for GPU inference: {exc}") from exc

    def detect_with_device(self, image, is_bgr: bool = False, **kwargs):
        prepared_image = _prepare_input(source=image, is_bgr=is_bgr)
        results = self.model.predict(
            source=prepared_image,
            conf=self._conf_th,
            iou=self._nms_iou,
            half=half,
            device=device,
            max_det=100,
            augment=False,
            agnostic_nms=True,
            classes=None,
            verbose=False,
        )
        if len(results) != 1:
            raise RuntimeError(f"Expected 1 result if no batch sent, got {len(results)}")
        parsed = _yolo_v8_results_to_dict(results=results[0], image=prepared_image)
        if kwargs.get("legacy"):
            return self._parse_legacy_results(results=parsed, **kwargs)
        return parsed

    detector.detect = MethodType(detect_with_device, detector)


def load_font(font_path: str | None, font_size: int):
    candidates = [font_path] if font_path else []
    candidates.extend(FONT_CANDIDATES)

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return ImageFont.truetype(candidate, font_size)
    return ImageFont.load_default()


def text_width(draw, text: str, font) -> int:
    left, _, right, _ = draw.textbbox((0, 0), text, font=font)
    return right - left


def wrap_text(text: str, draw, font, max_width: int | None, max_lines: int | None) -> list[str]:
    if not max_width or max_width <= 0 or text_width(draw, text, font) <= max_width:
        return [text]

    lines: list[str] = []
    current = ""
    for char in text:
        candidate = current + char
        if current and text_width(draw, candidate, font) > max_width:
            lines.append(current)
            current = char
            if max_lines and max_lines > 0 and len(lines) == max_lines:
                break
        else:
            current = candidate

    if not (max_lines and max_lines > 0 and len(lines) == max_lines):
        lines.append(current)

    if max_lines and max_lines > 0 and len(lines) > max_lines:
        lines = lines[:max_lines]

    if max_lines and max_lines > 0 and len("".join(lines)) < len(text):
        suffix = "..."
        last_line = lines[-1]
        while last_line and text_width(draw, last_line + suffix, font) > max_width:
            last_line = last_line[:-1]
        lines[-1] = last_line + suffix

    return lines


def count_wrapped_lines(text: str, font_path: str | None, font_size: int, max_width: int, max_lines: int) -> int:
    font = load_font(font_path, font_size)
    image = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(image)
    return len(wrap_text(text, draw, font, max_width, max_lines))


def draw_texts(
    frame: np.ndarray,
    texts: Iterable[
        tuple[str, tuple[int, int], tuple[int, int, int], int, int | None, int | None]
    ],
    font_path: str | None,
) -> np.ndarray:
    text_items = list(texts)
    if not text_items:
        return frame

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_image)
    fonts: dict[int, ImageFont.FreeTypeFont] = {}

    for text, position, bgr_color, font_size, max_width, max_lines in text_items:
        if font_size not in fonts:
            fonts[font_size] = load_font(font_path, font_size)
        rgb_color = (bgr_color[2], bgr_color[1], bgr_color[0])
        x, y = position
        line_height = int(font_size * 1.2)
        for line_index, line in enumerate(wrap_text(text, draw, fonts[font_size], max_width, max_lines)):
            draw.text((x, y + line_index * line_height), line, font=fonts[font_size], fill=rgb_color)

    return cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)


def resize_for_display(
    frame: np.ndarray,
    max_width: int,
    max_height: int,
) -> np.ndarray:
    if max_width <= 0 and max_height <= 0:
        return frame

    height, width = frame.shape[:2]
    width_scale = max_width / width if max_width > 0 else 1.0
    height_scale = max_height / height if max_height > 0 else 1.0
    scale = min(width_scale, height_scale, 1.0)
    if scale >= 1.0:
        return frame

    return cv2.resize(frame, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)


def clip_roi(frame: np.ndarray, roi: tuple[int, int, int, int] | None) -> tuple[np.ndarray, tuple[int, int]]:
    if roi is None:
        return frame, (0, 0)

    height, width = frame.shape[:2]
    x, y, w, h = roi
    x1 = min(max(x, 0), width)
    y1 = min(max(y, 0), height)
    x2 = min(max(x + w, x1), width)
    y2 = min(max(y + h, y1), height)
    return frame[y1:y2, x1:x2], (x1, y1)


def scale_frame(frame: np.ndarray, max_width: int | None) -> tuple[np.ndarray, float]:
    if max_width is None or max_width <= 0 or frame.shape[1] <= max_width:
        return frame, 1.0

    scale = max_width / frame.shape[1]
    resized = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return resized, scale


def unscale_detection(
    detection: dict,
    offset_xy: tuple[int, int],
    scale: float,
) -> tuple[tuple[int, int, int, int] | None, np.ndarray | None]:
    offset = np.array(offset_xy, dtype=np.float32)

    bbox = detection.get("bbox_xyxy")
    full_bbox = None
    if bbox is not None:
        bbox_np = np.array(bbox, dtype=np.float32) / scale
        bbox_np[[0, 2]] += offset[0]
        bbox_np[[1, 3]] += offset[1]
        full_bbox = tuple(int(round(v)) for v in bbox_np.tolist())

    quad = detection.get("quad_xy")
    if quad is None:
        quad = detection.get("padded_quad_xy")
    if quad is None:
        quad = detection.get("polygon_xy")
    full_quad = None
    if quad is not None:
        quad_np = np.array(quad, dtype=np.float32) / scale
        quad_np += offset
        full_quad = quad_np.astype(np.int32)

    return full_bbox, full_quad


def detect_qrs(
    qreader: QReader,
    frame: np.ndarray,
    frame_index: int,
    timestamp_sec: float,
    roi: tuple[int, int, int, int] | None,
    max_width: int | None,
) -> list[QRResult]:
    crop, offset_xy = clip_roi(frame, roi)
    if crop.size == 0:
        return []

    resized_crop, scale = scale_frame(crop, max_width)
    decoded_qrs, detections = qreader.detect_and_decode(
        image=resized_crop,
        return_detections=True,
        is_bgr=True,
    )

    results: list[QRResult] = []
    for text, detection in zip(decoded_qrs, detections):
        bbox, quad = unscale_detection(detection, offset_xy, scale)
        confidence = detection.get("confidence")
        results.append(
            QRResult(
                text=text,
                confidence=float(confidence) if confidence is not None else None,
                bbox_xyxy=bbox,
                quad_xy=quad,
                frame_index=frame_index,
                timestamp_sec=timestamp_sec,
            )
        )
    return results


def draw_results(
    frame: np.ndarray,
    results: Iterable[QRResult],
    roi: tuple[int, int, int, int] | None,
    fps: float,
    last_decoded: str | None,
    font_path: str | None,
    text_max_lines: int,
) -> np.ndarray:
    output = frame.copy()
    texts: list[tuple[str, tuple[int, int], tuple[int, int, int], int, int | None, int | None]] = []

    if roi is not None:
        x, y, w, h = roi
        cv2.rectangle(output, (x, y), (x + w, y + h), (80, 180, 255), 2)
        texts.append(("ROI", (x, max(y - 28, 4)), (80, 180, 255), 20, None, None))

    for result in results:
        label = result.text if result.text else "QR detected, decode failed"
        if result.confidence is not None:
            label = f"{label} | conf {result.confidence:.2f}"

        if result.quad_xy is not None and len(result.quad_xy) >= 4:
            cv2.polylines(output, [result.quad_xy.reshape((-1, 1, 2))], True, (0, 220, 0), 3)

        if result.bbox_xyxy is not None:
            x1, y1, x2, y2 = result.bbox_xyxy
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 255), 2)
            text_y = max(y1 - 28, 4)
            label_width = max(120, output.shape[1] - x1 - 12)
            texts.append((label, (x1, text_y), (0, 255, 255), 22, label_width, 2))

    panel_width = min(output.shape[1] - 8, max(520, int(output.shape[1] * 0.75)))
    last_text = last_decoded if last_decoded else "-"
    last_line = f"Last QR: {last_text}"
    panel_text_width = max(160, panel_width - 36)
    wrapped_lines = count_wrapped_lines(last_line, font_path, 20, panel_text_width, text_max_lines)
    panel_height = min(output.shape[0] - 8, max(72, 42 + wrapped_lines * 24 + 10))
    cv2.rectangle(output, (8, 8), (panel_width, panel_height), (0, 0, 0), -1)
    texts.append((f"FPS: {fps:.1f}", (18, 14), (255, 255, 255), 22, None, None))
    texts.append((last_line, (18, 42), (255, 255, 255), 20, panel_text_width, text_max_lines))
    return draw_texts(output, texts, font_path)


def write_csv_header(path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["frame_index", "timestamp_sec", "text", "confidence", "bbox_xyxy"])


def append_csv(path: str, results: Iterable[QRResult]) -> None:
    with open(path, "a", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.writer(csv_file)
        for result in results:
            if result.text is None:
                continue
            writer.writerow(
                [
                    result.frame_index,
                    f"{result.timestamp_sec:.3f}",
                    result.text,
                    "" if result.confidence is None else f"{result.confidence:.4f}",
                    "" if result.bbox_xyxy is None else ",".join(str(v) for v in result.bbox_xyxy),
                ]
            )


def open_capture(source: int | str) -> cv2.VideoCapture:
    if isinstance(source, int):
        capture = cv2.VideoCapture(source, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
    else:
        capture = cv2.VideoCapture(source)

    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")
    return capture


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect and decode QR codes from a moving logistics label in a camera or video stream."
    )
    parser.add_argument("--source", default="0", help="Camera index, RTSP/HTTP URL, or video file path. Default: 0")
    parser.add_argument("--model-size", default="s", choices=("n", "s", "m", "l"), help="QReader model size.")
    parser.add_argument("--confidence", type=float, default=0.5, help="Minimum QR detection confidence.")
    parser.add_argument("--detect-every", type=int, default=1, help="Run detection every N frames.")
    parser.add_argument("--max-width", type=int, default=0, help="Resize detection input to this max width. Use 0 to disable.")
    parser.add_argument("--roi", type=parse_roi, help="Optional detection ROI as x,y,w,h in original frame coordinates.")
    parser.add_argument("--device", default="auto", help="YOLO device: auto, cpu, cuda, cuda:0, etc. Default: auto")
    parser.add_argument("--half", action="store_true", help="Use FP16 YOLO inference. Only use on CUDA devices.")
    parser.add_argument("--font", help="Optional TrueType/OpenType font path for Chinese visualization text.")
    parser.add_argument("--text-max-lines", type=int, default=4, help="Max lines for the Last QR text. Use 0 for no limit.")
    parser.add_argument("--display-width", type=int, default=1280, help="Initial preview max width. Use 0 to disable.")
    parser.add_argument("--display-height", type=int, default=900, help="Initial preview max height. Use 0 to disable.")
    parser.add_argument("--output", help="Optional path to save the annotated video.")
    parser.add_argument("--csv", help="Optional path to save decoded QR records.")
    parser.add_argument("--no-window", action="store_true", help="Process without showing the visualization window.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.detect_every < 1:
        parser.error("--detect-every must be >= 1")
    if args.text_max_lines < 0:
        parser.error("--text-max-lines must be >= 0")
    if args.display_width < 0 or args.display_height < 0:
        parser.error("--display-width and --display-height must be >= 0")

    import_runtime_dependencies()

    device = resolve_device(args.device)
    if args.half and not device.startswith("cuda"):
        parser.error("--half is only supported with CUDA devices.")

    source = parse_source(args.source)
    capture = open_capture(source)
    qreader = QReader(model_size=args.model_size, min_confidence=args.confidence)
    configure_qreader_device(qreader, device=device, half=args.half)
    print(f"YOLO device: {device}", flush=True)

    writer = None
    output_fps = None
    if args.output:
        output_fps = capture.get(cv2.CAP_PROP_FPS)
        if not output_fps or output_fps <= 0:
            output_fps = 25.0

    if args.csv:
        write_csv_header(args.csv)

    frame_index = 0
    prev_time = time.perf_counter()
    fps = 0.0
    last_results: list[QRResult] = []
    last_decoded: str | None = None
    window_created = False

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            now = time.perf_counter()
            elapsed = max(now - prev_time, 1e-6)
            fps = 0.9 * fps + 0.1 * (1.0 / elapsed) if fps else 1.0 / elapsed
            prev_time = now

            timestamp_msec = capture.get(cv2.CAP_PROP_POS_MSEC)
            timestamp_sec = timestamp_msec / 1000.0 if timestamp_msec >= 0 else 0.0

            if frame_index % args.detect_every == 0:
                last_results = detect_qrs(
                    qreader=qreader,
                    frame=frame,
                    frame_index=frame_index,
                    timestamp_sec=timestamp_sec,
                    roi=args.roi,
                    max_width=args.max_width,
                )
                decoded_values = [result.text for result in last_results if result.text]
                if decoded_values:
                    last_decoded = decoded_values[-1]
                    print(f"[frame {frame_index}] {last_decoded}", flush=True)
            if args.csv:
                append_csv(args.csv, last_results)

            annotated = draw_results(
                frame,
                last_results,
                args.roi,
                fps,
                last_decoded,
                args.font,
                args.text_max_lines,
            )

            if args.output and writer is None:
                height, width = annotated.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(args.output, fourcc, output_fps, (width, height))
                if not writer.isOpened():
                    raise RuntimeError(f"Could not create output video: {args.output}")

            if writer is not None:
                writer.write(annotated)

            if not args.no_window:
                display_frame = resize_for_display(annotated, args.display_width, args.display_height)
                if not window_created:
                    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(WINDOW_NAME, display_frame.shape[1], display_frame.shape[0])
                    window_created = True
                cv2.imshow(WINDOW_NAME, display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

            frame_index += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if not args.no_window:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
