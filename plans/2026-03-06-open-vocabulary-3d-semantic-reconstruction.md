# Open-Vocabulary 3D Semantic Reconstruction Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a V1 prototype pipeline that takes a DJI video, reconstructs a unified scene point cloud with COLMAP poses + Pixel-Perfect Depth depths, and attaches open-vocabulary semantic embeddings that support coarse text-based 3D querying/export.

**Architecture:** Keep V1 point-based. Use COLMAP for camera intrinsics/extrinsics, `PixelPerfectDepth.infer_image()` for dense per-frame depth, lightweight depth-to-COLMAP alignment before world-space reprojection, Open3D for fusion/export, and a text-aligned 2D encoder for patch-level semantic embeddings that are projected onto 3D points. Execute the heavy stages in a streaming/chunked way so a single 16GB GPU is enough: process one frame or a very small batch at a time, flush depth/semantic artifacts to disk, and keep fused scene state on CPU. Keep DJI `.SRT` parsing auxiliary in V1: store metadata for future scale/georegistration, but do not make reconstruction depend on subtitle quality. Structure V1 around representation-agnostic interfaces so V2 can replace point fusion with sparse voxel or TSDF-lite fusion without rewriting the whole pipeline.

**Tech Stack:** Python, PyTorch, OpenCV, Open3D, NumPy, COLMAP CLI, pytest, `transformers` or another text-image model backend selected behind a semantic encoder interface.

---

## Scope

## V1 Success Criteria

- Operate on a short curated clip rather than an arbitrarily long flight.
- Produce a roughly aligned fused scene point cloud that is inspectable in Open3D or MeshLab.
- Support coarse open-vocabulary query visualization over points.
- Validate that the front-end interfaces work together:
  - frame extraction and manifests
  - COLMAP poses
  - PPD depth
  - depth alignment
  - 3D projection
  - semantic projection

## V1 Non-Goals

- Stable long-sequence semantic mapping.
- Region-level coherent semantic outputs.
- High-confidence real-world metric scale or georegistration.
- Publication-grade geometry quality.

### In Scope for V1
- Video frame extraction and workspace preparation.
- Optional DJI subtitle parsing into normalized metadata records.
- COLMAP command orchestration and model parsing.
- Dense depth inference cache for selected frames.
- Depth-scale alignment against COLMAP-visible sparse observations before world-space fusion.
- Reprojection of RGB-depth pixels into world coordinates using COLMAP poses.
- Scene-level point fusion, downsampling, filtering, and `.ply` export.
- Open-vocabulary 2D semantic backend abstraction.
- Projection and accumulation of semantic embeddings from 2D into 3D points.
- Text-query scoring over fused 3D points and export of colored query results.
- Targeted automated tests for pure parsing, geometry, and scoring modules, plus smoke tests for heavier pipeline boundaries.
- Streaming execution: never keep the full video, all depths, or all semantic features resident on GPU.
- Stable manifests and observation bookkeeping that V2 can reuse for scale alignment, chunking, and voxel/region fusion.

### Explicitly Out of Scope for V1
- TSDF fusion and mesh extraction.
- End-to-end neural field training.
- Real-time reconstruction.
- Absolute geo-registration using GPS as a hard constraint.
- Dense benchmark evaluation on public 3D semantic datasets.

## Resource Budget

- Assume one local GPU with 16GB VRAM.
- Prefer the image model path (`PixelPerfectDepth`) over video-window models for reconstruction because it is easier to run frame-by-frame.
- GPU stages must be frame-streamed:
  - decode one frame or a tiny batch
  - run depth or semantic encoding
  - move outputs to CPU/disk
  - free GPU memory before the next frame
- Fusion and semantic accumulation should run on CPU-backed arrays or chunked on-disk artifacts.
- The CLI should expose `--max-frames`, `--frame-stride`, and semantic feature downsampling controls so the user can trade coverage for memory.

## Testing Strategy

- Use selective TDD, not blanket TDD.
- Write focused unit tests for:
  - parsers and manifests
  - pose and camera math
  - depth-alignment math
  - reprojection math
  - semantic scoring and aggregation
- For heavyweight modules that depend on checkpoints, external tools, or long pipelines:
  - prefer smoke tests, invariants, and manual artifact inspection
  - avoid building a test suite larger than the feature itself
- The goal of V1 testing is to protect interfaces and math, not to simulate the full reconstruction stack in detail.

## V2 Compatibility Constraints

- Do not hard-code point-cloud assumptions into the top-level orchestration layer; keep scene fusion behind a replaceable interface.
- Persist frame IDs, timestamps, camera IDs, pose file paths, and optional DJI metadata in a normalized manifest.
- Keep per-observation metadata where cheap:
  - source frame ID
  - source pixel or patch index
  - observation count
  - confidence placeholder
- Separate geometry fusion, semantic fusion, and export code so V2 can swap scene representation without changing front-end inference.

## Directory Plan

### New Python packages
- Create: `ppd/reconstruction/__init__.py`
- Create: `ppd/reconstruction/config.py`
- Create: `ppd/reconstruction/frame_extractor.py`
- Create: `ppd/reconstruction/dji_srt.py`
- Create: `ppd/reconstruction/colmap_runner.py`
- Create: `ppd/reconstruction/pose_io.py`
- Create: `ppd/reconstruction/depth_inference.py`
- Create: `ppd/reconstruction/depth_alignment.py`
- Create: `ppd/reconstruction/reproject.py`
- Create: `ppd/reconstruction/fusion.py`
- Create: `ppd/reconstruction/io.py`
- Create: `ppd/semantics/__init__.py`
- Create: `ppd/semantics/open_vocab.py`
- Create: `ppd/semantics/project_to_3d.py`
- Create: `ppd/semantics/query.py`

### New CLI and tests
- Create: `tools/run_open_vocab_reconstruction.py`
- Create: `tests/reconstruction/test_config.py`
- Create: `tests/reconstruction/test_frame_extractor.py`
- Create: `tests/reconstruction/test_pose_io.py`
- Create: `tests/reconstruction/test_reproject.py`
- Create: `tests/reconstruction/test_depth_alignment.py`
- Create: `tests/reconstruction/test_fusion.py`
- Create: `tests/semantics/test_project_to_3d.py`
- Create: `tests/semantics/test_query.py`
- Modify: `requirements.txt`
- Modify: `README.md`

## Task 1: Add reconstruction config and workspace models

**Files:**
- Create: `ppd/reconstruction/__init__.py`
- Create: `ppd/reconstruction/config.py`
- Test: `tests/reconstruction/test_config.py`

**Step 1: Write the failing test**

```python
from pathlib import Path

from ppd.reconstruction.config import ReconstructionConfig


def test_config_builds_workspace_paths(tmp_path: Path):
    cfg = ReconstructionConfig(
        video_path=tmp_path / "clip.mp4",
        workspace=tmp_path / "workspace",
    )

    assert cfg.frames_dir == tmp_path / "workspace" / "frames"
    assert cfg.depth_dir == tmp_path / "workspace" / "depth"
    assert cfg.colmap_dir == tmp_path / "workspace" / "colmap"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/reconstruction/test_config.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'ppd.reconstruction'`

**Step 3: Write minimal implementation**

```python
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ReconstructionConfig:
    video_path: Path
    workspace: Path
    frame_stride: int = 10
    max_frames: int | None = None

    @property
    def frames_dir(self) -> Path:
        return self.workspace / "frames"

    @property
    def depth_dir(self) -> Path:
        return self.workspace / "depth"

    @property
    def colmap_dir(self) -> Path:
        return self.workspace / "colmap"
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/reconstruction/test_config.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/reconstruction/test_config.py ppd/reconstruction/__init__.py ppd/reconstruction/config.py
git commit -m "feat: add reconstruction config models"
```

## Task 2: Extract frames and optional DJI subtitle metadata

**Files:**
- Create: `ppd/reconstruction/frame_extractor.py`
- Create: `ppd/reconstruction/dji_srt.py`
- Test: `tests/reconstruction/test_frame_extractor.py`

**Step 1: Write the failing tests**

```python
from ppd.reconstruction.dji_srt import parse_srt_record
from ppd.reconstruction.frame_extractor import build_frame_manifest


def test_parse_srt_record_extracts_timestamp_and_altitude():
    block = "1\n00:00:00,000 --> 00:00:00,033\nHOME(31.0,121.0,45.3) REL_ALT:12.4"
    record = parse_srt_record(block)
    assert record.rel_altitude_m == 12.4


def test_build_frame_manifest_respects_stride():
    manifest = build_frame_manifest(total_frames=12, stride=5)
    assert manifest == [0, 5, 10]
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/reconstruction/test_frame_extractor.py -q`
Expected: FAIL because the modules do not exist

**Step 3: Write minimal implementation**

```python
import re


def build_frame_manifest(total_frames: int, stride: int) -> list[int]:
    return list(range(0, total_frames, stride))


def parse_srt_record(block: str):
    match = re.search(r"REL_ALT:(?P<alt>-?\d+(?:\.\d+)?)", block)
    altitude = float(match.group("alt")) if match else None
    return type("SrtRecord", (), {"rel_altitude_m": altitude})()
```

Implementation detail:
- Reuse `ppd/utils/video_utils.py` to decode frames.
- Save a JSON manifest containing `frame_index`, `timestamp_sec`, `image_path`, and optional DJI metadata fields.
- Include stable identifiers that V2 can reuse directly:
  - `frame_id`
  - `image_name`
  - `timestamp_sec`
  - `camera_id`
  - optional subtitle-derived metadata
- Keep subtitle parsing best-effort; never fail the run because an SRT block is malformed.
- Do not retain decoded video frames in a giant in-memory list for the reconstruction path; expose an iterator/generator form for streaming extraction.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/reconstruction/test_frame_extractor.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/reconstruction/test_frame_extractor.py ppd/reconstruction/frame_extractor.py ppd/reconstruction/dji_srt.py
git commit -m "feat: add frame extraction and srt parsing"
```

## Task 3: Orchestrate COLMAP and parse camera poses

**Files:**
- Create: `ppd/reconstruction/colmap_runner.py`
- Create: `ppd/reconstruction/pose_io.py`
- Test: `tests/reconstruction/test_pose_io.py`

**Step 1: Write the failing tests**

```python
from ppd.reconstruction.pose_io import parse_images_text


def test_parse_images_text_returns_pose_by_filename():
    images_txt = """
1 1 0 0 0 0 0 0 1 frame_0000.png

0 0 0
"""
    poses = parse_images_text(images_txt)
    assert "frame_0000.png" in poses
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/reconstruction/test_pose_io.py -q`
Expected: FAIL because parser and runner do not exist

**Step 3: Write minimal implementation**

```python
def build_colmap_commands(image_dir: str, workspace: str) -> list[list[str]]:
    return [
        ["colmap", "feature_extractor", "--database_path", f"{workspace}/database.db", "--image_path", image_dir],
        ["colmap", "exhaustive_matcher", "--database_path", f"{workspace}/database.db"],
        ["colmap", "mapper", "--database_path", f"{workspace}/database.db", "--image_path", image_dir, "--output_path", f"{workspace}/sparse"],
    ]
```

Implementation detail:
- Use `subprocess.run(..., check=True)` wrappers and return explicit command lists for testability.
- After sparse mapping, convert binary outputs to text via `colmap model_converter`.
- Parse `cameras.txt` and `images.txt` into normalized intrinsics/extrinsics keyed by frame filename.
- Expose camera-to-world transforms as `4x4 float64` matrices.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/reconstruction/test_pose_io.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/reconstruction/test_pose_io.py ppd/reconstruction/colmap_runner.py ppd/reconstruction/pose_io.py
git commit -m "feat: add colmap orchestration and pose parsing"
```

## Task 4: Cache dense PPD depth for selected frames

**Files:**
- Create: `ppd/reconstruction/depth_inference.py`
- Create: `ppd/reconstruction/io.py`
- Test: `tests/reconstruction/test_depth_inference.py`

**Step 1: Write the failing tests**

```python
import numpy as np

from ppd.reconstruction.depth_inference import infer_depth_batch


class StubDepthModel:
    def infer_image(self, image):
        depth = np.ones((1, 1, image.shape[0], image.shape[1]), dtype=np.float32)
        return depth, image


def test_infer_depth_batch_writes_outputs(tmp_path):
    images = [np.zeros((4, 5, 3), dtype=np.uint8)]
    manifest = [{"image_name": "frame_0000.png"}]
    outputs = infer_depth_batch(StubDepthModel(), images, manifest, tmp_path)
    assert outputs[0]["depth_path"].name == "frame_0000.npy"
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/reconstruction/test_depth_inference.py -q`
Expected: FAIL because module does not exist

**Step 3: Write minimal implementation**

```python
def infer_depth_batch(model, images, manifest, output_dir):
    records = []
    for image, meta in zip(images, manifest, strict=True):
        depth, _ = model.infer_image(image)
        depth_path = output_dir / meta["image_name"].replace(".png", ".npy")
        np.save(depth_path, depth)
        records.append({"image_name": meta["image_name"], "depth_path": depth_path})
    return records
```

Implementation detail:
- Add a thin adapter around `PixelPerfectDepth` so tests can inject a stub model.
- Save both raw `depth.npy` and a small JSON sidecar with original image size, resized image size, and any scale-alignment info.
- Include enough metadata for later V2 reuse:
  - `frame_id`
  - `camera_id`
  - exact intrinsics resolution used for depth
  - checkpoint/model identifier
- Do not load the actual checkpoint in unit tests.
- Ensure each inference call frees temporary tensors immediately; never batch the entire frame manifest.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/reconstruction/test_depth_inference.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/reconstruction/test_depth_inference.py ppd/reconstruction/depth_inference.py ppd/reconstruction/io.py
git commit -m "feat: add depth caching for reconstruction frames"
```

## Task 5: Align dense depth to the COLMAP scene scale

**Files:**
- Create: `ppd/reconstruction/depth_alignment.py`
- Test: `tests/reconstruction/test_depth_alignment.py`

**Step 1: Write the failing test**

```python
import numpy as np

from ppd.reconstruction.depth_alignment import fit_depth_scale


def test_fit_depth_scale_recovers_uniform_scale():
    pred = np.array([1.0, 2.0, 4.0], dtype=np.float32)
    ref = np.array([2.0, 4.0, 8.0], dtype=np.float32)
    scale = fit_depth_scale(pred, ref)
    assert np.isclose(scale, 2.0)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/reconstruction/test_depth_alignment.py -q`
Expected: FAIL because the alignment module does not exist

**Step 3: Write minimal implementation**

```python
def fit_depth_scale(pred_depth, ref_depth):
    pred = np.asarray(pred_depth, dtype=np.float64)
    ref = np.asarray(ref_depth, dtype=np.float64)
    return float(np.dot(pred, ref) / np.dot(pred, pred))
```

Implementation detail:
- This task is geometry-critical, not optional polish.
- Use COLMAP sparse points visible in each frame as the first reference source.
- Start with scale-only fitting; allow extension to robust scale-shift fitting if needed.
- Save alignment parameters per frame or per scene and record the number of supporting correspondences.
- If a frame has too few reliable correspondences, skip it or fall back to a scene-level estimate rather than silently fusing bad geometry.

**Step 4: Run test to verify it passes**

Run: `pytest tests/reconstruction/test_depth_alignment.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/reconstruction/test_depth_alignment.py ppd/reconstruction/depth_alignment.py
git commit -m "feat: add depth alignment against colmap scale"
```

## Task 6: Reproject RGB-depth frames into world-space points

**Files:**
- Create: `ppd/reconstruction/reproject.py`
- Test: `tests/reconstruction/test_reproject.py`

**Step 1: Write the failing tests**

```python
import numpy as np

from ppd.reconstruction.reproject import depth_to_world_points


def test_depth_to_world_points_identity_pose():
    depth = np.ones((2, 2), dtype=np.float32)
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    intrinsics = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]])
    pose = np.eye(4, dtype=np.float32)

    points, colors = depth_to_world_points(depth, rgb, intrinsics, pose)

    assert points.shape == (4, 3)
    assert colors.shape == (4, 3)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/reconstruction/test_reproject.py -q`
Expected: FAIL because reprojection code does not exist

**Step 3: Write minimal implementation**

```python
def depth_to_world_points(depth, rgb, intrinsics, camera_to_world):
    ys, xs = np.indices(depth.shape)
    z = depth.reshape(-1)
    x = (xs.reshape(-1) - intrinsics[0, 2]) * z / intrinsics[0, 0]
    y = (ys.reshape(-1) - intrinsics[1, 2]) * z / intrinsics[1, 1]
    points_cam = np.stack([x, y, z, np.ones_like(z)], axis=1)
    points_world = (camera_to_world @ points_cam.T).T[:, :3]
    colors = rgb.reshape(-1, 3).astype(np.float32) / 255.0
    return points_world, colors
```

Implementation detail:
- Add depth validity masks and max-depth clipping.
- Keep intrinsics consistent with the exact resolution used for depth inference.
- Return lightweight observation metadata along with points:
  - source `frame_id`
  - source pixel indices or flattened pixel ids
  - optional per-point confidence placeholder
- Store frame-level debug outputs for one or two frames to verify coordinate conventions visually.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/reconstruction/test_reproject.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/reconstruction/test_reproject.py ppd/reconstruction/reproject.py
git commit -m "feat: add world-space reprojection"
```

## Task 7: Fuse scene points and export geometry artifacts

**Files:**
- Create: `ppd/reconstruction/fusion.py`
- Test: `tests/reconstruction/test_fusion.py`
- Modify: `tools/run_open_vocab_reconstruction.py`

**Step 1: Write the failing tests**

```python
import numpy as np

from ppd.reconstruction.fusion import fuse_point_chunks


def test_fuse_point_chunks_downsamples_duplicates():
    chunk_a = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]], dtype=np.float32)
    chunk_b = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    fused = fuse_point_chunks([chunk_a, chunk_b], voxel_size=0.05)
    assert fused.shape[0] == 2
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/reconstruction/test_fusion.py -q`
Expected: FAIL because fusion code does not exist

**Step 3: Write minimal implementation**

```python
def fuse_point_chunks(chunks, voxel_size: float):
    stacked = np.concatenate(chunks, axis=0)
    voxel_keys = np.floor(stacked / voxel_size).astype(np.int64)
    _, unique_idx = np.unique(voxel_keys, axis=0, return_index=True)
    return stacked[np.sort(unique_idx)]
```

Implementation detail:
- Start with numpy voxel deduplication in tests and switch to Open3D downsampling in production code.
- Define a generic scene-fusion boundary in this module, for example:
  - `SceneAccumulator`
  - `PointSceneAccumulator`
- Export:
  - `scene_raw.ply`
  - `scene_fused.ply`
  - `frames_debug/frame_XXXX_world.ply`
  - `scene_observations.npz` or equivalent lightweight bookkeeping artifact
- Make filtering parameters configurable from the CLI.
- Fuse incrementally per frame chunk so geometry memory grows on CPU only.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/reconstruction/test_fusion.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/reconstruction/test_fusion.py ppd/reconstruction/fusion.py tools/run_open_vocab_reconstruction.py
git commit -m "feat: add scene fusion and geometry export"
```

## Task 8: Add open-vocabulary 2D encoder abstraction

**Files:**
- Create: `ppd/semantics/__init__.py`
- Create: `ppd/semantics/open_vocab.py`
- Modify: `requirements.txt`

**Step 1: Write the failing test**

```python
import numpy as np

from ppd.semantics.open_vocab import normalize_text_queries


def test_normalize_text_queries_strips_whitespace_and_deduplicates():
    queries = normalize_text_queries([" tree ", "tree", "road"])
    assert queries == ["tree", "road"]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/semantics/test_query.py -q`
Expected: FAIL because the semantics package does not exist

**Step 3: Write minimal implementation**

```python
def normalize_text_queries(queries):
    seen = set()
    normalized = []
    for query in queries:
        value = query.strip()
        if value and value not in seen:
            normalized.append(value)
            seen.add(value)
    return normalized
```

Implementation detail:
- Add a backend interface with methods:
  - `encode_text(list[str]) -> np.ndarray`
  - `encode_image(image: np.ndarray) -> FrameSemanticFeatures`
- Keep the first backend isolated behind the interface. Recommended first choice is a CLIP-family model that exposes patch tokens plus text embeddings; if patch tokens are awkward in the selected library, fall back to patch-crop batching in V1 rather than hard-coding to one API.
- Add the minimum dependency required by the chosen backend and document checkpoint download/setup in `README.md`.
- Semantic encoding must also run frame-by-frame or patch-chunk-by-patch-chunk; do not accumulate all frame embeddings on GPU.

**Step 4: Run test to verify it passes**

Run: `pytest tests/semantics/test_query.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add ppd/semantics/__init__.py ppd/semantics/open_vocab.py requirements.txt
git commit -m "feat: add open vocabulary semantic backend interface"
```

## Task 9: Project 2D semantic embeddings into 3D and query the scene

**Files:**
- Create: `ppd/semantics/project_to_3d.py`
- Create: `ppd/semantics/query.py`
- Test: `tests/semantics/test_project_to_3d.py`
- Test: `tests/semantics/test_query.py`
- Modify: `tools/run_open_vocab_reconstruction.py`
- Modify: `README.md`

**Step 1: Write the failing tests**

```python
import numpy as np

from ppd.semantics.project_to_3d import accumulate_semantic_features
from ppd.semantics.query import score_points_against_text


def test_accumulate_semantic_features_averages_duplicate_votes():
    point_ids = np.array([0, 0, 1])
    features = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    merged = accumulate_semantic_features(point_ids, features, num_points=2)
    assert np.allclose(merged[0], [1.0, 0.0])


def test_score_points_against_text_returns_similarity_per_point():
    point_features = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    text_features = np.array([[1.0, 0.0]], dtype=np.float32)
    scores = score_points_against_text(point_features, text_features)
    assert scores.shape == (2, 1)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/semantics/test_project_to_3d.py tests/semantics/test_query.py -q`
Expected: FAIL because 3D semantic projection code does not exist

**Step 3: Write minimal implementation**

```python
def accumulate_semantic_features(point_ids, features, num_points):
    accum = np.zeros((num_points, features.shape[1]), dtype=np.float32)
    counts = np.zeros((num_points, 1), dtype=np.float32)
    np.add.at(accum, point_ids, features)
    np.add.at(counts, point_ids, 1.0)
    return accum / np.clip(counts, 1.0, None)


def score_points_against_text(point_features, text_features):
    point_features = point_features / np.linalg.norm(point_features, axis=1, keepdims=True)
    text_features = text_features / np.linalg.norm(text_features, axis=1, keepdims=True)
    return point_features @ text_features.T
```

Implementation detail:
- During reprojection, keep the source pixel indices so each 3D point can inherit the corresponding semantic feature.
- Aggregate multi-view semantic votes with running mean plus observation counts.
- Keep semantic fusion logic independent from the concrete scene representation so V2 can move from points to voxels or regions.
- Export:
  - `scene_semantic_features.npy`
  - `query_tree_scores.npy`
  - `query_tree_vis.ply`
- Add README usage examples showing:
  - geometry-only run
  - geometry + semantic run
  - text-query export run
- Keep semantic accumulation in CPU memory or append-only chunks on disk; merge only the reduced per-point state.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/semantics/test_project_to_3d.py tests/semantics/test_query.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/semantics/test_project_to_3d.py tests/semantics/test_query.py ppd/semantics/project_to_3d.py ppd/semantics/query.py tools/run_open_vocab_reconstruction.py README.md
git commit -m "feat: add 3d semantic projection and query pipeline"
```

## Task 10: Wire the end-to-end CLI and verification flow

**Files:**
- Create: `tools/run_open_vocab_reconstruction.py`
- Modify: `README.md`

**Step 1: Write the failing smoke test**

```python
from tools.run_open_vocab_reconstruction import build_arg_parser


def test_build_arg_parser_exposes_geometry_and_semantic_flags():
    parser = build_arg_parser()
    args = parser.parse_args(["--video-path", "clip.mp4", "--workspace", "out", "--queries", "tree", "road"])
    assert args.video_path == "clip.mp4"
    assert args.queries == ["tree", "road"]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/semantics/test_query.py -q`
Expected: FAIL because the CLI parser does not exist yet

**Step 3: Write minimal implementation**

```python
def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--queries", nargs="*", default=[])
    parser.add_argument("--semantic-backend", default="clip")
    return parser
```

Implementation detail:
- Subcommands are unnecessary in V1. A single orchestrator script is enough.
- Add flags for:
  - `--skip-colmap`
  - `--skip-semantics`
  - `--max-frames`
  - `--voxel-size`
  - `--query-threshold`
- The CLI should emit a JSON summary file that records every produced artifact path.
- Keep the CLI/config naming scene-generic rather than point-specific, so V2 can add `voxel` backends without renaming the whole interface.

**Step 4: Run the focused tests**

Run: `pytest tests/reconstruction tests/semantics -q`
Expected: PASS

**Step 5: Run a manual smoke command**

Run:

```bash
python tools/run_open_vocab_reconstruction.py \
  --video-path /path/to/DJI_0544.MP4 \
  --workspace outputs/dji_0544 \
  --frame-stride 15 \
  --queries tree road building
```

Expected:
- `outputs/dji_0544/scene_fused.ply`
- `outputs/dji_0544/scene_semantic_features.npy`
- `outputs/dji_0544/query_tree_vis.ply`

**Step 6: Commit**

```bash
git add tools/run_open_vocab_reconstruction.py README.md tests/reconstruction tests/semantics
git commit -m "feat: add end to end open vocabulary reconstruction cli"
```

## Verification Checklist

- Run: `pytest tests/reconstruction tests/semantics -q`
- Run: `pytest tests/test_depth_anything_cpu_attention.py -q`
- Run the manual smoke command on the DJI sample after checkpoints and COLMAP are available.
- Inspect the fused point cloud in Open3D or MeshLab and confirm view overlap alignment.
- Inspect one text query result and confirm it highlights plausible 3D regions instead of random scatter.

## Risks and Mitigations

- **Monocular scale mismatch:** align dense depth against COLMAP-visible sparse depths before fusion; use DJI metadata later only for world-scale or gravity refinement.
- **Pose/image mismatch:** key all metadata by exact extracted frame filename and write manifests to disk.
- **Heavy model tests:** design semantic and depth modules with dependency injection so unit tests can use stubs.
- **Semantic drift in thin structures:** keep observation counts and optionally downweight grazing-angle observations in a later pass.
- **COLMAP failures on weak-texture frames:** expose frame stride and max frame count so the user can iterate on subsets quickly.
- **VRAM pressure on 16GB GPU:** process one frame or tiny micro-batches at a time, flush intermediate arrays to disk, and provide CLI knobs for stride, max frames, and semantic resolution.

## Recommended Execution Order

1. Tasks 1-7 to make geometry solid and debuggable.
2. Task 10 smoke run with `--skip-semantics` to validate the geometry pipeline on a short DJI segment.
3. Tasks 8-9 to add open-vocabulary semantics.
4. Task 10 full smoke run with text queries.
