# Open-Vocabulary 3D Semantic Reconstruction V2 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Upgrade the V1 point-based reconstruction pipeline into a scene-stable open-vocabulary 3D semantic mapping system with sparse voxel/TSDF fusion, scale-aware world alignment, region-level querying, and long-sequence chunked processing.

**Architecture:** Keep the V1 front-end interfaces for frame extraction, COLMAP poses, PPD depth, and V1's depth-to-COLMAP alignment, but replace the final scene representation with a sparse semantic voxel map. Geometry fusion becomes confidence-weighted and incremental; semantics are stored as fused embeddings per voxel or region rather than only per point. V2 also adds optional similarity alignment from DJI subtitle metadata so the reconstructed scene can have a more meaningful world scale, gravity axis, and future georegistration path.

**Tech Stack:** Python, PyTorch, NumPy, OpenCV, Open3D, COLMAP CLI, pytest, sparse voxel data structures implemented in Python first and optimized later, plus the same open-vocabulary backend selected in V1.

---

## Design Choice

### Alternatives Considered
- Improve V1 point cloud only: cheapest, but still noisy, duplicate-heavy, and weak for region-level semantics.
- Sparse voxel/TSDF semantic map: better scene stability, better multi-view fusion, still debuggable on one 16GB GPU. Recommended.
- Neural field or Gaussian scene representation: stronger expressiveness, but too expensive and too hard to debug at this stage.

### Recommendation
- Use a sparse voxel semantic map in V2.
- Keep V1 point export as a debug artifact, but treat voxel and region outputs as the primary scene representation.

## Scope

### In Scope for V2
- Sparse voxel or TSDF-lite scene representation.
- Confidence-weighted geometric fusion and semantic fusion.
- Optional COLMAP-to-DJI similarity alignment for scale and gravity.
- Build on top of V1 depth-to-COLMAP alignment rather than replacing it.
- Long-sequence chunking into local submaps plus global merge.
- Region-level semantic aggregation and open-vocabulary querying.
- Better export formats: voxel map, semantic mesh or semantic point cloud, region metadata.
- Regression tests for geometry alignment, fusion stability, and query behavior.

### Explicitly Out of Scope for V2
- Fully learned neural scene representation.
- End-to-end retraining of PPD.
- Real-time online SLAM.
- Multi-session map persistence across different flights.
- Large-benchmark training or publication-grade evaluation.

## Resource Budget

- Assume one local GPU with 16GB VRAM.
- Keep all heavy front-end inference frame-streamed as in V1.
- Store global map state on CPU in sparse arrays or chunk files.
- Fuse local windows incrementally and evict intermediate dense buffers as soon as each chunk is merged.

## Directory Plan

### New or expanded modules
- Modify: `ppd/reconstruction/config.py`
- Create: `ppd/reconstruction/scale_alignment.py`
- Create: `ppd/reconstruction/voxel_map.py`
- Create: `ppd/reconstruction/tsdf_fusion.py`
- Create: `ppd/reconstruction/chunk_manager.py`
- Create: `ppd/reconstruction/mesh_export.py`
- Create: `ppd/semantics/confidence.py`
- Create: `ppd/semantics/voxel_semantics.py`
- Create: `ppd/semantics/region_graph.py`
- Create: `ppd/semantics/region_query.py`

### CLI, docs, tests
- Modify: `tools/run_open_vocab_reconstruction.py`
- Create: `tests/reconstruction/test_scale_alignment.py`
- Create: `tests/reconstruction/test_voxel_map.py`
- Create: `tests/reconstruction/test_tsdf_fusion.py`
- Create: `tests/reconstruction/test_chunk_manager.py`
- Create: `tests/reconstruction/test_config_v2.py`
- Create: `tests/semantics/test_confidence.py`
- Create: `tests/semantics/test_voxel_semantics.py`
- Create: `tests/semantics/test_region_graph.py`
- Create: `tests/semantics/test_region_query.py`
- Modify: `README.md`

## Task 1: Extend reconstruction config for V2 modes

**Files:**
- Modify: `ppd/reconstruction/config.py`
- Test: `tests/reconstruction/test_config_v2.py`

**Step 1: Write the failing test**

```python
from pathlib import Path

from ppd.reconstruction.config import ReconstructionConfig


def test_config_exposes_v2_representation_settings(tmp_path: Path):
    cfg = ReconstructionConfig(
        video_path=tmp_path / "clip.mp4",
        workspace=tmp_path / "workspace",
        representation="voxel",
        voxel_size=0.1,
        chunk_size=32,
    )

    assert cfg.representation == "voxel"
    assert cfg.voxel_size == 0.1
    assert cfg.chunk_size == 32
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/reconstruction/test_config_v2.py -q`
Expected: FAIL because the new config fields are missing

**Step 3: Write minimal implementation**

```python
@dataclass(frozen=True)
class ReconstructionConfig:
    ...
    representation: str = "point"
    voxel_size: float = 0.10
    chunk_size: int = 32
    enable_scale_alignment: bool = False
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/reconstruction/test_config_v2.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add ppd/reconstruction/config.py tests/reconstruction/test_config_v2.py
git commit -m "feat: extend reconstruction config for v2 scene mapping"
```

## Task 2: Add COLMAP-to-DJI scale and gravity alignment

**Files:**
- Create: `ppd/reconstruction/scale_alignment.py`
- Test: `tests/reconstruction/test_scale_alignment.py`

**Step 1: Write the failing tests**

```python
import numpy as np

from ppd.reconstruction.scale_alignment import solve_similarity_scale


def test_solve_similarity_scale_recovers_uniform_scale():
    colmap_positions = np.array([[0.0], [1.0], [2.0]], dtype=np.float64)
    dji_positions = np.array([[0.0], [2.0], [4.0]], dtype=np.float64)
    scale = solve_similarity_scale(colmap_positions, dji_positions)
    assert np.isclose(scale, 2.0)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/reconstruction/test_scale_alignment.py -q`
Expected: FAIL because alignment code does not exist

**Step 3: Write minimal implementation**

```python
def solve_similarity_scale(colmap_positions, dji_positions):
    src_extent = float(colmap_positions.max() - colmap_positions.min())
    dst_extent = float(dji_positions.max() - dji_positions.min())
    return dst_extent / max(src_extent, 1e-8)
```

Implementation detail:
- Start with scale-only estimation from matched timestamp samples.
- Extend to full similarity transform with gravity alignment once the data path is stable.
- Keep alignment optional behind `enable_scale_alignment`; never block reconstruction if subtitle metadata is sparse.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/reconstruction/test_scale_alignment.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add ppd/reconstruction/scale_alignment.py tests/reconstruction/test_scale_alignment.py
git commit -m "feat: add optional scale alignment from dji metadata"
```

## Task 3: Introduce sparse voxel map core

**Files:**
- Create: `ppd/reconstruction/voxel_map.py`
- Test: `tests/reconstruction/test_voxel_map.py`

**Step 1: Write the failing tests**

```python
import numpy as np

from ppd.reconstruction.voxel_map import SparseVoxelMap


def test_sparse_voxel_map_merges_points_in_same_voxel():
    voxel_map = SparseVoxelMap(voxel_size=0.1, feature_dim=2)
    points = np.array([[0.01, 0.01, 0.01], [0.02, 0.02, 0.02]], dtype=np.float32)
    colors = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    features = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)

    voxel_map.integrate(points, colors, features)

    assert voxel_map.num_voxels == 1
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/reconstruction/test_voxel_map.py -q`
Expected: FAIL because the voxel map does not exist

**Step 3: Write minimal implementation**

```python
class SparseVoxelMap:
    def __init__(self, voxel_size: float, feature_dim: int):
        self.voxel_size = voxel_size
        self.feature_dim = feature_dim
        self._store = {}

    @property
    def num_voxels(self):
        return len(self._store)

    def integrate(self, points, colors, features):
        keys = np.floor(points / self.voxel_size).astype(np.int64)
        for key, point, color, feat in zip(keys, points, colors, features, strict=True):
            self._store.setdefault(tuple(key.tolist()), {"count": 0})
            self._store[tuple(key.tolist())]["count"] += 1
```

Implementation detail:
- V2 voxel entries should ultimately store:
  - fused position or TSDF state
  - color mean
  - semantic embedding mean
  - observation count
  - optional normal and confidence
- Keep the first implementation pure Python + NumPy for correctness, then optimize only if needed.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/reconstruction/test_voxel_map.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add ppd/reconstruction/voxel_map.py tests/reconstruction/test_voxel_map.py
git commit -m "feat: add sparse voxel map core"
```

## Task 4: Add TSDF-lite or occupancy-weighted geometry fusion

**Files:**
- Create: `ppd/reconstruction/tsdf_fusion.py`
- Test: `tests/reconstruction/test_tsdf_fusion.py`

**Step 1: Write the failing tests**

```python
import numpy as np

from ppd.reconstruction.tsdf_fusion import update_tsdf_value


def test_update_tsdf_value_computes_running_average():
    value, weight = update_tsdf_value(old_value=0.2, old_weight=2.0, new_value=0.8, new_weight=1.0)
    assert np.isclose(value, 0.4)
    assert np.isclose(weight, 3.0)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/reconstruction/test_tsdf_fusion.py -q`
Expected: FAIL because TSDF fusion code does not exist

**Step 3: Write minimal implementation**

```python
def update_tsdf_value(old_value, old_weight, new_value, new_weight):
    total_weight = old_weight + new_weight
    fused_value = (old_value * old_weight + new_value * new_weight) / total_weight
    return fused_value, total_weight
```

Implementation detail:
- If full TSDF is too heavy for the first V2 pass, start with occupancy-weighted voxel fusion and keep the module boundary stable.
- The integration API should not expose whether the backend is TSDF or occupancy averaging.

**Step 4: Run tests to verify it passes**

Run: `pytest tests/reconstruction/test_tsdf_fusion.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add ppd/reconstruction/tsdf_fusion.py tests/reconstruction/test_tsdf_fusion.py
git commit -m "feat: add tsdf lite fusion primitives"
```

## Task 5: Add chunked local-submap processing for long videos

**Files:**
- Create: `ppd/reconstruction/chunk_manager.py`
- Test: `tests/reconstruction/test_chunk_manager.py`
- Modify: `tools/run_open_vocab_reconstruction.py`

**Step 1: Write the failing tests**

```python
from ppd.reconstruction.chunk_manager import split_into_overlapping_chunks


def test_split_into_overlapping_chunks():
    chunks = split_into_overlapping_chunks(total_frames=10, chunk_size=4, overlap=1)
    assert chunks == [(0, 4), (3, 7), (6, 10)]
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/reconstruction/test_chunk_manager.py -q`
Expected: FAIL because chunk management code does not exist

**Step 3: Write minimal implementation**

```python
def split_into_overlapping_chunks(total_frames, chunk_size, overlap):
    chunks = []
    start = 0
    step = chunk_size - overlap
    while start < total_frames:
        end = min(start + chunk_size, total_frames)
        chunks.append((start, end))
        if end == total_frames:
            break
        start += step
    return chunks
```

Implementation detail:
- Each chunk builds a local voxel or point submap.
- Chunk merge uses shared global poses, not ICP-only alignment.
- This is a CPU-memory control feature as much as a sequence-quality feature.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/reconstruction/test_chunk_manager.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add ppd/reconstruction/chunk_manager.py tests/reconstruction/test_chunk_manager.py tools/run_open_vocab_reconstruction.py
git commit -m "feat: add chunked processing for long sequences"
```

## Task 6: Add confidence-weighted semantic fusion per voxel

**Files:**
- Create: `ppd/semantics/confidence.py`
- Create: `ppd/semantics/voxel_semantics.py`
- Test: `tests/semantics/test_confidence.py`
- Test: `tests/semantics/test_voxel_semantics.py`

**Step 1: Write the failing tests**

```python
import numpy as np

from ppd.semantics.confidence import combine_confidence_terms
from ppd.semantics.voxel_semantics import fuse_embedding


def test_combine_confidence_terms_multiplies_sources():
    value = combine_confidence_terms(depth_conf=0.5, view_conf=0.5, text_conf=1.0)
    assert np.isclose(value, 0.25)


def test_fuse_embedding_returns_running_average():
    fused = fuse_embedding(
        old_embedding=np.array([1.0, 0.0], dtype=np.float32),
        old_weight=2.0,
        new_embedding=np.array([0.0, 1.0], dtype=np.float32),
        new_weight=1.0,
    )
    assert fused[0].shape == (2,)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/semantics/test_confidence.py tests/semantics/test_voxel_semantics.py -q`
Expected: FAIL because the confidence fusion modules do not exist

**Step 3: Write minimal implementation**

```python
def combine_confidence_terms(depth_conf, view_conf, text_conf):
    return depth_conf * view_conf * text_conf


def fuse_embedding(old_embedding, old_weight, new_embedding, new_weight):
    total = old_weight + new_weight
    embedding = (old_embedding * old_weight + new_embedding * new_weight) / total
    return embedding, total
```

Implementation detail:
- Confidence terms can include depth confidence, viewing angle, semantic entropy, and mask cleanliness.
- Keep the API scalar-first so later calibration changes do not leak through the codebase.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/semantics/test_confidence.py tests/semantics/test_voxel_semantics.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add ppd/semantics/confidence.py ppd/semantics/voxel_semantics.py tests/semantics/test_confidence.py tests/semantics/test_voxel_semantics.py
git commit -m "feat: add confidence weighted voxel semantic fusion"
```

## Task 7: Extract semantic regions and support region-level querying

**Files:**
- Create: `ppd/semantics/region_graph.py`
- Create: `ppd/semantics/region_query.py`
- Test: `tests/semantics/test_region_graph.py`
- Test: `tests/semantics/test_region_query.py`

**Step 1: Write the failing tests**

```python
import numpy as np

from ppd.semantics.region_graph import build_region_ids
from ppd.semantics.region_query import score_regions


def test_build_region_ids_groups_adjacent_equal_labels():
    labels = np.array([1, 1, 2, 2], dtype=np.int32)
    neighbors = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2]}
    region_ids = build_region_ids(labels, neighbors)
    assert region_ids[0] == region_ids[1]


def test_score_regions_returns_one_score_per_region():
    region_features = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    text_feature = np.array([[1.0, 0.0]], dtype=np.float32)
    scores = score_regions(region_features, text_feature)
    assert scores.shape == (2, 1)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/semantics/test_region_graph.py tests/semantics/test_region_query.py -q`
Expected: FAIL because region modules do not exist

**Step 3: Write minimal implementation**

```python
def build_region_ids(labels, neighbors):
    return np.arange(len(labels), dtype=np.int32)


def score_regions(region_features, text_feature):
    region_features = region_features / np.linalg.norm(region_features, axis=1, keepdims=True)
    text_feature = text_feature / np.linalg.norm(text_feature, axis=1, keepdims=True)
    return region_features @ text_feature.T
```

Implementation detail:
- Region formation can start with connected components over voxel adjacency plus coarse argmax semantic labels.
- Region-level query is a major V2 improvement because it returns coherent 3D areas instead of scattered points.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/semantics/test_region_graph.py tests/semantics/test_region_query.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add ppd/semantics/region_graph.py ppd/semantics/region_query.py tests/semantics/test_region_graph.py tests/semantics/test_region_query.py
git commit -m "feat: add region level semantic query"
```

## Task 8: Add V2 exports and CLI modes

**Files:**
- Modify: `tools/run_open_vocab_reconstruction.py`
- Create: `ppd/reconstruction/mesh_export.py`
- Modify: `README.md`

**Step 1: Write the failing smoke test**

```python
from tools.run_open_vocab_reconstruction import build_arg_parser


def test_cli_supports_v2_voxel_mode():
    parser = build_arg_parser()
    args = parser.parse_args(["--video-path", "clip.mp4", "--workspace", "out", "--mode", "v2", "--representation", "voxel"])
    assert args.mode == "v2"
    assert args.representation == "voxel"
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/reconstruction/test_chunk_manager.py tests/semantics/test_region_query.py -q`
Expected: FAIL because the CLI does not expose the new flags

**Step 3: Write minimal implementation**

```python
parser.add_argument("--mode", choices=["v1", "v2"], default="v1")
parser.add_argument("--representation", choices=["point", "voxel"], default="point")
parser.add_argument("--chunk-size", type=int, default=32)
parser.add_argument("--overlap", type=int, default=4)
parser.add_argument("--enable-scale-alignment", action="store_true")
```

Implementation detail:
- V2 should export:
  - `scene_voxels.npz`
  - `scene_semantic_regions.json`
  - `scene_mesh.ply` or `scene_surface.ply`
  - query-specific region and voxel visualizations
- Keep V1 and V2 behind the same top-level CLI so users can compare them on the same video.

**Step 4: Run focused tests**

Run: `pytest tests/reconstruction tests/semantics -q`
Expected: PASS

**Step 5: Run a manual V2 smoke command**

Run:

```bash
python tools/run_open_vocab_reconstruction.py \
  --video-path /path/to/DJI_0544.MP4 \
  --workspace outputs/dji_0544_v2 \
  --mode v2 \
  --representation voxel \
  --frame-stride 15 \
  --chunk-size 24 \
  --queries tree road building
```

Expected:
- `outputs/dji_0544_v2/scene_voxels.npz`
- `outputs/dji_0544_v2/scene_semantic_regions.json`
- `outputs/dji_0544_v2/query_tree_regions.ply`

**Step 6: Commit**

```bash
git add tools/run_open_vocab_reconstruction.py ppd/reconstruction/mesh_export.py README.md
git commit -m "feat: add v2 voxel semantic mapping mode"
```

## Verification Checklist

- Run: `pytest tests/reconstruction tests/semantics -q`
- Run both V1 and V2 on the same DJI clip and compare artifact counts and visual coherence.
- Verify that V2 reduces duplicate surfaces relative to V1.
- Verify that region-level queries return contiguous scene structures, not isolated points.
- Inspect whether scale alignment changes only world scaling or orientation and does not damage local geometry.

## Risks and Mitigations

- **Voxel memory growth:** use sparse dictionaries first, add chunk flushing and on-disk serialization early.
- **Semantic oversmoothing:** keep observation counts and preserve per-voxel variance so later filtering can reject uncertain regions.
- **Weak DJI metadata quality:** make scale alignment optional and diagnostic-first.
- **Surface extraction complexity:** keep mesh export optional; semantic voxel and semantic point exports are sufficient for first validation.
- **Implementation spread:** reuse V1 interfaces aggressively instead of creating a second independent pipeline.

## Recommended Execution Order

1. Extend config and add scale alignment hooks.
2. Build sparse voxel map and TSDF-lite fusion primitives.
3. Add chunk manager and run V2 geometry-only smoke tests.
4. Add confidence-weighted semantic fusion.
5. Add region extraction and region-level querying.
6. Wire CLI exports and compare V1 vs V2 on the DJI sample.
