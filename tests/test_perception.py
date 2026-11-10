"""Perception: geometry, measurement accuracy, clustering, memory.

The accuracy assertions are tight on purpose. Every one of them corresponds to a decision
the grasp planner makes, and a centimetre of drift there is the difference between a
feasible grasp and a rejected one.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from rung.contracts.action import TCPTarget
from rung.mock import SCENES
from rung.mock.world import top_down_pose
from rung.perception import ObjectMemory, OracleGrounding, PerceptionService
from rung.perception.cloud import (
    connected_component,
    cross_section,
    min_width_direction,
    principal_axis_xy,
    rim_model,
)
from rung.perception.types import Catalog


def build(scene: str = "banana_in_bowl"):
    emb = SCENES[scene]()
    svc = PerceptionService(OracleGrounding(emb))
    return emb, svc, svc.build(emb.observe(), emb.instruction)


def test_catalog_lists_every_visible_body():
    emb, _, cat = build("cluttered_bin")
    assert {o.label for o in cat.objects} == {"apple", "lime", "mug", "bin"}
    assert not cat.failures


def test_measured_size_matches_the_truth_within_a_millimetre():
    """Across every scene, the width the jaw has to span is measured to 1 mm."""
    for name in SCENES:
        emb, _, cat = build(name)
        for o in cat.objects:
            truth = emb.scene.body(o.label)
            expected_long, expected_short = truth.footprint
            assert o.grasp_width == pytest.approx(expected_short, abs=0.002), (name, o.label)
            assert o.grasp_length == pytest.approx(expected_long, abs=0.002), (name, o.label)


def test_measured_top_height_matches_the_truth():
    for name in SCENES:
        emb, _, cat = build(name)
        for o in cat.objects:
            assert o.top_z == pytest.approx(emb.scene.body(o.label).top_z, abs=0.005), o.label


def test_min_width_beats_the_principal_axis_for_a_square_top():
    """A square cross-section has no meaningful principal axis; projecting onto a spurious
    one reports the diagonal, so a 7 cm bottle would measure 9.9 cm."""
    grid = np.array(
        [
            [x, y, 0.0]
            for x in np.linspace(-0.035, 0.035, 15)
            for y in np.linspace(-0.035, 0.035, 15)
        ]
    )
    width, length, _ = min_width_direction(grid)
    assert width == pytest.approx(0.07, abs=0.002)
    assert length == pytest.approx(0.07, abs=0.005)
    assert principal_axis_xy(grid) is None


def test_min_width_finds_the_narrow_direction_of_a_long_object():
    rng = np.random.default_rng(0)
    pts = np.stack(
        [rng.uniform(-0.09, 0.09, 400), rng.uniform(-0.02, 0.02, 400), np.zeros(400)], axis=1
    )
    width, length, closing = min_width_direction(pts)
    assert width == pytest.approx(0.04, abs=0.004)
    assert length == pytest.approx(0.18, abs=0.01)
    assert abs(closing[1]) > 0.95  # closes across the short side, along y


def test_connected_component_separates_neighbours():
    """Two clusters 13 cm apart are not merged."""
    a = np.random.default_rng(1).uniform(-0.02, 0.02, (200, 3))
    b = a + np.array([0.13, 0.0, 0.0])
    both = np.concatenate([a, b])
    got = connected_component(both, seed=a[0], link_m=0.02)
    assert len(got) == pytest.approx(len(a), rel=0.05)
    assert got[:, 0].max() < 0.05


def test_connected_component_returns_nothing_for_a_far_seed():
    pts = np.random.default_rng(2).uniform(0, 0.02, (50, 3))
    assert len(connected_component(pts, seed=[5.0, 5.0, 5.0])) == 0


def test_cross_section_takes_the_slab_below_the_top():
    pts = np.stack([np.zeros(100), np.zeros(100), np.linspace(0, 0.2, 100)], axis=1)
    slab = cross_section(pts, 0.2, slab_m=0.02)
    assert slab[:, 2].min() >= 0.18 - 1e-9


def test_rim_model_uses_the_median_so_a_partial_rim_still_centres():
    rng = np.random.default_rng(3)
    theta = rng.uniform(0, 1.2 * np.pi, 500)  # only part of the rim is visible
    pts = np.stack([0.1 * np.cos(theta), 0.1 * np.sin(theta), rng.uniform(0.09, 0.1, 500)], axis=1)
    centre, rim = rim_model(pts)
    assert rim == pytest.approx(0.1, abs=0.005)
    assert np.linalg.norm(centre[:2]) < 0.09


def test_hand_over_an_object_is_flagged_rather_than_measured_confidently():
    """With the gripper over an object, the hand exclusion removes part of the object too,
    so the measured width understates it. That has to be flagged, not reported as fact."""
    emb = SCENES["banana_in_bowl"]()
    banana = emb.scene.body("banana")
    clear = next(
        o
        for o in PerceptionService(OracleGrounding(emb)).build(emb.observe()).objects
        if o.label == "banana"
    )
    assert not clear.uncertain
    assert clear.grasp_width == pytest.approx(0.045, abs=0.003)

    emb.execute(
        TCPTarget(
            pose=top_down_pose(banana.pos + np.array([0, 0, 0.02]), banana.yaw),
            max_lin_speed_mps=0.3,
            timeout_s=10.0,
        )
    )
    obscured = next(
        o
        for o in PerceptionService(OracleGrounding(emb)).build(emb.observe()).objects
        if o.label == "banana"
    )
    assert obscured.uncertain
    assert "gripper is over it" in obscured.uncertainty_reason
    # never inflated toward the finger spacing, which would invent a too-wide object
    assert obscured.grasp_width < 0.06


def test_perception_is_deterministic():
    _, svc, first = build("cluttered_bin")
    emb2, svc2, second = build("cluttered_bin")
    assert [o.object_id for o in first.objects] == [o.object_id for o in second.objects]
    for a, b in zip(first.objects, second.objects, strict=True):
        assert np.allclose(a.centroid, b.centroid)
        assert a.grasp_width == b.grasp_width


def test_catalog_renders_for_a_prompt():
    _, _, cat = build("tall_bottle_and_crate")
    text = cat.render(jaw_m=0.085)
    assert "bottle" in text and "cm" in text
    assert "wider than the jaw" in text  # the crate is


def test_a_camera_without_depth_yields_an_empty_catalog_with_a_reason():
    emb = SCENES["banana_in_bowl"]()
    obs = emb.observe()
    obs.scene.depth = None
    cat = PerceptionService(OracleGrounding(emb)).build(obs)
    assert cat.objects == []
    assert "no depth" in cat.failures[0]


# ---- memory ---------------------------------------------------------------


def test_memory_keeps_ids_stable_across_frames():
    emb = SCENES["cluttered_bin"]()
    svc = PerceptionService(OracleGrounding(emb))
    mem = ObjectMemory()
    first = mem.update(svc.build(emb.observe()))
    ids = first.ids()
    emb.execute(
        TCPTarget(pose=top_down_pose([0.30, 0.30, 0.25]), max_lin_speed_mps=0.3, timeout_s=10.0)
    )
    second = mem.update(svc.build(emb.observe()))
    assert second.ids() == ids


def test_memory_reports_an_object_that_left_the_view():
    emb = SCENES["cluttered_bin"]()
    svc = PerceptionService(OracleGrounding(emb))
    mem = ObjectMemory()
    mem.update(svc.build(emb.observe()))
    emb.scene.bodies = [b for b in emb.scene.bodies if b.name != "lime"]
    mem.update(svc.build(emb.observe()))
    assert mem.missing("lime_0") == 1
    assert "lime_0" in mem.render()


def test_memory_renders_held_state_and_history():
    mem = ObjectMemory()
    mem.update(Catalog([], "test", "v", 0.0))
    assert "not holding anything" in mem.render()
    mem.set_held("apple_0")
    mem.record("grasp", "apple_0", "grasped", "rose 5 cm")
    text = mem.render()
    assert "holding apple_0" in text
    assert "grasp(apple_0) -> grasped" in text


def test_memory_history_is_capped_in_the_prompt():
    mem = ObjectMemory()
    mem.update(Catalog([], "test", "v", 0.0))
    for i in range(20):
        mem.record("grasp", f"o_{i}", "missed")
    assert mem.render(max_attempts=3).count("grasp(") == 3


# ---- box grounding --------------------------------------------------------


def test_boxes_parse_from_the_gemini_convention():
    from rung.perception import parse_boxes

    text = '[{"label": "banana", "box_2d": [250, 100, 500, 400]}]'
    dets = parse_boxes(text, width=640, height=480)
    assert len(dets) == 1
    left, top, right, bottom = dets[0].box
    assert (left, top, right, bottom) == pytest.approx((64.0, 120.0, 256.0, 240.0))


def test_boxes_parse_from_a_fenced_block_with_prose():
    from rung.perception import parse_boxes

    text = (
        'Here is what I see:\n```json\n[{"label": "bowl", "box_2d": [0, 0, 500, 500]}]\n```\nDone.'
    )
    assert len(parse_boxes(text, 640, 480)) == 1


def test_pixel_coordinates_are_accepted_too():
    from rung.perception import parse_boxes

    dets = parse_boxes(
        '[{"label": "cube", "box_2d": [40, 30, 90, 110]}]', 200, 150, coordinates="pixels"
    )
    assert dets[0].box == (30, 40, 110, 90)
    assert dets and dets[0].box[2] > dets[0].box[0]


def test_unusable_boxes_are_dropped_not_guessed():
    """A wrong box is worse than a missing one: the harness measures whatever is inside."""
    from rung.perception import parse_boxes

    text = json.dumps(
        [
            {"label": "ok", "box_2d": [100, 100, 300, 300]},
            {"label": "inverted", "box_2d": [500, 500, 100, 100]},
            {"label": "no box"},
            {"box_2d": [1, 2, 3, 4]},
            {"label": "not numbers", "box_2d": ["a", "b", "c", "d"]},
        ]
    )
    labels = [d.label for d in parse_boxes(text, 640, 480)]
    assert "ok" in labels
    assert "no box" not in labels and "not numbers" not in labels


def test_no_json_at_all_yields_nothing():
    from rung.perception import parse_boxes

    assert parse_boxes("I cannot see any objects.", 640, 480) == []


def test_duplicate_labels_are_made_unique():
    from rung.perception import parse_boxes

    text = json.dumps(
        [
            {"label": "cube", "box_2d": [100, 100, 200, 200]},
            {"label": "cube", "box_2d": [300, 300, 400, 400]},
        ]
    )
    labels = [d.label for d in parse_boxes(text, 640, 480)]
    assert len(set(labels)) == 2


def test_box_grounding_calls_the_model_and_builds_a_catalog():
    from rung.backends import MockBackend, ModelResponse
    from rung.perception import BoxGrounding

    emb = SCENES["banana_in_bowl"]()
    obs = emb.observe()
    boxes = []
    for body in emb.scene.bodies:
        from rung.contracts.frames import project

        uv, _ = project(np.asarray(body.pos)[None], obs.scene.K, obs.scene.pose)
        u, v = float(uv[0, 0]), float(uv[0, 1])
        boxes.append(
            {
                "label": body.name,
                "box_2d": [
                    (v - 25) / obs.scene.height * 1000,
                    (u - 25) / obs.scene.width * 1000,
                    (v + 25) / obs.scene.height * 1000,
                    (u + 25) / obs.scene.width * 1000,
                ],
            }
        )
    backend = MockBackend([ModelResponse(text=json.dumps(boxes))])
    grounding = BoxGrounding(backend)
    cat = PerceptionService(grounding).build(obs, emb.instruction)
    assert grounding.calls == 1
    assert {o.label for o in cat.objects} == {"banana", "bowl"}
    assert grounding.name == "boxes/mock-model"


def test_box_grounding_records_a_failure_rather_than_inventing_objects():
    from rung.backends import MockBackend, ModelResponse
    from rung.perception import BoxGrounding

    emb = SCENES["banana_in_bowl"]()
    grounding = BoxGrounding(MockBackend([ModelResponse(text="I see nothing.")]))
    cat = PerceptionService(grounding).build(emb.observe())
    assert cat.objects == []
    assert grounding.failures and "no usable boxes" in grounding.failures[0]


def test_normalized_boxes_scale_on_images_larger_than_1000_pixels():
    from rung.perception import parse_boxes

    boxes = parse_boxes('[{"label":"object","box_2d":[250,250,750,750]}]', width=1280, height=720)
    assert len(boxes) == 1
    assert boxes[0].box == (320, 180, 960, 540)


def test_reaching_over_an_object_does_not_rename_it():
    """Occlusion by the hand does not create a second object.

    With the gripper overhead, part of the cluster is hidden and the visible centroid can
    move further than the object's match distance. Proximity to the hand widens the
    tolerance, so the id is kept.
    """
    import numpy as np

    from rung.perception.memory import ObjectMemory
    from rung.perception.types import Catalog, ObjectDescriptor

    def catalog(centre, t):
        return Catalog(
            objects=[
                ObjectDescriptor(
                    object_id="raw",
                    label="banana",
                    centroid=np.array(centre),
                    top_z=0.04,
                    bottom_z=0.0,
                    length=0.17,
                    width=0.04,
                    height=0.04,
                    axis=np.array([1.0, 0.0, 0.0]),
                    grasp_length=0.17,
                    grasp_width=0.04,
                    grasp_centre=np.array(centre),
                    n_points=300,
                    visible_fraction=0.6,
                )
            ],
            source="test",
            version="test",
            t=t,
        )

    memory = ObjectMemory()
    first = memory.update(catalog([0.50, -0.12, 0.02], 0.0))
    original = first.objects[0].object_id

    # The hand arrives overhead and half the banana disappears, so its apparent centre
    # slides 9 cm, further than its own match distance of 8.5 cm.
    shadowed = memory.update(catalog([0.59, -0.12, 0.02], 1.0), tcp_pos=[0.59, -0.12, 0.10])
    assert shadowed.objects[0].object_id == original, "the hand's shadow renamed the object"

    # With the hand well away, that same jump is a different object, as it should be.
    memory2 = ObjectMemory()
    started = memory2.update(catalog([0.50, -0.12, 0.02], 0.0)).objects[0].object_id
    far = memory2.update(catalog([0.59, -0.12, 0.02], 1.0), tcp_pos=[0.20, 0.30, 0.40])
    assert far.objects[0].object_id != started


# ---- what one oblique camera can and cannot measure ---------------------------------


def test_the_table_is_not_part_of_the_object():
    """A flood fill from an object's base does not walk out across the table inside its box:
    the object's extent ends at the table."""
    for name in SCENES:
        emb, _, cat = build(name)
        for o in cat.objects:
            # An object standing on the table ends at the table, not at the cut, unless its
            # lower part is hidden behind a neighbour (the mug, behind the lime).
            truth = emb.scene.body(o.label)
            assert o.bottom_z >= truth.bottom_z - 0.006, (name, o.label, o.bottom_z)
            if o.label in ("banana", "bottle", "box", "apple"):
                assert o.bottom_z == pytest.approx(truth.bottom_z, abs=0.006), (name, o.label)
            long, short = truth.footprint
            assert o.length <= long + 0.03, (name, o.label, o.length, long)


def test_footprint_centre_ignores_how_densely_a_side_was_sampled():
    """A camera returns many points from the face it looks at and few from the others.
    The plain mean drifts toward the dense face; the area-weighted centre does not."""
    from rung.perception.cloud import footprint_centre

    rng = np.random.default_rng(1)
    top = np.stack(
        [rng.uniform(-0.03, 0.03, 300), rng.uniform(-0.03, 0.03, 300), np.full(300, 0.05)], axis=1
    )
    near_face = np.stack(
        [np.full(3000, -0.03), rng.uniform(-0.03, 0.03, 3000), rng.uniform(0.0, 0.05, 3000)], axis=1
    )
    cloud = np.vstack([top, near_face])
    biased = cloud[:, :2].mean(axis=0)
    centre = footprint_centre(cloud)
    assert abs(biased[0]) > 0.02, "the mean should be pulled toward the dense face"
    assert abs(centre[0]) < 0.006, centre
    assert abs(centre[1]) < 0.006, centre


def test_a_dome_is_not_measured_along_the_line_of_sight():
    """From one side, a ball's far half is hidden: the span along the line of sight is
    a lower bound while the silhouette across it is exact. The closing direction must
    be one the camera measured, and a planner must hear when it is not."""
    from rung.perception.cloud import top_slab, trusted_width_direction

    r = 0.035
    rng = np.random.default_rng(2)
    theta = rng.uniform(0, 2 * np.pi, 6000)
    phi = rng.uniform(0, np.pi / 2, 6000)  # upper hemisphere
    pts = np.stack(
        [r * np.cos(phi) * np.cos(theta), r * np.cos(phi) * np.sin(theta), r * np.sin(phi) + r],
        axis=1,
    )
    view = np.array([1.0, 0.0])  # the camera looks along +x, from -x, 30 degrees up
    cam = np.array([-1.0, 0.0, 0.58])
    normals = (pts - np.array([0.0, 0.0, r])) / r
    visible = pts[np.einsum("ij,ij->i", normals, cam - pts) > 0]
    band = visible[visible[:, 2] > 0.01]
    crown = top_slab(band, band[:, 2].max())
    width, _, closing, bounded = trusted_width_direction(band, crown, view, fits_m=0.085 - 0.016)
    assert abs(closing[1]) > 0.8, f"should close across the line of sight, got {closing}"
    assert width == pytest.approx(2 * r, abs=0.006)
    assert not bounded, "the silhouette is a measurement, not a bound"
    # Along the line of sight only the near half plus the top is seen: shorter, and
    # exactly the direction a naive minimum-width search would have chosen.
    naive_w, _, naive_dir = min_width_direction(band)
    assert abs(naive_dir[0]) > 0.8 and naive_w < 2 * r - 0.005


def test_a_flat_top_is_measured_in_every_direction():
    """A box's top face reaches its edges, so its extent along the line of sight is
    observed even though the far wall is not."""
    from rung.perception.cloud import top_slab, trusted_width_direction

    rng = np.random.default_rng(3)
    top = np.stack(
        [rng.uniform(-0.05, 0.05, 2000), rng.uniform(-0.02, 0.02, 2000), np.full(2000, 0.06)],
        axis=1,
    )
    near_wall = np.stack(
        [np.full(1500, -0.05), rng.uniform(-0.02, 0.02, 1500), rng.uniform(0.01, 0.06, 1500)],
        axis=1,
    )
    band = np.vstack([top, near_wall])
    view = np.array([0.0, 1.0])  # looking along +y: the narrow side is along the view
    width, length, closing, bounded = trusted_width_direction(band, top_slab(band, 0.06), view)
    assert not bounded
    assert width == pytest.approx(0.04, abs=0.004)
    assert abs(closing[1]) > 0.95


def test_memory_keeps_an_id_when_the_grounding_renames_the_object():
    """A label change at the same spot ('onion', then 'red onion') keeps the same id."""
    from rung.perception.memory import ObjectMemory
    from rung.perception.types import ObjectDescriptor

    def desc(label, xy):
        return ObjectDescriptor(
            object_id="?",
            label=label,
            centroid=np.array([xy[0], xy[1], 0.03]),
            top_z=0.06,
            bottom_z=0.01,
            length=0.07,
            width=0.06,
            height=0.05,
            axis=np.array([1.0, 0.0]),
            grasp_length=0.07,
            grasp_width=0.06,
            grasp_centre=np.array([xy[0], xy[1], 0.06]),
            n_points=500,
            visible_fraction=1.0,
        )

    mem = ObjectMemory()
    first = Catalog([desc("onion", (0.60, -0.29))], "test", "v", 0.0)
    mem.update(first)
    second = Catalog([desc("red onion", (0.605, -0.285))], "test", "v", 1.0)
    mem.update(second)
    assert second.objects[0].object_id == "onion_0"
    third = Catalog([desc("red onion", (0.70, -0.10))], "test", "v", 2.0)
    mem.update(third)
    assert third.objects[0].object_id != "onion_0", "ten centimetres away is not it"


def test_a_banana_along_the_line_of_sight_is_not_called_too_wide():
    """Seen from the side, an elongated object's silhouette is its length. The width along
    the line of sight is only partly seen, but a camera 30 degrees up hides at most a
    quarter of it, and that bound fits the jaw, so the direction is kept."""
    from rung.perception.cloud import top_slab, trusted_width_direction

    rng = np.random.default_rng(4)
    r, half_len = 0.022, 0.085
    theta = rng.uniform(0, np.pi, 6000)  # a half cylinder lying along x
    pts = np.stack(
        [rng.uniform(-half_len, half_len, 6000), r * np.cos(theta), r * np.sin(theta) + r], axis=1
    )
    view = np.array([0.0, 1.0])  # looking across the banana: its silhouette is its length
    cam = np.array([0.0, -1.5, 0.87])  # 30 degrees up
    normals = np.stack([np.zeros(6000), np.cos(theta), np.sin(theta)], axis=1)
    visible = pts[np.einsum("ij,ij->i", normals, cam - pts) > 0]
    band = visible[visible[:, 2] > 0.01]
    width, length, closing, bounded = trusted_width_direction(
        band,
        top_slab(band, band[:, 2].max()),
        view,
        fits_m=0.085 - 0.016,
        elevation_rad=np.radians(30),
    )
    assert abs(closing[1]) > 0.95, closing
    assert bounded
    assert 2 * r * 0.9 <= width <= 2 * r * 1.3, width


def test_a_banana_is_measured_where_the_fingers_close():
    """A curved banana spans more across its whole length than under the pads. The width
    the jaw meets is the local one, and the aim stays on the banana, not inside its curve."""
    from rung.perception.cloud import grasp_point, local_span

    t = np.linspace(-1.0, 1.0, 400)
    spine = np.stack([0.085 * t, 0.03 * (1 - t**2)], axis=1)  # 17 cm long, 3 cm bow
    rng = np.random.default_rng(5)
    off = rng.uniform(-0.02, 0.02, (400, 8))
    pts = np.concatenate(
        [
            np.stack(
                [spine[:, 0].repeat(8), spine[:, 1].repeat(8) + off.ravel(), np.full(3200, 0.04)],
                axis=1,
            )
        ]
    )
    whole, _, closing = min_width_direction(pts)
    aim = grasp_point(pts)
    local = local_span(pts, aim, closing)
    assert whole > 0.06, whole
    assert local == pytest.approx(0.04, abs=0.006), local
    assert np.linalg.norm(pts[:, :2] - aim, axis=1).min() < 0.005, "aim is on the banana"


def test_the_table_height_is_measured_not_assumed():
    """The table height is estimated from depth, including a table 5.5 cm above the robot
    base."""
    from rung.contracts.frames import project
    from rung.contracts.observation import CameraFrame
    from rung.perception.cloud import estimate_table_z

    emb = SCENES["banana_in_bowl"]()
    cam = emb.observe().scene
    H, W = cam.depth.shape
    # Re-render the depth of a plane at z = 0.055 seen by the same camera.
    K, pose = cam.K, cam.pose
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    rays = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u, float)], -1)
    R = pose.rot
    world = rays @ R.T
    t = (0.055 - pose.pos[2]) / world[..., 2]
    depth = np.where(t > 0, t, np.nan).astype(np.float32)
    frame = CameraFrame(name="scene", rgb=cam.rgb, t=0.0, depth=depth, K=K, pose=pose)
    z, share = estimate_table_z(frame)
    assert z == pytest.approx(0.055, abs=0.005)
    assert share > 0.9
    assert project  # imported for parity with the other geometry tests


def test_a_placed_object_keeps_its_id_when_the_grounding_renames_it():
    from rung.perception.memory import ObjectMemory
    from rung.perception.types import ObjectDescriptor

    def desc(label, xy):
        return ObjectDescriptor(
            object_id="?",
            label=label,
            centroid=np.array([xy[0], xy[1], 0.1]),
            top_z=0.12,
            bottom_z=0.08,
            length=0.17,
            width=0.05,
            height=0.04,
            axis=np.array([1.0, 0.0]),
            grasp_length=0.17,
            grasp_width=0.05,
            grasp_centre=np.array([xy[0], xy[1], 0.12]),
            n_points=500,
            visible_fraction=1.0,
        )

    mem = ObjectMemory()
    first = Catalog([desc("banana on the right table", (0.49, -0.25))], "t", "v", 0.0)
    mem.update(first)
    mem.relocate("banana_on_the_right_tabl_0", np.array([0.46, 0.25, 0.1]))
    after = Catalog([desc("banana", (0.46, 0.30))], "t", "v", 1.0)  # landed 5 cm off
    mem.update(after)
    assert after.objects[0].object_id == "banana_on_the_right_tabl_0"


def test_the_fast_grouping_finds_exactly_what_the_flood_fill_finds():
    """The KD-tree component labelling links exactly the same points as the flood fill."""
    pytest.importorskip("scipy")
    from rung.perception.cloud import _flood_fill, component_labels

    rng = np.random.default_rng(9)
    a = rng.uniform(0, 0.05, (1500, 3))
    b = rng.uniform(0, 0.05, (1500, 3)) + np.array([0.071, 0.0, 0.0])  # 2.1 cm gap
    c = rng.uniform(0, 0.05, (800, 3)) + np.array([0.0, 0.069, 0.0])  # 1.9 cm gap: touches a
    pts = np.vstack([a, b, c])
    labels = component_labels(pts, 0.02)
    for start in (0, 1600, 3200):
        fast = set(np.flatnonzero(labels == labels[start]))
        slow_pts = _flood_fill(pts, start, 0.02)
        slow = {int(np.flatnonzero((pts == q).all(axis=1))[0]) for q in slow_pts}
        assert fast == slow, start


def test_perception_of_a_cluttered_scene_is_fast():
    pytest.importorskip("scipy")
    import time

    emb, svc, _ = build("cluttered_bin")
    t0 = time.perf_counter()
    svc.build(emb.observe(), emb.instruction)
    assert time.perf_counter() - t0 < 1.0
