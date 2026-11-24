import warnings

import numpy as np
import numpy.testing as npt
import pytest
from scipy.spatial import Delaunay

import discretize
from discretize.utils import example_simplex_mesh


def _signed_volumes(nodes, simplices):
    v = nodes[simplices]
    return np.linalg.det(v[:, 1:] - v[:, :1])


def _non_delaunay_mesh(dim, n_points, seed):
    """A Delaunay mesh of random points whose interior nodes are then moved as
    far as possible without inverting any simplex, so most of it is no longer
    Delaunay."""
    rng = np.random.default_rng(seed)
    corners = np.array(np.meshgrid(*([[0.0, 1.0]] * dim))).reshape(dim, -1).T
    nodes = np.vstack([corners, rng.uniform(size=(n_points, dim))])
    simplices = Delaunay(nodes).simplices
    sign = np.sign(_signed_volumes(nodes, simplices))
    interior = np.all((nodes > 0.02) & (nodes < 0.98), axis=1)
    for _ in range(10):
        step = rng.normal(scale=0.3 * n_points ** (-1 / dim), size=nodes.shape)
        moved = nodes + np.where(interior[:, None], step, 0.0)
        # Undo the moves of any simplex that flips. Undoing some moves can flip
        # other simplices, so repeat until none do.
        while True:
            flipped = np.sign(_signed_volumes(moved, simplices)) != sign
            if not np.any(flipped):
                break
            undo = np.unique(simplices[flipped].ravel())
            moved[undo] = nodes[undo]
        nodes = moved
    assert np.all(np.sign(_signed_volumes(nodes, simplices)) == sign)
    return discretize.SimplexMesh(nodes, simplices)


def _remove_cells(nodes, simplices, remove):
    simplices = simplices[~remove]
    used = np.unique(simplices)
    renumber = np.full(len(nodes), -1)
    renumber[used] = np.arange(len(used))
    return discretize.SimplexMesh(nodes[used], renumber[simplices])


def _assert_contained(mesh, locs, inds):
    transform, shift = mesh.transform_and_shift
    b = np.einsum("nij,nj->ni", transform[inds], locs - shift[inds])
    barys = np.c_[b, 1 - b.sum(axis=1)]
    assert barys.min() > -1e-12


def _linear(x):
    return 1.0 + x @ np.arange(2.0, 2.0 + x.shape[1])


@pytest.fixture
def walk_counts(monkeypatch):
    """Record, for each call, how many times the walk jumped across a gap in
    the mesh, and how many points it had to find by testing every simplex."""
    import discretize.unstructured_mesh as um

    walk = um._directed_search
    counts = []

    def counted(*args, **kwargs):
        out = walk(*args, **kwargs)
        counts.append({"jumps": out[3], "searched": out[4]})
        return out

    monkeypatch.setattr(um, "_directed_search", counted)
    return counts


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("spatial_sort", [True, False])
def test_non_delaunay_mesh(dim, spatial_sort):
    # Always crossing the first face the point is beyond can cycle on meshes
    # like this (rarely), and a walk that gave up was then interpolated from an
    # unrelated simplex.
    mesh = _non_delaunay_mesh(dim, 4000 if dim == 2 else 1500, seed=dim)
    locs = np.random.default_rng(10).uniform(0.02, 0.98, size=(5000, dim))
    inds = mesh.point2index(locs, spatial_sort=spatial_sort)
    _assert_contained(mesh, locs, inds)
    P = mesh.get_interpolation_matrix(locs, "nodes", spatial_sort=spatial_sort)
    npt.assert_allclose(P @ _linear(mesh.nodes), _linear(locs), rtol=1e-12)


def test_concave_mesh(walk_counts):
    # A U shaped mesh. Walking the points in the given order alternates
    # between the arms, so every walk runs into the slot between them.
    nodes, simplices = example_simplex_mesh((40, 40))
    centers = nodes[simplices].mean(axis=1)
    slot = (centers[:, 0] > 0.3) & (centers[:, 0] < 0.7) & (centers[:, 1] < 0.8)
    mesh = _remove_cells(nodes, simplices, slot)
    assert not mesh._is_convex

    rng = np.random.default_rng(0)
    left = np.c_[rng.uniform(0.01, 0.29, 500), rng.uniform(0.01, 0.79, 500)]
    right = np.c_[rng.uniform(0.71, 0.99, 500), rng.uniform(0.01, 0.79, 500)]
    inside = np.stack([left, right], axis=1).reshape(-1, 2)
    in_slot = np.c_[rng.uniform(0.32, 0.68, 200), rng.uniform(0.0, 0.78, 200)]
    locs = np.vstack([inside, in_slot])

    for spatial_sort in (True, False):
        P = mesh.get_interpolation_matrix(
            locs, "nodes", zeros_outside=True, spatial_sort=spatial_sort
        )
        values = P @ _linear(mesh.nodes)
        npt.assert_allclose(values[: len(inside)], _linear(inside), rtol=1e-12)
        npt.assert_equal(values[len(inside) :], 0.0)
    # Walks that stop at the slot continue across it along a ray to the point,
    # without testing every simplex. Sorted, they only have to cross it a few
    # times, since each walk starts from where the previous point was found.
    sorted_counts, given_counts = walk_counts
    assert sorted_counts["searched"] == 0 and given_counts["searched"] == 0
    assert 0 < sorted_counts["jumps"] < 50
    assert given_counts["jumps"] >= len(inside) // 2


def test_interior_void(walk_counts):
    nodes, simplices = example_simplex_mesh((10, 10, 10))
    assert discretize.SimplexMesh(nodes, simplices)._is_convex
    centers = nodes[simplices].mean(axis=1)
    void = np.all((centers > 0.4) & (centers < 0.6), axis=1)
    mesh = _remove_cells(nodes, simplices, void)
    assert not mesh._is_convex

    # Points on either side of the void, alternating so the walks cross it,
    # and points inside it.
    rng = np.random.default_rng(1)
    yz = rng.uniform(0.41, 0.59, (300, 2))
    sides = np.stack(
        [
            np.c_[rng.uniform(0.05, 0.38, 300), yz],
            np.c_[rng.uniform(0.62, 0.95, 300), yz],
        ],
        axis=1,
    ).reshape(-1, 3)
    in_void = rng.uniform(0.42, 0.58, (100, 3))
    locs = np.vstack([sides, in_void])

    for location_type in ("nodes", "cell_centers"):
        P = mesh.get_interpolation_matrix(
            locs, location_type, zeros_outside=True, spatial_sort=False
        )
        rows = np.asarray(abs(P).sum(axis=1)).ravel()
        assert np.all(rows[: len(sides)] > 0)
        npt.assert_equal(rows[len(sides) :], 0.0)
    _assert_contained(mesh, sides, mesh.point2index(sides, spatial_sort=False))
    assert all(c["searched"] == 0 for c in walk_counts)
    assert all(c["jumps"] > 0 for c in walk_counts)


def test_crack(walk_counts):
    # Cut the mesh along y = 0.5 for 0.25 < x < 0.75 by giving the cells above
    # the cut their own copies of the nodes on it. The cut adds no volume, but
    # the walk cannot cross it.
    nodes, simplices = example_simplex_mesh((20, 20))
    on_cut = np.flatnonzero(
        np.isclose(nodes[:, 1], 0.5) & (nodes[:, 0] > 0.26) & (nodes[:, 0] < 0.74)
    )
    copies = np.arange(len(nodes), len(nodes) + len(on_cut))
    renumber = np.arange(len(nodes))
    renumber[on_cut] = copies
    above = nodes[simplices].mean(axis=1)[:, 1] > 0.5
    simplices = simplices.copy()
    simplices[above] = renumber[simplices[above]]
    mesh = discretize.SimplexMesh(np.vstack([nodes, nodes[on_cut]]), simplices)
    assert not mesh._is_convex

    rng = np.random.default_rng(2)
    x = rng.uniform(0.3, 0.7, 400)
    dy = rng.uniform(0.001, 0.02, 400)
    locs = np.stack([np.c_[x, 0.5 + dy], np.c_[x, 0.5 - dy]], axis=1).reshape(-1, 2)
    inds, _, outside = mesh._locate(locs, spatial_sort=False)
    assert not np.any(outside)
    _assert_contained(mesh, locs, inds)
    assert walk_counts[0]["searched"] == 0
    assert walk_counts[0]["jumps"] > 0


def test_outside_points():
    nodes, simplices = example_simplex_mesh((8, 8, 8))
    mesh = discretize.SimplexMesh(nodes, simplices)
    rng = np.random.default_rng(3)
    inside = rng.uniform(0.0, 1.0, (100, 3))
    outside = rng.uniform(1.1, 3.0, (100, 3)) * rng.choice([-1, 1], (100, 3))
    locs = np.vstack([inside, outside])
    inds, _, is_outside = mesh._locate(locs)
    npt.assert_equal(is_outside, np.r_[np.zeros(100, bool), np.ones(100, bool)])
    # Outside points still get a (boundary) cell to extrapolate from.
    assert np.all((inds >= 0) & (inds < mesh.n_cells))
    P = mesh.get_interpolation_matrix(locs, "cell_centers", zeros_outside=True)
    rows = np.asarray(abs(P).sum(axis=1)).ravel()
    assert np.all(rows[:100] > 0)
    npt.assert_equal(rows[100:], 0.0)


@pytest.mark.parametrize("zeros_outside", [True, False])
@pytest.mark.parametrize(
    "location_type", ["nodes", "cell_centers", "edges_x", "faces_y"]
)
def test_non_finite_points(location_type, zeros_outside):
    nodes, simplices = example_simplex_mesh((4, 4, 4))
    mesh = discretize.SimplexMesh(nodes, simplices)
    locs = np.full((5, 3), 0.4)
    locs[1, 0] = np.nan
    locs[3, 2] = np.inf
    locs[4, 1] = -np.inf
    bad = np.array([False, True, False, True, True])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        inds, barys, outside = mesh._locate(locs)
        P = mesh.get_interpolation_matrix(
            locs, location_type, zeros_outside=zeros_outside
        )
    npt.assert_equal(inds[bad], -1)
    npt.assert_equal(outside, bad)
    npt.assert_equal(barys[bad], 0.0)
    P = P.toarray()
    assert np.all(np.isfinite(P))
    npt.assert_equal(P[bad], 0.0)
    # The finite points are unaffected.
    P_good = mesh.get_interpolation_matrix(
        locs[~bad], location_type, zeros_outside=zeros_outside
    ).toarray()
    npt.assert_allclose(P[~bad], P_good)


def test_all_non_finite_points():
    nodes, simplices = example_simplex_mesh((3, 3))
    mesh = discretize.SimplexMesh(nodes, simplices)
    npt.assert_equal(mesh.point2index(np.full((3, 2), np.nan)), -1)


@pytest.mark.parametrize(
    "location_type", ["nodes", "cell_centers", "edges_x", "faces_y"]
)
def test_spatial_sort_does_not_change_result(location_type):
    nodes, simplices = example_simplex_mesh((6, 6, 6))
    mesh = discretize.SimplexMesh(nodes, simplices)
    locs = np.random.default_rng(4).uniform(0.05, 0.95, (300, 3))
    a = mesh.get_interpolation_matrix(locs, location_type, spatial_sort=True)
    b = mesh.get_interpolation_matrix(locs, location_type, spatial_sort=False)
    npt.assert_allclose(a.toarray(), b.toarray(), atol=1e-14)


def test_empty_locations():
    nodes, simplices = example_simplex_mesh((3, 3))
    mesh = discretize.SimplexMesh(nodes, simplices)
    assert mesh.point2index(np.empty((0, 2))).shape == (0,)


def test_walk_failures_fall_back(monkeypatch):
    # Cap the walk at two steps so most walks give up, which leaves those
    # points to testing every simplex.
    import discretize.unstructured_mesh as um

    walk = um._directed_search
    searched = []

    def short_walk(locs, start, neighbors, transform, shift, max_steps, *args):
        out = walk(locs, start, neighbors, transform, shift, 2, *args)
        searched.append(out[4])
        return out

    monkeypatch.setattr(um, "_directed_search", short_walk)
    mesh = _non_delaunay_mesh(2, 2000, seed=5)
    locs = np.random.default_rng(6).uniform(0.02, 0.98, size=(500, 2))
    inds, _, outside = mesh._locate(locs, spatial_sort=False)
    assert searched[0] > 100
    assert not np.any(outside)
    _assert_contained(mesh, locs, inds)


@pytest.mark.parametrize("dim", [2, 3])
def test_morton_order_matches_bit_by_bit(dim):
    # _morton_order builds its keys with libmorton's "magic bits" shifts and
    # masks; check them against interleaving the bits one at a time.
    from discretize.unstructured_mesh import _morton_order

    rng = np.random.default_rng(7)
    lower, upper = np.full(dim, -1.0), np.full(dim, 2.0)
    # include points outside the box, which are clamped to it
    locs = rng.uniform(-1.5, 2.5, (20000, dim))

    bits = 63 // dim
    n_bins = (1 << bits) - 1
    q = np.clip((locs - lower) / (upper - lower) * n_bins, 0, n_bins).astype(np.uint64)
    key = np.zeros(len(locs), dtype=np.uint64)
    for b in range(bits):
        for d in range(dim):
            bit = (q[:, d] >> np.uint64(b)) & np.uint64(1)
            key |= bit << np.uint64(b * dim + d)

    npt.assert_equal(_morton_order(locs, lower, upper), np.argsort(key, kind="stable"))


def _contains(mesh, locs, tol=1e-12):
    """Whether each point is in some simplex, by testing every simplex."""
    transform, shift = mesh.transform_and_shift
    found = np.zeros(len(locs), dtype=bool)
    for start in range(0, len(locs), 200):
        x = locs[start : start + 200]
        b = np.einsum("cij,ncj->nci", transform, x[:, None, :] - shift[None])
        barys = np.concatenate([b, 1 - b.sum(axis=-1, keepdims=True)], axis=-1)
        found[start : start + 200] = np.any(barys.min(axis=-1) >= -tol, axis=1)
    return found


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("spatial_sort", [True, False])
def test_matches_exhaustive_search_on_irregular_mesh(dim, spatial_sort):
    # An irregular mesh with voids inside it and bites out of its surface,
    # checked point by point against testing every simplex.
    rng = np.random.default_rng(8)
    nodes = rng.uniform(size=(3000, dim))
    simplices = Delaunay(nodes).simplices
    centers = nodes[simplices].mean(axis=1)
    holes = rng.uniform(size=(8, dim))
    radii = rng.uniform(0.08, 0.2, 8)
    removed = np.any(
        np.linalg.norm(centers[:, None] - holes[None], axis=-1) < radii, axis=1
    )
    mesh = _remove_cells(nodes, simplices, removed)
    assert not mesh._is_convex

    # Include points outside the mesh's bounding box.
    locs = rng.uniform(-0.2, 1.2, size=(3000, dim))
    truth = _contains(mesh, locs)
    assert 0 < np.sum(~truth) < len(locs)
    inds, _, outside = mesh._locate(locs, spatial_sort=spatial_sort)
    npt.assert_equal(outside, ~truth)
    _assert_contained(mesh, locs[truth], inds[truth])


def test_points_on_gap_walls():
    # Points exactly on the walls of the U shaped mesh's slot, at nodes and
    # on edges, alternating between the arms. The rays across the slot hit
    # the far wall exactly on its facets' vertices and edges.
    nodes, simplices = example_simplex_mesh((40, 40))
    centers = nodes[simplices].mean(axis=1)
    slot = (centers[:, 0] > 0.3) & (centers[:, 0] < 0.7) & (centers[:, 1] < 0.8)
    mesh = _remove_cells(nodes, simplices, slot)

    # Take the walls' own nodes (and their edges' midpoints), so the points
    # are exactly on them.
    def wall(x):
        on = np.isclose(mesh.nodes[:, 0], x) & (mesh.nodes[:, 1] < 0.81)
        w = mesh.nodes[on][np.argsort(mesh.nodes[on, 1])]
        return np.vstack([w, 0.5 * (w[1:] + w[:-1])])

    left, right = wall(0.3), wall(0.7)
    locs = np.stack([left, right], axis=1).reshape(-1, 2)
    inds, _, outside = mesh._locate(locs, spatial_sort=False)
    assert not np.any(outside)
    _assert_contained(mesh, locs, inds)


def _topography(x, y):
    return 0.15 * np.sin(6 * x) * np.cos(5 * y) + 0.05 * x


def _topography_mesh():
    h = np.linspace(0, 1, 21)
    nodes = np.stack(np.meshgrid(h, h, np.linspace(-0.6, 0.3, 19)), -1).reshape(-1, 3)
    nodes = nodes[nodes[:, 2] <= _topography(nodes[:, 0], nodes[:, 1])]
    sx, sy = np.meshgrid(h, h)
    surface = np.c_[sx.ravel(), sy.ravel(), _topography(sx, sy).ravel()]
    nodes = np.vstack([nodes, surface])
    nodes += np.random.default_rng(10).uniform(-1e-3, 1e-3, nodes.shape)
    simplices = Delaunay(nodes).simplices
    c = nodes[simplices].mean(axis=1)
    return _remove_cells(nodes, simplices, c[:, 2] > _topography(c[:, 0], c[:, 1]))


def test_points_above_topography(walk_counts):
    # Draped points above a surface: whether they are outside can't be read
    # off the walk, as the mesh is not convex.
    mesh = _topography_mesh()
    assert not mesh._is_convex
    rng = np.random.default_rng(11)
    xy = rng.uniform(0.02, 0.98, (1000, 2))
    z = _topography(xy[:, 0], xy[:, 1])
    clearance = rng.uniform(0.01, 0.1, 1000)
    locs = np.vstack([np.c_[xy, z + clearance], np.c_[xy, z - clearance]])
    truth = _contains(mesh, locs)
    inds, _, outside = mesh._locate(locs)
    npt.assert_equal(outside, ~truth)
    _assert_contained(mesh, locs[truth], inds[truth])
    assert walk_counts[0]["searched"] == 0

    # Points above the highest node are outside the bounding box, so they
    # never need a jump.
    high = np.c_[xy, np.full(1000, mesh.nodes[:, 2].max() + 0.05)]
    _, _, outside = mesh._locate(high)
    assert np.all(outside)
    assert walk_counts[1]["jumps"] == 0


@pytest.mark.parametrize("convex", [True, False])
def test_gave_up_outside_points_get_boundary_cell(monkeypatch, convex):
    # When a walk gives up and the point turns out to be outside the mesh, it
    # still gets a nearby boundary cell to extrapolate from.
    import discretize.unstructured_mesh as um

    walk = um._directed_search

    def short_walk(locs, start, neighbors, transform, shift, max_steps, *args):
        return walk(locs, start, neighbors, transform, shift, 2, *args)

    monkeypatch.setattr(um, "_directed_search", short_walk)
    if convex:
        nodes, simplices = example_simplex_mesh((20, 20))
        mesh = discretize.SimplexMesh(nodes, simplices)
    else:
        mesh = _topography_mesh()
    rng = np.random.default_rng(12)
    if convex:
        locs = rng.uniform(1.05, 1.2, (200, 2)) * rng.choice([-1, 1], (200, 2))
    else:
        xy = rng.uniform(0.02, 0.98, (200, 2))
        z = _topography(xy[:, 0], xy[:, 1]) + rng.uniform(0.01, 0.1, 200)
        locs = np.c_[xy, z]
    inds, barys, outside = mesh._locate(locs, spatial_sort=False)
    assert np.all(outside)
    # The barycentric coordinates are in the returned cell, and the point is
    # beyond one of its boundary faces.
    npt.assert_allclose(
        np.einsum("ni,nij->nj", barys, mesh.nodes[mesh.simplices[inds]]),
        locs,
        atol=1e-12,
    )
    assert np.all(np.any((mesh.neighbors[inds] == -1) & (barys < 0), axis=1))
