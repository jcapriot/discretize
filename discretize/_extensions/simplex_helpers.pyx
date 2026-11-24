# distutils: language=c++
# cython: embedsignature=True, language_level=3
# cython: linetrace=True
# cython: freethreading_compatible = True

from libcpp.pair cimport pair
from libcpp.unordered_map cimport unordered_map
from cython.operator cimport dereference
cimport cython
cimport numpy as np
from libc.math cimport sqrt, floor, INFINITY

cdef extern from "triplet.h":
    cdef cppclass triplet[T, U, V]:
        T v1
        U v2
        V v3
        triplet()
        triplet(T, U, V)
import numpy as np

ctypedef fused ints:
    size_t
    np.int32_t
    np.int64_t

@cython.boundscheck(False)
def _build_faces_edges(ints[:, :] simplices):
    # the node index in each simplex must be in increasing order
    cdef:
        int dim = simplices.shape[1] - 1
        ints n_simplex = simplices.shape[0]
        ints[:] simplex
        ints v1, v2, v3

        unordered_map[pair[ints, ints], ints] edges
        pair[ints, ints] edge
        ints n_edges = 0
        ints edges_per_simplex = 3 if dim==2 else 6
        ints[:, :] simplex_edges

        unordered_map[triplet[ints, ints, ints], ints] faces
        triplet[ints, ints, ints] face
        ints n_faces = 0
        ints faces_per_simplex = dim + 1
        ints[:, :] simplex_faces

    if ints is size_t:
        int_type = np.uintp
    elif ints is np.int32_t:
        int_type = np.int32
    elif ints is np.int64_t:
        int_type = np.int64

    simplex_edges = np.empty((n_simplex, edges_per_simplex), dtype=int_type)

    if dim == 3:
        simplex_faces = np.empty((n_simplex, faces_per_simplex), dtype=int_type)
    else:
        simplex_faces = simplex_edges

    cdef ints[:,:] edge_pairs = np.array(
        [[1, 2], [0, 2], [0, 1], [0, 3], [1, 3], [2, 3]],
        dtype=int_type
    )

    for i_simp in range(n_simplex):
        simplex = simplices[i_simp]

        # build edges
        for i_edge in range(edges_per_simplex):

            v1 = simplex[edge_pairs[i_edge, 0]]
            v2 = simplex[edge_pairs[i_edge, 1]]
            edge = pair[ints, ints](v1, v2)

            edge_search = edges.find(edge)
            if edge_search != edges.end():
                ind = dereference(edge_search).second
            else:
                ind = n_edges
                edges[edge] = ind
                n_edges += 1
            simplex_edges[i_simp, i_edge] = ind

        # build faces in 3D
        if dim == 3:
            for i_face in range(4):
                if i_face == 0:
                    v1 = simplex[1]
                    v2 = simplex[2]
                    v3 = simplex[3]
                elif i_face == 1:
                    v1 = simplex[0]
                    v2 = simplex[2]
                    v3 = simplex[3]
                elif i_face == 2:
                    v1 = simplex[0]
                    v2 = simplex[1]
                    v3 = simplex[3]
                else:
                    v1 = simplex[0]
                    v2 = simplex[1]
                    v3 = simplex[2]
                face = triplet[ints, ints, ints](v1, v2, v3)

                face_search = faces.find(face)
                if face_search != faces.end():
                    ind = dereference(face_search).second
                else:
                    ind = n_faces
                    faces[face] = ind
                    n_faces += 1
                simplex_faces[i_simp, i_face] = ind

    cdef ints[:, :] _edges = np.empty((n_edges, 2), dtype=int_type)
    for edge_it in edges:
        _edges[edge_it.second, 0] = edge_it.first.first
        _edges[edge_it.second, 1] = edge_it.first.second

    cdef ints[:, :] _faces
    if dim == 3:
        _faces =  np.empty((n_faces, 3), dtype=int_type)
        for face_it in faces:
            _faces[face_it.second, 0] = face_it.first.v1
            _faces[face_it.second, 1] = face_it.first.v2
            _faces[face_it.second, 2] = face_it.first.v3
    else:
        _faces = _edges

    cdef ints[:, :] face_edges
    cdef ints[:] _face

    if dim == 3:
        face_edges = np.empty((n_faces, 3), dtype=int_type)
        for i_face in range(n_faces):
            _face = _faces[i_face]
            # get indices of each edge in the face
            # 3 edges per face
            for i_edge in range(3):
                if i_edge == 0:
                    v1 = _face[1]
                    v2 = _face[2]
                elif i_edge == 1:
                    v1 = _face[0]
                    v2 = _face[2]
                elif i_edge == 2:
                    v1 = _face[0]
                    v2 = _face[1]
                # because of how faces were constructed, v1 < v2 always
                edge = pair[ints, ints](v1, v2)
                ind = edges[edge]
                face_edges[i_face, i_edge] = ind
    else:
        face_edges = np.empty((1, 1), dtype=int_type)

    return simplex_faces, _faces, simplex_edges, _edges, face_edges

@cython.boundscheck(False)
def _build_adjacency(ints[:, :] simplex_faces, n_faces):
    cdef:
        size_t n_cells = simplex_faces.shape[0]
        int dim = simplex_faces.shape[1] - 1
        np.int64_t[:, :] neighbors
        np.int64_t[:] visited
        ints[:] simplex

        ints i_cell, j, k, i_face, i_other

    if ints is size_t:
        int_type = np.uintp
    elif ints is np.int32_t:
        int_type = np.int32
    elif ints is np.int64_t:
        int_type = np.int64

    neighbors = np.full((n_cells, dim + 1), -1, dtype=np.int64)
    visited = np.full((n_faces), -1, dtype=np.int64)

    for i_cell in range(n_cells):
        simplex = simplex_faces[i_cell]
        for j in range(dim + 1):
            i_face = simplex[j]
            i_other = visited[i_face]
            if i_other == -1:
                visited[i_face] = i_cell
            else:
                neighbors[i_cell, j] = i_other
                k = 0
                while (k < dim + 1) and (simplex_faces[i_other, k] != i_face):
                    k += 1
                neighbors[i_other, k] = i_cell
    return neighbors

cdef inline np.uint64_t _xorshift64(np.uint64_t x) noexcept nogil:
    x ^= x << 13
    x ^= x >> 7
    x ^= x << 17
    return x


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline bint _barycentric(
    np.float64_t[:, :] locs,
    Py_ssize_t i,
    np.float64_t[:, :, :] transform,
    np.float64_t[:, :] shift,
    np.int64_t c,
    np.float64_t eps,
    np.float64_t *barys,
    bint stop_early=False,
) noexcept nogil:
    """Point i's barycentric coordinates in simplex c; whether it is inside.

    With ``stop_early``, returns as soon as a coordinate shows the point is
    outside, leaving ``barys`` incomplete.
    """
    cdef:
        int a, b, dim = locs.shape[1]
        bint inside = True
        np.float64_t v, bsum = 0.0
    for a in range(dim):
        v = 0.0
        for b in range(dim):
            v = v + transform[c, a, b] * (locs[i, b] - shift[c, b])
        if v < -eps:
            if stop_early:
                return False
            inside = False
        barys[a] = v
        bsum = bsum + v
    barys[dim] = 1.0 - bsum
    return inside and barys[dim] >= -eps


@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.int64_t _search_all(
    np.float64_t[:, :] locs,
    Py_ssize_t i,
    np.float64_t[:, :, :] transform,
    np.float64_t[:, :] shift,
    np.float64_t eps,
    np.float64_t *barys,
) noexcept nogil:
    """A simplex containing point i found by testing every simplex, or -1."""
    cdef np.int64_t c
    for c in range(transform.shape[0]):
        if _barycentric(locs, i, transform, shift, c, eps, barys, True):
            return c
    return -1


@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.int64_t _walk(
    np.float64_t[:, :] locs,
    Py_ssize_t i,
    np.int64_t *i_simp,
    np.int64_t[:, :] neighbors,
    np.float64_t[:, :, :] transform,
    np.float64_t[:, :] shift,
    np.int64_t max_steps,
    np.float64_t eps,
    np.uint64_t *rng,
    np.float64_t *barys,
) noexcept nogil:
    """Walk from simplex ``i_simp[0]`` towards point i.

    At each step the walk crosses a randomly chosen face that the point lies
    beyond, without stepping straight back to the simplex it came from. Unlike
    always crossing the first such face, this terminates on any valid
    triangulation, not only Delaunay ones (Devillers, Pion and Teillaud, 2002).

    Returns the simplex containing the point, -1 if the walk stopped at the
    boundary (every face the point is beyond is a boundary face), or -2 if it
    took more than ``max_steps`` steps. ``i_simp[0]`` is left at the simplex
    the walk ended in, and ``barys`` holds the point's coordinates in it.
    """
    cdef:
        int j, k, k0, dim = locs.shape[1]
        np.int64_t c = i_simp[0], prev = -1, nb, choice, steps = 0
        bint hit_boundary
    while True:
        i_simp[0] = c
        if _barycentric(locs, i, transform, shift, c, eps, barys):
            return c

        hit_boundary = False
        choice = -1
        rng[0] = _xorshift64(rng[0])
        k0 = <int>(rng[0] % <np.uint64_t>(dim + 1))
        for k in range(dim + 1):
            j = (k0 + k) % (dim + 1)
            if barys[j] < -eps:
                nb = neighbors[c, j]
                if nb == -1:
                    hit_boundary = True
                elif nb != prev:
                    choice = nb
                    break
        if choice == -1:
            if hit_boundary:
                return -1
            # The only face the point is beyond leads back to the previous
            # simplex, so it is on their shared face to within round off.
            return c
        if steps >= max_steps:
            return -2
        prev = c
        c = choice
        steps += 1


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline bint _enters(
    np.float64_t *o,
    np.float64_t *d,
    np.float64_t[:, :, :] facets,
    Py_ssize_t f,
    int dim,
    np.float64_t tol,
    np.float64_t *t_out,
) noexcept nogil:
    """Whether the segment ``o + t * d`` (0 < t <= 1) enters the mesh through
    boundary facet f, and where (``t_out``).

    ``facets`` (n_facets, dim, dim) are the boundary faces (segments in 2D,
    triangles in 3D), each oriented with its normal pointing out of the mesh,
    so the segment enters through a facet when it crosses it against its
    normal. Crossings within ``tol`` of a facet's edges count, so a segment
    through an edge shared by two facets enters through one of them.
    """
    cdef np.float64_t t, s, u, w, det, inv
    cdef np.float64_t ex, ey, ez, gx, gy, gz, hx, hy, hz, ox, oy, oz, qx, qy, qz
    if dim == 2:
        # Solve o + t * d = a + s * (b - a).
        ex = facets[f, 1, 0] - facets[f, 0, 0]
        ey = facets[f, 1, 1] - facets[f, 0, 1]
        det = d[0] * ey - d[1] * ex
        # The outward normal is (ey, -ex), so entering means
        # d . (ey, -ex) = det < 0.
        if det >= 0.0:
            return False
        ox = facets[f, 0, 0] - o[0]
        oy = facets[f, 0, 1] - o[1]
        t = (ox * ey - oy * ex) / det
        s = (ox * d[1] - oy * d[0]) / det
        if s < -tol or s > 1.0 + tol:
            return False
    else:
        # Moller-Trumbore ray-triangle intersection.
        ex = facets[f, 1, 0] - facets[f, 0, 0]
        ey = facets[f, 1, 1] - facets[f, 0, 1]
        ez = facets[f, 1, 2] - facets[f, 0, 2]
        gx = facets[f, 2, 0] - facets[f, 0, 0]
        gy = facets[f, 2, 1] - facets[f, 0, 1]
        gz = facets[f, 2, 2] - facets[f, 0, 2]
        hx = d[1] * gz - d[2] * gy
        hy = d[2] * gx - d[0] * gz
        hz = d[0] * gy - d[1] * gx
        # det = -d . (e x g), and e x g is the outward normal, so the
        # segment enters through the facet when det > 0.
        det = ex * hx + ey * hy + ez * hz
        if det <= 0.0:
            return False
        inv = 1.0 / det
        ox = o[0] - facets[f, 0, 0]
        oy = o[1] - facets[f, 0, 1]
        oz = o[2] - facets[f, 0, 2]
        u = (ox * hx + oy * hy + oz * hz) * inv
        if u < -tol or u > 1.0 + tol:
            return False
        qx = oy * ez - oz * ey
        qy = oz * ex - ox * ez
        qz = ox * ey - oy * ex
        w = (d[0] * qx + d[1] * qy + d[2] * qz) * inv
        if w < -tol or u + w > 1.0 + tol:
            return False
        t = (gx * qx + gy * qy + gz * qz) * inv
    t_out[0] = t
    return t > 0.0 and t <= 1.0 + tol


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef np.int64_t _reentry_facet(
    np.float64_t[:, :] locs,
    Py_ssize_t i,
    np.float64_t[:] origin,
    np.float64_t[:, :, :] facets,
    np.float64_t[:] grid_origin,
    np.float64_t grid_h,
    np.int64_t[:] grid_shape,
    np.int64_t[:] grid_ptr,
    np.int64_t[:] grid_facets,
    np.float64_t tol,
    np.int64_t *used,
    int n_used,
) noexcept nogil:
    """The first boundary facet through which the segment from ``origin`` to
    point i enters the mesh, skipping the ``n_used`` facets in ``used``.

    Only the facets listed in the grid cells the segment passes through are
    tested (``grid_ptr`` and ``grid_facets`` hold, in CSR form, the facets
    whose bounding box overlaps each cell of a uniform grid), visiting the
    cells in order along the segment (Amanatides and Woo, 1987). A facet hit
    no further along than the end of the current grid cell is the first one,
    since every facet the segment crosses before there is listed in a cell
    already visited.

    Returns -1 if the segment does not enter the mesh, and -2 if it only
    enters through facets in ``used``.
    """
    cdef:
        int a, k, dim = locs.shape[1], axis
        Py_ssize_t j, f, cell, n_steps = 0, max_grid_steps = 0
        bint is_used, saw_used = False
        np.int64_t best = -1
        np.int64_t ix[3]
        np.int64_t step[3]
        np.float64_t o[3]
        np.float64_t d[3]
        np.float64_t t_max[3]
        np.float64_t t_delta[3]
        np.float64_t best_t = 2.0, t, t_end
    for a in range(dim):
        o[a] = origin[a]
        d[a] = locs[i, a] - origin[a]
        ix[a] = <np.int64_t>((o[a] - grid_origin[a]) / grid_h)
        if ix[a] < 0:
            ix[a] = 0
        if ix[a] >= grid_shape[a]:
            ix[a] = grid_shape[a] - 1
        if d[a] > 0.0:
            step[a] = 1
            t_max[a] = (grid_origin[a] + (ix[a] + 1) * grid_h - o[a]) / d[a]
            t_delta[a] = grid_h / d[a]
        elif d[a] < 0.0:
            step[a] = -1
            t_max[a] = (grid_origin[a] + ix[a] * grid_h - o[a]) / d[a]
            t_delta[a] = -grid_h / d[a]
        else:
            step[a] = 0
            t_max[a] = 2.0
            t_delta[a] = 2.0
        max_grid_steps += grid_shape[a]

    while True:
        cell = ix[0]
        for a in range(1, dim):
            cell = cell * grid_shape[a] + ix[a]
        for j in range(grid_ptr[cell], grid_ptr[cell + 1]):
            f = grid_facets[j]
            if not _enters(o, d, facets, f, dim, tol, &t) or t >= best_t:
                continue
            is_used = False
            for k in range(n_used):
                if used[k] == f:
                    is_used = True
                    break
            if is_used:
                saw_used = True
            else:
                best_t = t
                best = f
        axis = 0
        for a in range(1, dim):
            if t_max[a] < t_max[axis]:
                axis = a
        t_end = t_max[axis]
        if best != -1 and best_t <= t_end:
            return best
        if t_end >= 1.0:
            break
        ix[axis] += step[axis]
        if ix[axis] < 0 or ix[axis] >= grid_shape[axis]:
            break
        t_max[axis] += t_delta[axis]
        # Each step moves to a new cell along the segment, so this only
        # guards against round off.
        n_steps += 1
        if n_steps > max_grid_steps:
            break
    if best == -1 and saw_used:
        return -2
    return best


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline np.float64_t _facet_distance2(
    np.float64_t *p,
    np.float64_t[:, :, :] facets,
    Py_ssize_t f,
    int dim,
    np.float64_t *height,
) noexcept nogil:
    """Squared distance from p to boundary facet f.

    Also sets ``height``, p's signed distance from the facet's plane along its
    outward normal.
    """
    cdef:
        int a
        np.float64_t ab[3]
        np.float64_t ac[3]
        np.float64_t ap[3]
        np.float64_t q[3]
        np.float64_t n[3]
        np.float64_t d1, d2, d3, d4, d5, d6, va, vb, vc, v, w, denom, nn, dist2
    for a in range(dim):
        ab[a] = facets[f, 1, a] - facets[f, 0, a]
        ap[a] = p[a] - facets[f, 0, a]
    if dim == 2:
        d1 = ab[0] * ap[0] + ab[1] * ap[1]
        nn = ab[0] * ab[0] + ab[1] * ab[1]
        v = d1 / nn
        if v < 0.0:
            v = 0.0
        elif v > 1.0:
            v = 1.0
        q[0] = ap[0] - v * ab[0]
        q[1] = ap[1] - v * ab[1]
        # outward normal (ab_y, -ab_x)
        height[0] = (ap[0] * ab[1] - ap[1] * ab[0]) / sqrt(nn)
        return q[0] * q[0] + q[1] * q[1]

    for a in range(3):
        ac[a] = facets[f, 2, a] - facets[f, 0, a]
    n[0] = ab[1] * ac[2] - ab[2] * ac[1]
    n[1] = ab[2] * ac[0] - ab[0] * ac[2]
    n[2] = ab[0] * ac[1] - ab[1] * ac[0]
    nn = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2])
    height[0] = (ap[0] * n[0] + ap[1] * n[1] + ap[2] * n[2]) / nn

    # Closest point on a triangle (Ericson, Real-Time Collision Detection,
    # section 5.1.5), found from its barycentric coordinates v and w.
    d1 = ab[0] * ap[0] + ab[1] * ap[1] + ab[2] * ap[2]
    d2 = ac[0] * ap[0] + ac[1] * ap[1] + ac[2] * ap[2]
    # bp = ap - ab, cp = ap - ac
    d3 = d1 - (ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2])
    d4 = d2 - (ac[0] * ab[0] + ac[1] * ab[1] + ac[2] * ab[2])
    d5 = d1 - (ab[0] * ac[0] + ab[1] * ac[1] + ab[2] * ac[2])
    d6 = d2 - (ac[0] * ac[0] + ac[1] * ac[1] + ac[2] * ac[2])
    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4
    if d1 <= 0.0 and d2 <= 0.0:
        v = 0.0
        w = 0.0
    elif d3 >= 0.0 and d4 <= d3:
        v = 1.0
        w = 0.0
    elif vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        w = 0.0
    elif d6 >= 0.0 and d5 <= d6:
        v = 0.0
        w = 1.0
    elif vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        v = 0.0
        w = d2 / (d2 - d6)
    elif va <= 0.0 and d4 - d3 >= 0.0 and d5 - d6 >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        v = 1.0 - w
    else:
        denom = 1.0 / (va + vb + vc)
        v = vb * denom
        w = vc * denom
    dist2 = 0.0
    for a in range(3):
        q[a] = ap[a] - v * ab[a] - w * ac[a]
        dist2 = dist2 + q[a] * q[a]
    return dist2


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void _nearest_in_cell(
    np.float64_t *p,
    Py_ssize_t cell,
    np.float64_t[:, :, :] facets,
    np.int64_t[:] grid_ptr,
    np.int64_t[:] grid_facets,
    int dim,
    np.int64_t *best,
    np.float64_t *best_d2,
    np.float64_t *best_height,
) noexcept nogil:
    """Update the nearest facet so far with the facets in one grid cell."""
    cdef:
        Py_ssize_t j, f
        np.float64_t d2, height, tie
    for j in range(grid_ptr[cell], grid_ptr[cell + 1]):
        f = grid_facets[j]
        d2 = _facet_distance2(p, facets, f, dim, &height)
        tie = 1E-12 * best_d2[0]
        # Several facets are equally near a point closest to an edge or node
        # they share; take the one the point is most directly in front of.
        if d2 < best_d2[0] - tie or (
            d2 <= best_d2[0] + tie and height > best_height[0]
        ):
            best[0] = f
            best_d2[0] = d2
            best_height[0] = height


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef np.int64_t _nearest_facet(
    np.float64_t[:, :] locs,
    Py_ssize_t i,
    np.float64_t[:, :, :] facets,
    np.float64_t[:] grid_origin,
    np.float64_t grid_h,
    np.int64_t[:] grid_shape,
    np.int64_t[:] grid_ptr,
    np.int64_t[:] grid_facets,
) noexcept nogil:
    """The boundary facet nearest to point i.

    Searches the grid cells in rings (cubic shells) around the cell nearest to
    the point, until the next ring can't hold anything nearer. Every facet is
    listed in each cell its bounding box overlaps, so after the ring at
    distance r (in cells) every facet within ``r * grid_h`` of the point has
    been seen. For a point outside the grid that holds for its projection onto
    the grid, which is no farther than the point from anything in the grid.
    """
    cdef:
        int a, dim = locs.shape[1]
        np.int64_t r, r_max = 0, i0, i1, i2, n0, n1, n2
        np.int64_t c0[3]
        np.int64_t lo[3]
        np.int64_t hi[3]
        np.int64_t best = -1
        bint on0, on1
        np.float64_t p[3]
        np.float64_t x, best_d2 = INFINITY, best_height = -INFINITY
    for a in range(dim):
        p[a] = locs[i, a]
        x = floor((p[a] - grid_origin[a]) / grid_h)
        if x < 0.0:
            x = 0.0
        if x > grid_shape[a] - 1:
            x = grid_shape[a] - 1
        c0[a] = <np.int64_t>x
        r_max = max(r_max, grid_shape[a])
    n0 = grid_shape[0]
    n1 = grid_shape[1]
    n2 = grid_shape[2] if dim == 3 else 1

    for r in range(r_max + 1):
        for a in range(dim):
            lo[a] = c0[a] - r
            hi[a] = c0[a] + r
        # Visit only the cells on the shell of the box of cells within r.
        for i0 in range(max(lo[0], 0), min(hi[0], n0 - 1) + 1):
            on0 = i0 == lo[0] or i0 == hi[0]
            if dim == 2:
                if on0:
                    for i1 in range(max(lo[1], 0), min(hi[1], n1 - 1) + 1):
                        _nearest_in_cell(
                            p, i0 * n1 + i1, facets, grid_ptr, grid_facets, dim,
                            &best, &best_d2, &best_height
                        )
                else:
                    if lo[1] >= 0:
                        _nearest_in_cell(
                            p, i0 * n1 + lo[1], facets, grid_ptr, grid_facets, dim,
                            &best, &best_d2, &best_height
                        )
                    if hi[1] < n1:
                        _nearest_in_cell(
                            p, i0 * n1 + hi[1], facets, grid_ptr, grid_facets, dim,
                            &best, &best_d2, &best_height
                        )
                continue
            for i1 in range(max(lo[1], 0), min(hi[1], n1 - 1) + 1):
                on1 = i1 == lo[1] or i1 == hi[1]
                if on0 or on1:
                    for i2 in range(max(lo[2], 0), min(hi[2], n2 - 1) + 1):
                        _nearest_in_cell(
                            p, (i0 * n1 + i1) * n2 + i2, facets, grid_ptr,
                            grid_facets, dim, &best, &best_d2, &best_height
                        )
                else:
                    if lo[2] >= 0:
                        _nearest_in_cell(
                            p, (i0 * n1 + i1) * n2 + lo[2], facets, grid_ptr,
                            grid_facets, dim, &best, &best_d2, &best_height
                        )
                    if hi[2] < n2:
                        _nearest_in_cell(
                            p, (i0 * n1 + i1) * n2 + hi[2], facets, grid_ptr,
                            grid_facets, dim, &best, &best_d2, &best_height
                        )
        if best != -1 and best_d2 <= (r * grid_h) * (r * grid_h):
            break
    return best


@cython.boundscheck(False)
@cython.wraparound(False)
cdef bint _outside_hull(
    np.float64_t[:, :] locs,
    Py_ssize_t i,
    np.float64_t[:] lower,
    np.float64_t[:] upper,
    np.float64_t[:, :] hull_equations,
    np.float64_t tol,
) noexcept nogil:
    """Whether point i is clearly outside the mesh's bounding box or convex
    hull (``hull_equations`` are scipy's ConvexHull.equations, with unit
    outward normals)."""
    cdef:
        int a, dim = locs.shape[1]
        Py_ssize_t h
        np.float64_t v
    for a in range(dim):
        if locs[i, a] < lower[a] - tol or locs[i, a] > upper[a] + tol:
            return True
    for h in range(hull_equations.shape[0]):
        v = hull_equations[h, dim]
        for a in range(dim):
            v = v + hull_equations[h, a] * locs[i, a]
        if v > tol:
            return True
    return False


# A point whose walk needs more than this many jumps across gaps in the mesh
# is found by testing every simplex instead.
cdef enum:
    _MAX_JUMPS = 64


@cython.boundscheck(False)
@cython.wraparound(False)
def _directed_search(
    np.float64_t[:, :] locs,
    np.int64_t start,
    np.int64_t[:, :] neighbors,
    np.float64_t[:, :, :] transform,
    np.float64_t[:, :] shift,
    np.int64_t max_steps,
    np.float64_t[:, :] cell_centers,
    np.float64_t[:, :, :] facets,
    np.int64_t[:] facet_cells,
    bint verify_outside,
    np.float64_t[:] lower,
    np.float64_t[:] upper,
    np.float64_t[:, :] hull_equations,
    np.float64_t[:] grid_origin,
    np.float64_t grid_h,
    np.int64_t[:] grid_shape,
    np.int64_t[:] grid_ptr,
    np.int64_t[:] grid_facets,
    bint nearest_outside,
    np.float64_t eps=1E-15,
):
    """Locate points by walking from simplex to simplex.

    The points are walked in the given order, each one starting from the simplex
    the previous point was found in (the first starts at ``start``). A walk that
    takes more than ``max_steps`` steps falls back to testing every simplex.

    A walk that stops at the boundary proves the point is outside only on a
    convex mesh. Otherwise (``verify_outside``) a point outside the mesh's
    bounding box (``lower``, ``upper``) or convex hull (``hull_equations``) is
    still outside. If not, the segment from the center of the simplex the walk
    stopped in to the point is tested against the boundary ``facets`` (through
    a uniform grid over them, see ``_reentry_facet``): it leaves the mesh
    through the face the walk stopped at, and if it enters again, the walk
    continues from the simplex (``facet_cells``) behind the first facet it
    enters through. If it never enters again, the point is outside. A point
    never jumps through the same facet twice, so a walk that would circle back
    falls back to testing every simplex. Either way, the next point's walk
    starts from the simplex actually found.

    With ``nearest_outside``, a point outside the mesh is given the simplex
    behind the boundary facet nearest to it (see ``_nearest_facet``), so it
    doesn't depend on where its walk started. Otherwise it keeps the simplex
    its walk ended in.

    Returns
    -------
    status : (n_locs) numpy.ndarray of int
        The simplex containing each point, or -1 if it is outside the mesh.
    last : (n_locs) numpy.ndarray of int
        The simplex containing each point, or for points outside the mesh the
        simplex given to it (see ``nearest_outside``).
    barys : (n_locs, dim + 1) numpy.ndarray of float
        Each point's barycentric coordinates in ``last``.
    n_jumps : int
        How many times a walk had to continue across a gap in the mesh.
    n_searched : int
        How many points needed testing every simplex.
    """
    cdef:
        Py_ssize_t i
        int a, jumps
        int n_locs = locs.shape[0], dim = locs.shape[1]
        np.int64_t i_simp = start, state, f
        Py_ssize_t n_jumps = 0, n_searched = 0
        np.uint64_t rng = 0x9E3779B97F4A7C15
        np.float64_t barys[4]
        np.float64_t found_barys[4]
        np.int64_t used[_MAX_JUMPS]
        np.float64_t hull_tol = 0.0
        np.int64_t[:] status = np.empty(n_locs, dtype=np.int64)
        np.int64_t[:] last = np.empty(n_locs, dtype=np.int64)
        np.float64_t[:, :] all_barys = np.empty((n_locs, dim + 1), dtype=np.float64)

    if verify_outside:
        for a in range(dim):
            hull_tol = max(hull_tol, upper[a] - lower[a])
        hull_tol = 1E-10 * hull_tol

    with nogil:
        for i in range(n_locs):
            state = _walk(
                locs, i, &i_simp, neighbors, transform, shift, max_steps, eps,
                &rng, barys
            )
            if (
                state == -1 and verify_outside
                and _outside_hull(locs, i, lower, upper, hull_equations, hull_tol)
            ):
                # Nothing to cross over to.
                pass
            else:
                jumps = 0
                while state == -1 and verify_outside:
                    if jumps == _MAX_JUMPS:
                        state = -2
                        break
                    f = _reentry_facet(
                        locs, i, cell_centers[i_simp], facets, grid_origin, grid_h,
                        grid_shape, grid_ptr, grid_facets, 1E-12, used, jumps
                    )
                    if f == -1:
                        break
                    if f == -2:
                        # Every way back in is a facet this point already
                        # jumped through, so the walk would go in circles.
                        state = -2
                        break
                    used[jumps] = f
                    jumps += 1
                    n_jumps += 1
                    i_simp = facet_cells[f]
                    state = _walk(
                        locs, i, &i_simp, neighbors, transform, shift, max_steps,
                        eps, &rng, barys
                    )

            if state == -2:
                n_searched += 1
                state = _search_all(locs, i, transform, shift, eps, found_barys)
                if state >= 0:
                    i_simp = state
                    for a in range(dim + 1):
                        barys[a] = found_barys[a]
            if state == -1 and nearest_outside:
                f = _nearest_facet(
                    locs, i, facets, grid_origin, grid_h, grid_shape, grid_ptr,
                    grid_facets
                )
                if f >= 0:
                    i_simp = facet_cells[f]
                    _barycentric(locs, i, transform, shift, i_simp, eps, barys)
            status[i] = state
            last[i] = i_simp
            for a in range(dim + 1):
                all_barys[i, a] = barys[a]

    return (
        np.asarray(status), np.asarray(last), np.asarray(all_barys), n_jumps, n_searched
    )


@cython.boundscheck(False)
@cython.cdivision(True)
def _interp_cc(
  np.float64_t[:, :] locs,
  np.float64_t[:, :] cell_centers,
  np.float64_t[:] mat_data,
  ints[:] mat_indices,
  ints[:] mat_indptr,
):
    cdef:
        ints i, j, diff, start, stop, i_d
        ints n_max_per_row = 0
        ints[:] close_cells
        int dim = locs.shape[1]
        np.float64_t[:, :] drs
        np.float64_t[:] rs
        np.float64_t[:] rhs
        np.float64_t[:] lambs
        np.float64_t[:] weights
        np.float64_t[:] point
        np.float64_t[:] close_cell
        np.float64_t det, weight_sum
        np.float64_t xx, xy, xz, yy, yz, zz
        bint too_close
        np.float64_t eps = 1E-15
    # Find maximum number per row to pre-allocate a storage
    for i in range(locs.shape[0]):
        diff = mat_indptr[i+1] - mat_indptr[i]
        if diff > n_max_per_row:
          n_max_per_row = diff
    #
    drs = np.empty((n_max_per_row, dim), dtype=np.float64)
    rs = np.empty((n_max_per_row,), dtype=np.float64)
    rhs = np.empty((dim,),dtype=np.float64)
    lambs = np.empty((dim,),dtype=np.float64)

    for i in range(locs.shape[0]):
        point = locs[i]
        start = mat_indptr[i]
        stop = mat_indptr[i+1]
        diff = stop-start
        close_cells = mat_indices[start:stop]
        for j in range(diff):
            rs[j] = 0.0
            close_cell = cell_centers[close_cells[j]]
            for i_d in range(dim):
                drs[j, i_d] = close_cell[i_d] - point[i_d]
                rs[j] += drs[j, i_d]*drs[j, i_d]
            rs[j] = sqrt(rs[j])
        weights = mat_data[start:stop]
        weights[:] = 0.0
        too_close = False
        i_d = 0
        for j in range(diff):
            if rs[j] < eps:
                too_close = True
                i_d = j
        if too_close:
            weights[i_d] = 1.0
        else:
            for j in range(diff):
                for i_d in range(dim):
                    drs[j, i_d] /= rs[j]
            xx = xy = yy = 0.0
            rhs[:] = 0.0
            if dim == 2:
                for j in range(diff):
                    xx += drs[j, 0] * drs[j, 0]
                    xy += drs[j, 0] * drs[j, 1]
                    yy += drs[j, 1] * drs[j, 1]
                    rhs[0] -= drs[j, 0]
                    rhs[1] -= drs[j, 1]
                det = xx * yy - xy * xy
                lambs[0] = (yy * rhs[0] - xy * rhs[1])/det
                lambs[1] = (-xy * rhs[0] + xx * rhs[1])/det

            if dim == 3:
                zz = xz = yz = 0.0
                for j in range(diff):
                    xx += drs[j, 0] * drs[j, 0]
                    xy += drs[j, 0] * drs[j, 1]
                    yy += drs[j, 1] * drs[j, 1]
                    xz += drs[j, 0] * drs[j, 2]
                    yz += drs[j, 1] * drs[j, 2]
                    zz += drs[j, 2] * drs[j, 2]
                    rhs[0] -= drs[j, 0]
                    rhs[1] -= drs[j, 1]
                    rhs[2] -= drs[j, 2]

                det = (
                      xx * (yy * zz - yz * yz)
                      + xy * (xz * yz - xy * zz)
                      + xz * (xy * yz - xz * yy)
                )
                lambs[0] = (
                    (yy * zz - yz * yz) * rhs[0]
                    + (xz * yz - xy * zz) * rhs[1]
                    + (xy * yz - xz * yy) * rhs[2]
                )/det
                lambs[1] = (
                    (xz * yz - xy * zz) * rhs[0]
                    + (xx * zz - xz * xz) * rhs[1]
                    + (xy * xz - xx * yz) * rhs[2]
                )/det
                lambs[2] = (
                    (xy * yz - xz * yy) * rhs[0]
                    + (xz * xy - xx * yz) * rhs[1]
                    + (xx * yy - xy * xy) * rhs[2]
                )/det

            weight_sum = 0.0
            for j in range(diff):
                weights[j] = 1.0
                for i_d in range(dim):
                    weights[j] += lambs[i_d] * drs[j, i_d]
                weights[j] /= rs[j]
                weight_sum += weights[j]
            for j in range(diff):
                weights[j] /= weight_sum
