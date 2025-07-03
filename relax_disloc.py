import gmsh
import meshio
import numpy as np
import pandas as pd
import scipy
import addict
from matplotlib import path
import scipy

N_MESH_DIM = 3


def make_tri_horizon(x, y, z, el_size, ztilt=0.01):
    """
    Makes a simple ~horizon of triangular dislocation elements using Gmsh
    x, y are 2-element arrays defining horizontal bounds
    z is a scalar giving depth
    el_size gives nominal element size
    """
    # Mesh construction using Gmsh
    if gmsh.isInitialized() == 0:
        gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.clear()
    # Define points
    gmsh.model.geo.addPoint(x[0], y[0], z - ztilt, el_size, 1)
    gmsh.model.geo.addPoint(x[1], y[0], z + ztilt, el_size, 2)
    gmsh.model.geo.addPoint(x[1], y[1], z + ztilt, el_size, 3)
    gmsh.model.geo.addPoint(x[0], y[1], z - ztilt, el_size, 4)
    # Boundary lines
    gmsh.model.geo.addLine(1, 2, 1)
    gmsh.model.geo.addLine(2, 3, 2)
    gmsh.model.geo.addLine(3, 4, 3)
    gmsh.model.geo.addLine(4, 1, 4)
    # Perimeter
    gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 1)
    # Surface
    gmsh.model.geo.addPlaneSurface([1], 1)
    gmsh.model.geo.synchronize()
    # Generate and write. Writing a file allows use of meshio, consistent with celeri codes
    gmsh.model.mesh.generate(2)
    gmsh.write("horiz.msh")
    gmsh.finalize()


def read_mesh(meshfile):
    # Read and parse mesh
    mesh = meshio.read(meshfile)
    fault_pts = mesh.points
    fault_tri = meshio.CellBlock("triangle", mesh.get_cells_type("triangle")).data
    return fault_pts, fault_tri


def make_chichi_mesh(sourcename, source_el_size):
    """
    Make a Chi Chi source fault mesh, based on a convex hull around the patches from Rousset et al.
    """

    colnames = [
        "no",
        "slip",
        "ys",
        "xs",
        "zs",
        "length",
        "width",
        "strike",
        "dip",
        "rake",
    ]
    faults = pd.read_csv(sourcename, sep="\\s+", header=None, names=colnames)

    # Find convex hull around x, y points of faults
    faultpoints = faults[["xs", "ys"]].to_numpy()
    hull = scipy.spatial.ConvexHull(faultpoints)

    # Mesh convex hull as a network of TDEs

    if gmsh.isInitialized() == 0:
        gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.clear()
    # Define points
    for i in range(len(hull.vertices)):
        gmsh.model.geo.addPoint(
            faultpoints[hull.vertices[i], 0],
            faultpoints[hull.vertices[i], 1],
            0,
            source_el_size,
            i,
        )
    # Define lines
    for i in range(len(hull.vertices) - 1):
        gmsh.model.geo.addLine(i, i + 1, i)
    gmsh.model.geo.addLine(i + 1, 0, i + 1)
    # Define curve loop
    gmsh.model.geo.addCurveLoop(list(range(0, i + 2)), 1)
    # Surface
    gmsh.model.geo.addPlaneSurface([1], 1)
    gmsh.model.geo.synchronize()
    # Generate and write. Writing a file allows use of meshio, consistent with celeri codes
    gmsh.model.mesh.generate(2)

    # Access node coordinates and interpolate depths
    nodetags, nodecoords, _ = gmsh.model.mesh.getNodes(-1, -1)
    interp = scipy.interpolate.NearestNDInterpolator(
        faultpoints, faults[["zs"]].to_numpy()
    )
    nodexy = np.array((nodecoords[0::3], nodecoords[1::3]))
    interp_depths = interp(nodexy.T)
    nodecoords[2::3] = -interp_depths[:, 0]

    # Reassign depth-interpolated coordinates
    for j in range(len(nodetags)):
        gmsh.model.mesh.setNode(nodetags[j], nodecoords[3 * j : 3 * j + 3], [])
    gmsh.write("chichi.msh")
    gmsh.finalize()


def make_chelungpu_mesh():
    """
    Make the Hsu et al. fault for kinematic afterslip inversion
    Geometry is based on digitized fault corners, from Rousset Figure 7
    """
    corners = np.array(
        (
            (-32.068311195445915, -25.265553869499243),
            (-25.237191650853887, 54.40060698027315),
            (27.514231499051235, 49.848254931714735),
            (19.924098671726753, -30.197268588770868),
        )
    )
    strike = np.arctan2(corners[1, 0] - corners[0, 0], corners[1, 1] - corners[0, 1])

    # Approximated, using just SW corner and projecting
    strike = np.deg2rad(5)
    top_panel_width = 20
    top_panel_dip = 30
    top_panel_horiz = top_panel_width * np.cos(np.deg2rad(top_panel_dip))
    top_panel_bot = top_panel_width * np.sin(np.deg2rad(top_panel_dip))
    bot_panel_width = 35
    bot_panel_dip = 5
    bot_panel_horiz = bot_panel_width * np.cos(np.deg2rad(bot_panel_dip))
    bot_panel_bot = bot_panel_width * np.sin(np.deg2rad(bot_panel_dip))
    fault_length = 80

    # Panel coordinates
    top_panel_rect = np.array(
        (
            (0, 0),
            (top_panel_horiz, 0),
            (top_panel_horiz, fault_length),
            (0, fault_length),
        )
    )
    bot_panel_rect = np.array(
        (
            (0, 0),
            (bot_panel_horiz, 0),
            (bot_panel_horiz, fault_length),
            (0, fault_length),
        )
    )

    # Rotate by strike
    top_panel_rect = np.array(
        (
            np.cos(strike) * top_panel_rect[:, 0]
            + np.sin(strike) * top_panel_rect[:, 1],
            -np.sin(strike) * top_panel_rect[:, 0]
            + np.cos(strike) * top_panel_rect[:, 1],
        )
    ).T
    bot_panel_rect = np.array(
        (
            np.cos(strike) * bot_panel_rect[:, 0]
            + np.sin(strike) * bot_panel_rect[:, 1],
            -np.sin(strike) * bot_panel_rect[:, 0]
            + np.cos(strike) * bot_panel_rect[:, 1],
        )
    ).T

    # Shift by SW corners
    top_panel_rect = top_panel_rect + corners[0, :]
    bot_panel_rect = bot_panel_rect + top_panel_rect[1, :]

    # Add depths
    top_panel_rect = np.hstack(
        (top_panel_rect, np.array([[0], [top_panel_bot], [top_panel_bot], [0]]))
    )
    bot_panel_rect = np.hstack(
        (
            bot_panel_rect,
            top_panel_bot + np.array([[0], [bot_panel_bot], [bot_panel_bot], [0]]),
        )
    )

    # Create mesh
    if gmsh.isInitialized() == 0:
        gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.clear()
    # Define points
    # From top
    for i in range(4):
        gmsh.model.geo.addPoint(
            top_panel_rect[i, 0], top_panel_rect[i, 1], -top_panel_rect[i, 2], 5, i
        )
    # From bottom
    gmsh.model.geo.addPoint(
        bot_panel_rect[1, 0], bot_panel_rect[1, 1], -bot_panel_rect[1, 2], 5, 4
    )
    gmsh.model.geo.addPoint(
        bot_panel_rect[2, 0], bot_panel_rect[2, 1], -bot_panel_rect[2, 2], 5, 5
    )
    # Define lines
    for i in range(3):
        gmsh.model.geo.addLine(i, i + 1, i)
    # Top edge
    gmsh.model.geo.addLine(3, 0, i + 1)
    # Bottoms
    gmsh.model.geo.addLine(1, 4, 5)
    gmsh.model.geo.addLine(4, 5, 6)
    gmsh.model.geo.addLine(5, 2, 7)
    # Define curve loops
    gmsh.model.geo.addCurveLoop(list(range(0, 4)), 1)
    gmsh.model.geo.addCurveLoop((5, 6, 7, -1), 2)
    # Surfaces
    gmsh.model.geo.addPlaneSurface([1], 1)
    gmsh.model.geo.addPlaneSurface([2], 2)
    gmsh.model.geo.synchronize()
    gmsh.write("chelungpu.geo_unrolled")
    # Generate and write. Writing a file allows use of meshio, consistent with celeri codes
    gmsh.model.mesh.generate(2)
    gmsh.write("chelungpu.msh")
    gmsh.finalize()


def get_shared_sides(vertices):
    """
    Determine the indices of the triangular elements sharing
    one side with a particular element.
    Inputs:
    vertices: n x 3 array containing the 3 vertex indices of the n elements,
        assumes that values increase monotonically from 1:n

    Outputs:
    share: n x 3 array containing the indices of the m elements sharing a
        side with each of the n elements.  "-1" values in the array
        indicate elements with fewer than m neighbors (i.e., on
        the edge of the geometry).

    In general, elements will have 1 (mesh corners), 2 (mesh edges), or 3
    (mesh interiors) neighbors, but in the case of branching faults that
    have been adjusted with mergepatches, it's for edges and corners to
    also up to 3 neighbors.

    From https://github.com/brendanjmeade/celeri/
    """
    # Make side arrays containing vertex indices of sides
    side_1 = np.sort(np.vstack((vertices[:, 0], vertices[:, 1])).T, 1)
    side_2 = np.sort(np.vstack((vertices[:, 1], vertices[:, 2])).T, 1)
    side_3 = np.sort(np.vstack((vertices[:, 0], vertices[:, 2])).T, 1)
    sides_all = np.vstack((side_1, side_2, side_3))

    # Find the unique sides - each side can part of at most 2 elements
    _, first_occurence_idx = np.unique(sides_all, return_index=True, axis=0)
    _, last_occurence_idx = np.unique(np.flipud(sides_all), return_index=True, axis=0)
    last_occurence_idx = sides_all.shape[0] - last_occurence_idx - 1

    # Shared sides are those whose first and last indices are not equal
    shared = np.where((last_occurence_idx - first_occurence_idx) != 0)[0]

    # These are the indices of the shared sides
    sside1 = first_occurence_idx[shared]  # What should I name these variables?
    sside2 = last_occurence_idx[shared]

    el1, sh1 = np.unravel_index(
        sside1, vertices.shape, order="F"
    )  # "F" is for fortran ordering.  What should I call this variables?
    el2, sh2 = np.unravel_index(sside2, vertices.shape, order="F")
    share = -1 * np.ones((vertices.shape[0], 3))
    for i in range(el1.size):
        share[el1[i], sh1[i]] = el2[i]
        share[el2[i], sh2[i]] = el1[i]
    share = share.astype(int)
    return share


def get_all_mesh_smoothing_matrices_simple(meshes, operators):
    """
    Build smoothing matrices for each of the triangular meshes
    stored in meshes
    These are the simple not distance weighted meshes

    From https://github.com/brendanjmeade/celeri/
    """
    for i in range(len(meshes)):
        if len(meshes[i].coords) > 0:
            # Get smoothing operator for a single mesh.
            meshes[i].share = get_shared_sides(meshes[i].verts)
            operators.smoothing_matrix[i] = get_tri_smoothing_matrix_simple(
                meshes[i].share, N_MESH_DIM
            )


def get_tri_smoothing_matrix_simple(share, n_dim):
    """
    Produces a smoothing matrix based without scale-dependent
    weighting.

    Inputs:
    share: n x 3 array of indices of the up to 3 elements sharing a side
        with each of the n elements

    Outputs:
    smoothing matrix: n_dim * n x n_dim * n smoothing matrix

    From https://github.com/brendanjmeade/celeri/
    """

    # Allocate sparse matrix for contructing smoothing matrix
    n_shared_tri = share.shape[0]
    smoothing_matrix = scipy.sparse.lil_matrix(
        (n_dim * n_shared_tri, n_dim * n_shared_tri)
    )

    for j in range(n_dim):
        for i in range(n_shared_tri):
            smoothing_matrix[n_dim * i + j, n_dim * i + j] = 3
            if share[i, j] != -1:
                k = n_dim * i + np.arange(n_dim)
                m = n_dim * share[i, j] + np.arange(n_dim)
                smoothing_matrix[k, m] = -1
    return smoothing_matrix


def get_ordered_edge_nodes(meshes):
    """Find exterior edges of each mesh and return them in the dictionary
    for each mesh.

    Args:
        meshes (List): list of mesh dictionaries

    From https://github.com/brendanjmeade/celeri/
    """

    for i in range(len(meshes)):
        # Make side arrays containing vertex indices of sides
        vertices = meshes[i].verts
        side_1 = np.sort(np.vstack((vertices[:, 0], vertices[:, 1])).T, 1)
        side_2 = np.sort(np.vstack((vertices[:, 1], vertices[:, 2])).T, 1)
        side_3 = np.sort(np.vstack((vertices[:, 2], vertices[:, 0])).T, 1)
        all_sides = np.vstack((side_1, side_2, side_3))
        unique_sides, sides_count = np.unique(all_sides, return_counts=True, axis=0)
        edge_nodes = unique_sides[np.where(sides_count == 1)]

        meshes[i].ordered_edge_nodes = np.zeros_like(edge_nodes)
        meshes[i].ordered_edge_nodes[0, :] = edge_nodes[0, :]
        last_row = 0
        for j in range(1, len(edge_nodes)):
            idx = np.where(
                (edge_nodes == meshes[i].ordered_edge_nodes[j - 1, 1])
            )  # Edge node indices the same as previous row, second column
            next_idx = np.where(
                idx[0][:] != last_row
            )  # One of those indices is the last row itself. Find the other row index
            next_row = idx[0][next_idx]  # Index of the next ordered row
            next_col = idx[1][next_idx]  # Index of the next ordered column (1 or 2)
            if next_col == 1:
                next_col_ord = [1, 0]  # Flip edge ordering
            else:
                next_col_ord = [0, 1]
            meshes[i].ordered_edge_nodes[j, :] = edge_nodes[next_row, next_col_ord]
            last_row = (
                next_row  # Update last_row so that it's excluded in the next iteration
            )


def matrix_assembly(meshes, disp_mat, smoothing_weight):
    """
    Assembles elastic partials and smoothing matrix together

    Returns assembled matrix and smoothing vector
    """
    # Get triangular smoothing matrix
    operators = addict.Dict()
    get_all_mesh_smoothing_matrices_simple(meshes, operators)

    ndisp_comp = np.size(disp_mat, axis=1)
    nslip_comp = np.size(disp_mat, axis=3)

    if np.size(smoothing_weight) == 1:
        smoothing_weight = [smoothing_weight, smoothing_weight]

    nobs = ndisp_comp * np.size(disp_mat, axis=0)
    ntri = np.size(disp_mat, axis=2)
    nsource_tri = ntri - np.size(meshes[1].verts, axis=0)

    # Assemble matrices
    assembled_mat = np.zeros((nobs + nslip_comp * ntri, nslip_comp * ntri))
    # Insert elastic partials
    assembled_mat[0:nobs, :] = disp_mat.reshape((-1, nslip_comp * ntri))
    # Insert smoothing matrices
    source_row_start = nobs
    source_row_end = source_row_start + nslip_comp * nsource_tri
    horiz_row_start = source_row_end
    if nsource_tri != 0:
        assembled_mat[source_row_start:source_row_end, 0 : nslip_comp * nsource_tri] = (
            operators.smoothing_matrix[0].toarray()
        )
    assembled_mat[source_row_end:, nslip_comp * nsource_tri :] = (
        operators.smoothing_matrix[1].toarray()
    )

    # Assemble weighting vector
    weights = np.ones((np.shape(assembled_mat)[0], 1))
    weights[source_row_start:source_row_end] = smoothing_weight[0]
    weights[horiz_row_start:] = smoothing_weight[1]

    return assembled_mat, weights


def adjust_low_visc_coords(meshes, rampheight):

    # Outer bounds, digitized from Rousset et al. Fig. 11 using plotdigitizer.com
    lvob = np.array(
        [
            [-32.59259259259259, -88.58195211786372],
            [15.925925925925926, -106.26151012891344],
            [84.44444444444444, 83.05709023941068],
            [37.03703703703704, 100],
        ]
    )

    # Dimensions, width and length
    lvw = np.sqrt((lvob[1, 0] - lvob[0, 0]) ** 2 + (lvob[1, 1] - lvob[0, 1]) ** 2)
    lvl = np.sqrt((lvob[3, 0] - lvob[0, 0]) ** 2 + (lvob[3, 1] - lvob[0, 1]) ** 2)
    # Slope of side
    l_slope = (lvob[3, 0] - lvob[0, 0]) / (lvob[3, 1] - lvob[0, 1])
    l_ang = np.arctan(l_slope)
    cl_ang = np.cos(np.pi / 4 - l_ang)
    sl_ang = np.sin(np.pi / 4 - l_ang)

    # Inner bounds
    ib_buffer = 10
    ib_shift = np.sqrt(2 * ib_buffer**2) * np.array(
        [[cl_ang, sl_ang], [-sl_ang, cl_ang], [-cl_ang, -sl_ang], [sl_ang, -cl_ang]]
    )
    lvib = lvob + ib_shift

    # Outer outer bounds (grow LV region so we can have a smoother depth transition)
    oob_buffer = 10
    oob_shift = np.sqrt(2 * oob_buffer**2) * -np.array(
        [[cl_ang, sl_ang], [-sl_ang, cl_ang], [-cl_ang, -sl_ang], [sl_ang, -cl_ang]]
    )
    lvoob = lvob + oob_shift

    # Define paths using matplotlib.path
    outer_path = path.Path(np.vstack((lvoob, lvoob[0, :])))
    inner_path = path.Path(np.vstack((lvib, lvib[0, :])))

    # Find mesh nodes within these regions
    outnodes = outer_path.contains_points(meshes[1].coords[:, 0:2])
    innodes = inner_path.contains_points(meshes[1].coords[:, 0:2])
    # Transitional nodes are outnodes that aren't innodes
    rampnodes = outnodes
    rampnodes[innodes] = False

    # Interpolated inner path, for calculating distance from nodes
    inner_path_interp = inner_path.interpolated(100).vertices
    # Calculate minimum distance between the rampnodes and the interpolated inner path
    all_dists = np.min(
        scipy.spatial.distance.cdist(
            meshes[1].coords[rampnodes, 0:2], inner_path_interp
        ),
        axis=1,
    )
    # Scale depth based on distance
    meshes[1].coords[rampnodes, 2] += rampheight * (1 - all_dists / np.max(all_dists))
    meshes[1].coords[innodes, 2] += rampheight


def get_3component_index(indices):
    """Returns indices into 3-component array, where each entry of input array
    corresponds to three entries in the 3-component array
    Given indices = [0, 2, 10, 6].

    Returns:
    [0, 1, 2, 6, 7, 8, 27, 28, 29, 15, 16, 17]
    This is useful for referencing velocity/slip components corresponding to a set
    of stations/faults.

    Args:
        indices (np.array): Element index array

    Returns:
        idx (np.array): Component index array (3 * length of indices)

    From https://github.com/brendanjmeade/celeri/
    """
    idx = np.sort(
        np.append(3 * (indices + 1) - 3, (3 * (indices + 1) - 2, 3 * (indices + 1) - 1))
    )
    return idx
