import numpy as np
import okada_wrapper
from tqdm import tqdm

KM2M = 1.0e3


def get_segment_station_operator_okada(segment, station, command):
    """
    Calculates the elastic displacement partial derivatives based on the Okada
    formulation, using the source and receiver geometries defined in
    dicitonaries segment and stations.

    The linear operator is structured as ():

                ss(segment1)  ds(segment1) ts(segment1) ... ss(segmentN) ds(segmentN) ts(segmentN)
    ve(station 1)
    vn(station 1)
    vu(station 1)
    .
    .
    .
    ve(station N)
    vn(station N)
    vu(station N)

    """
    if not station.empty:
        okada_segment_operator = np.ones((3 * len(station), 3 * len(segment)))
        # Loop through each segment and calculate displacements for each slip component
        for i in tqdm(
            range(len(segment)),
            desc="Calculating Okada partials for segments",
            colour="cyan",
        ):
            (
                u_east_strike_slip,
                u_north_strike_slip,
                u_up_strike_slip,
            ) = get_okada_displacements(
                segment.x1[i],
                segment.y1[i],
                segment.x2[i],
                segment.y2[i],
                segment.locking_depth[i],
                segment.burial_depth[i],
                segment.dip[i],
                command.material_lambda,
                command.material_mu,
                1,
                0,
                0,
                station.x,
                station.y,
            )
            (
                u_east_dip_slip,
                u_north_dip_slip,
                u_up_dip_slip,
            ) = get_okada_displacements(
                segment.x1[i],
                segment.y1[i],
                segment.x2[i],
                segment.y2[i],
                segment.locking_depth[i],
                segment.burial_depth[i],
                segment.dip[i],
                command.material_lambda,
                command.material_mu,
                0,
                1,
                0,
                station.x,
                station.y,
            )
            (
                u_east_tensile_slip,
                u_north_tensile_slip,
                u_up_tensile_slip,
            ) = get_okada_displacements(
                segment.x1[i],
                segment.y1[i],
                segment.x2[i],
                segment.y2[i],
                segment.locking_depth[i],
                segment.burial_depth[i],
                segment.dip[i],
                command.material_lambda,
                command.material_mu,
                0,
                0,
                1,
                station.x,
                station.y,
            )
            segment_column_start_idx = 3 * i
            okada_segment_operator[0::3, segment_column_start_idx] = np.squeeze(
                u_east_strike_slip
            )
            okada_segment_operator[1::3, segment_column_start_idx] = np.squeeze(
                u_north_strike_slip
            )
            okada_segment_operator[2::3, segment_column_start_idx] = np.squeeze(
                u_up_strike_slip
            )
            okada_segment_operator[0::3, segment_column_start_idx + 1] = np.squeeze(
                u_east_dip_slip
            )
            okada_segment_operator[1::3, segment_column_start_idx + 1] = np.squeeze(
                u_north_dip_slip
            )
            okada_segment_operator[2::3, segment_column_start_idx + 1] = np.squeeze(
                u_up_dip_slip
            )
            okada_segment_operator[0::3, segment_column_start_idx + 2] = np.squeeze(
                u_east_tensile_slip
            )
            okada_segment_operator[1::3, segment_column_start_idx + 2] = np.squeeze(
                u_north_tensile_slip
            )
            okada_segment_operator[2::3, segment_column_start_idx + 2] = np.squeeze(
                u_up_tensile_slip
            )
    else:
        okada_segment_operator = np.empty(1)
    return okada_segment_operator


def get_okada_displacements(
    segment_x1,
    segment_y1,
    segment_x2,
    segment_y2,
    segment_locking_depth,
    segment_burial_depth,
    segment_dip,
    material_lambda,
    material_mu,
    strike_slip,
    dip_slip,
    tensile_slip,
    station_x,
    station_y,
):
    """
    Caculate elastic displacements in a homogeneous elastic half-space.
    Inputs are in geographic coordinates and then projected into a local
    xy-plane using a oblique Mercator projection that is tangent and parallel
    to the trace of the fault segment.  The elastic calculation is the
    original Okada 1992 Fortran code acceccesed through T. Ben Thompson's
    okada_wrapper: https://github.com/tbenthompson/okada_wrapper
    """
    segment_locking_depth *= KM2M
    segment_burial_depth *= KM2M

    # Calculate geometric fault parameters
    segment_strike = np.arctan2(
        segment_y2 - segment_y1, segment_x2 - segment_x1
    )  # radians
    segment_length = np.sqrt(
        (segment_y2 - segment_y1) ** 2.0 + (segment_x2 - segment_x1) ** 2.0
    )
    segment_up_dip_width = (segment_locking_depth - segment_burial_depth) / np.sin(
        np.deg2rad(segment_dip)
    )

    # Translate stations and segment so that segment mid-point is at the origin
    segment_x_mid = (segment_x1 + segment_x2) / 2.0
    segment_y_mid = (segment_y1 + segment_y2) / 2.0
    station_x -= segment_x_mid
    station_y -= segment_y_mid
    segment_x1 -= segment_x_mid
    segment_x2 -= segment_x_mid
    segment_y1 -= segment_y_mid
    segment_y2 -= segment_y_mid

    # Unrotate coordinates to eliminate strike, segment will lie along y = 0
    rotation_matrix = np.array(
        [
            [np.cos(segment_strike), -np.sin(segment_strike)],
            [np.sin(segment_strike), np.cos(segment_strike)],
        ]
    )
    station_x_rotated, station_y_rotated = np.hsplit(
        np.einsum("ij,kj->ik", np.dstack((station_x, station_y))[0], rotation_matrix.T),
        2,
    )

    # Elastic displacements from Okada 1992
    alpha = (material_lambda + material_mu) / (material_lambda + 2 * material_mu)
    u_x = np.zeros_like(station_x)
    u_y = np.zeros_like(station_x)
    u_up = np.zeros_like(station_x)
    for i in range(len(station_x)):
        _, u, _ = okada_wrapper.dc3dwrapper(
            alpha,  # (lambda + mu) / (lambda + 2 * mu)
            [
                station_x_rotated[i],
                station_y_rotated[i],
                0,
            ],  # (meters) observation point
            segment_locking_depth,  # (meters) depth of the fault origin
            segment_dip,  # (degrees) the dip-angle of the rectangular dislocation surface
            [
                -segment_length / 2,
                segment_length / 2,
            ],  # (meters) the along-strike range of the surface (al1,al2 in the original)
            [
                0,
                segment_up_dip_width,
            ],  # (meters) along-dip range of the surface (aw1, aw2 in the original)
            [strike_slip, dip_slip, tensile_slip],
        )  # (meters) strike-slip, dip-slip, tensile-slip
        u_x[i] = u[0]
        u_y[i] = u[1]
        u_up[i] = u[2]

    # Un-rotate displacement to account for projected fault strike
    u_east, u_north = np.hsplit(
        np.einsum("ij,kj->ik", np.dstack((u_x, u_y))[0], rotation_matrix), 2
    )
    return u_east, u_north, u_up


def get_segment_oblique_projection(lon1, lat1, lon2, lat2, skew=True):
    """
    Use pyproj oblique mercator: https://proj.org/operations/projections/omerc.html

    According to: https://proj.org/operations/projections/omerc.html
    This is this already rotated by the fault strike but the rotation can be undone with +no_rot
    > +no_rot
    > No rectification (not “no rotation” as one may well assume).
    > Do not take the last step from the skew uv-plane to the map XY plane.
    > Note: This option is probably only marginally useful,
    > but remains for (mostly) historical reasons.

    The version with north still pointing "up" appears to be called the
    Rectified skew orthomorphic projection or Hotine oblique Mercator projection
    https://pro.arcgis.com/en/pro-app/latest/help/mapping/properties/rectified-skew-orthomorphic.htm
    """
    if lon1 > 180.0:
        lon1 = lon1 - 360
    if lon2 > 180.0:
        lon2 = lon2 - 360

    # Check if latitudes are too close to identical
    # If lat1 and lat2 are the same at the 5 decimal place proj with fail
    # Perturb lat2 slightly to avoid this.
    if np.isclose(lat1, lat2):
        latitude_offset = 0.001
        lat2 += latitude_offset

    projection_string = (
        "+proj=omerc "
        + "+lon_1="
        + str(lon1)
        + " "
        + "+lat_1="
        + str(lat1)
        + " "
        + "+lon_2="
        + str(lon2)
        + " "
        + "+lat_2="
        + str(lat2)
        + " "
        + "+ellps=WGS84"
    )
    if not skew:
        projection_string += " +no_rot"
    projection = pyproj.Proj(pyproj.CRS.from_proj4(projection_string))
    return projection
