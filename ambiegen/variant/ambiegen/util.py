import importlib

import numpy as np

def test_to_road_points(test, step_length, map_size, initial_point="bottom", rotate=False):
    """Converts a test to road points.

    Args:
      test (list):         List of floats in the curvature range.
      step_length (float): Distance between two road points.
      map_size (int):      Size of the map.
      initial_point (str): Tells where the initial point is located on the map
                           (top, left, right, bottom, middle).
      rotate (bool):       Whether the first component signifies a rotation
                           angle or not.

    Returns:
      output (list): List of length len(test) of coordinate tuples."""

    # This is the same code as in the Frenetic algorithm.
    # https://github.com/ERATOMMSD/frenetic-sbst21/blob/main/src/generators/base_frenet_generator.py
    # We integrate curvature (acceleratation) to get an angle (speed) and
    # then we move one step to this direction to get position. The
    # integration is done using the trapezoid rule with step given by the
    # first component of the test. Now we use a fixed step size.
    step = step_length
    # We assume that denormalization from [-1, 1] to the actual curvature
    # range has already been done.
    if rotate:
        curvature = test[1:]
        rotation_angle = test[0]
    else:
        curvature = test
        rotation_angle = 0

    # 10 is the margin for not being out of bounds
    if initial_point == "top":
        initial_point = (map_size/2, map_size - 10)
    elif initial_point == "left":
        initial_point = (10, map_size/2)
    elif initial_point == "right":
        initial_point = (map_size - 10, map_size/2)
    elif initial_point == "bottom":
        initial_point = (map_size/2, 10)
    elif initial_point == "middle":
        initial_point = (map_size/2, map_size/2)
    else:
        raise ValueErrore("Initial point must take value 'top', 'left', 'right' or 'bottom'.")

    # The initial angle is 90 degrees and this is rotated by the angle
    # (degrees) given by the first component of the test if rotate is True.
    points = [initial_point]
    angles = [np.math.pi/2 + rotation_angle*(np.math.pi/180)]
    # Add the second point.
    points.append((points[-1][0] + step * np.cos(angles[-1]), points[-1][1] + step * np.sin(angles[-1])))
    # Find the remaining points.
    for i in range(curvature.shape[0] - 1):
        angles.append(angles[-1] + step * (curvature[i + 1] + curvature[i]) / 2)
        x = points[-1][0] + step * np.cos(angles[-1])
        y = points[-1][1] + step * np.sin(angles[-1])
        points.append((x, y))

    return points

def sbst_test_to_image(test, map_size):
    """Visualizes the road described as points in the plane in the map of
    specified size."""

    # We load the modules here to allow imports on machines where the BeamNG
    # simulator is not set up.
    load = {
        "code_pipeline.tests_generation": ["RoadTestFactory"],
        "code_pipeline.validation": ["TestValidator"],
        "shapely.geometry": ["LineString", "Polygon"],
        "descartes": ["PolygonPatch"]
    }
    for package, modules in load.items():
        for module in modules:
            if module in globals(): continue
            tmp = importlib.import_module(package)
            globals()[module] = getattr(tmp, module)

    little_triangle = Polygon([(10, 0), (0, -5), (0, 5), (10, 0)])
    square = Polygon([(5, 5), (5, -5), (-5, -5), (-5, 5), (5, 5)])

    V = TestValidator(map_size=map_size)
    try:
        the_test = RoadTestFactory.create_road_test(test)
        valid, msg = V.validate_test(the_test)
    except:
        return

    # This code is mainly from https://github.com/se2p/tool-competition-av/code_pipeline/visualization.py
    plt.figure()

    # plt.gcf().set_title("Last Generated Test")
    plt.gca().set_aspect("equal", "box")
    plt.gca().set(xlim=(-30, map_size + 30), ylim=(-30, map_size + 30))

    # Add information about the test validity
    title_string = "Test is " + ("valid" if valid else "invalid")
    if not valid:
        title_string = title_string + ":" + msg

    plt.suptitle(title_string, fontsize=14)
    plt.draw()

    # Plot the map. Trying to re-use an artist in more than one Axes which is supported
    map_patch = patches.Rectangle(
        (0, 0),
        map_size,
        map_size,
        linewidth=1,
        edgecolor="black",
        facecolor="none",
    )
    plt.gca().add_patch(map_patch)

    # Road Geometry.
    road_poly = LineString(
        [(t[0], t[1]) for t in the_test.interpolated_points]
    ).buffer(8.0, cap_style=2, join_style=2)
    road_patch = PolygonPatch(
        road_poly, fc="gray", ec="dimgray"
    )  # ec='#555555', alpha=0.5, zorder=4)
    plt.gca().add_patch(road_patch)

    # Interpolated Points
    sx = [t[0] for t in the_test.interpolated_points]
    sy = [t[1] for t in the_test.interpolated_points]
    plt.plot(sx, sy, "yellow")

    # Road Points
    x = [t[0] for t in the_test.road_points]
    y = [t[1] for t in the_test.road_points]
    plt.plot(x, y, "wo")

    # Plot the little triangle indicating the starting position of the ego-vehicle
    delta_x = sx[1] - sx[0]
    delta_y = sy[1] - sy[0]

    current_angle = atan2(delta_y, delta_x)

    rotation_angle = degrees(current_angle)
    transformed_fov = rotate(
        little_triangle, origin=(0, 0), angle=rotation_angle
    )
    transformed_fov = translate(transformed_fov, xoff=sx[0], yoff=sy[0])
    plt.plot(*transformed_fov.exterior.xy, color="black")

    # Plot the little square indicating the ending position of the ego-vehicle
    delta_x = sx[-1] - sx[-2]
    delta_y = sy[-1] - sy[-2]

    current_angle = atan2(delta_y, delta_x)

    rotation_angle = degrees(current_angle)
    transformed_fov = rotate(square, origin=(0, 0), angle=rotation_angle)
    transformed_fov = translate(transformed_fov, xoff=sx[-1], yoff=sy[-1])
    plt.plot(*transformed_fov.exterior.xy, color="black")

    plt.suptitle(title_string, fontsize=14)
    plt.draw()

    return plt.gcf()

def frechet_distance(P, Q):
    """
    Computes the discrete Fréchet distance between the polygonal curves defined
    by the point sequences P and Q.
    """

    # The implementation is based on
    # T. Eiter, H. Mannila. Computing discrete Fréchet distance.
    # Technical report CD-TR 94/64. Technical University of Vienna (1994).
    # http://www.kr.tuwien.ac.at/staff/eiter/et-archive/cdtr9464.pdf

    def C(ca, i, j, P, Q):
        # We use the Euclidean distance.
        if ca[i, j] > -1:
            return ca[i, j]

        if i == 0 and j == 0:
            m = 0
        elif i > 0 and j == 0:
            m = C(ca, i - 1, 0, P, Q)
        elif i == 0 and j > 0:
            m = C(ca, 0, j - 1, P, Q)
        else:
            m = min(
                C(ca, i - 1, j, P, Q),
                C(ca, i - 1, j - 1, P, Q),
                C(ca, i, j - 1, P, Q),
            )

        ca[i, j] = max(np.linalg.norm(P[i] - Q[j]), m)

        return ca[i, j]

    if len(P) == 0 or len(Q) == 0:
        raise ValueError("The input sequences must be nonempty.")

    ca = -1 * np.ones(shape=(len(P), len(Q)))
    return C(ca, len(P) - 1, len(Q) - 1, np.array(P), np.array(Q))

