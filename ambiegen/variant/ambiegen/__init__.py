import copy, importlib, importlib, math, multiprocessing, os, sys, time, traceback
import numpy as np
from ambiegen.variant import RevisionSingleFile, Variant
from stgem.sut import SUT, SUTOutput, SUTInput
from scipy.interpolate import splprep, splev
from shapely.geometry import Point, LineString
import shutil

class AmbieGenRevision(RevisionSingleFile):

    sut_parameters = {
        "curvature_points": 5,
        "curvature_range": 0.07,
        "step_length": 15,
        "map_size": 200
    }

    @classmethod
    def from_source_file(C, source_file, remove_files_on_unload=False):
        head, tail = os.path.split(source_file)
        pcs = [tail]
        while head != "":
            head, tail = os.path.split(head)
            pcs.insert(0, tail)
        module_string = ".".join(pcs)[:-3]
        module = importlib.import_module(module_string)

        sut = module.AMBIEGEN_SUT(copy.deepcopy(C.sut_parameters))
        return C(sut, source_file, module_string, remove_files_on_unload=remove_files_on_unload)

    @staticmethod
    def get_original_revision():
        # Get the AmbieGen SUT from variant/ambiegen/original.py (currently the
        # same as in the original SUT).
        return AmbieGenRevision.from_source_file("variant/ambiegen/original.py")

class AmbieGenVariant(Variant):

    @staticmethod
    def _get_variant_path():
        return os.path.join("ambiegen","variant", "ambiegen")

    @staticmethod
    def get_revision_class():
        return AmbieGenRevision


class AmbiegenSurrogateModel(SUT):
    """Implements the Ambiegen surrogate model. The input representation is
    curvature-based as in SBSTSUT."""

    default_parameters = {"curvature_range": 0.07,
                          "step_length": 15,
                          "initial_point": "bottom",
                          "rotate": False,
                          "max_rotation": 180,
                          "map_size": 200}

    # The code is from
    # https://github.com/dgumenyuk/tool-competition-av/blob/main/ambiegen/vehicle.py
    # under GPL license. Some fixes applied.

    def __init__(self, parameters=None):
        super().__init__(parameters)

        sys.path.append(os.path.dirname(__file__))
        package = importlib.import_module("util")
        globals()["test_to_road_points"] = package.test_to_road_points

        # These are from https://github.com/dgumenyuk/tool-competition-av/blob/main/ambiegen/config.py
        self.init_speed = 9
        self.init_str_ang = 12

        if not "curvature_points" in self.parameters:
            raise Exception("Number of curvature points not defined.")
        if self.curvature_points <= 0:
            raise ValueError("The number of curvature points must be positive.")
        if self.curvature_range <= 0:
            raise ValueError("The curvature range must be positive.")

        self.input_type = "vector"
        if self.rotate:
            self.idim = 1 + self.curvature_points
            range = [-self.curvature_range, self.curvature_range]
            self.input_range = [[-self.max_rotation, self.max_rotation]] + [range]*(self.idim - 1)
        else:
            self.idim = self.curvature_points
            range = [-self.curvature_range, self.curvature_range]
            self.input_range = [range] * self.idim

        self.output_type = "signal"
        self.odim = 2
        self.outputs = ["distance", "angle"]
        self.output_range = [[0, 3], [0, 360]]  # Maximum observed value 3.79 (mean 2.26, std 0.48)

        if self.map_size <= 0:
            raise ValueError("The map size must be positive.")

    def interpolate_road(self, road):
        test_road = LineString([(t[0], t[1]) for t in road])

        length = test_road.length

        num_nodes = int(length)
        if num_nodes < 20:
            num_nodes = 20

        old_x_vals = [t[0] for t in road]
        old_y_vals = [t[1] for t in road]

        if len(old_x_vals) == 2:
            k = 1
        elif len(old_x_vals) == 3:
            k = 2
        else:
            k = 3
        f2, u = splprep([old_x_vals, old_y_vals], s=0, k=k)
        # step_size = 1 / (length) * 8

        step_size = 1 / num_nodes

        xnew = np.ma.arange(0, 1 + step_size, step_size)

        x2, y2 = splev(xnew, f2)

        nodes = list(zip(x2, y2))

        return nodes

    def find_circle(self, p1, p2, p3):
        """
        Returns the center and radius of the circle passing the given 3 points.
        In case the 3 points form a line, returns (None, infinity).
        """
        temp = p2[0] * p2[0] + p2[1] * p2[1]
        bc = (p1[0] * p1[0] + p1[1] * p1[1] - temp) / 2
        cd = (temp - p3[0] * p3[0] - p3[1] * p3[1]) / 2
        det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])

        if abs(det) < 1.0e-6:
            return np.inf

        # Center of circle
        cx = (bc * (p2[1] - p3[1]) - cd * (p1[1] - p2[1])) / det
        cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det

        radius = np.sqrt((cx - p1[0]) ** 2 + (cy - p1[1]) ** 2)
        return radius

    def min_radius(self, x, w=5):
        mr = np.inf
        nodes = x
        for i in range(len(nodes) - w):
            p1 = nodes[i]
            p2 = nodes[i + int((w - 1) / 2)]
            p3 = nodes[i + (w - 1)]
            radius = self.find_circle(p1, p2, p3)
            if radius < mr:
                mr = radius
        if mr == np.inf:
            mr = 0

        return mr * 3.280839895  # , mincurv

    def is_too_sharp(self, the_test, TSHD_RADIUS=47):
        return TSHD_RADIUS > self.min_radius(the_test) > 0.0

    def get_distance(self, road, x, y):
        p = Point(x, y)
        return p.distance(road)

    def get_angle(self, node_a, node_b):
        vector = np.array(node_b) - np.array(node_a)
        cos = vector[0] / (np.linalg.norm(vector))

        angle = math.degrees(math.acos(cos))

        if node_a[1] > node_b[1]:
            return -angle
        else:
            return angle

    def go_straight(self):
        self.x = self.speed * np.cos(math.radians(self.angle)) / 2.3 + self.x
        self.y = self.speed * np.sin(math.radians(self.angle)) / 2.3 + self.y
        self.tot_x.append(self.x)
        self.tot_y.append(self.y)

    def turn_right(self):
        self.str_ang = math.degrees(math.atan(1 / self.speed * 2 * self.distance))
        self.angle = -self.str_ang + self.angle
        self.x = self.speed * np.cos(math.radians(self.angle)) / 3 + self.x
        self.y = self.speed * np.sin(math.radians(self.angle)) / 3 + self.y
        self.tot_x.append(self.x)
        self.tot_y.append(self.y)

    def turn_left(self):
        self.str_ang = math.degrees(math.atan(1 / self.speed * 2 * self.distance))
        self.angle = self.str_ang + self.angle
        self.x = self.speed * np.cos(math.radians(self.angle)) / 3 + self.x
        self.y = self.speed * np.sin(math.radians(self.angle)) / 3 + self.y

        self.tot_x.append(self.x)
        self.tot_y.append(self.y)

    def _execute_test_surrogate(self, nodes):
        # Notice that this is only called after the test is checked to be valid.
        self.x = 0
        self.y = 0

        old_x_vals = [t[0] for t in nodes]
        old_y_vals = [t[1] for t in nodes]

        self.road_x = old_x_vals
        self.road_y = old_y_vals

        self.speed = self.init_speed
        self.str_ang = self.init_str_ang

        self.angle = 0
        self.tot_x = []
        self.tot_y = []
        self.tot_dist = []
        self.tot_angle = []
        self.final_dist = []
        self.distance = 0

        road = LineString([(t[0], t[1]) for t in nodes])
        mini_nodes1 = nodes[: round(len(nodes) / 2)]
        mini_nodes2 = nodes[round(len(nodes) / 2):]
        if (len(mini_nodes1) < 2) or (len(mini_nodes2) < 2):
            return 0, []
        mini_road1 = LineString([(t[0], t[1]) for t in mini_nodes1])
        mini_road2 = LineString([(t[0], t[1]) for t in mini_nodes2])
        road_split = [mini_road1, mini_road2]

        if (road.is_simple is False) or (self.is_too_sharp(nodes) is True):
            # This will never execute as we ensure that the road is valid
            # before executing this function.
            fitness = 0
        else:
            init_pos = nodes[0]
            self.x = init_pos[0]
            self.y = init_pos[1]

            self.angle = self.get_angle(nodes[0], nodes[1])

            #self.tot_x.append(self.x)
            #self.tot_y.append(self.y)

            i = 0

            for p, mini_road in enumerate(road_split):

                current_length = 0
                if p == 1:
                    self.x = mini_nodes2[0][0]
                    self.y = mini_nodes2[0][1]
                    self.angle = self.get_angle(mini_nodes1[-1], mini_nodes2[0])

                current_pos = [(self.x, self.y)]

                while (current_length < mini_road.length) and i < 1000:
                    distance = self.get_distance(mini_road, self.x, self.y)
                    self.distance = distance

                    self.tot_dist.append(distance)
                    if self.angle < 0:
                        self.tot_angle.append(self.angle + 360)
                    else:
                        self.tot_angle.append(self.angle)
                    if distance <= 1:
                        self.go_straight()
                        current_pos.append((self.x, self.y))
                        self.speed += 0.3

                    else:
                        angle = -1 + self.angle
                        x = self.speed * np.cos(math.radians(angle)) + self.x
                        y = self.speed * np.sin(math.radians(angle)) + self.y

                        distance_right = self.get_distance(mini_road, x, y)

                        angle = 1 + self.angle
                        x = self.speed * np.cos(math.radians(angle)) + self.x
                        y = self.speed * np.sin(math.radians(angle)) + self.y

                        distance_left = self.get_distance(mini_road, x, y)

                        if distance_right < distance_left:
                            self.turn_right()
                            current_pos.append((self.x, self.y))
                        else:
                            self.turn_left()
                            current_pos.append((self.x, self.y))

                        self.speed -= 0.1

                    current_road = LineString(current_pos)
                    current_length = current_road.length

                    i += 1

            # Add one more distance so that self.tot_x, self.tot_y and
            # self.tot_dist have the same length.
            distance = self.get_distance(mini_road, self.x, self.y)
            self.tot_dist.append(distance)
            self.tot_angle.append(self.angle)

            fitness = max(self.tot_dist) * (-1)

            car_path = LineString(zip(self.tot_x, self.tot_y))
            if car_path.is_simple == False:
                # This cannot be checked in advance and currently we're unsure
                # what to do here. Ideally this should never happen, so we
                # return an error.
                return SUTOutput(None, None, None, "car trajectory intersected itself")

        # Uncomment to visualize the original road and the simulated
        # trajectory.

        #import matplotlib.pyplot as plt
        #plt.plot(list(t[0] for t in nodes), list(t[1] for t in nodes))
        #plt.plot(self.tot_x, self.tot_y)
        #plt.show()

        # The positions of the car are in self.tot_x and self.tot_y.
        # We now return the distance signal.
        signals1 = np.array(self.tot_dist[:-1]).reshape(1, -1)
        signals2 = np.array(self.tot_angle[:-1]).reshape(1, -1)
        #signals = np.row_stack((signals1, signals2))
        pos = np.row_stack((self.tot_x, self.tot_y))
        signals = np.row_stack((signals1, self.tot_x, self.tot_y))
        timestamps = np.arange(len(self.tot_dist) - 1)

        return SUTOutput(signals, timestamps, None, None)

    def _execute_test(self, test):
        denormalized = self.descale(test.inputs.reshape(1, -1), self.input_range).reshape(-1)
        interpolated_road = self.interpolate_road(test_to_road_points(denormalized, self.step_length, self.map_size, initial_point=self.initial_point, rotate=self.rotate))
        test.input_denormalized = interpolated_road
        if self.validity(test.inputs) == 1:
            output = self._execute_test_surrogate(interpolated_road)
        else:
            output = SUTOutput(None, None, None, "invalid road")

        return output

    def validity(self, test):
        if isinstance(test, SUTInput):
            test = test.inputs
        denormalized = self.descale(test.reshape(1, -1), self.input_range).reshape(-1)
        interpolated_road = self.interpolate_road(test_to_road_points(denormalized, self.step_length, self.map_size, initial_point=self.initial_point, rotate=self.rotate))
        road = LineString([(t[0], t[1]) for t in interpolated_road])
        return 1 if road.is_simple and not self.is_too_sharp(interpolated_road) else 0

