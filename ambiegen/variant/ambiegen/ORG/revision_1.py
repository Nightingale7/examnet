from ambiegen.variant.ambiegen import AmbiegenSurrogateModel # pragma: no mutate
import numpy as np # pragma: no mutate
import math # pragma: no mutate

class AMBIEGEN_SUT(AmbiegenSurrogateModel): # pragma: no mutate
    
    def __init__(self, parameters=None):
        super().__init__(parameters)

    def go_straight(self): # pragma: no mutate
        self.x = self.speed * np.cos(math.radians(self.angle)) / 2.3 + self.x
        self.y = self.speed * np.sin(math.radians(self.angle)) / 2.3 + self.y
        self.tot_x.append(self.x)
        self.tot_y.append(self.y)

    def turn_right(self): # pragma: no mutate
        self.str_ang = math.degrees(math.atan(1 / self.speed * 2 * self.distance))
        self.angle = self.angle - self.str_ang
        self.x = self.speed * np.cos(math.radians(self.angle)) / 3 + self.x
        self.y = self.speed * np.sin(math.radians(self.angle)) / 3 + self.y
        self.tot_x.append(self.x)
        self.tot_y.append(self.y)

    def turn_left(self): # pragma: no mutate
        self.str_ang = math.degrees(math.atan(1 / self.speed * 2 * self.distance))
        self.angle = self.str_ang + self.angle
        self.x = self.speed * np.cos(math.radians(self.angle)) / 3 + self.x
        self.y = self.speed * np.sin(math.radians(self.angle)) / 3 + self.y

        self.tot_x.append(self.x)
        self.tot_y.append(self.y)
        
