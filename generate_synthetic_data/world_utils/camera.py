import mujoco as mj
from mujoco import MjvCamera
from .configurations_and_constants import ORIGIN


class Camera:
    def __init__(self, model, data, specs):
        self.model = model
        self.data = data

        self.renderer = mj.Renderer(model, specs['height'], specs['width'])

        self.camera = MjvCamera()
        self.camera.lookat = ORIGIN
        self.camera.distance = specs['dist']
        self.camera.elevation = specs['elevation']
        self.camera.azimuth = specs['azimuth']

    def __call__(self):
        self.renderer.update_scene(self.data, camera=self.camera)
        return self.renderer.render()
