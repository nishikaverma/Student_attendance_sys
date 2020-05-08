# the front face of application

import controller

class view:
    def __init__(self):
        self.obj_controller = controller.controller()
    
    def Start(self):
        self.obj_controller.Video_capture()

if __name__ == '__main__':
    obj_view = view()
    obj_view.Start()
    