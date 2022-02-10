class SaveFeatures():
    ''' Save Forward Progapation Activations of Feature Maps of the requested module of the model'''
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.features = None
        self.params = None

    def hook_fn(self, module, input, output):
        ''' Save output of the requested layer (module) to self.features '''
        self.features = output 
        for p in module.parameters():
          if len(p.shape)==4:
            self.params = p
            break

    def close(self):
        self.hook.remove()