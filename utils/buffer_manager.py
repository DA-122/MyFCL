import numpy as np


# class Buffer():
#     def __init__(self, args):
#         super().__init__()
#         self.buffer_size = args["memory_size"]
#         print('buffer has %d slots' % self.buffer_size)
#         self.buffer_data = np.array()
#         self.buffer_label = np.array()


#         # define update and retrieve method
#         self.update_method = get_update_methods[args.update](args["update"])
#         self.retrieve_method = get_retrieve_methods[args.retrieve]("retrieve")

#         if self.params.buffer_tracker:
#             self.buffer_tracker = BufferClassTracker(n_classes[params.data], self.device)

#     def update(self, x, y,**kwargs):
#         return self.update_method.update(buffer=self, x=x, y=y, **kwargs)


#     def retrieve(self, **kwargs):
#         return self.retrieve_method.retrieve(buffer=self, **kwargs)