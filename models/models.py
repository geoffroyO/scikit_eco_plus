from scikit_eco_plus.utils.utils import get_kernel


class Model:
    def __init__(self, kernel='linear', **kwargs):
        self.kernel = get_kernel(kernel, **kwargs)
