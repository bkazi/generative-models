class HParams(object):
    """Dictionary of hyperparameters."""

    def __init__(self, **kwargs):
        self.dict_ = kwargs
        self.__dict__.update(self.dict_)

    def update_config(self, in_string):
        """Update the dictionary with a comma separated list."""
        pairs = in_string.split(",")
        pairs = [pair.split("=") for pair in pairs]
        for key, val in pairs:
            self.dict_[key] = type(self.dict_[key])(val)
        self.__dict__.update(self.dict_)
        return self

    def __getitem__(self, key):
        return self.dict_[key]

    def __setitem__(self, key, val):
        self.dict_[key] = val
        self.__dict__.update(self.dict_)


def get_default_hparams():
    """Get the default hyperparameters."""
    return HParams(
        batch_size=64,
        n_couplings=2,
        learning_rate=0.001,
        l2_coeff=0.00005,
        clip_gradient=100.,
        optimizer="adam",
        num_parallel_calls=2,
        num_epochs=int(1e3))
