from qfin.assets import Asset


class Spot(Asset):
    def __init__(self, s0, rate, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.s0 = s0
        self.rate = rate

    @property
    def asset_name(self):
        return f"SPOT"

    def generate(self):

        msg = f"Generation of the paths for the model {self.model.name} is not implemented."
        assert hasattr(self.model, 'paths'), msg

        self.model.paths(self.paths, self.period, self.s0, self.rate, self.npaths)
