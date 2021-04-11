from datetime import timedelta


class Period:

    def __init__(self, t0, t1, name):
        self.t0 = t0
        self.t1 = t1
        self.name = name

    @property
    def days(self):
        dt = self.t1 - self.t0
        return dt.days

    @property
    def date_range(self):
        return [self.t0 + timedelta(days=i) for i in range(self.days)]
