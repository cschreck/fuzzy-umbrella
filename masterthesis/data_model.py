class Point:
    def __init__(self, lat, lon, datetime):
        self.lat = lat
        self.lon = lon
        self.datetime = datetime

    def __str__(self):
        return '{},{},{}'.format(self.lat, self.lon, self.datetime)


class Trajectory:
    def __init__(self, points):
        self.points = points

    def __iter__(self):
        return self.points.__iter__()

    def __getitem__(self, item):
        return self.points[item]


