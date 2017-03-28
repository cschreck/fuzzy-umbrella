from masterthesis.util import calc_heading
import math
from tpm.data_model import R


def add_dist(p1, p2, dist):
    radius = R
    bearing = math.radians(-1 * calc_heading(p1, p2)) - math.radians(90)

    lat1 = math.radians(p1[0])
    lon1 = math.radians(p1[1])

    lat2 = math.asin(math.sin(lat1) * math.cos(dist / radius) +
                     math.cos(lat1) * math.sin(dist / radius) * math.cos(bearing))

    lon2 = lon1 + math.atan2(math.sin(bearing) * math.sin(dist / radius) * math.cos(lat1),
                             math.cos(dist / radius) - math.sin(lat1) * math.sin(lat2))

    lat2 = math.degrees(lat2)
    lon2 = math.degrees(lon2)

    return (lat2, lon2)


def same_direction(rs, re, ps, pe, threshold):
    bearing1 = calc_heading(rs, re)
    bearing2 = calc_heading(ps, pe)

    return abs(bearing1 - bearing2) < threshold


def calculate_rectangle(a, b, alpha):
    delta_one = calc_heading(a, b) - alpha
    delta_two = calc_heading(a, b) + alpha

    m_one = math.tan(delta_one + (math.pi / 2))
    m_two = math.tan(delta_two + (math.pi / 2))

    c_x = (m_one * a[0] - m_two * b[0] - a[1] + b[1]) / (m_one - m_two)
    c_y = m_one * (c_x - a[0]) + a[1]

    d_x = (m_two * a[0] - m_one * b[0] - a[1] + b[1]) / (m_two - m_one)
    d_y = m_two * (d_x - a[0]) + a[1]

    return a, (c_x, c_y), b, (d_x, d_y)


def is_in_rectangle(rec, p):
    # divde into two triangles
    x = p[0]
    y = p[1]

    x1 = rec[0][0]
    x2 = rec[1][0]
    x3 = rec[2][0]
    x4 = rec[3][0]
    y1 = rec[0][1]
    y2 = rec[1][1]
    y3 = rec[2][1]
    y4 = rec[3][1]

    a_1 = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / ((y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3))
    b_1 = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / ((y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3))
    c_1 = 1 - a_1 - b_1

    a_2 = ((y4 - y3) * (x - x3) + (x3 - x4) * (y - y3)) / ((y4 - y3) * (x1 - x3) + (x3 - x4) * (y1 - y3))
    b_2 = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / ((y4 - y3) * (x1 - x3) + (x3 - x4) * (y1 - y3))
    c_2 = 1 - a_2 - b_2

    in_first_triangle = 0 <= a_1 <= 1 and 0 <= b_1 <= 1 and 0 <= c_1 <= 1
    in_second_triangle = 0 <= a_2 <= 1 and 0 <= b_2 <= 1 and 0 <= c_2 <= 1

    return in_first_triangle or in_second_triangle


def candidate_selection(rs, re, ps, pe, dist_delta=200, bearing_threshold=0.5):
    if not same_direction(rs, re, ps, pe, bearing_threshold):
        return False
    rs = add_dist(rs, re, dist=dist_delta)
    re = add_dist(re, rs, dist=dist_delta)
    rec = calculate_rectangle(rs, re, 0.5)
    return is_in_rectangle(rec, ps) and is_in_rectangle(rec, pe)
