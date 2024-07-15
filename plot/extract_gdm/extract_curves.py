import re
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ORIGIN = pd.Series(dict(x=26.637, y=153.741))
MAXES = pd.Series(dict(x=303.818, y=5.911))


def parse_path_data(path_data):
    coords = re.findall(
        r"([Mlm])? ([-+]?\d*\.?\d+(e[-+]?\d+)?),([-+]?\d*\.?\d+(e[-+]?\d+)?)|([hvHV])? ([-+]?\d*\.?\d+(e[-+]?\d+)?)", path_data
    )

    points = []
    prev_n1 = prev_n2 = 0
    for mlm, n1, _, n2, _, hor_vert, nhor_vert, _ in coords:
        if not mlm and not n1 and not n2 and not hor_vert:
            assert float(nhor_vert.strip()) == 0.0
            continue

        if hor_vert == "":
            if mlm == "m" or mlm == "" or mlm == "l":
                points.append((prev_n1 + float(n1.strip()), prev_n2 + float(n2.strip())))
            elif mlm == "M":
                points.append((float(n1.strip()), float(n2.strip())))
            else:
                raise ValueError((mlm, hor_vert))
        else:
            if hor_vert == "h":
                points.append((prev_n1 + float(nhor_vert.strip()), prev_n2))
            elif hor_vert == "H":
                points.append((float(nhor_vert.strip()), prev_n2))
            elif hor_vert == "v":
                points.append((prev_n1, prev_n2 + float(nhor_vert.strip())))
            elif hor_vert == "V":
                points.append((prev_n1, float(nhor_vert.strip())))
            else:
                raise ValueError((mlm, hor_vert))

        prev_n1, prev_n2 = points[-1]
    return points


def extract_properties(style):
    if style is None:
        return {}
    properties = str(style).split(";")
    out = {}
    for prop in properties:
        k, v = prop.split(":")
        out[k.strip()] = v.strip()
    return out


# Parse the SVG file
FNAME, x_axis_max = "dm-learning-curves-resized", 1e9
# FNAME, x_axis_max = "dm-drc-zoomed", 1e8
tree = ET.parse(f"{FNAME}.svg")
root = tree.getroot()

# Find all <path> elements with stroke colors matching the lines
lines = []
confidences = []
points = []

grid_mins = pd.Series(dict(x=np.inf, y=np.inf))
grid_maxs = pd.Series(dict(x=-np.inf, y=-np.inf))

stroke_colors = set(["#377eb8", "#984ea3", "#4daf4a", "#f781bf", "#a65628", "#cccccc", "#999999", "#e41a1c"])
for path in root.findall(".//{http://www.w3.org/2000/svg}path"):
    style = path.get("style")
    props = extract_properties(style)

    fill = props.get("fill", None)
    if (stroke := props.get("stroke", None)) in stroke_colors or fill in stroke_colors:
        path_data = path.get("d")

        parsed_data = parse_path_data(str(path_data))
        df = pd.DataFrame(parsed_data, columns=pd.Index(["x", "y"])).round(2)

        # Transform points
        transform = str(path.get("transform"))
        assert transform.startswith("matrix(")
        a, b, c, d, e, f = [float(n) for n in transform.strip("matrix()").split(",")]

        new_y = a * df.x + c * df.y + e
        new_x = b * df.x + d * df.y + f
        df.y = -new_x
        df.x = new_y

        if stroke == "#ffffff":
            top_curve = df.groupby("x").max()
            bottom_curve = df.groupby("x").min()
            confidences.append((fill, bottom_curve.index, bottom_curve["y"], top_curve["y"]))
            points.append((fill, df.x, df.y))
        elif stroke == "#cccccc":
            uniques = df.nunique("rows")
            if uniques["x"] > 1 and uniques["y"] > 1:
                print(path.get("id"))

            grid_mins = pd.concat([grid_mins, df.min()], axis=1).min(axis=1)
            grid_maxs = pd.concat([grid_maxs, df.max()], axis=1).max(axis=1)
            lines.append((stroke, df.x, df.y))
        else:
            lines.append((stroke, df.x, df.y))


# nx and ny have to be defined after `grid_mins` and `grid_maxs`
def nx(x):
    return (x - grid_mins.x) / (grid_maxs.x - grid_mins.x) * x_axis_max


def ny(y):
    return (y - grid_mins.y) / (grid_maxs.y - grid_mins.y)


def to_series(xs, ys, name):
    aa = pd.DataFrame(data=dict(x=nx(xs), y=ny(ys)))
    aa = aa.round(dict(x=0, y=10))
    aa = aa.drop_duplicates(subset=["x"])
    out = aa.set_index(["x"]).y
    out.name = name
    return out


all_data = pd.DataFrame()

for color, xs, ys in lines:
    plt.plot(nx(xs), ny(ys), color=color)
    if len(xs) > 5:
        all_data = pd.concat([all_data, to_series(xs, ys, color)], axis=1)

for color, xs, ys_bottom, ys_top in confidences:
    assert len(xs) == len(ys_bottom)
    assert len(xs) == len(ys_top)
    plt.fill_between(nx(xs), ny(ys_bottom), ny(ys_top), color=color, alpha=0.2)

    all_data = pd.concat([all_data, to_series(xs, ys_top, color + "_max"), to_series(xs, ys_bottom, color + "_min")], axis=1)


print("NA:", np.sum(all_data.isna()))
all_data.to_csv(f"{FNAME}.csv")

plt.xlim((0.0, x_axis_max))
plt.ylim((0.0, 1.0))
plt.show()
