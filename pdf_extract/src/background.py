"""Conservative vector fill sampling; no OCR or text semantics involved."""
from __future__ import annotations


def rgb(color):
    if isinstance(color, (int, float)):
        color = [color]
    if not isinstance(color, (list, tuple)):
        return None
    if len(color) == 1:
        result = list(color) * 3
    elif len(color) == 3:
        result = list(color)
    elif len(color) == 4:
        c, m, y, k = color
        result = [1 - min(1, v + k) for v in (c, m, y)]
    else:
        return None  # Pattern/spot/ICC colours need a colour-space decoder.
    return [max(0, min(1, float(v))) for v in result]


def fill_regions(page):
    regions = []
    # Keep white paints as well: a later white rectangle can cover a gray one.
    objects = sorted(list(page.rects) + list(page.curves),
                     key=lambda obj: obj.get('object_index', 0))
    for obj in objects:
        if not obj.get('fill'):
            continue
        color = rgb(obj.get('non_stroking_color'))
        if color is None:
            continue
        box = [obj['x0'], obj['top'], obj['x1'], obj['bottom']]
        if min(box[2]-box[0], box[3]-box[1]) < 3:
            continue  # Filled border strips are not a cell background.
        if obj.get('object_type') == 'curve':
            # Only accept rectangular, axis-aligned polygons. The bounding
            # box of an arbitrary filled logo is NOT a background rectangle.
            pts = obj.get('pts', [])
            if len(pts) < 4 or any(
                abs(a[0]-b[0]) > .5 and abs(a[1]-b[1]) > .5
                for a, b in zip(pts, pts[1:] + pts[:1])
            ):
                continue
            corners = {(round(p[0], 1), round(p[1], 1)) for p in pts}
            if len(corners) != 4:
                continue
        regions.append((box, color))
    return regions


def annotate_cells(cells, page, white_tolerance=.02, coverage=.80):
    """Sample cell interiors, ignoring thin rules and tiny coloured marks.

    This supports ordinary solid vector fills, not raster backgrounds,
    gradients, transparency groups or arbitrary overlapping clipping paths.
    """
    regions = fill_regions(page)
    for cell in cells:
        if cell.get('is_page_text'):
            continue
        x0, y0, x1, y1 = (cell[k] for k in ('x0', 'y0', 'x1', 'y1'))
        colors = []
        for fx in (.15, .325, .5, .675, .85):
            for fy in (.15, .325, .5, .675, .85):
                x, y = x0+(x1-x0)*fx, y0+(y1-y0)*fy
                color = [1., 1., 1.]
                for box, paint in regions:
                    if box[0] <= x <= box[2] and box[1] <= y <= box[3]:
                        color = paint
                colors.append(color)
        tinted = [c for c in colors if max(1-v for v in c) > white_tolerance]
        fraction = len(tinted)/len(colors)
        if fraction >= coverage:
            mean = [sum(c[i] for c in tinted)/len(tinted) for i in range(3)]
            cell['background'] = '#'+''.join(f'{round(v*255):02x}' for v in mean)
            cell['background_coverage'] = round(fraction, 3)
