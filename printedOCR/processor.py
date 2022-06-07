import numpy as np

def estimate_page_angle(polys):
    """Takes a batch of rotated previously ORIENTED polys (N, 4, 2) (rectified by the classifier) and return the
    estimated angle ccw in degrees
    """
    # Compute mean left points and mean right point with respect to the reading direction (oriented polygon)
    xleft = polys[:, 0, 0] + polys[:, 3, 0]
    yleft = polys[:, 0, 1] + polys[:, 3, 1]
    xright = polys[:, 1, 0] + polys[:, 2, 0]
    yright = polys[:, 1, 1] + polys[:, 2, 1]
    return np.median(np.arctan(
        (yleft - yright) / (xright - xleft)  # Y axis from top to bottom!
    )) * 180 / np.pi

def remap_boxes(loc_preds,orig_shape,dest_shape):
    """ Remaps a batch of rotated locpred (N, 4, 2) expressed for an origin_shape to a destination_shape.
    This does not impact the absolute shape of the boxes, but allow to calculate the new relative RotatedBbox
    coordinates after a resizing of the image.
    Args:
        loc_preds: (N, 4, 2) array of RELATIVE loc_preds
        orig_shape: shape of the origin image
        dest_shape: shape of the destination image
    Returns:
        A batch of rotated loc_preds (N, 4, 2) expressed in the destination referencial
    """

    if len(dest_shape) != 2:
        raise ValueError(f"Mask length should be 2, was found at: {len(dest_shape)}")
    if len(orig_shape) != 2:
        raise ValueError(f"Image_shape length should be 2, was found at: {len(orig_shape)}")
    orig_height, orig_width = orig_shape
    dest_height, dest_width = dest_shape
    mboxes = loc_preds.copy()
    mboxes[:, :, 0] = ((loc_preds[:, :, 0] * orig_width) + (dest_width - orig_width) / 2) / dest_width
    mboxes[:, :, 1] = ((loc_preds[:, :, 1] * orig_height) + (dest_height - orig_height) / 2) / dest_height

    return mboxes


def rotate_boxes(
    loc_preds,
    angle,
    orig_shape,
    min_angle=1,
    target_shape= None):
    """Rotate a batch of straight bounding boxes (xmin, ymin, xmax, ymax, c) or rotated bounding boxes
    (4, 2) of an angle, if angle > min_angle, around the center of the page.
    If target_shape is specified, the boxes are remapped to the target shape after the rotation. This
    is done to remove the padding that is created by rotate_page(expand=True)
    Args:
        loc_preds: (N, 5) or (N, 4, 2) array of RELATIVE boxes
        angle: angle between -90 and +90 degrees
        orig_shape: shape of the origin image
        min_angle: minimum angle to rotate boxes
    Returns:
        A batch of rotated boxes (N, 4, 2): or a batch of straight bounding boxes
    """

    # Change format of the boxes to rotated boxes
    _boxes = loc_preds.copy()
    if _boxes.ndim == 2:
        _boxes = np.stack(
            [
                _boxes[:, [0, 1]],
                _boxes[:, [2, 1]],
                _boxes[:, [2, 3]],
                _boxes[:, [0, 3]],
            ],
            axis=1
        )
    # If small angle, return boxes (no rotation)
    if abs(angle) < min_angle or abs(angle) > 90 - min_angle:
        return _boxes
    # Compute rotation matrix
    angle_rad = angle * np.pi / 180.  # compute radian angle for np functions
    rotation_mat = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ], dtype=_boxes.dtype)
    # Rotate absolute points
    points = np.stack((_boxes[:, :, 0] * orig_shape[1], _boxes[:, :, 1] * orig_shape[0]), axis=-1)
    image_center = (orig_shape[1] / 2, orig_shape[0] / 2)
    rotated_points = image_center + np.matmul(points - image_center, rotation_mat)
    rotated_boxes = np.stack(
        (rotated_points[:, :, 0] / orig_shape[1], rotated_points[:, :, 1] / orig_shape[0]), axis=-1
    )

    # Apply a mask if requested
    if target_shape is not None:
        rotated_boxes = remap_boxes(rotated_boxes, orig_shape=orig_shape, dest_shape=target_shape)

    return rotated_boxes

def _sort_boxes(boxes):
    """Sort bounding boxes from top to bottom, left to right
    Args:
        boxes: bounding boxes of shape (N, 4) or (N, 4, 2) (in case of rotated bbox)
    Returns:
        tuple: indices of ordered boxes of shape (N,), boxes
            If straight boxes are passed tpo the function, boxes are unchanged
            else: boxes returned are straight boxes fitted to the straightened rotated boxes
            so that we fit the lines afterwards to the straigthened page
    """
    if boxes.ndim == 3:
        boxes = rotate_boxes(
            loc_preds=boxes,
            angle=-estimate_page_angle(boxes),
            orig_shape=(1024, 1024),
            min_angle=5.,
        )
        boxes = np.concatenate((boxes.min(1), boxes.max(1)), -1)
    return (boxes[:, 0] + 2 * boxes[:, 3] / np.median(boxes[:, 3] - boxes[:, 1])).argsort(), boxes



def _resolve_sub_lines(boxes, word_idcs,paragraph_break=0.035):
    """Split a line in sub_lines
    Args:
        boxes: bounding boxes of shape (N, 4)
        word_idcs: list of indexes for the words of the line
    Returns:
        A list of (sub-)lines computed from the original line (words)
    """
    lines = []
    # Sort words horizontally
    word_idcs = [word_idcs[idx] for idx in boxes[word_idcs, 0].argsort().tolist()]

    # Eventually split line horizontally
    if len(word_idcs) < 2:
        lines.append(word_idcs)
    else:
        sub_line = [word_idcs[0]]
        for i in word_idcs[1:]:
            horiz_break = True

            prev_box = boxes[sub_line[-1]]
            # Compute distance between boxes
            dist = boxes[i, 0] - prev_box[2]
            # If distance between boxes is lower than paragraph break, same sub-line
            if dist < paragraph_break:
                horiz_break = False

            if horiz_break:
                lines.append(sub_line)
                sub_line = []

            sub_line.append(i)
        lines.append(sub_line)

    return lines

def resolve_lines(boxes):
    """Order boxes to group them in lines
    Args:
        boxes: bounding boxes of shape (N, 4) or (N, 4, 2) in case of rotated bbox
    Returns:
        nested list of box indices
    """

    # Sort boxes, and straighten the boxes if they are rotated
    idxs, boxes = _sort_boxes(boxes)

    # Compute median for boxes heights
    y_med = np.median(boxes[:, 3] - boxes[:, 1])

    lines = []
    words = [idxs[0]]  # Assign the top-left word to the first line
    # Define a mean y-center for the line
    y_center_sum = boxes[idxs[0]][[1, 3]].mean()

    for idx in idxs[1:]:
        vert_break = True

        # Compute y_dist
        y_dist = abs(boxes[idx][[1, 3]].mean() - y_center_sum / len(words))
        # If y-center of the box is close enough to mean y-center of the line, same line
        if y_dist < y_med / 2:
            vert_break = False

        if vert_break:
            # Compute sub-lines (horizontal split)
            lines.extend(_resolve_sub_lines(boxes, words))
            words = []
            y_center_sum = 0

        words.append(idx)
        y_center_sum += boxes[idx][[1, 3]].mean()

    # Use the remaining words to form the last(s) line(s)
    if len(words) > 0:
        # Compute sub-lines (horizontal split)
        lines.extend(_resolve_sub_lines(boxes, words))

    return lines