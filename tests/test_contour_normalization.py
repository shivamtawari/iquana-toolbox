import numpy as np

from iquana_toolbox.schemas.database.contours import Contour


def test_from_binary_mask_preserves_fractional_coordinates_for_multiple_contours():
    mask = np.zeros((20, 30), dtype=np.uint8)
    mask[2:8, 4:10] = 1
    mask[11:18, 18:27] = 1

    contours = Contour.from_binary_mask(
        mask,
        only_return_biggest_contour=False,
    )

    assert len(contours) == 2
    assert max(max(contour.x) for contour in contours) > 0.5
    assert max(max(contour.y) for contour in contours) > 0.5
    assert all(0.0 <= coordinate <= 1.0 for contour in contours for coordinate in contour.x)
    assert all(0.0 <= coordinate <= 1.0 for contour in contours for coordinate in contour.y)
