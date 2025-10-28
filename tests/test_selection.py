from utils.selection import image_uncertainty
def test_uncertainty_no_boxes():
    assert abs(image_uncertainty({'boxes': []}) - 1.0) < 1e-9
