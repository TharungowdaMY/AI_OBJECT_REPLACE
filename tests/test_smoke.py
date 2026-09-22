def test_environment_and_package_import():
    """The interpreter, core deps, and the project package all import and work."""
    import cv2
    import numpy as np

    import src

    assert isinstance(src.__version__, str) and src.__version__

    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    assert cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).shape == (4, 4)
