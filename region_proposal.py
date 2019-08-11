import cv2


def region_proposal(scene):
    """
    Image size of 720p is required.
    """
    circles = cv2.HoughCircles(cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY), cv2.HOUGH_GRADIENT, 1, 100, param1=50,
                               param2=30, minRadius=55, maxRadius=65)
    if circles is None:
        return []
    return [(int(c[0]-int(c[2] * 2.4)//2), int(c[1]-int(c[2] * 2.4)//2), int(c[2] * 2.4)) for c in circles[0]]
