![](assets/timber.jpg?raw=true)

A project using a low-cost [Arducam ToF depth camera](https://blog.arducam.com/time-of-flight-camera-raspberry-pi/) with a Raspberry Pi to measure the area of circular objects (such as cross-sectional area of timber logs).

Place the circular object along the optical axis of the camera (in the center of the screen), launch the UI (`python tk.py`), adjust the necessary threshold values using the sliders, and click the "Measure" button. The measurement result will include the area of the shape in mm², as well as the lengths of the semi-major axes a and b in mm for an [area-equivalent ellipse](https://en.wikipedia.org/wiki/Ellipse) for manual size verification.

![](assets/ui.png?raw=true)

The measurement error of the Arducam ToF depth camera, depending on environmental conditions, typically does not exceed 10%.