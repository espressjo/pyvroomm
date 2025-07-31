def stretched_wavelength_to_rgb(wavelength_nm: float, gamma: float = 0.8) -> tuple[int, int, int]:
    """
    Convert a wavelength (350–930 nm) to RGB by stretching the visible spectrum (380–780 nm)
    over the full input range.

    Parameters:
    - wavelength_nm: float, wavelength in nanometers (350–930)
    - gamma: float, gamma correction factor

    Returns:
    - (R, G, B): tuple[int, int, int] in range 0–255
    """

    # Clamp input to supported range
    wavelength_nm = max(350, min(930, wavelength_nm))

    # Linearly map 350–930 nm to visible 380–780 nm range
    visible_start = 380
    visible_end = 780
    stretch_start = 350
    stretch_end = 930

    # Stretch the input wavelength into the visible range
    wavelength = visible_start + (wavelength_nm - stretch_start) * (visible_end - visible_start) / (stretch_end - stretch_start)

    # Now apply standard visible-light to RGB conversion
    if wavelength < 440:
        R, G, B = -(wavelength - 440) / (440 - 380), 0.0, 1.0
    elif wavelength < 490:
        R, G, B = 0.0, (wavelength - 440) / (490 - 440), 1.0
    elif wavelength < 510:
        R, G, B = 0.0, 1.0, -(wavelength - 510) / (510 - 490)
    elif wavelength < 580:
        R, G, B = (wavelength - 510) / (580 - 510), 1.0, 0.0
    elif wavelength < 645:
        R, G, B = 1.0, -(wavelength - 645) / (645 - 580), 0.0
    else:
        R, G, B = 1.0, 0.0, 0.0

    # Intensity correction (full across stretched range)
    if wavelength < 420:
        factor = 0.3 + 0.7 * (wavelength - 380) / (40)
    elif wavelength > 700:
        factor = 0.3 + 0.7 * (780 - wavelength) / (80)
    else:
        factor = 1.0

    def correct(c):
        return (c * factor)**gamma #removed *255

    return correct(R), correct(G), correct(B)


if "__main__" == __name__:
    print(stretched_wavelength_to_rgb(929.96670814))