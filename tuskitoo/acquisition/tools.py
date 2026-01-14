from astropy.coordinates import SkyCoord,Angle


def convert_to_degrees(ra_hms, dec_dms):
    """
    Convert RA and Dec from sexagesimal format (HHMMSS.sss, ±DDMMSS.sss) to decimal degrees.

    Parameters:
    ra_hms (float): Right Ascension in HHMMSS.sss format.
    dec_dms (float): Declination in ±DDMMSS.sss format.

    Returns:
    tuple: RA and Dec in decimal degrees.
    """
    # Parse the RA into hours, minutes, and seconds
    ra_hours = int(ra_hms // 10000)
    ra_minutes = int((ra_hms % 10000) // 100)
    ra_seconds = ra_hms % 100

    # Parse the Dec into degrees, minutes, and seconds
    dec_degrees = int(dec_dms // 10000)
    dec_minutes = int((abs(dec_dms) % 10000) // 100)
    dec_seconds = abs(dec_dms) % 100

    # Format them into strings for SkyCoord
    ra_str = f"{ra_hours}h{ra_minutes}m{ra_seconds:.3f}s"
    dec_str = f"{dec_degrees}d{dec_minutes}m{dec_seconds:.3f}s"

    # Create SkyCoord object
    coord = SkyCoord(ra=ra_str, dec=dec_str)

    # Get RA and Dec in degrees
    ra_degrees = coord.ra.deg
    dec_degrees = coord.dec.deg

    return ra_degrees, dec_degrees