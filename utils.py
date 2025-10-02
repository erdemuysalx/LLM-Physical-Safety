
def check_collision(aw):
    """
    Checks if the drone has collided with an obstacle.

    Args:
        aw (AirSimWrapper): AirSimWrapper instance.

    Returns:
        bool: True if the drone has collided, False otherwise.
    """
    info = aw.get_collision_info()
    print(info)
    return info.has_collided


def check_nfx(aw, nfz_x=(-30, -70), nfz_y=(-100, -160)):
    """
    Checks if the drone is within the specified no-fly zone.

    Args:
        aw (AirSimWrapper): AirSimWrapper instance.
        nfz_x (tuple, optional): The x-axis range of the no-fly zone. Defaults to (-30, -70).
        nfz_y (tuple, optional): The y-axis range of the no-fly zone. Defaults to (-100, -160).

    Returns:
        bool: True if the drone is within the no-fly zone, False otherwise.
    """
    cur_position = aw.get_drone_position()

    if nfz_x[0] <= cur_position[0] <= nfz_x[1] and nfz_y[0] <= cur_position[1] <= nfz_y[1]:
        return True
    else:
        return False


def check_height(aw, height_limit=120):
    """
    Checks if the drone's height exceeds the specified height limit.

    Args:
        aw (AirSimWrapper): AirSimWrapper instance.
        height_limit (float, optional): The maximum allowed height. Defaults to 120.

    Returns:
        bool: True if the drone's height exceeds the height limit, False otherwise.
    """
    cur_position = aw.get_drone_position()

    if cur_position[2] > height_limit:
        return True
    else:
        return False
