"""Contains transforms for spatially manipulating the root system.

This module defines the spatial transforms needed for
manipulating the root system using affline transformation matrices.

"""

import numpy as np


def get_x_rotation_matrix(theta: float) -> np.ndarray:
    """
    Construct a rotation matrix for the x axis.

    Args:
        theta (float):
            The x rotation angle in degrees.

    Returns:
        np.ndarray:
            The x rotation matrix.
    """
    cos = np.cos(np.radians(theta))
    sin = np.sin(np.radians(theta))

    x_rotate = np.eye(4)
    x_rotate[1:3, 1:3] = np.array([[cos, sin], [-sin, cos]])
    return x_rotate


def get_y_rotation_matrix(theta: float) -> np.ndarray:
    """
    Construct a rotation matrix for the y axis.

    Args:
        theta (float):
            The y rotation angle in degrees.

    Returns:
        np.ndarray:
            The y rotation matrix.
    """
    cos = np.cos(np.radians(theta))
    sin = np.sin(np.radians(theta))

    y_rotate = np.eye(4)
    y_rotate[0, 0:3] = [cos, 0, -sin]
    y_rotate[2, 0:3] = [sin, 0, cos]
    return y_rotate


def get_z_rotation_matrix(theta: float) -> np.ndarray:
    """
    Construct a rotation matrix for the z axis.

    Args:
        theta (float):
            The z rotation angle in degrees.

    Returns:
        np.ndarray:
            The z rotation matrix.
    """
    cos = np.cos(np.radians(theta))
    sin = np.sin(np.radians(theta))

    z_rotate = np.eye(4)
    z_rotate[0:2, 0:2] = np.array([[cos, sin], [-sin, cos]])
    return z_rotate


def get_transform_matrix(
    roll: float = 0,
    pitch: float = 0,
    yaw: float = 0,
    translation: list[float] = [0, 0, 0],
    reflect: list[float] = [1, 1, 1, 1],
    scale: list[float] = [1, 1, 1, 1],
) -> np.ndarray:
    """
    Updates the transformation matrix.

    Args:
        roll (float):
            Rotation about the x-axis in degrees.
        pitch (float):
            Rotation about the y-axis in degrees.
        yaw (float):
            Rotation about the z-axis in degrees.
        translation (list[float]):
            Translation matrix.
        reflect (list[float]):
            Reflection matrix.
        scale (list[float]):
            Scaling matrix.

    Returns:
        np.ndarray:
            Transformation matrix.
    """
    # Rotations
    x_rotate = get_x_rotation_matrix(roll)
    y_rotate = get_y_rotation_matrix(pitch)
    z_rotate = get_z_rotation_matrix(yaw)
    # Translate
    translate = np.eye(4)
    translate[:-1, 3] = np.array(translation)

    transformation_matrix = (
        translate @ np.diag(scale) @ np.diag(reflect) @ x_rotate @ y_rotate @ z_rotate
    )
    return transformation_matrix


def make_homogenous(arr: np.array) -> np.ndarray:
    """
    Adds an additional 'W' dimension of ones to the array.
    Performs a conversion from Cartesian coordinates to homogenous coordinates.

    Args:
        arr (float):
            The array of Cartesian coordinates.

    Returns:
        np.ndarray:
            The homogenous array.
    """
    ones = np.ones((len(arr), 1))
    homogenous_coordinates = np.hstack((arr, ones)).T
    return homogenous_coordinates
