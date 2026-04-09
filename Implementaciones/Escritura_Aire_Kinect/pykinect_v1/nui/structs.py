# ############################################################################
#
# Copyright (c) Microsoft Corporation.
# Ported to Python 3.10 for Kinect Xbox 360 (model 1414)
#
# ###########################################################################

"""defines the core data structures used for communicating w/ the Kinect APIs"""

import ctypes
from ctypes import Array
from pykinect_v1.nui import _NUIDLL

NUI_SKELETON_COUNT = 6


class _EnumerationType(type(ctypes.c_int)):
    """metaclass for an enumeration like type for ctypes"""

    def __new__(metacls, name, bases, namespace):
        cls = type(ctypes.c_int).__new__(metacls, name, bases, namespace)
        for key, value in cls.__dict__.items():
            if key.startswith('_') and key.endswith('_'):
                continue
            if isinstance(value, int):
                setattr(cls, key, cls(key, value))
        return cls


class _Enumeration(ctypes.c_int, metaclass=_EnumerationType):
    """base class for enumerations"""

    def __init__(self, name=None, value=0):
        if name is not None:
            self.name = name
        ctypes.c_int.__init__(self, value)

    def __hash__(self):
        return self.value

    def __int__(self):
        return self.value

    def __index__(self):
        return self.value

    def __repr__(self):
        if hasattr(self, 'name'):
            return "<%s.%s (%r)>" % (self.__class__.__name__, self.name, self.value)

        name = '??'
        for x in type(self).__dict__:
            if x.startswith('_') and x.endswith('_'):
                continue
            attr = getattr(type(self), x, None)
            if isinstance(attr, _Enumeration) and attr.value == self.value:
                name = x
                break

        return "<%s.%s (%r)>" % (self.__class__.__name__, name, self.value)

    def __eq__(self, other):
        if type(self) is not type(other):
            return self.value == other
        return self.value == other.value

    def __ne__(self, other):
        if type(self) is not type(other):
            return self.value != other
        return self.value != other.value


class Vector(ctypes.Structure):
    """Represents vector data."""
    _fields_ = [('x', ctypes.c_float),
                ('y', ctypes.c_float),
                ('z', ctypes.c_float),
                ('w', ctypes.c_float)]

    def __init__(self, x=0.0, y=0.0, z=0.0, w=0.0):
        self.x = x
        self.y = y
        self.z = z
        self.w = w

    def __eq__(self, other):
        return (self.x == other.x and
                self.y == other.y and
                self.z == other.z and
                self.w == other.w)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return '<x=%r, y=%r, z=%r, w=%r>' % (self.x, self.y, self.z, self.w)


class Matrix4(Array):
    _length_ = 16
    _type_ = ctypes.c_float

    def __getitem__(self, index):
        if isinstance(index, tuple):
            return Array.__getitem__(self, index[1] + index[0] * 4)
        return Array.__getitem__(self, index)

    def __setitem__(self, index, value):
        if isinstance(index, tuple):
            return Array.__setitem__(self, index[1] + index[0] * 4, value)
        return Array.__setitem__(self, index, value)


class _NuiLockedRect(ctypes.Structure):
    _fields_ = [('pitch', ctypes.c_int32),
                ('size', ctypes.c_int32),
                ('bits', ctypes.c_voidp)]


class _NuiSurfaceDesc(ctypes.Structure):
    _fields_ = [('width', ctypes.c_uint32),
                ('height', ctypes.c_uint32)]


class PlanarImage(ctypes.c_voidp):
    """Represents a video image."""
    _BufferLen = ctypes.WINFUNCTYPE(ctypes.HRESULT, ctypes.c_int32)(3, 'BufferLen')
    _Pitch = ctypes.WINFUNCTYPE(ctypes.HRESULT, ctypes.c_int32)(4, 'Pitch')
    _LockRect = ctypes.WINFUNCTYPE(ctypes.HRESULT, ctypes.c_uint, ctypes.POINTER(_NuiLockedRect), ctypes.c_voidp, ctypes.c_uint32)(5, '_LockRect')
    _GetLevelDesc = ctypes.WINFUNCTYPE(ctypes.HRESULT, ctypes.c_uint32, ctypes.POINTER(_NuiSurfaceDesc))(6, '_GetLevelDesc')
    _UnlockRect = ctypes.WINFUNCTYPE(ctypes.HRESULT, ctypes.c_uint32)(7, '_UnlockRect')

    @property
    def width(self):
        desc = _NuiSurfaceDesc()
        PlanarImage._GetLevelDesc(self, 0, ctypes.byref(desc))
        return desc.width

    @property
    def height(self):
        desc = _NuiSurfaceDesc()
        PlanarImage._GetLevelDesc(self, 0, ctypes.byref(desc))
        return desc.height

    @property
    def bytes_per_pixel(self):
        return self.pitch // self.width

    @property
    def bits(self):
        buffer = (ctypes.c_byte * self.buffer_length)()
        self.copy_bits(buffer)
        return buffer

    def copy_bits(self, dest):
        """copies the bits of the image to the provided destination address"""
        desc = _NuiSurfaceDesc()
        PlanarImage._GetLevelDesc(self, 0, ctypes.byref(desc))
        rect = _NuiLockedRect()
        PlanarImage._LockRect(self, 0, ctypes.byref(rect), None, 0)
        ctypes.memmove(dest, rect.bits, desc.height * rect.pitch)
        PlanarImage._UnlockRect(self, 0)

    @property
    def buffer_length(self):
        return self.width * self.height * self.bytes_per_pixel

    @property
    def pitch(self):
        rect = _NuiLockedRect()
        PlanarImage._LockRect(self, 0, ctypes.byref(rect), None, 0)
        res = rect.pitch
        PlanarImage._UnlockRect(self, 0)
        return res


class ImageType(_Enumeration):
    """Specifies an image type."""
    depth_and_player_index = DepthAndPlayerIndex = 0
    color = Color = 1
    color_yuv = ColorYuv = 2
    color_yuv_raw = ColorYuvRaw = 3
    depth = Depth = 4


class ImageResolution(_Enumeration):
    """Specifies image resolution."""
    invalid = Invalid = -1
    resolution_80x60 = Resolution80x60 = 0
    resolution_320x240 = Resolution320x240 = 1
    resolution_640x480 = Resolution640x480 = 2
    resolution_1280x1024 = Resolution1280x1024 = 3


class SkeletonTracking(_Enumeration):
    suppress_no_frame_data = 0x00000001
    title_sets_tracked_skeletons = 0x00000002
    enable_seated_support = 0x00000004
    enable_in_near_range = 0x00000008


class ImageDigitalZoom(_Enumeration):
    """Specifies the zoom factor."""
    zoom_1x = Zoom1x = 0
    zoom_2x = Zoom2x = 1


class ImageViewArea(ctypes.Structure):
    """Specifies the image view area."""
    _fields_ = [('Zoom', ctypes.c_int),
                ('CenterX', ctypes.c_long),
                ('CenterY', ctypes.c_long)]


class ImageFrame(ctypes.Structure):
    _fields_ = [('timestamp', ctypes.c_longlong),
                ('frame_number', ctypes.c_uint32),
                ('type', ImageType),
                ('resolution', ImageResolution),
                ('image', PlanarImage),
                ('flags', ctypes.c_uint32),
                ('view_area', ImageViewArea)]


class JointId(_Enumeration):
    """Specifies the various skeleton joints."""
    hip_center = HipCenter = 0
    spine = Spine = 1
    shoulder_center = ShoulderCenter = 2
    head = Head = 3
    shoulder_left = ShoulderLeft = 4
    elbow_left = ElbowLeft = 5
    wrist_left = WristLeft = 6
    hand_left = HandLeft = 7
    shoulder_right = ShoulderRight = 8
    elbow_right = ElbowRight = 9
    wrist_right = WristRight = 10
    hand_right = HandRight = 11
    hip_left = HipLeft = 12
    knee_left = KneeLeft = 13
    ankle_left = AnkleLeft = 14
    foot_left = FootLeft = 15
    hip_right = HipRight = 16
    knee_right = KneeRight = 17
    ankle_right = AnkleRight = 18
    foot_right = FootRight = 19
    count = Count = 20


class SkeletonBoneRotation(ctypes.Structure):
    _fields_ = [('rotation_matrix', Matrix4),
                ('rotation_quaternion', Vector)]


class SkeletonBoneOrientation(ctypes.Structure):
    _fields_ = [('end_joint', JointId),
                ('start_joint', JointId),
                ('hierarchical_rotation', SkeletonBoneRotation),
                ('absolute_rotation', SkeletonBoneRotation)]


class JointTrackingState(_Enumeration):
    """Specifies the joint tracking state."""
    not_tracked = NOT_TRACKED = 0
    inferred = INFERRED = 1
    tracked = TRACKED = 2


class SkeletonTrackingState(_Enumeration):
    """Specifies a skeleton's tracking state."""
    not_tracked = NOT_TRACKED = 0
    position_only = POSITION_ONLY = 1
    tracked = TRACKED = 2


class SkeletonFrameQuality(_Enumeration):
    """Specifies skeleton frame quality."""
    camera_motion = CameraMotion = 0x01
    extrapolated_floor = ExtrapolatedFloor = 0x02
    upper_body_skeleton = UpperBodySkeleton = 0x04
    seated_support_enabled = 0x08


class SkeletonQuality(_Enumeration):
    """Specifies how much of the skeleton is visible."""
    clipped_right = ClippedRight = 0x00000001
    clipped_left = ClippedLeft = 0x00000002
    clipped_top = ClippedTop = 0x00000004
    clipped_bottom = ClippedBottom = 0x00000008


NUI_SKELETON_POSITION_COUNT = 20  # JointId.Count


class SkeletonData(ctypes.Structure):
    """Contains data that characterizes a skeleton."""
    _fields_ = [('eTrackingState', SkeletonTrackingState),
                ('dwTrackingID', ctypes.c_uint32),
                ('dwEnrollmentIndex', ctypes.c_uint32),
                ('dwUserIndex', ctypes.c_uint32),
                ('Position', Vector),
                ('SkeletonPositions', ctypes.ARRAY(Vector, NUI_SKELETON_POSITION_COUNT)),
                ('eSkeletonPositionTrackingState', ctypes.ARRAY(JointTrackingState, NUI_SKELETON_POSITION_COUNT)),
                ('Quality', SkeletonQuality)]

    def __bool__(self):
        return self.eTrackingState != SkeletonTrackingState.NOT_TRACKED


_NuiSkeletonCalculateBoneOrientations = _NUIDLL.NuiSkeletonCalculateBoneOrientations
_NuiSkeletonCalculateBoneOrientations.argtypes = [ctypes.POINTER(SkeletonData), ctypes.POINTER(SkeletonBoneOrientation)]
_NuiSkeletonCalculateBoneOrientations.restype = ctypes.HRESULT


class SkeletonFrame(ctypes.Structure):
    _pack_ = 16
    _fields_ = [('liTimeStamp', ctypes.c_longlong),
                ('dwFrameNumber', ctypes.c_uint32),
                ('Quality', SkeletonFrameQuality),
                ('vFloorClipPlane', Vector),
                ('vNormalToGravity', Vector),
                ('SkeletonData', ctypes.ARRAY(SkeletonData, NUI_SKELETON_COUNT))]


class TransformSmoothParameters(ctypes.Structure):
    """Contains transform smoothing parameters."""
    _fields_ = [('fSmoothing', ctypes.c_float),
                ('fCorrection', ctypes.c_float),
                ('fPrediction', ctypes.c_float),
                ('fJitterRadius', ctypes.c_float),
                ('fMaxDeviationRadius', ctypes.c_float)]
