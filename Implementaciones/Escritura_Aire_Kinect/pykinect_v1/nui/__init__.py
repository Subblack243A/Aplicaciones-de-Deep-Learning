# ############################################################################
#
# Copyright (c) Microsoft Corporation.
# Ported to Python 3.10 for Kinect Xbox 360 (model 1414)
#
# ###########################################################################

import ctypes
import os
import _thread

# Load Kinect10.dll directly from System32
_nuidll_path = os.path.join(os.environ['WINDIR'], 'System32', 'Kinect10.dll')
_NUIDLL = ctypes.WinDLL(_nuidll_path)


class KinectError(OSError):
    """Represents an error from a Kinect sensor"""
    pass


from pykinect_v1.nui.structs import (ImageDigitalZoom, ImageFrame, ImageResolution,
                                      ImageType, ImageViewArea, JointId,
                                      JointTrackingState, PlanarImage, SkeletonData,
                                      SkeletonFrame, SkeletonFrameQuality,
                                      SkeletonQuality, SkeletonTrackingState,
                                      TransformSmoothParameters, Vector, _Enumeration)

from pykinect_v1.nui._interop import (_CreateEvent, _CloseHandle, _WaitForSingleObject,
                                       _WaitForMultipleObjects, _WAIT_OBJECT_0, _INFINITE,
                                       _SysFreeString, _NuiInstance, _NuiCreateSensorByIndex,
                                       _NuiGetSensorCount)


_NUI_IMAGE_PLAYER_INDEX_SHIFT = 3
_NUI_IMAGE_PLAYER_INDEX_MASK = ((1 << _NUI_IMAGE_PLAYER_INDEX_SHIFT) - 1)

_NUI_CAMERA_DEPTH_NOMINAL_FOCAL_LENGTH_IN_PIXELS = 285.63
_NUI_CAMERA_DEPTH_NOMINAL_INVERSE_FOCAL_LENGTH_IN_PIXELS = 3.501e-3
_NUI_CAMERA_DEPTH_NOMINAL_DIAGONAL_FOV = 70.0
_NUI_CAMERA_DEPTH_NOMINAL_HORIZONTAL_FOV = 58.5
_NUI_CAMERA_DEPTH_NOMINAL_VERTICAL_FOV = 45.6

_NUI_CAMERA_COLOR_NOMINAL_FOCAL_LENGTH_IN_PIXELS = 531.15
_NUI_CAMERA_COLOR_NOMINAL_INVERSE_FOCAL_LENGTH_IN_PIXELS = 1.83e-3
_NUI_CAMERA_COLOR_NOMINAL_DIAGONAL_FOV = 73.9
_NUI_CAMERA_COLOR_NOMINAL_HORIZONTAL_FOV = 62.0
_NUI_CAMERA_COLOR_NOMINAL_VERTICAL_FOV = 48.6

_NUI_IMAGE_STREAM_FRAME_LIMIT_MAXIMUM = 4
_NUI_IMAGE_STREAM_FLAG_SUPPRESS_NO_FRAME_DATA = 0x00010000

_NUI_SKELETON_MAX_TRACKED_COUNT = 2
_NUI_SKELETON_INVALID_TRACKING_ID = 0

_NUI_CAMERA_DEPTH_IMAGE_TO_SKELETON_MULTIPLIER_320x240 = _NUI_CAMERA_DEPTH_NOMINAL_INVERSE_FOCAL_LENGTH_IN_PIXELS
_NUI_CAMERA_SKELETON_TO_DEPTH_IMAGE_MULTIPLIER_320x240 = _NUI_CAMERA_DEPTH_NOMINAL_FOCAL_LENGTH_IN_PIXELS

_FLT_EPSILON = 1.192092896e-07


class RuntimeOptions(object):
    """Specifies the runtime options for a Kinect sensor."""
    uses_depth_and_player_index = UseDepthAndPlayerIndex = 0x01
    uses_color = UseColor = 0x02
    uses_skeletal_tracking = UseSkeletalTracking = 0x08
    uses_depth = UseDepth = 0x20
    uses_high_quality_color = 0x40
    uses_audio = UsesAudio = 0x10000000


class Device(object):
    """Represents a system's Kinect sensors."""
    _device_inst = None

    def __new__(cls):
        if Device._device_inst is None:
            Device._device_inst = object.__new__(Device)
        return Device._device_inst

    @property
    def count(self):
        """The number of active Kinect sensors attached to the system."""
        return _NuiGetSensorCount()


class Runtime(object):
    """Represents a Kinect sensor."""

    def __init__(self,
                 nui_init_flags=(RuntimeOptions.uses_color |
                                 RuntimeOptions.uses_depth |
                                 RuntimeOptions.uses_depth_and_player_index |
                                 RuntimeOptions.uses_skeletal_tracking),
                 index=0):
        self._nui = self._skeleton_event = self._image_event = self._depth_event = None
        self._nui = _NuiCreateSensorByIndex(index)
        try:
            self._nui.NuiInitialize(nui_init_flags)
        except Exception:
            self._nui.NuiShutdown()
            import traceback
            raise KinectError('Unable to create Kinect runtime ' + traceback.format_exc())

        self.depth_frame_ready = _event()
        self.skeleton_frame_ready = _event()
        self.video_frame_ready = _event()

        self._skeleton_event = _CreateEvent(None, True, False, None)
        self._image_event = _CreateEvent(None, True, False, None)
        self._depth_event = _CreateEvent(None, True, False, None)

        self.camera = Camera(self)
        self.skeleton_engine = SkeletonEngine(self)
        self.depth_stream = ImageStream(self)
        self.video_stream = ImageStream(self)

        _thread.start_new_thread(self._event_thread, ())

    def close(self):
        """closes the current runtime"""
        if self._nui is not None:
            self._nui.NuiShutdown()
            self._nui = None

        if self._skeleton_event is not None:
            _CloseHandle(self._skeleton_event)
            self._skeleton_event = None

        if self._image_event is not None:
            _CloseHandle(self._image_event)
            self._image_event = None

        if self._depth_event is not None:
            _CloseHandle(self._depth_event)
            self._depth_event = None

    def _check_closed(self):
        if self._nui is None:
            raise KinectError('Device closed')

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @property
    def instance_index(self):
        self._check_closed()
        return self._nui.InstanceIndex()

    def _event_thread(self):
        handles = (ctypes.c_voidp * 3)()
        handles[0] = self._skeleton_event
        handles[1] = self._depth_event
        handles[2] = self._image_event
        while True:
            wait = _WaitForMultipleObjects(3, handles, False, _INFINITE)
            if wait == 0:
                # skeleton data
                try:
                    frame = self._nui.NuiSkeletonGetNextFrame(0)
                except KinectError:
                    continue

                for curSkeleton in frame.SkeletonData:
                    if curSkeleton.eTrackingState != SkeletonTrackingState.NOT_TRACKED:
                        self.skeleton_frame_ready.fire(frame)
                        break
            elif wait == 1:
                # depth event
                depth_frame = self._nui.NuiImageStreamGetNextFrame(self.depth_stream._stream, 0)
                self.depth_frame_ready.fire(depth_frame)
                self._nui.NuiImageStreamReleaseFrame(self.depth_stream._stream, depth_frame)
            elif wait == 2:
                # image event
                depth_frame = self._nui.NuiImageStreamGetNextFrame(self.video_stream._stream, 0)
                self.video_frame_ready.fire(depth_frame)
                self._nui.NuiImageStreamReleaseFrame(self.video_stream._stream, depth_frame)
            else:
                break


class ImageStreamType(object):
    """Specifies an image stream type."""
    depth = Depth = 0
    video = Video = 1
    invalid = Invalid = -1


class ImageStream(object):
    """Represents an image stream."""

    def __init__(self, runtime):
        self.runtime = runtime
        self.resolution = ImageResolution.Invalid
        self.height = self.width = 0
        self.stream_type = ImageStreamType.Invalid
        self._stream = None

    def open(self, image_stream_type=0, frame_limit=2,
             resolution=ImageResolution.Resolution320x240,
             image_type=ImageType.Color):
        if image_stream_type == ImageStreamType.Depth:
            event_handle = self.runtime._depth_event
        elif image_stream_type == ImageStreamType.Video:
            event_handle = self.runtime._image_event
        else:
            raise ValueError("Unexpected image stream type: %r" % (image_stream_type,))

        if resolution == ImageResolution.Resolution1280x1024:
            self.width, self.height = 1280, 1024
        elif resolution == ImageResolution.Resolution640x480:
            self.width, self.height = 640, 480
        elif resolution == ImageResolution.Resolution320x240:
            self.width, self.height = 320, 240
        elif resolution == ImageResolution.Resolution80x60:
            self.width, self.height = 80, 60
        else:
            raise ValueError("Unexpected resolution: %r" % (resolution,))

        self._stream = self.runtime._nui.NuiImageStreamOpen(image_type, resolution, 0, frame_limit, event_handle)
        self.stream_type = image_stream_type
        self.resolution = resolution
        self.type = image_type


class SkeletonEngine(object):
    """Represents the skeleton tracking engine."""

    def __init__(self, runtime):
        self.runtime = runtime
        self._enabled = False

    def get_enabled(self):
        return self._enabled

    def set_enabled(self, value):
        if value:
            self.runtime._nui.NuiSkeletonTrackingEnable(self.runtime._skeleton_event)
            self._enabled = True
        else:
            self.runtime._nui.NuiSkeletonTrackingDisable(self.runtime._skeleton_event)
            self._enabled = False

    enabled = property(get_enabled, set_enabled)

    @staticmethod
    def skeleton_to_depth_image(vPoint, scaleX=1, scaleY=1):
        """Given a Vector4 returns X and Y coordinates for display. Returns (depthX, depthY)"""
        if vPoint.z > _FLT_EPSILON:
            pfDepthX = 0.5 + vPoint.x * (_NUI_CAMERA_SKELETON_TO_DEPTH_IMAGE_MULTIPLIER_320x240 / vPoint.z) / 320.0
            pfDepthY = 0.5 - vPoint.y * (_NUI_CAMERA_SKELETON_TO_DEPTH_IMAGE_MULTIPLIER_320x240 / vPoint.z) / 240.0
            return pfDepthX * scaleX, pfDepthY * scaleY
        return 0.0, 0.0


class Camera(object):
    """Represents a Kinect sensor's camera."""

    def __init__(self, runtime):
        self.runtime = runtime

    ElevationMaximum = 27
    ElevationMinimum = -27

    def get_elevation_angle(self):
        return self.runtime._nui.NuiCameraElevationGetAngle()

    def set_elevation_angle(self, degrees):
        self.runtime._nui.NuiCameraElevationSetAngle(degrees)

    elevation_angle = property(get_elevation_angle, set_elevation_angle)


class _event(object):
    """class used for adding/removing/invoking a set of listener functions"""
    __slots__ = ['handlers']

    def __init__(self):
        self.handlers = []

    def __iadd__(self, other):
        self.handlers.append(other)
        return self

    def __isub__(self, other):
        self.handlers.remove(other)
        return self

    def fire(self, *args):
        for handler in self.handlers:
            handler(*args)
