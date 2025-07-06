

from __future__ import print_function


try:
  import OpenGL
except:
  print('This module depends on PyOpenGL.')
  print('Please run "\033[1m!pip install -q pyopengl\033[0m" '
        'prior importing this module.')
  raise

import ctypes
from ctypes import pointer, util
import os

os.environ['PYOPENGL_PLATFORM'] = 'egl'
_find_library_old = ctypes.util.find_library
try:

  def _find_library_new(name):
    return {
        'GL': 'libOpenGL.so',
        'EGL': 'libEGL.so',
    }.get(name, _find_library_old(name))
  util.find_library = _find_library_new
  import OpenGL.GL as gl
  import OpenGL.EGL as egl
  from OpenGL import error
  from OpenGL.EGL.EXT.device_base import egl_get_devices
  from OpenGL.raw.EGL.EXT.platform_device import EGL_PLATFORM_DEVICE_EXT
except:
  print('Unable to load OpenGL libraries. '
        'Make sure you use GPU-enabled backend.')
  print('Press "Runtime->Change runtime type" and set '
        '"Hardware accelerator" to GPU.')
  raise
finally:
  util.find_library = _find_library_old

def create_initialized_headless_egl_display():
  for device in egl_get_devices():
    display = egl.eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, device, None)

    if display != egl.EGL_NO_DISPLAY and egl.eglGetError() == egl.EGL_SUCCESS:
      try:
        initialized = egl.eglInitialize(display, None, None)
      except error.GLError:
        pass
      else:
        if initialized == egl.EGL_TRUE and egl.eglGetError() == egl.EGL_SUCCESS:
          return display
  return egl.EGL_NO_DISPLAY

def create_opengl_context(surface_size=(640, 480)):
  egl_display = create_initialized_headless_egl_display()
  if egl_display == egl.EGL_NO_DISPLAY:
    raise ImportError('Cannot initialize a headless EGL display.')

  major, minor = egl.EGLint(), egl.EGLint()
  egl.eglInitialize(egl_display, pointer(major), pointer(minor))

  config_attribs = [
      egl.EGL_SURFACE_TYPE, egl.EGL_PBUFFER_BIT, egl.EGL_BLUE_SIZE, 8,
      egl.EGL_GREEN_SIZE, 8, egl.EGL_RED_SIZE, 8, egl.EGL_DEPTH_SIZE, 24,
      egl.EGL_RENDERABLE_TYPE, egl.EGL_OPENGL_BIT, egl.EGL_NONE
  ]
  config_attribs = (egl.EGLint * len(config_attribs))(*config_attribs)

  num_configs = egl.EGLint()
  egl_cfg = egl.EGLConfig()
  egl.eglChooseConfig(egl_display, config_attribs, pointer(egl_cfg), 1,
                      pointer(num_configs))

  width, height = surface_size
  pbuffer_attribs = [
      egl.EGL_WIDTH,
      width,
      egl.EGL_HEIGHT,
      height,
      egl.EGL_NONE,
  ]
  pbuffer_attribs = (egl.EGLint * len(pbuffer_attribs))(*pbuffer_attribs)
  egl_surf = egl.eglCreatePbufferSurface(egl_display, egl_cfg, pbuffer_attribs)

  egl.eglBindAPI(egl.EGL_OPENGL_API)

  egl_context = egl.eglCreateContext(egl_display, egl_cfg, egl.EGL_NO_CONTEXT,
                                     None)
  egl.eglMakeCurrent(egl_display, egl_surf, egl_surf, egl_context)
