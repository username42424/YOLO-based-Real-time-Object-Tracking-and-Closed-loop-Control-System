# -*- coding: utf-8 -*-
"""Physical relative mouse capture using Windows Raw Input.

The capture owns a message-only window on a dedicated thread.  Its callback is
kept deliberately small: decode one relative RAWMOUSE packet and forward the
timestamped delta.  It never touches Tk and never performs fitting or logging.
"""

import ctypes
from ctypes import wintypes
import sys
import threading
import time


WM_INPUT = 0x00FF
WM_APP_STOP = 0x8001
RID_INPUT = 0x10000003
RIM_TYPEMOUSE = 0
RIDEV_REMOVE = 0x00000001
RIDEV_INPUTSINK = 0x00000100
MOUSE_MOVE_ABSOLUTE = 0x0001
RI_MOUSE_LEFT_BUTTON_DOWN = 0x0001
RI_MOUSE_LEFT_BUTTON_UP = 0x0002
HID_USAGE_PAGE_GENERIC = 0x01
HID_USAGE_GENERIC_MOUSE = 0x02


class RAWINPUTDEVICE(ctypes.Structure):
    _fields_ = [
        ("usUsagePage", wintypes.USHORT),
        ("usUsage", wintypes.USHORT),
        ("dwFlags", wintypes.DWORD),
        ("hwndTarget", wintypes.HWND),
    ]


class RAWINPUTHEADER(ctypes.Structure):
    _fields_ = [
        ("dwType", wintypes.DWORD),
        ("dwSize", wintypes.DWORD),
        ("hDevice", wintypes.HANDLE),
        ("wParam", wintypes.WPARAM),
    ]


class _RAWMOUSE_BUTTON_FIELDS(ctypes.Structure):
    _fields_ = [
        ("usButtonFlags", wintypes.USHORT),
        ("usButtonData", wintypes.USHORT),
    ]


class _RAWMOUSE_BUTTONS(ctypes.Union):
    _anonymous_ = ("fields",)
    _fields_ = [
        ("ulButtons", wintypes.ULONG),
        ("fields", _RAWMOUSE_BUTTON_FIELDS),
    ]


class RAWMOUSE(ctypes.Structure):
    _anonymous_ = ("buttons",)
    _fields_ = [
        ("usFlags", wintypes.USHORT),
        ("buttons", _RAWMOUSE_BUTTONS),
        ("ulRawButtons", wintypes.ULONG),
        ("lLastX", wintypes.LONG),
        ("lLastY", wintypes.LONG),
        ("ulExtraInformation", wintypes.ULONG),
    ]


class _RAWINPUT_DATA(ctypes.Union):
    _fields_ = [("mouse", RAWMOUSE)]


class RAWINPUT(ctypes.Structure):
    _anonymous_ = ("data",)
    _fields_ = [("header", RAWINPUTHEADER), ("data", _RAWINPUT_DATA)]


if sys.platform == "win32":
    LRESULT = ctypes.c_ssize_t
    WNDPROC = ctypes.WINFUNCTYPE(
        LRESULT, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)

    class WNDCLASSW(ctypes.Structure):
        _fields_ = [
            ("style", wintypes.UINT),
            ("lpfnWndProc", WNDPROC),
            ("cbClsExtra", ctypes.c_int),
            ("cbWndExtra", ctypes.c_int),
            ("hInstance", wintypes.HINSTANCE),
            ("hIcon", wintypes.HICON),
            ("hCursor", wintypes.HANDLE),
            ("hbrBackground", wintypes.HBRUSH),
            ("lpszMenuName", wintypes.LPCWSTR),
            ("lpszClassName", wintypes.LPCWSTR),
        ]

    _user32 = ctypes.windll.user32
    _kernel32 = ctypes.windll.kernel32
    _kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
    _kernel32.GetModuleHandleW.restype = wintypes.HMODULE
    _user32.RegisterClassW.argtypes = [ctypes.POINTER(WNDCLASSW)]
    _user32.RegisterClassW.restype = wintypes.ATOM
    _user32.CreateWindowExW.argtypes = [
        wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        wintypes.HWND, wintypes.HMENU, wintypes.HINSTANCE, ctypes.c_void_p,
    ]
    _user32.CreateWindowExW.restype = wintypes.HWND
    _user32.DefWindowProcW.argtypes = [
        wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    _user32.DefWindowProcW.restype = LRESULT
    _user32.RegisterRawInputDevices.argtypes = [
        ctypes.POINTER(RAWINPUTDEVICE), wintypes.UINT, wintypes.UINT]
    _user32.RegisterRawInputDevices.restype = wintypes.BOOL
    _user32.GetRawInputData.argtypes = [
        wintypes.HANDLE, wintypes.UINT, ctypes.c_void_p,
        ctypes.POINTER(wintypes.UINT), wintypes.UINT,
    ]
    _user32.GetRawInputData.restype = wintypes.UINT
    _user32.GetMessageW.argtypes = [
        ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT]
    _user32.GetMessageW.restype = wintypes.BOOL
    _user32.PostMessageW.argtypes = [
        wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    _user32.PostMessageW.restype = wintypes.BOOL
    _user32.DestroyWindow.argtypes = [wintypes.HWND]
    _user32.DestroyWindow.restype = wintypes.BOOL
else:
    WNDPROC = None


class RawMouseInput:
    """Capture physical relative mouse packets and forward them to ``callback``.

    ``callback`` receives ``(dx, dy, monotonic_ns, device_id)``.  The class is
    intentionally Windows-only; callers can inspect ``available`` and keep the
    feature disabled when startup fails.
    """

    def __init__(self, callback, button_callback=None):
        self.callback = callback
        self.button_callback = button_callback
        self._ready = threading.Event()
        self._thread = None
        self._hwnd = None
        self._wndproc = None
        self._healthy = False
        self._last_error = None
        self._event_serial = 0
        self._device_less_serial = 0
        self._serial_lock = threading.Lock()

    @property
    def available(self):
        return bool(self._healthy and self._hwnd)

    @property
    def healthy(self):
        return bool(self._healthy)

    @property
    def last_error(self):
        return self._last_error

    @property
    def event_serial(self):
        with self._serial_lock:
            return self._event_serial

    def start(self, timeout=2.0):
        if sys.platform != "win32":
            self._last_error = "Raw Input 仅支持 Windows"
            return False
        if self._thread and self._thread.is_alive():
            return self.available
        self._ready.clear()
        self._thread = threading.Thread(
            target=self._message_loop, name="RawMouseInput", daemon=True)
        self._thread.start()
        self._ready.wait(max(0.1, float(timeout)))
        return self.available

    def stop(self, timeout=1.0):
        hwnd = self._hwnd
        if hwnd and sys.platform == "win32":
            try:
                ctypes.windll.user32.PostMessageW(hwnd, WM_APP_STOP, 0, 0)
            except Exception:
                pass
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(max(0.0, float(timeout)))
        self._healthy = False
        self._hwnd = None

    def verify_injection_isolated(self, send_fn, attempts=3, settle_ms=35.0):
        """Return true when a +1/-1 SendInput pair produces no Raw Input packet.

        Multiple attempts tolerate incidental physical movement.  If injected
        motion reaches Raw Input, every attempt changes the packet serial and the
        probe fails closed.
        """

        if not self.available:
            self._last_error = self._last_error or "Raw Input 未就绪"
            return False
        delay = max(0.005, float(settle_ms) / 1000.0)
        for _ in range(max(1, int(attempts))):
            # Let any earlier packets drain, but do not require the user to
            # hold the physical mouse perfectly still.  The probe below has a
            # stronger signal: our SendInput packet is device-less and filtered.
            time.sleep(delay)
            before = self.event_serial
            with self._serial_lock:
                device_less_before = self._device_less_serial
            if not send_fn(1, 0) or not send_fn(-1, 0):
                self._last_error = "SendInput 自检发送失败"
                return False
            time.sleep(delay)
            with self._serial_lock:
                device_less_seen = (
                    self._device_less_serial > device_less_before)
            # On Windows, SendInput WM_INPUT packets normally have hDevice=0.
            # Seeing that filtered packet proves isolation even if the user
            # moved the physical mouse during this short probe.
            if device_less_seen or self.event_serial == before:
                return True
        self._last_error = "Raw Input 注入隔离自检失败（未确认程序输入已被过滤）"
        self._healthy = False
        return False

    def _dispatch_relative_packet(self, mouse, device, event_ns):
        """Forward one physical packet in causal button/motion order.

        A down edge is delivered before movement from the same packet so bin 0
        cannot lose the first physical delta.  An up edge is delivered after
        movement so the final delta is still part of the burst.
        """
        device = int(device)
        flags = int(mouse.usButtonFlags)
        is_absolute = bool(int(mouse.usFlags) & MOUSE_MOVE_ABSOLUTE)
        dx, dy = int(mouse.lLastX), int(mouse.lLastY)
        has_motion = not is_absolute and (dx != 0 or dy != 0)
        has_left_edge = bool(flags & (
            RI_MOUSE_LEFT_BUTTON_DOWN | RI_MOUSE_LEFT_BUTTON_UP))
        if not has_motion and not has_left_edge:
            return

        with self._serial_lock:
            if device == 0:
                self._device_less_serial += 1
                return
            self._event_serial += 1
        if flags & RI_MOUSE_LEFT_BUTTON_DOWN and self.button_callback:
            self.button_callback(True, int(event_ns), device)
        if has_motion:
            self.callback(dx, dy, int(event_ns), device)
        if flags & RI_MOUSE_LEFT_BUTTON_UP and self.button_callback:
            self.button_callback(False, int(event_ns), device)

    def _handle_raw_input(self, lparam):
        user32 = ctypes.windll.user32
        size = wintypes.UINT(0)
        header_size = ctypes.sizeof(RAWINPUTHEADER)
        result = user32.GetRawInputData(
            wintypes.HANDLE(lparam), RID_INPUT, None,
            ctypes.byref(size), header_size)
        if result == 0xFFFFFFFF or size.value < header_size:
            raise ctypes.WinError()
        buffer = ctypes.create_string_buffer(size.value)
        result = user32.GetRawInputData(
            wintypes.HANDLE(lparam), RID_INPUT, buffer,
            ctypes.byref(size), header_size)
        if result == 0xFFFFFFFF:
            raise ctypes.WinError()
        raw = ctypes.cast(buffer, ctypes.POINTER(RAWINPUT)).contents
        if raw.header.dwType != RIM_TYPEMOUSE:
            return
        mouse = raw.mouse
        device = int(ctypes.cast(raw.header.hDevice, ctypes.c_void_p).value or 0)
        # Windows reports SendInput-generated WM_INPUT packets without a
        # physical device handle.  Calibration binds to a real hDevice, so
        # device-less packets are never eligible learning samples.
        self._dispatch_relative_packet(
            mouse, device=device, event_ns=time.perf_counter_ns())

    def _unregister(self):
        if sys.platform != "win32":
            return
        device = RAWINPUTDEVICE(
            HID_USAGE_PAGE_GENERIC, HID_USAGE_GENERIC_MOUSE,
            RIDEV_REMOVE, None)
        ctypes.windll.user32.RegisterRawInputDevices(
            ctypes.byref(device), 1, ctypes.sizeof(RAWINPUTDEVICE))

    def _message_loop(self):
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        class_name = f"AimRawMouse_{id(self):x}"

        @WNDPROC
        def wndproc(hwnd, msg, wparam, lparam):
            if msg == WM_INPUT:
                try:
                    self._handle_raw_input(lparam)
                except Exception as exc:
                    self._healthy = False
                    self._last_error = f"Raw Input 读取失败: {exc}"
                return 0
            if msg == WM_APP_STOP:
                self._unregister()
                user32.DestroyWindow(hwnd)
                user32.PostQuitMessage(0)
                return 0
            return user32.DefWindowProcW(hwnd, msg, wparam, lparam)

        self._wndproc = wndproc  # keep the callback alive for the whole window lifetime
        try:
            hinstance = kernel32.GetModuleHandleW(None)
            wc = WNDCLASSW()
            wc.lpfnWndProc = wndproc
            wc.hInstance = hinstance
            wc.lpszClassName = class_name
            if not user32.RegisterClassW(ctypes.byref(wc)):
                raise ctypes.WinError()
            hwnd_message = ctypes.c_void_p(-3 & ((1 << (ctypes.sizeof(ctypes.c_void_p) * 8)) - 1))
            hwnd = user32.CreateWindowExW(
                0, class_name, class_name, 0,
                0, 0, 0, 0, hwnd_message, None, hinstance, None)
            if not hwnd:
                raise ctypes.WinError()
            self._hwnd = hwnd
            device = RAWINPUTDEVICE(
                HID_USAGE_PAGE_GENERIC, HID_USAGE_GENERIC_MOUSE,
                RIDEV_INPUTSINK, hwnd)
            if not user32.RegisterRawInputDevices(
                    ctypes.byref(device), 1, ctypes.sizeof(RAWINPUTDEVICE)):
                raise ctypes.WinError()
            self._healthy = True
            self._ready.set()
            msg = wintypes.MSG()
            while True:
                result = user32.GetMessageW(ctypes.byref(msg), None, 0, 0)
                if result <= 0:
                    break
                user32.TranslateMessage(ctypes.byref(msg))
                user32.DispatchMessageW(ctypes.byref(msg))
        except Exception as exc:
            self._healthy = False
            self._last_error = str(exc)
            self._ready.set()
        finally:
            self._unregister()
            self._hwnd = None
            self._healthy = False
