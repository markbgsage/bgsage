# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Win32 Batch Analyze transport for an XG process owned by the runner.

The dialog and control flow follows the audited standalone XG controller.
"""

import ctypes as C
from contextlib import contextmanager
from ctypes import wintypes as W
import time

u32 = C.WinDLL("user32", use_last_error=True)
u32.GetMenu.argtypes = [W.HWND]
u32.GetMenu.restype = W.HMENU
u32.GetClassNameW.argtypes = [W.HWND, W.LPWSTR, C.c_int]
u32.GetWindowTextW.argtypes = [W.HWND, W.LPWSTR, C.c_int]
u32.GetWindowThreadProcessId.argtypes = [W.HWND, C.POINTER(W.DWORD)]
u32.IsWindowVisible.argtypes = [W.HWND]
u32.PostMessageW.argtypes = [W.HWND, C.c_uint, W.WPARAM, W.LPARAM]
u32.SendMessageTimeoutW.argtypes = [W.HWND, C.c_uint, W.WPARAM, C.c_void_p,
                                  C.c_uint, C.c_uint, C.POINTER(C.c_ssize_t)]
u32.SendMessageTimeoutW.restype = C.c_ssize_t

WM_COMMAND, CMD_BATCH_ANALYZE = 0x0111, 281
BM_CLICK, BM_GETCHECK = 0x00F5, 0x00F0
CB_GETCOUNT, CB_GETLBTEXT, CB_GETLBTEXTLEN, CB_SETCURSEL, CB_GETCURSEL = 0x0146, 0x0148, 0x0149, 0x014E, 0x0147
LB_GETCOUNT = 0x018B
WM_SETTEXT, WM_CLOSE = 0x000C, 0x0010
_P = C.WINFUNCTYPE(C.c_bool, W.HWND, W.LPARAM)


def _send(hwnd, message, parameter=0, data=None):
    result = C.c_ssize_t()
    # SMTO_ABORTIFHUNG avoids stranding the game streams in a synchronous UI call.
    if not u32.SendMessageTimeoutW(hwnd, message, parameter, data, 2, 5000, C.byref(result)):
        raise OSError(f"XG did not respond to window message {message:#x}")
    return result.value


@contextmanager
def lease():
    kernel = C.WinDLL("kernel32", use_last_error=True)
    kernel.CreateMutexW.argtypes = [C.c_void_p, W.BOOL, W.LPCWSTR]
    kernel.CreateMutexW.restype = W.HANDLE
    kernel.ReleaseMutex.argtypes = [W.HANDLE]
    kernel.CloseHandle.argtypes = [W.HANDLE]
    handle = kernel.CreateMutexW(None, True, r"Local\bgsage-xg-head-to-head")
    status = C.get_last_error()
    if not handle:
        raise C.WinError(status)
    try:
        if status == 183:  # ERROR_ALREADY_EXISTS
            raise RuntimeError("Another bgsage run owns the XG desktop session")
        try:
            yield
        finally:
            kernel.ReleaseMutex(handle)
    finally:
        kernel.CloseHandle(handle)


def find_main(pid=None):
    """All TMainX windows (optionally for one pid): [(hwnd, pid, title)]."""
    out = []
    P = C.WINFUNCTYPE(C.c_bool, W.HWND, W.LPARAM)

    def cb(h, _):
        b = C.create_unicode_buffer(256)
        u32.GetClassNameW(h, b, 256)
        if b.value == "TMainX":
            p = W.DWORD()
            u32.GetWindowThreadProcessId(h, C.byref(p))
            t = C.create_unicode_buffer(512)
            u32.GetWindowTextW(h, t, 512)
            if pid is None or p.value == pid:
                out.append((h, p.value, t.value))
        return True

    u32.EnumWindows(P(cb), 0)
    return out


def cmd(hwnd, command_id):
    """Post a menu command. Goes to the window, so no focus is required."""
    if not u32.PostMessageW(hwnd, WM_COMMAND, W.WPARAM(command_id), W.LPARAM(0)):
        raise OSError("PostMessage failed for id %d" % command_id)


def visible_dialogs(pid, ignore=("TMainX", "TApplication")):
    """Any visible top-level window of XG other than the board -- a modal popup."""
    out = []
    P = C.WINFUNCTYPE(C.c_bool, W.HWND, W.LPARAM)

    def cb(h, _):
        p = W.DWORD()
        u32.GetWindowThreadProcessId(h, C.byref(p))
        if p.value == pid and u32.IsWindowVisible(h):
            b = C.create_unicode_buffer(256)
            u32.GetClassNameW(h, b, 256)
            if b.value not in ignore:
                t = C.create_unicode_buffer(256)
                u32.GetWindowTextW(h, t, 256)
                out.append((h, b.value, t.value))
        return True

    u32.EnumWindows(P(cb), 0)
    return out


def _kids(hwnd):
    out, seen = [], set()

    def cb(c, _):
        if c in seen:
            return True
        seen.add(c)
        b = C.create_unicode_buffer(64)
        u32.GetClassNameW(c, b, 64)
        t = C.create_unicode_buffer(256)
        u32.GetWindowTextW(c, t, 256)
        out.append((c, b.value, t.value))
        return True

    u32.EnumChildWindows(hwnd, _P(cb), 0)
    return out


def _dialogs(pid, cls=None):
    return [(h, c, t) for h, c, t in visible_dialogs(pid)
            if cls is None or c == cls]


def _button(dlg, *labels):
    want = {l.lower() for l in labels}
    for c, cls, txt in _kids(dlg):
        if cls.lower().startswith("button") or cls == "TButton":
            if txt.replace("&", "").strip().lower() in want:
                return c
    return None


def dismiss_msgboxes(pid):
    """Click OK/No on any stray message box (notably 'Batch Analyze completed')."""
    n = 0
    for dh, _cls, _t in _dialogs(pid, "#32770"):
        b = _button(dh, "ok", "no", "close")
        if b:
            u32.PostMessageW(b, BM_CLICK, 0, 0)
            n += 1
    if n:
        time.sleep(0.15)
    return n


def close_start_dialog(pid, wait=0.0):
    """Close XG's startup dialog (TStartDlg) and CONFIRM it is gone.

    It is created a beat AFTER the main window's menu, so a single WM_CLOSE
    posted the instant the menu appears can find nothing to close -- the dialog
    then comes up afterwards and stays, and while it is up the Batch Analyze
    command is swallowed and the "completed" box can go unnoticed. That stalled a
    run for minutes with nothing in the output to show it. So: wait up to `wait`
    seconds for it to appear, close it, and verify.
    """
    end = time.time() + wait
    while True:
        dlgs = _dialogs(pid, "TStartDlg")
        if dlgs:
            for dh, _cls, _t in dlgs:
                u32.PostMessageW(dh, WM_CLOSE, 0, 0)
            for _ in range(20):
                time.sleep(0.1)
                if not _dialogs(pid, "TStartDlg"):
                    return True
            for dh, _cls, _t in _dialogs(pid, "TStartDlg"):
                u32.PostMessageW(dh, 0x0112, 0xF060, 0)        # WM_SYSCOMMAND SC_CLOSE
            time.sleep(0.3)
            return not _dialogs(pid, "TStartDlg")
        if time.time() >= end:
            return True
        time.sleep(0.2)


def _clear_dialogs(pid):
    """Return XG to a state with none of our dialogs open."""
    for _ in range(8):
        dismiss_msgboxes(pid)
        open_ = _dialogs(pid, "TBatchDlg")
        if not open_:
            return True
        for dh, _c, _t in open_:
            b = _button(dh, "close", "cancel")
            if b:
                u32.PostMessageW(b, BM_CLICK, 0, 0)
            u32.PostMessageW(dh, WM_CLOSE, 0, 0)
        time.sleep(0.3)
    return not _dialogs(pid, "TBatchDlg")


def _find_file_dialog(pid, timeout):
    """The file-open dialog: a #32770 that owns a filename Edit."""
    end = time.time() + timeout
    while time.time() < end:
        for dh, _c, _t in _dialogs(pid, "#32770"):
            for c, k, _tt in _kids(dh):
                if k == "Edit":
                    return dh, c
            for c, k, _tt in _kids(dh):
                if k == "ComboBoxEx32":
                    return dh, c
        time.sleep(0.05)
    return None, None


def _combo_items(c):
    n = _send(c, CB_GETCOUNT)
    out = []
    for i in range(max(n, 0)):
        ln = _send(c, CB_GETLBTEXTLEN, i)
        b = C.create_unicode_buffer(max(ln, 0) + 8)
        _send(c, CB_GETLBTEXT, i, b)
        out.append(b.value)
    return out


def _file_rows(dlg):
    return max((_send(c, LB_GETCOUNT)
                for c, cls, _ in _kids(dlg) if cls == "TGListBox"), default=0)


def _open_batch_dialog(pid, hwnd, timeout=20.0):
    """Open TBatchDlg and wait until its CONTROLS exist.

    Post the command exactly ONCE. Re-posting while a dialog is already up spawns
    duplicates whose children are still being created, which then fail the button
    lookup -- that is what made XG sit there flashing dialogs in a loop. Waiting
    for the buttons rather than the window is the other half: the frame appears
    before its children do.
    """
    if not _clear_dialogs(pid):
        raise RuntimeError("XG's previous batch dialog is still open")
    if not close_start_dialog(pid):
        raise RuntimeError("XG's start dialog is still open")
    cmd(hwnd, CMD_BATCH_ANALYZE)
    end = time.time() + timeout
    while time.time() < end:
        time.sleep(0.1)
        d = _dialogs(pid, "TBatchDlg")
        if d and _button(d[0][0], "choose") and _button(d[0][0], "start"):
            return d[0][0]
    windows = [(kind, title) for _, kind, title in _dialogs(pid)]
    raise RuntimeError(f"Batch Analyze dialog did not open (visible XG dialogs: {windows})")


def _add_file(pid, dlg, xg_path, tries=4):
    """Push one path through the file dialog; verified by the file-list count."""
    for _ in range(tries):
        before = _file_rows(dlg)
        choose = _button(dlg, "choose")
        if choose is None:
            raise RuntimeError("no Choose button")
        u32.PostMessageW(choose, BM_CLICK, 0, 0)
        fd, edit = _find_file_dialog(pid, 6.0)
        if fd is None:
            continue                         # click dropped while initialising
        _send(edit, WM_SETTEXT, data=C.c_wchar_p(str(xg_path)))
        time.sleep(0.05)
        ok = _button(fd, "open")
        if ok:
            u32.PostMessageW(ok, BM_CLICK, 0, 0)
        else:
            u32.PostMessageW(fd, 0x0111, W.WPARAM(1), W.LPARAM(0))     # IDOK
        end = time.time() + 6.0
        while time.time() < end:
            time.sleep(0.05)
            if _file_rows(dlg) > before:
                return True
        dismiss_msgboxes(pid)
    return False


def ready(process, timeout=40):
    """Wait for the newly launched process to own a working desktop window."""
    end = time.monotonic() + timeout
    while time.monotonic() < end and process.poll() is None:
        windows = find_main(process.pid)
        if len(windows) == 1 and u32.GetMenu(windows[0][0]):
            if not close_start_dialog(process.pid, wait=4):
                raise RuntimeError("XG's startup dialog did not close")
            return
        time.sleep(0.2)
    raise RuntimeError("The new XG process did not open a usable desktop window")


def close(process):
    """Close only the XG process launched for this run."""
    from subprocess import TimeoutExpired

    if process.poll() is not None:
        return
    for hwnd, _, _ in find_main(process.pid):
        u32.PostMessageW(hwnd, WM_CLOSE, 0, 0)
    try:
        process.wait(timeout=3)
    except TimeoutExpired:
        process.terminate()
        process.wait(timeout=10)


def run_batch(pid, path, profile, timeout=240):
    """Analyze exactly one shard in place, then wait for the saved result."""
    windows = find_main(pid)
    if len(windows) != 1:
        raise RuntimeError("The dedicated XG window is unavailable")
    dlg = _open_batch_dialog(pid, windows[0][0])
    try:
        if not _add_file(pid, dlg, path) or _file_rows(dlg) != 1:
            raise RuntimeError("XG Batch Analyze did not select exactly one shard")
        selected = 0
        for control, kind, label in _kids(dlg):
            if kind == "TComboBox":
                choices = _combo_items(control)
                matching = [i for i, value in enumerate(choices)
                            if value == profile or value.startswith(profile + " (")]
                if len(matching) == 1:
                    index = matching[0]
                    _send(control, CB_SETCURSEL, index)
                    if _send(control, CB_GETCURSEL) != index:
                        raise RuntimeError("XG did not select the requested batch profile")
                    selected += 1
            if kind == "TCheckBox" and label.replace("&", "").strip().lower().startswith(
                    ("save games", "override")):
                if not _send(control, BM_GETCHECK):
                    u32.PostMessageW(control, BM_CLICK, 0, 0)
                    time.sleep(0.03)
                if not _send(control, BM_GETCHECK):
                    raise RuntimeError("Could not enable XG's batch save/override option")
        if not selected:
            raise RuntimeError(f"XG did not offer the profile {profile!r} in Batch Analyze")
        before = path.stat().st_mtime_ns
        u32.PostMessageW(_button(dlg, "start"), BM_CLICK, 0, 0)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            time.sleep(0.12)
            closed = not _dialogs(pid, "TBatchDlg")
            completed = any("completed" in title.lower() for _, _, title in _dialogs(pid, "#32770"))
            try:
                saved = path.stat().st_mtime_ns != before
            except OSError:
                saved = False
            if saved and (closed or completed):
                break
            if not find_main(pid):
                raise RuntimeError("XG exited during Batch Analyze")
        else:
            raise TimeoutError("XG Batch Analyze did not save the analyzed shard")
    finally:
        _clear_dialogs(pid)
