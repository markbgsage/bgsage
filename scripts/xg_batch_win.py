#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Drive eXtreme Gammon's Batch Analyze dialog deterministically (pywinauto,
Win32 backend) -- no vision model in the loop.

XG's dialogs are ordinary Delphi/Win32 windows: the "Batch Analyze" form
(class ``TBatchDlg``) exposes its ``Choose`` / ``Start`` / ``Close`` buttons,
the ``Save Games after analyze`` / ``Add to this Profile:`` / ``Override
Previous Analyze`` check boxes and the per-player ``Analyze Level`` combo box
as real controls, and ``Choose`` opens a standard Windows Open dialog whose
``File name`` box accepts a multi-file selection (``"a.xg" "b.xg" ...``).
The only custom-drawn widgets are the two grids (file list, player list);
those are driven by clicking at known offsets.

    py -3.14 scripts/xg_batch_win.py run --folder data/backgame_xg_flat/xg2ply --level Sage2P
    py -3.14 scripts/xg_batch_win.py run --folder ... --level "Custom Setting ++" --no-start --shot dialog.png
    py -3.14 scripts/xg_batch_win.py start            # press Start on the open dialog
    py -3.14 scripts/xg_batch_win.py dismiss          # click OK on a leftover message box
    py -3.14 scripts/xg_batch_win.py shot out.png     # screenshot of the Batch Analyze dialog

XG must be running (the script does not launch it). Steps are verified by
reading the controls back; a failed check raises rather than guessing.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from pywinauto import Desktop, keyboard
from pywinauto.timings import wait_until

#: The Analyze Level combo lists XG analysis PROFILES (the built-in presets plus
#: the ones defined under Options); pass the profile's exact name.
LEVEL_LABELS = None


def _desktop():
    return Desktop(backend="win32")


def _windows(retries: int = 4):
    """Top-level windows, tolerating one that vanishes mid-enumeration.

    pywinauto builds and wraps the WHOLE list inside ``windows()``, so a window
    destroyed between the enumeration and the wrapping raises
    ``InvalidWindowHandle`` out of that call -- the per-window ``try/except`` in
    the loops below never gets a chance. It is a race, not a state, so retrying
    clears it. Seen when the session unlocks and transient windows churn, which
    is exactly when an unattended chain resumes.
    """
    for _ in range(retries):
        try:
            return _desktop().windows()
        except Exception:
            time.sleep(0.4)
    return []


def main_window():
    for w in _windows():
        try:
            if w.class_name() == "TMainX" and w.window_text().startswith("eXtreme Gammon"):
                return w
        except Exception:
            continue
    raise SystemExit("eXtreme Gammon is not running")


def _find(cls: str, title: str | None = None, visible_only: bool = True):
    for w in _windows():
        try:
            if w.class_name() != cls:
                continue
            if title is not None and w.window_text().strip() != title:
                continue
            if visible_only and not w.is_visible():
                continue
            return w
        except Exception:
            continue
    return None


def _child(win, cls: str, title: str | None = None, visible: bool = True, timeout: float = 8.0):
    """First descendant control of ``win`` with class ``cls`` (and text ``title``),
    retrying while a freshly created form is still building its controls."""
    end = time.time() + timeout
    while True:
        for c in win.descendants():
            try:
                if c.class_name() != cls:
                    continue
                if title is not None and c.window_text().strip() != title:
                    continue
                if visible and not c.is_visible():
                    continue
                return c
            except Exception:
                continue
        if time.time() > end:
            raise RuntimeError(f"no {cls} {title!r} in {win.window_text()!r}")
        time.sleep(0.4)


BM_GETCHECK = 0x00F0


def _checked(ctrl) -> int:
    import win32gui
    return int(win32gui.SendMessage(ctrl.handle, BM_GETCHECK, 0, 0))


def batch_dialog(timeout: float = 10.0):
    """The visible Batch Analyze form, or None."""
    end = time.time() + timeout
    while time.time() < end:
        d = _find("TBatchDlg", "Batch Analyze")
        if d is not None:
            return d
        time.sleep(0.5)
    return None


def dismiss_message() -> bool:
    """Click OK on a leftover XG message box -- the 'Batch Analyze completed'
    box is a plain Win32 dialog (#32770) owned by XG; XG's own TMessageDlgG
    boxes are handled too."""
    pid = main_window().process_id()
    done = False
    for w in _windows():
        try:
            if w.process_id() != pid or not w.is_visible():
                continue
            if w.class_name() == "#32770" and w.window_text().strip() != "Open":
                for cls in ("Button", "TButton"):
                    try:
                        _child(w, cls, "OK", timeout=1).click_input(); done = True; break
                    except Exception:
                        continue
                else:
                    w.type_keys("{ENTER}"); done = True
                time.sleep(0.5)
            elif w.class_name() == "TMessageDlgG":
                for name in ("OK", "&OK", "Yes", "&Yes"):
                    try:
                        _child(w, "TButton", name, timeout=1).click_input(); done = True; break
                    except Exception:
                        continue
                time.sleep(0.5)
        except Exception:
            continue
    return done


def close_open_dialog() -> bool:
    """Cancel a leftover file Open dialog.

    An attempt killed part-way through ``choose_files`` leaves the Open dialog
    sitting there. It is MODAL, so every later attempt blocks on it -- the next
    ``choose_files`` waits for a dialog that will never close, and neither
    ``dismiss_message`` (which skips windows titled "Open") nor
    ``close_batch_dialog`` touches it. That wedges the chain permanently, so
    clearing it is part of starting a run, not a manual repair.
    """
    d = _find("#32770", "Open")
    if d is None:
        return False
    try:
        _child(d, "Button", "Cancel", timeout=2).click_input()
    except Exception:
        try:
            d.set_focus(); d.type_keys("{ESC}")
        except Exception:
            return False
    time.sleep(1.0)
    return _find("#32770", "Open") is None


def close_batch_dialog() -> None:
    """Close a Batch Analyze form left over from a finished run, so each run
    starts from a fresh dialog with an empty file list."""
    d = batch_dialog(timeout=1)
    if d is None:
        return
    for name in ("Close", "Stop and Cancel"):
        try:
            _child(d, "TButton", name, timeout=1).click_input(); break
        except Exception:
            continue
    end = time.time() + 10
    while time.time() < end and batch_dialog(timeout=0.5) is not None:
        dismiss_message(); time.sleep(0.5)


def open_batch_dialog():
    """Open a fresh Analyze > Batch Analyze... form and return it."""
    dismiss_message()
    close_open_dialog()          # modal; blocks everything below if left over
    close_batch_dialog()
    main = main_window()
    main.set_focus(); time.sleep(0.5)
    # XG's menu bar is a Delphi action bar, not an HMENU: click "Analyze" (its
    # label sits ~353 px from the window's left edge, ~60 px below the top; the
    # accelerator is Alt+N) and take the last item of the popup, Batch Analyze...
    r = main.rectangle()
    import pyautogui
    pyautogui.click(r.left + 353, r.top + 60); time.sleep(0.8)
    if _find("#32768") is None:
        keyboard.send_keys("%n"); time.sleep(0.8)
    pop = _find("#32768")
    if pop is None:
        raise RuntimeError("the Analyze menu did not open")
    # owner-drawn items carry no text; "Batch Analyze..." is the last entry
    pr = pop.rectangle()
    pyautogui.click(pr.left + 110, pr.bottom - 18)
    d = batch_dialog(timeout=10)
    if d is None:
        raise RuntimeError("could not open the Batch Analyze dialog")
    return d


def choose_files(dlg, folder: Path, shot: Path | None = None) -> int:
    """Choose -> every .xg in ``folder`` (multi-selection through the File name box)."""
    files = sorted(folder.glob("*.xg"))
    if not files:
        raise SystemExit(f"no .xg files in {folder}")
    _child(dlg, "TButton", "Choose").click_input()
    od = None
    end = time.time() + 15
    while time.time() < end and od is None:
        od = _find("#32770", "Open"); time.sleep(0.3)
    if od is None:
        raise RuntimeError("the Open dialog did not appear")
    edit = _child(od, "Edit")
    # navigate into the folder first ...
    edit.set_edit_text(str(folder).replace("/", "\\")); time.sleep(0.2)
    _child(od, "Button", "&Open").click_input()
    time.sleep(1.5)
    # ... then select every file in the list view (the File-name box is capped at
    # 260 characters, so typing the names only takes the first handful)
    view = _child(od, "SHELLDLL_DefView")
    vr = view.rectangle()
    view.click_input(coords=(vr.width() // 4, 75))      # the first file row, below the column header
    time.sleep(0.4)
    keyboard.send_keys("^a"); time.sleep(0.8)
    chosen = _child(od, "Edit").window_text()
    if chosen.count('"') < 4:
        raise RuntimeError(f"select-all did not take: file name box reads {chosen[:80]!r}")
    _child(od, "Button", "&Open").click_input()
    end = time.time() + 30
    while time.time() < end and _find("#32770", "Open") is not None:
        time.sleep(0.3)
    if _find("#32770", "Open") is not None:
        raise RuntimeError("the Open dialog stayed open -- XG rejected the selection")
    time.sleep(1.0)
    if shot:
        screenshot(dlg, shot)
    return len(files)


def set_level(dlg, level: str, shot: Path | None = None) -> None:
    """Set the Analyze Level combo for every player row of the list (the players
    grid is custom-drawn, so rows are selected by clicking at row offsets)."""
    combos = dlg.descendants(class_name="TComboBox")
    # the level combo sits beside the players grid (upper right); the profile
    # combo ('None') below it -- pick by vertical position
    combos = sorted(combos, key=lambda c: c.rectangle().top)
    level_combo = combos[0]
    if level not in level_combo.item_texts():
        raise SystemExit(f"XG has no analysis profile {level!r}; it offers {level_combo.item_texts()}")
    grid = [g for g in dlg.descendants(class_name="TGListBox")]
    grid = sorted(grid, key=lambda g: g.rectangle().left)[-1]     # right-hand grid = players
    r = grid.rectangle()
    row_h = 22
    seen = []
    for row in range(6):     # a handful of player rows at most
        y = r.top + 6 + row * row_h
        if y > r.bottom - 4:
            break
        grid.click_input(coords=(r.width() // 2, y - r.top))
        time.sleep(0.3)
        level_combo.select(level)
        time.sleep(0.3)
        seen.append(level_combo.selected_text())
    if any(s != level for s in seen):
        raise RuntimeError(f"level combo reads {seen!r} after selecting {level!r}")
    if shot:
        screenshot(dlg, shot)


def set_checkboxes(dlg) -> None:
    for title, want in (("Save Games after analyze", 1), ("Add to this Profile:", 0),
                        ("Override Previous Analyze", 1)):
        cb = _child(dlg, "TCheckBox", title)
        if _checked(cb) != want:
            cb.click_input(); time.sleep(0.2)
        if _checked(cb) != want:
            raise RuntimeError(f"could not set {title!r} to {want}")


def start(dlg) -> None:
    _child(dlg, "TButton", "Start").click_input()


def screenshot(win, path: Path) -> None:
    r = win.rectangle()
    import pyautogui
    img = pyautogui.screenshot(region=(r.left, r.top, r.width(), r.height()))
    img.save(path)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="open the dialog, choose every .xg of a folder, set the level, start")
    r.add_argument("--folder", type=Path, required=True)
    r.add_argument("--level", required=True, help="XG analysis profile name, e.g. Sage3P")
    r.add_argument("--no-start", action="store_true")
    r.add_argument("--shot", type=Path, default=None, help="screenshot of the dialog before Start")
    sub.add_parser("start")
    sub.add_parser("dismiss")
    s = sub.add_parser("shot"); s.add_argument("path", type=Path)
    args = ap.parse_args(argv)

    if args.cmd == "dismiss":
        print("dismissed" if dismiss_message() else "no message box"); return
    if args.cmd == "shot":
        d = batch_dialog(timeout=1) or main_window(); screenshot(d, args.path); print(args.path); return
    if args.cmd == "start":
        d = batch_dialog(timeout=2)
        if d is None:
            raise SystemExit("no Batch Analyze dialog is open")
        start(d); print("started"); return

    dismiss_message()
    d = open_batch_dialog()
    n = choose_files(d, args.folder.resolve())
    print(f"chose {n} files from {args.folder}")
    set_level(d, args.level)
    print(f"level set to {args.level!r} on every player row")
    set_checkboxes(d)
    print("check boxes set (Save Games on, Add to Profile off, Override on)")
    if args.shot:
        screenshot(d, args.shot); print(f"screenshot: {args.shot}")
    if args.no_start:
        print("not started (--no-start)"); return
    start(d); print("started")


if __name__ == "__main__":
    main()
