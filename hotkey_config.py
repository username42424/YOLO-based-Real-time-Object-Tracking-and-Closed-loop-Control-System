"""Shared hotkey defaults and validation.

The GUI and the CLI use different Windows API wrappers, but they must agree on
the names that are persisted in config.json.  Keeping the policy here avoids a
missing-key fallback silently changing an action (for example ``quit``).
"""

HOTKEY_ACTIONS = ("aim", "screenshot", "quit", "recoil", "trigger")
HOTKEY_DEFAULTS = {
    "aim": "mouse_right",
    "screenshot": "mouse_x2",
    "quit": "`",
    "recoil": "mouse_left",
    "trigger": "f6",
}


def normalize_hotkeys(config, keymap):
    """Return a complete, normalized hotkey mapping or raise ``ValueError``.

    Duplicate physical keys are rejected deliberately.  The pollers perform
    edge detection, so allowing two actions to share a key can make one action
    consume the edge state of the other and look like a random remap.
    """
    source = config if isinstance(config, dict) else {}
    normalized = {}
    for action in HOTKEY_ACTIONS:
        value = str(source.get(action, HOTKEY_DEFAULTS[action]) or "").strip().lower()
        if value not in keymap:
            raise ValueError(f"未知热键: {action}={value or '<empty>'}")
        normalized[action] = value

    owners = {}
    conflicts = []
    for action, value in normalized.items():
        previous = owners.get(value)
        if previous is not None:
            conflicts.append(f"{previous}={value} 与 {action}={value}")
        else:
            owners[value] = action
    if conflicts:
        raise ValueError("热键冲突: " + "; ".join(conflicts))
    return normalized
