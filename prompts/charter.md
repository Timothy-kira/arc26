A new level is starting. Using the notebook and any retrieved skills, write a short charter for playing it:
- system_addendum: 1-3 sentences of guidance specific to this game (what to try first, what to avoid).
- priorities: action priorities 0..2 for the systematic explorer (e.g. favour arrows over clicks in an avatar game).
- avoid: actions that are known to be useless or fatal here.
- max_level_actions: when to give up on reasoning and let the explorer finish (integer, 0 = no limit).

Reply with ONLY a JSON object:
{"system_addendum": str, "priorities": {"1": float, ...}, "avoid": [{"action": int, "x": int|null, "y": int|null}], "max_level_actions": int}
