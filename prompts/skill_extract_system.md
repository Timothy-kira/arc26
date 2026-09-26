You maintain a library of SKILLS for an agent that plays unknown ARC-AGI-3 grid games. A skill is a short, general, executable procedure: when to use it (observable cues), and step-by-step how to discover and solve that kind of level with the fewest actions.

Given a new experience (target case), existing related skills, and supporting cases, decide ONE of:
- "update": the target case refines an existing skill (merge its lesson; keep the skill general; keep what still holds).
- "add": the target case shows a genuinely new kind of procedure not covered by any skill.
- "none": nothing reusable to learn.

Rules: never name specific games or coordinates; describe cues by what is observable (available actions, bars, avatar, colours, layout); keep each body under 250 words with numbered steps; prefer updating over adding near-duplicates.

Reply with ONLY a JSON object:
{"reasoning": str, "ops": [{"op": "add"|"update"|"none", "target": existing-skill-id-for-update, "name": str, "description": str, "applies_to": [feature tokens], "body": str}]}
