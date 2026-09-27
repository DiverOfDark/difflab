# difflab

## Comments and docstrings

The reader is a developer with twenty years of experience. Write for them.

Never describe what the code does. Comment only to carry what the code cannot: where a constant's
value came from, what was measured, which upstream bug a workaround exists for, what breaks if this
changes. If none of those apply, write nothing.

Docstrings are one line or absent. A module docstring earns its place only when the module's reason
for existing is not obvious from its name and its contents.

Keep facts that are expensive to rediscover — measured numbers, cache-key consequences, hardware
limits, licence constraints. Delete everything else.

This governs existing code as much as new. When you touch a file, bring its prose to this bar.

What survives a trim, always: a measured number and what produced it; anything that names a
cache-key or fingerprint consequence; a licence or hardware limit; the reason a workaround exists.
What goes: restatement, narration of control flow, history of what the code used to be, and any
sentence a competent reader would produce themselves from the next five lines.
