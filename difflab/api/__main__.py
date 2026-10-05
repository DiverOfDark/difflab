from difflab.api.serve import serve_forever
from difflab.core import events

# The page watches runs rather than being one.
events.disable()
serve_forever()
