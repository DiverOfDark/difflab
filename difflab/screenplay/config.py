"""The director's numbers; its words are in `words.py`.

Every DIRECTOR_* here is hashed by name and value into a direction's fingerprint
(`stages._fingerprint` reflects over this module), so adding, renaming or retuning one re-plans every
direction on disk. The module they live in is not hashed - only the names and the values.
"""

# Used only when a job forces a shape (`Job.aspect`). Left alone, a take is cut at the first
# photograph's own aspect: 4:3 forced to 9:16 throws away a third of the picture.
DIRECTOR_ASPECT_TALL = 9 / 16
DIRECTOR_ASPECT_WIDE = 16 / 9
DIRECTOR_SURVEY_PIXELS = 400_000  # per photo when the VLM sees them all at once (~400 vision tokens each)
DIRECTOR_PHOTO_PIXELS = 1024 * 1024  # per photo when it sees one: grounding, trait cards, the opening frame
DIRECTOR_ID_LENGTH = 24  # letters in a member id ("man", "hookah", "shisha_lounge")
DIRECTOR_REPAIRS = 2  # re-asks with the gate's issue list before giving up
DIRECTOR_PLAN_REPAIRS = 3  # the plan answers to the most rules at once, and has needed two repairs on its own
DIRECTOR_TOKENS = {"survey": 1024, "ground": 1024, "card": 2048, "opening": 512, "tidy": 2048, "plan": 8192}
DIRECTOR_WRITE_TOKENS = 2048
# A beat's prose is asked as 35-60 words. Unbounded, the writer once listed clothes and hair for 1,482
# words inside one string, ran out of tokens and left the JSON open, three times in a row. The schema
# closes each string at DIRECTOR_WRITE_CHARS; asked again to shorten, it wrote the same 115-125 words, so
# a beat is cut in code to the whole sentences that fit DIRECTOR_WRITE_WORDS.
DIRECTOR_WRITE_CHARS = 700
DIRECTOR_WRITE_WORDS = 60
DIRECTOR_WRITE_TEMPERATURE = 0.5  # beat prose only; every other first answer decodes greedily
# Greedy decoding asked again repeats the rejected answer word for word, so a repair samples.
DIRECTOR_REPAIR_TEMPERATURE = 0.4
# The opening frame in words: the VLM is asked for OPENING_WORDS and the gate accepts OPENING_WORDS_HARD;
# the writer turns it into one sentence of ARRANGE_WORDS (accepted: ARRANGE_WORDS_HARD).
DIRECTOR_OPENING_WORDS = (40, 80)
DIRECTOR_OPENING_WORDS_HARD = (30, 160)
DIRECTOR_ARRANGE_WORDS = (30, 60)
DIRECTOR_ARRANGE_WORDS_HARD = (25, 120)

# Crops, cut in code from the VLM's boxes (`crops.py`).
DIRECTOR_FACE_MIN = 48  # px of head width: anything smaller gets no face crop
# Face crop width in head widths. A cat's head box already takes in its ruff, and 1.9 of those is mostly blanket.
DIRECTOR_FACE_WIDTH = {"person": 1.9, "animal": 1.4}
DIRECTOR_FACE_HEADROOM = 0.2  # space above the head, in head heights
# An animal's head box is often drawn from the eyes down; its portrait reaches up to the top of the
# animal's own box for the ears, but at most this many head heights above the head box.
DIRECTOR_FACE_EARS = 0.5
DIRECTOR_FACE_SIZE = (512, 640)  # or the largest 128k x 160k within DIRECTOR_MAX_UPSCALE and the canvas
DIRECTOR_FACE_NARROW = 0.7  # a second try at a face crop that caught a bystander: the head box this wide
DIRECTOR_BODY_MARGIN = 0.06  # left, right and top, in box widths/heights; none below, where the floor or blanket is
DIRECTOR_OBJECT_MARGIN = 0.10
DIRECTOR_VIEW_PIXELS = 300_000  # a body, view or object crop (the hand-cut first cast used 0.2-0.33 MP)
DIRECTOR_ENVIRONMENT_PIXELS = 800_000  # the first photo, whole: the room, the colour anchor, who is in it
# The room cut clear of the people and animals in it, for a chunk one of them is out of: made only
# when that leaves at least this much of the photo.
DIRECTOR_ROOM_MIN = 0.2
DIRECTOR_MAX_UPSCALE = 2.5  # a 120 px head still makes a 512x640 face crop (the hand-cut ones went to 3.4x)
DIRECTOR_SHOWS = 0.4  # a crop shows a member when this much of its box lies inside ...
DIRECTOR_SHOWS_HEAD = 0.5  # ... or this much of its head box
DIRECTOR_SHOWS_WHOLE = 0.9  # ... and shows all of it with this much inside (else "partly outside the frame")
# "other" is the same thing as a member when this much of it lies inside the member's box and that much
# of the member's box inside it: the VLM listing a member a second time, not a bystander.
DIRECTOR_SAME_THING = (0.7, 0.5)
DIRECTOR_PORTRAIT_COVERS = 0.7  # no view crop when the face crop already holds this much of it
DIRECTOR_CLEAR_KEEP = 0.6  # a body or view crop cut back past a bystander must keep this much of its member
DIRECTOR_OPENING_LIVING_WEIGHT = 2.0  # in the opening cut, a person's or animal's box against an object's
# An object counts as kept once this much of it is inside, so a prop at the far edge does not push the
# rest of the room (the table it is passed across) out of the frame.
DIRECTOR_OPENING_KEEP = 0.8
# What a chunk may hold, in reference rows: (width/32) * (height/32) per picture. The hand-cut shisha
# cast used 1835 and 2579 and rendered at a 101k-row peak; ~115k did not fit.
DIRECTOR_REFERENCE_ROWS = 3000
DIRECTOR_WORDS = (350, 520)  # detailed_description: official 350-500, plus the compiler's fixed sentences
DIRECTOR_WORDS_HARD = (300, 650)
DIRECTOR_MAX_NEGATIONS = 3  # in a detailed_description: a negated noun tends to be rendered

# The plan's rules (lint.plan_rules), stated in the plan prompt with these same numbers.
DIRECTOR_AIRLOCK = (2.0, 1.5)  # the opening beat of every chunk after the first: asked, least accepted (s)
DIRECTOR_SETTLE = (2.0, 1.5)  # the closing beat of every chunk: asked, least accepted (s)
# Longest airlock or settle accepted. With none, a 60 s take of five events came back as five 12.2 s
# clips whose settles ran up to 6.2 s: the lengths were the budget divided by the count, padded with stillness.
DIRECTOR_QUIET_MAX = 3.0
DIRECTOR_LOOK_WORDS = 15  # a lasting change to a member's look, restated in every later clip
DIRECTOR_HANDOFF_MIN = 4.0  # s: reach, grip and release all visible
DIRECTOR_BEAT_MIN = 1.0  # s
# The plan chooses how many clips and how long each; the longest is the render settings' duration.
# 6 s holds a 2 s airlock, a main beat and a 2 s settle.
DIRECTOR_CLIP_MIN = 6.0
DIRECTOR_TAKE_SLACK = 0.1  # the take's length is a ceiling; the clips may run over it by this fraction
# The main beat starts mid-chunk, because a chunk's last beat gets squeezed; it may start right after a
# 2 s airlock (0.16 of a 12.25 s chunk).
DIRECTOR_MAIN_BEAT_WINDOW = (0.15, 0.65)
DIRECTOR_MAX_LIVING = 3  # people and animals on screen per chunk (the official shot-table rule)
DIRECTOR_WORDS_PER_SECOND = 2.5  # dialogue
DIRECTOR_DIALOGUE_TAIL = 2.0  # s of silence after a chunk's last line
# A closing places every member and object on screen, so three people and a prop need room.
DIRECTOR_CLOSING_WORDS = 75

# The order traits are said in: a person's hair, face, clothes; an animal's coat and markings before
# its face; a place's landmarks before its furniture and surfaces.
DIRECTOR_TRAIT_ORDER = (
    "hair",
    "coat",
    "markings",
    "face",
    "top",
    "bottom",
    "footwear",
    "accessory",
    "tail",
    "ears",
    "component",
    "material",
    "landmark",
    "furniture",
    "surface",
)
DIRECTOR_SHORT_TRAITS = {"person": 4, "animal": 4, "object": 2, "environment": 4}  # restated at first mention
DIRECTOR_RETAIN_TRAITS = {"person": 6, "animal": 5, "object": 4, "environment": 7}  # in retention_analysis

# The script supervisor (`tools/ruling.py`). Not DIRECTOR_: in "shadow" the word-list checks still decide
# and each ruling that disagrees is only recorded, as a "ruling" event; "on" lets the rulings decide, and
# their RULING_ words join the fingerprint.
RULINGS = "shadow"
RULING_TOKENS = 1024

# The idea editor's numbers (`tools/refine_idea.py`). IDEA_, not DIRECTOR_: they rewrite ideas and
# re-plan nothing.
IDEA_TOKENS = 2048
# Warm enough that asking again gives another version; every round samples on its own seed.
IDEA_TEMPERATURE = 0.5
IDEA_BEAT_WORDS = (70, 110)  # asked, accepted: the hand-written target's beats ran 25 to 65 words
IDEA_PLACE_WORDS = (15, 140)  # accepted; the prompt asks 35 to 80
