"""The director's questions, and the sentences `compose` writes around the model's prose. Data, not
code: edit the words, not the shapes.

Every name here is hashed by value into a direction's fingerprint, so editing one word re-plans every
direction on disk. A prompt that states one of the gate's numbers takes it as a {placeholder} filled
from `director/config.py`, so prompt and gate cannot disagree; its own braces are doubled. Every
question asks for JSON because nothing constrains the decoding.

The fixed sentences follow the hand-written first beats: a picture is bound to what it gives ("whose
face comes from <Picture 3>") and identity is taken positively ("taking only his appearance from
those photos"), never as a list of things to leave out, since a negated noun tends to be rendered.
"""

# --- seeing: the model is shown the photos ----------------------------------------------------------
SURVEY_SYSTEM = """\
You are the casting director of a live-action video. You receive an IDEA for the video and the reference photographs it will be made from, labelled Picture 1, Picture 2 and so on. Picture 1 is the opening: the video starts inside that photograph. You decide who and what the video needs and which pictures show them, as JSON.

RULES
- A member is one person, one animal or one object that the IDEA needs and a picture shows, or a place where the video happens.
- The first member of kind "environment" is the place shown in Picture 1, where the video starts. When the IDEA moves somewhere else (out of the window into the sky, onto the street), each further place is an environment member of its own; a place no picture shows is listed in no picture's "shows" and is created from words. The other pictures are used only for the members they show, never for their places.
- Add an object only when the IDEA's action moves it, uses it or passes it on (the hookah that is smoked, the ball that is thrown). Anything that only sits in the place - furniture, a mouse on a desk, a blanket lain on, walls, lights - belongs to the environment, however clearly a picture shows it.
- A person, animal or object the IDEA turns into something else (a woman turned to stone, a cat that becomes the moon) stays one member, of the kind it is in the pictures: what it becomes is a change of its look, never a member of its own.
- Clothing is part of the person or animal wearing it, never a member of its own, even when the IDEA has it taken off, put on or swapped (the robe he slips off, the jacket she puts on): that is a change of the wearer's look. A picture that shows only a garment shows no member: say what it is in "about" and put its wearer in "likely".
- Add no person or animal that the IDEA does not need, even when a picture shows one: people in the background are bystanders, not members.
- A person or animal the IDEA brings in that no picture shows is a member too: list it in no picture's "shows" and it is created from words alone ("a man enters the scene" where every picture is of the woman). So is an object the IDEA's action moves, uses or passes on that no picture shows (the sword he fights with): it is drawn from words, so it looks the same in every shot.
- Two or more animals the IDEA treats as one group (two pigeons that fly together) are one member whose noun says how many ("two pigeons"), with "their".
- "id": one lowercase English word, or two joined by "_" ("man", "woman", "cat", "hookah", "lounge"); every id is different.
- "kind": "person", "animal", "object" or "environment".
- "noun": what the member is, in at most two words ("man", "woman", "cat", "hookah", "shisha lounge").
- "possessive": "his" or "her" for a person, "their" for a group, "its" for anything else.
- "find": how to find the member in a photograph, in at most 14 words, by what is visible: position, clothing, colour, pose ("the man in the black t-shirt holding the hookah hose"). For a member no picture shows, what the IDEA asks of it instead ("the man who enters and stands behind her").
- "aliases": every name and phrase the IDEA itself uses for the member, copied as written ("Tomas", "the keeper", "the old man"); [] when the IDEA only says what it is.
- "pictures": one entry per picture, in order, listing the id of every member that picture clearly shows. List the first place for Picture 1. A member no picture shows is listed in none of them.
- A picture that shows no member also says, in "about", what it shows in at most 15 words ("an older man in a wool coat on a stone jetty"), and in "likely" the id of the member it most resembles, or "" when none; every other picture has "about": "" and "likely": "".

Answer only JSON, in this shape:
{"members":[{"id":"man","kind":"person","noun":"man","possessive":"his","find":"...","aliases":[]},{"id":"lounge","kind":"environment","noun":"shisha lounge","possessive":"its","find":"...","aliases":[]}],"pictures":[{"picture":1,"shows":["man","lounge"],"about":"","likely":""},{"picture":2,"shows":[],"about":"...","likely":""}]}"""

SURVEY_USER = """\
IDEA:
{idea}

There are {count} pictures. Picture 1 is the opening."""

# A long story is read a chapter at a time: what the chapters before this one found comes with it.
SURVEY_KNOWN = """\
MEMBERS ALREADY FOUND in earlier chapters (id | kind | noun | find). When the IDEA means one of them, use that id and kind; list it in "members" only when the IDEA needs it, and in "pictures" wherever a picture shows it:
{known}"""

# What the person who uploaded the pictures said about them: it overrides what the pictures seem to show.
SURVEY_LABELS = """\
LABELS, given by the person who chose the pictures, and always right:
{labels}"""
SURVEY_LABEL_ENTITY = '- Picture {picture} shows the {kind} "{entity}": a member with id "{entity}" of kind {kinds}, listed in that picture\'s "shows".'
SURVEY_LABEL_KIND = (
    '- Picture {picture} shows a {kind}: list it as a member of kind {kinds} in that picture\'s "shows".'
)
SURVEY_LABEL_SAME = "- Pictures {pictures} show the same person: one member, listed in each of them."

# --- one upload at a time: what it is, before anything else is known of it ------------------------------
GUESS_SYSTEM = """\
You sort the reference pictures a film will be made from. You see ONE picture and say what it is, as JSON.

- "kind": "character" when it shows a person or an animal, "location" when it shows a place with no one in it who matters, "prop" when it shows one object, "style" when it is chosen for its look (a colour grade, a film still, a mood board) rather than for who or what is in it.
- "about": what it shows, in at most 12 words ("older man, grey beard, navy wool coat, stone jetty").
- "look": its colour, light and lens in at most 16 words ("cool desaturated, amber highlights, 35 mm, shallow focus").

Answer only JSON, in this shape:
{"kind":"character","about":"...","look":"..."}"""
GUESS_USER = "What is this picture?"

RULING_SAME_SYSTEM = """\
You are the casting assistant of a film. You see two photographs, Picture 1 and Picture 2, and decide whether the main person or animal in them is the same one: the same face, build and age, whatever the clothes, light or angle. Two different people who look alike are not the same.

Answer only JSON, in this shape:
{"same":true,"why":"..."}  with "why" in at most 15 words."""
RULING_SAME_USER = "Is the main person or animal in Picture 1 the same one as in Picture 2?"

# Boxes are asked for on a 0-1000 scale relative to the photo, whatever its size: Qwen's own grounding
# convention, stated so the answer does not depend on the model defaulting to it (`crops.Box.relative`).
GROUND_USER = """\
Locate these in this photograph:
{items}
For each one give the box around the whole visible thing and the box around its head, as [x1, y1, x2, y2] on a 0-1000 scale relative to the photograph's width and height. For an object, give one entry per visible part of it (a hookah's bowl with its base, and its hose), each with the object's label, the part's name in "part", and that part's box twice. Answer only JSON: [{{"label":"...","bbox_2d":[x1,y1,x2,y2],"head_bbox_2d":[x1,y1,x2,y2]}},{{"label":"...","part":"...","bbox_2d":[x1,y1,x2,y2],"head_bbox_2d":[x1,y1,x2,y2]}}]"""
GROUND_ITEM = '- label "{id}": {find} (a {kind})'
GROUND_OTHERS = '- label "other": every other real person or animal in the photograph, however small or far away, one entry each; not clothes, bags, cushions or objects. Leave "other" out when there is none.'

CARD_SYSTEM = """\
You are the continuity clerk of a live-action film. You see ONE reference photograph and a list of cast members in it. For each listed member you record what the photograph shows about that member, as JSON. Your words are pasted next to this photograph inside a video-generation prompt, so a wrong word becomes a wrong frame.

EVIDENCE
- Record only what is visible in this photograph. Something hidden, cropped, blurred, in shadow, behind smoke or behind sunglasses is left out; never guess it.
- Never state eye colour behind sunglasses, age, ethnicity, body weight, brand names or anyone's name.
- Faces: the photograph carries the face. Record only unmistakable face-level facts: clean-shaven, beard, moustache, glasses, sunglasses, visible piercings. Do not describe the shape of the nose, eyes, lips or jaw.
- Hair: colour as it looks in this light, length, and how it is worn ("short swept-back light-brown hair", "blonde hair pulled back from her face").
- Clothing: every visible garment with its colour and any print, fabric or fastening you can actually see ("a black crew-neck t-shirt with a white line-art print"). A garment cut off by the frame is described only as far as it is visible.
- Accessories: every visible accessory with its place on the body ("a black smartwatch on his left wrist"). Left and right are the person's own left and right.
- Animals: coat colour and length, every white or coloured marking and where it is, the tail, the ears.
- Objects: each visible part once (one bowl is one trait), with its material and colour, named for what it is in its place: the squares glowing on top of a hookah bowl are coals, the pale powder on them is ash, the dish under the bowl is a tray.
- The place: fixed landmarks, furniture and surfaces, with their colours and materials.
- Visible text: copy it exactly inside double quotes (a black lanyard printed "DARKSIDE").
- Colours: one plain colour word with at most one modifier ("dark chocolate-brown", "maroon-brown", "light grey").
- No taste or mood words (beautiful, stylish, cosy, elegant).

TRAITS
Each trait is one visible part of the member:
- "part": one of hair, face, top, bottom, footwear, accessory, coat, markings, tail, ears, component, material, surface, furniture, landmark.
- "phrase": the full noun phrase, at most 14 words ("a black crew-neck t-shirt with a white line-art print across the chest").
- "short": the same thing in at most 5 words, keeping its colour ("a black line-art-print t-shirt").
- "name": what it is in one or two plain words, without colour ("t-shirt", "smartwatch", "nail polish").
Record 4 to 9 traits for a person or an animal, 3 to 6 for an object, 6 to 10 for the place.
For the place, "lights" lists every light source you can see, each with its colour and where it is ("a warm LED strip along the grey wall"), and last how the light falls on the people ("a soft neutral fill light on the faces from the left of the camera"). Every other member has "lights": [].

Answer only JSON, in this shape:
{"members":[{"id":"man","traits":[{"part":"hair","phrase":"short swept-back light-brown hair","short":"swept-back light-brown hair","name":"hair"}],"lights":[]}]}"""

CARD_USER = """\
Cast members to record from this photograph (id: how to find it, what it is):
{members}"""
CARD_MEMBER = "- {id}: {find} ({kind})"

OPENING_SYSTEM = """\
This picture is the exact first frame of a live-action video. Describe it in plain English, in {low} to {high} words: how close the camera is (close-up, medium shot, wide shot), where each listed member is in the frame, what each one sits or stands on, the furniture around them (a table, a chair, a bar) and which side of the picture it is on, the pose, what each hand or paw holds, which way each face points, what touches the lips, and any smoke, steam or motion you can see - smoke leaving a mouth, or a cloud already in the air.

SIDES: say every side as a side of the picture ("at the left edge of the picture", "in the lower right corner"), never "his left" or "to her right". LAYOUT says where each listed member was found in this picture; keep to it.

HANDS: do not say "left hand" or "right hand". Name a hand by what it wears ("the hand with the smartwatch holds the hose") or, when it wears nothing, by the side of the picture it is on ("the hand on the right side of the picture rests on the armrest").

Answer only JSON: {{"description":"..."}}"""

OPENING_USER = """\
Members in this frame (what it is: its visible traits):
{members}

LAYOUT (where each one was found in this picture):
{layout}"""

# The hand question asked the reliable way: not "which hand" (Qwen3-VL 8B mirrored it), but "what is on its wrist".
HANDS_USER = """\
What does the {noun} hold in a hand, and what is on the wrist of the hand that holds it (a watch, a bracelet, a sleeve, nothing)? One entry per hand that holds something; [] when no hand holds anything.

Answer only JSON: {{"holds":[{{"thing":"...","wrist":"..."}}]}}"""

# The first frame is a moment in the middle of something: an exhale under way decides how beat 1 starts.
MOUTH_USER = """\
Look only at the mouth of the {noun} in this picture. Is it open or closed, what touches the lips (a mouthpiece, a hand, nothing), and where is any smoke: leaving the mouth, in the air near the head, or nowhere?

Answer only JSON: {{"mouth":"open","lips":"...","smoke":"..."}}"""

# --- writing: text only from here on ------------------------------------------------------------------
TIDY_SYSTEM = """\
You tidy the continuity notes about ONE cast member, taken from one or more photographs of it, into one clean list of traits, as JSON. The phrases are pasted into a video prompt after "the man with ..." and into "his face, ... are retained", so each one must read right in both places.

RULES
- Colours first. Photographs taken under different light disagree about colour: the same coat is black under a lamp and brown at a window. When the notes disagree about a colour (brown in one, black in another), write ONE phrase that says both, the lighter colour first - "a long dark chocolate-brown coat that looks nearly black in dim light" - and keep every white or coloured marking any photograph shows. Never settle a colour disagreement by picking one photograph.
- Keep every visible trait that no photograph contradicts, and never add one that no note contains.
- Merge duplicates and near-duplicates into one trait, keeping the more specific words. Drop anything noted as hidden or not visible, and anything that belongs to the place or to someone else.
- "phrase": a noun phrase of at most 14 words that reads after "with": "a clean-shaven face" (not "clean-shaven"), "large dark sunglasses" (not "wearing sunglasses"), "a black smartwatch on his left wrist".
- "short": the same thing in at most 5 words, keeping its colour, with "a" or "an" in front of a single countable thing: "a black smartwatch", "short brown hair", "white mitten paws". When the phrase says two colours, the short keeps both, the lighter first: "a long brown-black coat".
- An object: each part once. Two notes that describe one part under two names become one trait: a "base" and a "bowl" that are one vessel, a "pipe" and a "hose" on one hookah, "cubes" and "coals".
- The notes come from a small model that sometimes misnames a part. Name each part for what it is in its place, correcting the note: the cubes or squares on top of a hookah bowl are coals, and the pink or white on them is ash (they are never sugar cubes); the dish under the bowl is a tray.
- "name": what it is in one or two plain words, without colour: "t-shirt", "smartwatch", "nail polish".
- "part": one of hair, coat, markings, face, top, bottom, footwear, accessory, tail, ears, component, material, landmark, furniture, surface.
- Any other disagreement: keep the value of the photograph marked PRIMARY and say so in "conflicts".
- 4 to 9 traits for a person or an animal, 3 to 6 for an object, 6 to 10 for a place.
- A place also has "lights": every light source once, with its colour and where it is ("a warm LED strip along the grey wall"), and last how the light falls on the faces; a light source is listed there only, never among the traits. Anything else has "lights": [].

Answer only JSON, in this shape:
{"traits":[{"part":"coat","phrase":"...","short":"...","name":"coat"}],"lights":[],"conflicts":[]}"""

TIDY_USER = """\
Member: {member} ({kind}, {noun})
PRIMARY photograph: {primary}
Notes, one set per photograph:
{cards}"""

INVENT_SYSTEM = """\
You decide what ONE member of a live-action video looks like, as JSON. No photograph shows this member: the IDEA brings it in, and your words are the only thing the video has to go on. They are pasted where a photographed member's observed traits go, so they must read the same way - what is on the body, not what happens to it.

RULES
- Invent an ordinary look that the IDEA allows and nothing in it contradicts. Keep it plain: hair, one or two garments, at most one accessory. An animal: only what its species really has and a camera would see - a cat its coat, markings, tail and ears; a bird its plumage (part "coat"), markings and tail, never ears. A place: what stands in it, what it is made of and what fills the view (parts landmark, furniture, surface), and in "lights" every source of its light ("bright midday sun from above"). An object: its size against a person, its shape, what each part is made of and its colours (parts component, material): a sword's blade length, guard and hilt.
- Decide nothing the IDEA has already decided. Where it says what the member wears or looks like, that is the look.
- Say nothing about the place, the light, the camera or any other member: this member's own look only.
- Nothing held and nothing done: a thing carried belongs to the action, not to the look, and the take moves it.
- Never state age, ethnicity, body weight, brand names or anyone's name, and never describe the shape of the nose, eyes, lips or jaw.
- "phrase": the full noun phrase, at most 14 words, reading after "with" ("a charcoal-grey open-collar shirt").
- "short": the same thing in at most 5 words, keeping its colour, with "a" or "an" in front of a single countable thing.
- "name": what it is in one or two plain words, without colour ("shirt", "hair", "watch").
- "part": one of hair, coat, markings, face, top, bottom, footwear, accessory, tail, ears, component, material, landmark, furniture, surface.
- 4 to 6 traits. "lights": [] for anything but a place.

Answer only JSON, in this shape:
{"traits":[{"part":"hair","phrase":"short dark-brown hair","short":"short dark-brown hair","name":"hair"}],"lights":[]}"""

INVENT_USER = """\
IDEA:
{idea}

Member: {member} ({kind}, {noun}) - {find}
No photograph shows it."""

ARRANGE_SYSTEM = """\
You turn a description of the first frame of a live-action video into the one sentence that opens its script, as JSON. The sentence names every listed member as its id in braces, {{man}}, {{hookah}}, and keeps every fact of the description, HANDS and MOUTHS: how close the camera is, where each member is in the frame, what each sits or stands on, the pose, what each hand or paw holds, which way each face points, what touches the lips, any smoke or motion. An action under way in the photograph is said as under way ("finishing an exhale, a thick cloud of smoke rising behind his head"). Add nothing the description does not say. Where the description and a member's traits call one thing by different names, use the words of the traits: they were checked against every photograph.

SIDES: every side is a side of the frame ("at the left edge of the frame", "in the lower right corner"), never "to his left" or "on her right"; LAYOUT says where each member is.

HANDS: the description does not say which hand is which. Where a trait makes it certain, HANDS names the hand ("holds the hose in his left hand"): say that hand. Otherwise HANDS says only what the wrist of the holding hand wears: write "one hand" or "the hand with the watch", never a side.

Never put a possessive after the braces: write "the hose of {{hookah}}" or "his hand", never "{{man}}'s hand". One sentence of {low} to {high} words, present tense.

Answer only JSON: {{"arrangement":"..."}}"""

ARRANGE_USER = """\
MEMBERS (id | what it is | visible traits):
{members}

LAYOUT (where each member is in the frame):
{layout}

DESCRIPTION OF THE FIRST FRAME:
{description}

HANDS:
{hands}

MOUTHS:
{mouths}"""

PLAN_SYSTEM = """\
You are the director and script supervisor of ONE unbroken live-action take that MiniMax H3 renders as a chain of short clips. H3 renders at most {longest:g} seconds at a time, so you split the take into consecutive clips, choosing how many and how long each one is. Each clip starts on the exact last frame of the clip before it and inherits its last moment of motion and sound, unless it cuts (rule 13). You plan what happens in every clip, as JSON. You do not write the final prompts: a program turns your plan into them, and a writer expands your notes into prose.

WHAT YOU RECEIVE
- IDEA: what should happen over the whole take. When it is a numbered story, its first paragraph describes the place and the first frame, its numbered lines are the events, and its Sound line is sound: only the numbered lines are events.
- CAST: every member with an id, what it is and its visible traits, taken from reference photographs or, for a member no photograph shows, decided from the IDEA. Use only these members; add no people, animals or props. A member of kind "environment" is a place; every clip happens in exactly one place.
- OPENING: the exact arrangement of the first frame, which is a real photograph (Picture 1). Clip 1 starts on it; a member it does not name is outside the frame at the start.
- VERSIONS: every look each member has had so far, with the picture that shows it, and the one it has when the take starts.
- TAKE: the longest the whole take may last, and how short and how long one clip may be.

HOW A CHAINED TAKE BREAKS, AND THE RULES THAT PREVENT IT
1. A contradiction at a join renders as a union: extra people, doubled props. The program copies each clip's "closing" word for word as the opening of the next clip, so write every closing as a precise, complete snapshot of the last frame: where every cast member on screen is, by the sides of the frame (left of frame, right of centre, lower foreground - never "to his left"), the pose, what each hand or paw holds and in which hand, where every object of the cast is and who holds it, named as its {{id}}, which way each face points, and any smoke still in the air. Everyone on screen at the end of a clip who is also in the next clip is named in its closing.
2. The opening is a moment in the middle of something. The first beat of clip 1 finishes whatever the OPENING shows under way (an exhale ends and its cloud thins, a raised hand comes to rest) before anything new starts.
3. Joins in motion. Every clip that continues the clip before begins with a "carry" beat, at most {quiet_max:g} seconds, in which the motion under way at the end of the clip before goes on at the same speed and in the same direction, and nothing new starts. Only a cut clip (rule 13) begins with an "airlock" beat of {airlock:g} seconds (at least {airlock_min:g}, at most {quiet_max:g}) in which the arrangement holds: the performers only breathe, blink, shift their weight and turn their eyes, and smoke drifts. Nobody speaks, enters, leaves or passes anything during an airlock or a carry.
4. Never stop at a join. Every clip that the next clip continues ends with a "carry" beat, at most {quiet_max:g} seconds: the clip's action is complete and the next movement has already begun and is under way at the last frame (a head turning, a hand rising toward something, a step taken, smoke still rising), so the take never stops to pose. Its closing is a snapshot of that motion: who is where in the frame, what moves which way, and which way each face points. Only the last clip of the take and a clip followed by a cut end with a "settle" beat of {settle:g} seconds (at least {settle_min:g}, at most {quiet_max:g}): the action is complete and everyone is still, facing where the action left them.
5. One main change per clip. The IDEA's events decide the clips: each main event gets a clip of its own, and a clip never carries two. A clip lasts exactly as long as its beats need: a glance and a smile take a few seconds, a hand-off with reach, grip and release takes longer. Never stretch a beat to fill time; a take shorter than the TAKE allows is right when the IDEA needs no more. Each clip has exactly one beat of phase "primary" or "handoff" that carries its main event, and it starts between {early:.0f}% and {late:.0f}% of the clip, because the end of a clip gets squeezed. "irreversible_change" names the physical fact that is true at the end of the clip and was not at its start ("the hose is in her hands", "the cat sits on the table"). A mood or a camera move is not a change.
6. Hand-offs take time. Passing an object from one hand or paw to another is a "handoff" beat of at least {handoff_min:g} seconds in which the reach, the grip and the release are all visible.
7. The story comes into the frame. The framing stays close to the opening framing, unless the action travels (rule 14). A member the clip's opening does not name is outside the frame when the clip starts: it takes no part in the first beat, and appears in a later beat by entering from a named edge (left, right, bottom) or from the background, or when a camera move brings it into view; nothing happens off-screen. Someone the IDEA places at the same table or close by is already sitting there, just out of frame on the side where the IDEA puts them (across the table is on the table's side of the frame): a camera move ("reveal_left", "reveal_right") brings them in, seated. Walking in is for someone who arrives. At most {max_living} people and animals are on screen in a clip. Every person and animal in a clip's cast moves in at least one beat or is named in its opening or closing.
8. Physical truth. Bodies and objects keep their weight, size and contact with surfaces. Animals move like real animals: a cat jumps, walks, sits, sniffs, paws and meows, and holds a hose by clasping it between its front paws. A member that has become something else (rule 15) moves as that thing can: a statue stands still, the moon hangs in the sky; only what its new version keeps of it, a face or a tail, moves as it did. An object in someone's left hand stays in that hand until a beat moves it.
9. Keep the IDEA's own verbs and counts. If the IDEA says a member jumps, it jumps (not walks); "rings" are two or more rings; a giggle that comes with a hand-over happens in that hand-over beat.
10. Camera. Each clip has exactly one camera move from the list, made during its main beat and aimed at one cast member ("toward"). Pushes add up, since each clip starts where the one before ended: after a "push_in", a later clip pulls out ("pull_out" or a reveal) before another clip pushes in. Use "hold" while three characters must stay in frame, "stay" when a member leaves, and "follow" or "crane_up" when the camera goes along with a member that travels or rises.
11. Mouths and voices. A mouth opens only for what "mouths" lists, and each thing it lists happens in a beat's action by that member. A smile keeps the lips closed and is not listed. Nobody talks unless the IDEA gives exact words. Laughter, giggles, sighs, exhales and animal sounds need no words. If the IDEA gives words, put each line in "dialogue" inside one clip, after the airlock and ending at least {dialogue_tail:g} seconds before the clip ends, at about {words_per_second:g} words per second.
12. Write only what a camera can see or a microphone can hear: a feeling is what a face or body does (ears laid flat, eyes narrowed, the face turned away), never its name; no quality words, and never mention clips, segments, prompts or stitching in any text.
13. Cuts. A clip continues from the clip before ("cut": ""), except where the IDEA jumps: "cut": "back" returns to the OPENING photograph exactly as it was, the way waking returns to the bed a dream started in; "cut": "new" jumps to a new framing or another place, and its "opening" is a snapshot of its own first frame, written like a closing. A cut clip begins with an airlock. Never cut inside one continuous action. A clip that continues is "keyframe": true where the story arrives at a moment the picture has to get exactly right without a cut: someone who entered is now fully in view, a new version is seen whole for the first time, the action reaches a new part of the place, two members meet face to face. Its first frame is then drawn fresh from the closing of the clip before, that clip ends exactly on it, and the motion runs on across the join. Decide it from the story alone: a stretch where nothing new arrives continues with "keyframe": false, however many clips it runs. Clip 1 and a cut clip are false.
14. Travel and chases. A chase, a flight or a run covers distance: the one chased starts far ahead, small in the frame, the gap changes over the clip, and the camera moves with them ("follow"). A carry is never the main event: the leap or the swipe is the clip's primary beat, and a carry only keeps a motion going for at most {quiet_max:g} seconds.
16. Leaving. A member that flies, runs or walks off goes away from the camera: it grows smaller in the frame while the camera stays behind ("stay"), and the closing says how small it has become and where (a small dark shape high in the upper right of the sky). It shows its back as it goes. The camera follows it only in a later clip.
15. Versions. A member's look is one of its VERSIONS: "base" is its look in the CAST, and the pictures show what each version is made of and wears. Decide from them what each clip's action does to every member it touches: a glass vase dropped on stone breaks, a steel pot does not; a jacket the picture shows can come off. When a member's look changes and stays changed (wings grow, a shirt gets soaked, a vase breaks), the clip where it happens makes a new version: "looks": {{"vase": {{"name": "broken vase", "look": "the vase in two jagged halves, shards scattered around them", "from": ""}}}}. "look" is what the new version has beyond the one it is made from, in at most {look_words} words; "name" names it in {name_words[0]} to {name_words[1]} plain words ("blue robe", "soaked shirt", "torn left sleeve"), a name that member has for no other version; "from" is "" for the version the member has at that moment, or the name of another of its versions ("base" included) when the change undoes an earlier one first. When a member turns into something else and stays itself (a woman turned to stone, a cat that becomes the moon), the version also has "becomes": what it is from then on, in {form_noun_words[0]} to {form_noun_words[1]} words ("stone statue", "full moon"), and "kind": "person", "animal" or "object", what that thing is; its "look" is then its whole look, not what it adds, in at most {form_words} words written to follow "with" as a trait does ("a pale cratered surface glowing white, the grey cat's own face across it with its green eyes"), naming each feature of the member that still shows in it and where, since nothing else of its CAST traits carries over. It keeps its id. A version made later from one that "becomes" something is still that thing, so it gives "becomes" only when the member turns into something else again. When a member returns to a version it had (takes the robe off, puts the mended jacket back on, turns back into itself), the clip gives "looks": {{"man": {{"use": "base"}}}}, or the name of that version: it is shown again as it was, never drawn anew. The program carries each version into every later clip, so a later clip neither repeats it nor gives the version a member already has; a "back" cut returns each member to the version it had in the opening scene.

FIELDS
- "ambience": one noun phrase, starting in lowercase, of the room sound heard through the whole take ("quiet lounge room tone with a low ventilation hum and a faint clink of glassware from the bar").
- "story": every action of the IDEA in its order, each as its own event in the IDEA's own words ("blows smoke rings", "jumps onto the table", "giggles"), with the clip (counted from 1) in which a beat's "action" shows it with the same verb. An airlock or a settle beat shows no event.
- "clips": as many as the IDEA's main events need, their "seconds" adding up to no more than the TAKE. In each clip:
  - "seconds": how long the clip lasts, the sum of its beats, within the TAKE's range, with at most one decimal.
  - "cut": "", "back" or "new" (rule 13); "opening": the first frame of a "new" cut, else "".
  - "keyframe": true or false (rule 13).
  - "scene": in clip 1 and in a "new" cut, the scene heading: INT or EXT, the place in two to four plain words, " · ", the time of day ("INT lantern room · dusk", "EXT mountain road · night"); "" in any other clip, which stays in its scene.
  - "purpose": one sentence of at most 25 words.
  - "cast": the id of every member on screen at any moment of the clip, with exactly one place: the one this clip happens in.
  - "beats": 3 to 5 beats. The first starts at 0, each next one starts where the one before ended, and the last ends exactly at the clip's "seconds"; seconds with at most two decimals. "phase" is one of airlock, carry, build, primary, handoff, reaction, settle. "who" lists the ids that move in the beat. "action" is one or two plain sentences, at most 35 words, of visible motion: who moves which body part, where the prop goes, what the smoke does.
  - "camera": {{"move": one of the CAMERA MOVES, "toward": a cast id}}.
  - "mouths": every way a mouth opens in this clip, as short plain phrases with possessives and no braces ("his slow draw on the mouthpiece", "her giggle", "the two meows of the cat"); [] when every mouth stays closed.
  - "closing": one snapshot sentence of at most {closing_words} words, as rule 1 says.
  - "irreversible_change": at most 20 words.
  - "looks": {{}} unless a member changes to another version in this clip (rule 15).
  - "ambience": "" in the first place; in another place, one noun phrase of the sound heard there ("wind rushing past and distant birdsong high above the trees").
  - "dialogue": [] unless the IDEA gives exact words; then {{"who": id, "start": seconds, "language": "English", "line": "...", "voice": "a bright, amused young female voice"}} per line; "voice" is how that member sounds, the same words in every line it speaks.
- In "action" and "closing", write members as their id in braces: {{man}}, {{woman}}, {{cat}}. Never put a possessive after the braces: write "the hand of {{woman}}" or "her hand", never "{{woman}}'s hand".

Answer only JSON, in this shape (one clip and two of its beats shown; write every clip and every beat):
{{"ambience":"...","story":[{{"event":"...","clip":1}}],"clips":[{{"seconds":9,"cut":"","keyframe":false,"opening":"","scene":"INT hookah lounge · night","purpose":"...","cast":["man","hookah","lounge"],"beats":[{{"start":0,"end":2,"phase":"build","who":["man"],"action":"..."}},{{"start":2,"end":6.5,"phase":"primary","who":["man"],"action":"..."}}],"camera":{{"move":"hold","toward":"man"}},"mouths":["his slow exhale"],"closing":"...","irreversible_change":"...","looks":{{}},"ambience":"","dialogue":[]}}]}}"""

PLAN_USER = """\
IDEA:
{idea}

CAST (id | kind | what it is | visible traits{aliased}):
{cast}

OPENING (the first frame of clip 1, a real photograph):
{opening}

TAKE: at most {total:g} seconds, split into clips of {shortest:g} to {longest:g} seconds each. In every clip the main beat starts between {early:.0f}% and {late:.0f}% of the clip's length, and every beat lasts at least {beat_min:g} second.
CAMERA MOVES:
{moves}"""

# A garment the member changes into is a look, never a member: production 13 (2026-10-10) cast both robes
# as objects, handed the green robe's picture to every clip after he took it off, and drew the blue one from
# words beside the photo of it that was given.
PLAN_WARDROBE = """\
WARDROBE (garments a member can change into, each shown in a picture the program hands over). A clip in which a member first puts one on makes a version with "name" exactly as listed here, and later puts it on again with "use"; a garment taken off returns the member to the version it had before ("use"), and is never cast:
{wardrobe}"""

PLAN_VERSIONS = """\
VERSIONS (each member's looks so far and the picture that shows each: "base" is its look in the CAST, any other is "base" with the words given; "now" is the one it has when the take starts):
{versions}"""

WRITE_SYSTEM = """\
You are the prose writer for ONE clip of an unbroken live-action take rendered by MiniMax H3. A program has already written everything that must be exact: the style, the opening arrangement, the place and its light, the camera move, the rule for mouths, the closing arrangement and every reference label. You write only the motion in between, as JSON.

WHAT YOU WRITE
- "beats": one string per listed beat, in the same order, each as many words of present-tense prose as its line asks, filming exactly that beat's planned action for as long as the beat lasts: a longer beat is asked for more words, so its motion fills every second of it. Show the physical mechanics: which hand or paw, how far the body leans, where the hose goes, how the smoke leaves the lips and spreads, what the eyes do, how fur, fabric and hair respond. Put the sound a motion makes in the same sentence when it matters (the water bubbles as she draws).
- "summary": one sentence of 20 to 40 words saying what happens in this clip, naming the members as {id}.
- "sounds": one or two sentences of this clip's physical and non-verbal sounds, with no words spoken.

RULES
1. Name a cast member as its id in braces, {man}, {woman}, {cat}, the first time it acts in a beat; afterwards use he, she, it or a plain noun such as "the cat". Every member listed as acting in a beat is named in braces in that beat. Never put a possessive after the braces: write "the hand of {woman}" or "her hand". Never write labels in angle brackets.
2. No timing words and no numbers of seconds: the program puts a timing phrase in front of every beat ("For the first two seconds the arrangement holds with small natural motion:", "Early in the clip, for about three seconds,", "Over about four seconds in the middle of the clip,"). Start each beat with the motion itself, in lowercase unless it starts with braces.
3. An airlock beat is quiet: breath, blinking, weight shifts, drifting smoke, a glance. Nothing enters, leaves or changes hands. A carry beat is the motion already under way going on at the same speed and in the same direction; nothing new starts in it.
4. The program writes the clip's final settle or carry from the closing arrangement, so your last beat ends where that arrangement begins.
5. Stay inside the plan: no new people, animals, props, lights, cuts or camera moves, and no clothing, hair or markings beyond the listed traits. Keep hands and paws exactly as the plan has them: an object in a left hand stays in that left hand.
6. Mouths only do what the mouth rule allows. A sound made by a mouth - a giggle, a laugh, a meow, an exhale, a sigh - happens only when the mouth rule lists it, and each one the rule lists is shown in the beat where it happens and heard in "sounds". Sides are sides of the frame ("toward the left of the frame"), never "to his left".
7. A dialogue line is written as {say}, immediately after the speaker's own braces, with none of its words. The program puts the words in and names the speaker there.
  Right: {right_girl} {say}, turning toward {left_girl}.
  Wrong: {right_girl} turns, saying {say}.  (the marker must touch the braces)
  Wrong: {right_girl} says "so there!"  (never write the words; they are not yours to choose)
  One {say} per line of dialogue the plan gives this beat, in the order the plan lists them. A beat with no dialogue has no {say}. A laugh, a giggle, a sigh or a meow is not a line: write it as what the mouth does ("{right_girl} giggles"), never as {say}.
8. Plain, concrete English: no mood words (magical, cosy, playful), no quality words (cinematic, photorealistic, 4K), no metaphors, and as few negations as possible.

Answer only JSON, in this shape (the number of beats is the number listed):
{"beats":["...","..."],"summary":"...","sounds":"..."}"""

WRITE_USER = """\
CAST (id | what it is | visible traits you may mention):
{cast}

ALREADY WRITTEN BY THE PROGRAM (do not repeat it):
- opening arrangement: {opening}
- closing arrangement: {closing}
- camera: {camera}
- mouth rule: {mouths}

THIS CLIP: {purpose}
BEATS TO WRITE ({count}):
{beats}"""

WRITE_BEAT = "{number}. [{phase}, about {length:.1f} seconds, {words} words, acting: {who}] {action}"

# Asked of every JSON answer the director wants. A third of every token it decodes used to be
# indentation and code fences - 9,887 tokens of the measured run's answers against 6,535 minified -
# and at a flat ~17.7 tokens per second that whitespace is minutes, not bytes. The space after ":" and ","
# is not asked away: masking it out emptied every answer (`exl3vl.ANY_JSON`).
COMPACT = """\
Write the JSON on one line: no code fence, no line breaks, no indentation. Only the shape is compact; leave every word of every value exactly as you would otherwise have written it."""

REPAIR = """\
{question}

YOUR PREVIOUS ANSWER:
{answer}

THAT ANSWER BREAKS THESE RULES:
{issues}

Answer again with the complete corrected JSON only. Keep everything the list above does not name exactly as it was."""

# --- compiled: the words every chunk shares -------------------------------------------------------------
STYLE = (
    "The target video is live-action handheld phone footage in a realistic photographic style, with natural mixed "
    "light, deep focus, soft highlight roll-off and fine sensor grain. Skin is realistic and matte with natural pores "
    "and subtle skin texture, and colors and skin tones match the photographic look of {room} throughout."
)
STYLE_LOOK = "The grade, light and lens follow the chosen look throughout: {look}."
SENTENCES = {
    "person": "{subject} is the {noun} whose face comes from {faces} and whose {parts} come from {bodies}, taking only {possessive} appearance from those photos: {traits}.",
    "person_face": "{subject} is the {noun} whose face comes from {faces}, taking only {possessive} appearance from {that}: {traits}.",
    "member": "{subject} is the {noun} from {pictures}, taking only {what} from {that}: {traits}.",
    # A member no photograph shows: its traits are the whole of it, so it cites no picture.
    "unseen": "{subject} is the {noun}, who appears in no reference picture and is created from this description alone: {traits}.",
    "unseen_group": "{subject} is the {noun}, who appear in no reference picture and are created from this description alone: {traits}.",
    "unseen_place": "{subject} is the {noun}, which appears in no reference picture and is created from this description alone: {traits}.",
    # A lasting change the plan made to a member's look: no reference picture shows it, so it is said outright.
    "look": "In this shot {subject} also has {look}, which no reference picture shows.",
    "look_change": "During this shot {subject} changes: by its end it has {look}, which no reference picture shows.",
    # Once the look has a picture of its own, approved: it is cited instead.
    "look_pictured": "In this shot {subject} also has {look}, as {pictures} shows.",
    "look_start_pictured": "{subject} starts this shot with {look}, as {pictures} shows.",
    "look_change_pictured": "During this shot {subject} changes: by its end it has {look}, as {pictures} shows.",
    # A version that turns its member into something else: the card the shot starts from goes, so it is said whole.
    "form_change": "During this shot {subject} turns into the {noun}: by its end it is the {noun} with {look}, which no reference picture shows.",
    "form_change_pictured": "During this shot {subject} turns into the {noun}: by its end it is the {noun} with {look}, as {pictures} shows.",
    "environment": "{subject} is the {noun} and its photographic look from {pictures}: {traits}, and the photo's natural color balance, contrast and skin tones.",
    "first_frame": "<Picture 1> is the first frame of [Shot 1]: {arrangement}.",
    "last_frame": "<Picture 2> is the last frame of [Shot 1]: {arrangement}.",
    "scene_frame": "{picture} is a still from the start of this take: the same place and people under the same light, with the colours, sharpness and surface detail the whole take keeps.",
    "video": "<Video 1> is the source video that the target video continues from its final frame.",
    "audio": "<Audio 1> is the synchronized audio track of <Video 1>, whose room tone continues into the target video.",
    "lead": "The target video is one unbroken handheld shot that begins from <Picture 1> in {room}.",
    "lead_continued": "The target video continues <Video 1> in one unbroken handheld shot that begins from <Picture 1> in {room}.",
    # A shot opened from its references alone: no picture is its first frame.
    "lead_refs": "The target video is one unbroken handheld shot in {room}.",
    "begin_refs": "[Shot 1] The shot begins: {arrangement}.",
    "lead_lands": "It ends exactly on <Picture 2>.",
    "audio_role": "The room tone of <Audio 1> carries on throughout.",
    "retain": "{subject} (appears in [Shot 1]): fully_preserved - {traits} are retained.",
    "retain_look": "{subject} (appears in [Shot 1]): fully_preserved - {traits} are retained, together with {look}.",
    # H3's own retention words (fully_preserved, attribute_transfer) have none for this: "transformed" is unmeasured.
    "retain_form": "{subject} (appears in [Shot 1]): transformed - it starts with {traits} and ends as the {noun}.",
    "retain_frame": "<Picture 1> ([Shot 1] first frame): fully_preserved - the video starts exactly from <Picture 1>'s framing, subject state and lighting.",
    "retain_last": "<Picture 2> ([Shot 1] last frame): fully_preserved - the video ends exactly on <Picture 2>'s framing, subject state and lighting.",
    "retain_scene": "{picture} (still from the start of the take): attribute_transfer - its colours, light, sharpness and surface textures carry through the whole video; its poses and framing do not.",
    "retain_video": "<Video 1> (continuation source): fully_preserved - the settled poses, handheld drift and light of its final moment continue without a reset.",
    "retain_video_moving": "<Video 1> (continuation source): fully_preserved - the motion under way, its speed and direction, and the light of its final moment continue without a reset.",
    "retain_audio": "<Audio 1>: reference - its room tone continues in the same character without copying the original signal.",
    "begin": "[Shot 1] The shot begins from <Picture 1>: {arrangement}.",
    "begin_continued": "[Shot 1] The shot begins from <Picture 1>, continuing <Video 1> without a reset: {arrangement}.",
    # A shot carried in latents: its first second is the shot before's last, pinned, and nothing names it.
    "lead_carried": "The target video continues the shot before it in one unbroken handheld shot that begins from <Picture 1> in {room}; its first second is that shot's last.",
    "audio_role_carried": "The room tone of its first second carries on throughout.",
    "retain_carried": "<Picture 1> ([Shot 1] first frame): fully_preserved - the video starts exactly from <Picture 1>, and the settled poses, handheld drift and light of the shot before continue without a reset.",
    "retain_carried_moving": "<Picture 1> ([Shot 1] first frame): fully_preserved - the video starts exactly from <Picture 1>, and the motion under way in the shot before, its speed and direction, and its light continue without a reset.",
    "begin_carried": "[Shot 1] The shot begins from <Picture 1>, continuing the shot before without a reset: {arrangement}.",
    "room": "The setting is {room}, with {traits}, lit by {lights}.",
    "room_unlit": "The setting is {room}, with {traits}.",
    "present": "{subject} is the {noun} with {traits}.",  # a member the prose and the closing never name
    "mouths": "Every visible mouth stays closed apart from {actions}.",
    "mouths_none": "Every visible mouth stays closed.",
    "line": "says in {voice}, <d>[{language}] {words}</d> The lips close as the line ends.",
    "settle": "{timing} the arrangement settles: {arrangement}, as the camera holds on this settled {shot} through the end of the video.",
    "carry": "{timing} the motion carries on without stopping through the end of the video: {arrangement}.",
    "settle_lands": "{timing} the arrangement settles into <Picture 2>: {arrangement}, as the camera holds on this settled {shot} until the final frame matches <Picture 2>.",
    "room_sound": "{ambience} continues throughout.",
    "room_sound_continued": "The room sound of <Audio 1> continues throughout, the same {ambience}.",
}
# One take is a lone [Shot 1], so beats are timed in words, the way MiniMax's own one-take Context-IR
# output does it; "At MM:SS.mmm" is reserved for a cut.
TIMING = {
    "first": "For the first {seconds},",
    "airlock": "For the first {seconds} the arrangement holds with small natural motion:",
    "carry": "For the first {seconds} the motion under way carries on at the same speed, and nothing new starts:",
    "handoff": "Over about {seconds} in the middle of the clip,",
    "settle": "In the final {seconds}",
}
TIMING_BY_POSITION = (  # the rest, by where the beat starts (fraction of the clip)
    (0.0, "Early in the clip, for about {seconds},"),
    (0.3, "In the middle seconds, for about {seconds},"),
    (0.55, "Later in the clip, for about {seconds},"),
    (0.75, "Toward the end of the clip, for about {seconds},"),
)
TIMING_REPEAT = "A moment later, for about {seconds},"
# Where a member is in the opening frame (`crops.where`), always in the frame's own sides: fraction of
# the frame -> words, across by its head and down by its box.
FRAME_ACROSS = (
    (0.12, "at the left edge"),
    (0.38, "left of centre"),
    (0.62, "in the centre"),
    (0.88, "right of centre"),
)
FRAME_ACROSS_LAST = "at the right edge"
FRAME_DOWN = ((0.08, "the top edge"), (0.35, "the upper third"), (0.65, "the middle"), (0.92, "the lower third"))
FRAME_DOWN_LAST = "the bottom edge"
FRAME_PARTLY = ", partly outside the frame"
# The official camera vocabulary: motion type + amplitude + speed. One move per chunk, in its main beat.
CAMERA_MOVES = {
    "hold": "The camera holds its framing steadily.",
    "push_in": "The camera pushes in with small amplitude at slow speed toward {toward}.",
    "pull_out": "The camera pulls out with small amplitude at slow speed, keeping {toward} in frame.",
    "reveal_left": "The camera pulls out with large amplitude at slow speed while trucking left, bringing {toward} into the frame.",
    "reveal_right": "The camera pulls out with large amplitude at slow speed while trucking right, bringing {toward} into the frame.",
    "pan_left": "The camera pans left with small amplitude at slow speed to follow {toward}.",
    "pan_right": "The camera pans right with small amplitude at slow speed to follow {toward}.",
    "truck_left": "The camera trucks left with small amplitude at slow speed, keeping {toward} centred.",
    "truck_right": "The camera trucks right with small amplitude at slow speed, keeping {toward} centred.",
    "tilt_down": "The camera tilts down with small amplitude at slow speed toward {toward}.",
    # Moves that travel: a chase or a flight covers distance, and a camera that holds makes it hover.
    "follow": "The camera travels with {toward} at its speed, keeping it in frame as the background streams past.",
    "stay": "The camera stays where it is while {toward} moves away from it, growing smaller in the frame.",
    "crane_up": "The camera rises with large amplitude at medium speed, following {toward} upward.",
}
MUSIC = "N/A"  # a score regenerated per chunk cannot be continuous; add music after the join


# --- the story against the photographs: what it says of a member's look that a photograph does not show ------
RULING_NOVEL_SYSTEM = """\
You are the continuity clerk of a film made from a written story and reference photographs. The photographs decide how a member looks; the story was written without them. You find where the STORY describes the member's look differently from the TRAITS read off its photographs, as JSON.

RULES
- A conflict is a visible fact the STORY states about this member that one of the TRAITS contradicts: a red beard where a trait says grey, brass buttons where a trait says horn. Something the traits do not mention, an action, a mood or a passing state (wet, bloody, bareheaded for a moment) is not a conflict.
- "says": the story's words for it, copied exactly ("his red beard, gone to rust at the chin").
- "trait": the number of the TRAIT it contradicts, from 1.
- "shows": that trait's own words as the photographs give them.
- [] when the story agrees with every trait or says nothing of the look.

Answer only JSON, in this shape:
{"conflicts":[{"says":"...","trait":1,"shows":"..."}]}"""

RULING_NOVEL_USER = """\
STORY:
{story}

MEMBER: {member} ({kind}, {noun}){aliases}
TRAITS:
{traits}"""

# One question per conflict: the sentences that state it, each rewritten to the photograph, and the
# trait that would keep the story instead.
RULING_NOVEL_LINES_SYSTEM = """\
You reconcile ONE conflict between a written story and a member's reference photograph, as JSON.

- "lines": every place in the STORY that states the story's version: "line" the fewest words that say it whole, copied exactly ("short red hair is cropped close"), and "rewritten" those words changed to say what the photograph shows, so they still read right in their sentence ("long black hair falls loose"). Never take in words about anything else: each conflict rewrites only its own words.
- "trait": the STORY's version as a trait, never the photograph's: story "his red beard, gone to rust at the chin" is the trait "a rust-red beard". "phrase" a noun phrase of at most 14 words that reads after "with", "short" the same in at most 5 words keeping its colour, "name" what it is in one or two plain words without colour.

Answer only JSON, in this shape:
{"lines":[{"line":"...","rewritten":"..."}],"trait":{"phrase":"...","short":"...","name":"..."}}"""

RULING_NOVEL_LINES_USER = """\
STORY:
{story}

MEMBER: {member} ({kind}, {noun})
THE STORY SAYS: {says}
THE PHOTOGRAPH SHOWS: {shows}"""

# --- judging: a keyframe or a clip against the shot it was made for ---------------------------------------
JUDGE_SYSTEM = """\
You are a continuity supervisor on a film set. You are shown a still, or a few frames of a clip in order, and told what the shot was meant to show and who is in it, with how each of them looks; after the frames may come the cast's own reference photos, one per member, named in WHO IS IN IT. You judge only what you can see. Score 1 when the picture shows the shot as described with every listed member present and looking exactly as described and as in their reference photo; score 0 when it shows something else, someone missing, or a member unrecognisable. Score in between for a picture that mostly shows it: name every mismatch you see as an issue, in a few words each.

A clip moves: it opens on the pose the shot before ended in, its action carries it across the frames, and the arrangement is its last frame. Judge whether the action happens over the clip and ends as arranged; never fault an early frame for not yet showing the end.

Two failures cap the score at 0.4, whatever else is right: any person or animal in the frame who is not listed, and a member wearing a different garment or colour from their reference photo. Name each as an issue a painter could fix ("remove the monks behind him", "the robe must be the dark green one with black embroidery from his photo")."""

JUDGE_USER = """\
{references}This is a {kind}: {count} picture(s), in order.

WHAT THE SHOT SHOWS
Purpose: {purpose}
Arrangement: {arrangement}

WHO IS IN IT
{cast}

Answer JSON: {{"score": 0.0-1.0, "cast": [{{"id": "<id>", "present": true|false, "likeness": 0.0-1.0, "outfit": true|false}}, ...], "strangers": ["a few words for each person or animal not listed"], "issues": ["..."], "fixes": ["..."], "notes": "one sentence"}}. One "cast" entry per listed member; "outfit" is false only when a garment is a different garment or a different colour from the one in their photo (a bright robe with gold medallions for a dark embroidered one) and WHO IS IN IT does not say it has changed; a missing accessory, a detail of embroidery or the way the hair is worn is an issue, not a changed outfit. {fixes}"""
JUDGE_FORM = (
    "; it is the {was} of its reference photo turned into this, so only what these words name comes from that photo,"
    " and no {was} is left anywhere: a {was} still shown beside, behind or under it fails the picture"
)
JUDGE_FIXES_SCENE = '"fixes" are the issues a painter is told, about who is there, where each stands, what each holds and does; never about clothes, hair or a face, which the painter copies from the photos.'
# A look or a drawn reference is redrawn alone from its own words, so there the clothes, hair and face are
# exactly what the editor must be told.
JUDGE_FIXES_PICTURE = '"fixes" are what the editor redrawing this one picture is told, each a short instruction about the member\'s clothes, colours, hair, face, build, pose or backdrop that brings it to the description and the reference photo ("make the robe deep sapphire blue, not teal").'

# A clip's retake is rendered from these on a new seed (`judge.revise`).
CLIP_REVISE_SYSTEM = """\
You are the director of a live-action video. A video model rendered one shot from the SUMMARY and DESCRIPTION below, and the continuity supervisor rejected the take: you are shown its frames, in order, with what the supervisor found wrong and wants fixed. You rewrite the SUMMARY and the DESCRIPTION so the next take, rendered from them on a new seed, shows the shot as it was meant, as JSON.

RULES
- Change only what the issues and fixes need: who is in the frame, where each stands, what each holds and does, how the action runs. Keep every other sentence word for word, in its order, with its timing words ("For the first two seconds", "In the final two seconds").
- An action the take skipped or blurred is said plainer and earlier, as its own sentence; one it got wrong is said again the way it should happen.
- Say what is there and what happens, never what is not: a model renders a thing it is told is absent. A person who should not be there is fixed by saying who fills the frame ("only <Subject 1> and <Subject 3> are on the beach").
- Keep every <Subject N>, <Picture N>, <Video N> and <Audio N> label as it is written, and add none. Add nothing about faces, hair or clothes: those come from the pictures.
- The SUMMARY keeps its square-bracketed task list first; the DESCRIPTION keeps its style sentences before [Shot 1] and stays about as long as it is.

Answer only JSON, in this shape:
{"summary": "...", "description": "..."}"""

CLIP_REVISE_USER = """\
The take: {count} frame(s), in order.

WHAT THE SUPERVISOR FOUND WRONG
{issues}

WHAT THE SUPERVISOR WANTS FIXED
{fixes}

SUMMARY:
{summary}

DESCRIPTION:
{description}"""

# --- ruling: what a gate asks about meaning, answered by the model instead of a word list ---------------------
RULING_EVENTS_SYSTEM = """\
You are the script supervisor of a live-action video. You list the events of an IDEA, as JSON.

RULES
- An event is something that happens: a member moves, acts, changes or makes a sound (a moan, a giggle), or something appears, bursts, rises, falls or changes. The place, a light, a look, clothing, a pose or state already true when the video begins, an action the OPENING FRAME already shows under way (a man already smoking), and a rule ("nobody speaks") are not events.
- "events" lists every event of the IDEA in its order, each in the IDEA's own words with who does it named ("she giggles" is "the woman giggles"): the fewest words that say who does what ("the cat leaps out through the open window"). An action and the result it causes at once are one event; two actions joined by "and" or "then" are two.

Answer only JSON, in this shape:
{"events":["the cat lifts its head","the window swings open"]}"""

RULING_EVENTS_USER = """\
IDEA:
{idea}

OPENING FRAME: {opening}"""

RULING_STORY_SYSTEM = """\
You are the script supervisor of a live-action video. A planner split an IDEA into clips of beats. You find the beat that shows one EVENT of the IDEA, as JSON. You rewrite nothing.

RULES
- "clip" and "beat" are the numbers of the beat whose action shows the EVENT happening; both 0 when no beat's action shows it. Naming a thing is not showing the event: "the cat sits by the hookah" does not show "the cat smokes the hookah". The STORY is the planner's own note of where each event is: look there first, but go by the beats' actions.
- "faithful" is false when that beat shows the event with another action or another count than the EVENT's (the event's cat jumps and the beat's cat walks; the event's two rings and the beat's one); true when it does the same thing, in any words, and true when no beat shows it.

Answer only JSON, in this shape:
{"clip":2,"beat":3,"faithful":true}"""

RULING_STORY_USER = """\
THE CLIPS
{clips}

STORY
{story}

EVENT: {event}"""

RULING_MOUTHS_SYSTEM = """\
You are the script supervisor of a live-action video. A clip lists its MOUTHS: every way a mouth opens in it. The video model opens a mouth only for what that list says, and fills a mouth listed but never used with mumbling. You check the list against the clip's beats, as JSON.

RULES
- A mouth opens to speak, laugh, giggle, chuckle, sigh, gasp, cough, yawn, exhale or blow smoke, draw on a pipe or a mouthpiece, eat, drink, bite, take or hold something in it, or make an animal's call (a meow, a purr, a bark), and wherever an action says a mouth is open. A smile, breathing, and lips resting on a mouthpiece keep the mouth closed. Go by what an action says happens, and infer nothing it does not say.
- "openings" lists every mouth opening in the beats' actions, in order: its beat; "quote", the words of that beat's action that say it, word for word; "what", a short phrase for it with its owner ("her giggle"); and "entries", the numbers of the MOUTHS entries that name it, in any words ("exhales and blows two rings" is both "his exhale" and "his two rings"), or [] when none does.

Answer only JSON, in this shape:
{"openings":[{"beat":2,"quote":"giggles softly","what":"her giggle","entries":[1]}]}"""

# Asked beside the beats, "does it open a mouth" came back false for "the cat's meow" and "his exhale"
# whenever no beat had them done, and the planner deleted the meow; asked alone, it is about the words.
RULING_OPENS_SYSTEM = """\
You read one phrase from a video's list of the ways a mouth opens, and say whether the thing it names opens a mouth, as JSON. Speaking, laughing, giggling, sighing, gasping, coughing, yawning, exhaling or blowing smoke, drawing on a pipe, eating, drinking, biting, spitting, holding something in the mouth and an animal's call (a meow, a purr, a bark) open a mouth, and so does a phrase that names what a mouth makes or does ("the two smoke rings of the cat", "his blow", "her spit"). A smile, breathing through the nose, and a sound something else makes (the rip of cloth, a shove) do not.

Answer only JSON, in this shape:
{"opens":true}"""

RULING_MOUTHS_USER = """\
BEATS
{beats}

MOUTHS
{mouths}"""

RULING_VOICES_SYSTEM = """\
You are the script supervisor of a live-action video. A clip's mouth rule lists every way a mouth opens in it, and a writer then wrote the clip's prose and its sounds. The video model opens a mouth only for what the rule allows, and fills a mouth the rule allows but the prose never uses with mumbling. You check the prose and the sounds against the rule, as JSON.

RULES
- For each ALLOWED entry: "shown" is true when the prose has its owner doing it. "audible" is false for what can be done with breath alone: an exhale, a yawn, a draw on a pipe or a mouthpiece, blowing smoke or smoke rings. "heard" is true when the SOUNDS line has its sound.
- "extra" quotes word for word, from the prose or the sounds, each mouth opening the rule does not allow: speaking, laughing, giggling, sighing, gasping, coughing, yawning, exhaling, eating, biting, taking or holding something in a mouth, an animal's call, or any other sound or breath made with an open mouth. A smile, breathing, and lips resting on a mouthpiece keep the mouth closed and are never extra. [] when there is none.

Answer only JSON, in this shape:
{"allowed":[{"shown":true,"audible":true,"heard":true}],"extra":[]}"""

RULING_VOICES_USER = """\
ALLOWED, one "allowed" entry for each, in this order:
{allowed}

PROSE
{prose}

SOUNDS
{sounds}"""

RULING_CHANGED_SYSTEM = """\
You are the script supervisor of a live-action video. A clip ends with a CHANGE, a fact true at its end that was not at its start, and the clip's last frame in words. You say which members of the clip look different from here on because of it, as JSON.

RULES
- A member looks different when its own body, clothes or material is changed and stays so: a garment put on, taken off, torn, soaked or stained; a thing broken, bent, burnt, opened for good, filled or emptied; wings or a wound that stay; a member turned into something else. Where a member is, how it stands or sits, what it holds or where it was put, a door or a lid that can simply be closed again, an expression, light, smoke and the camera change nothing about its look.
- "changed" lists the id of each member that looks different from here on, [] when none does.

Answer only JSON, in this shape:
{"changed":[]}"""

RULING_CHANGED_USER = """\
MEMBERS: {members}
CHANGE: {change}
LAST FRAME: {closing}"""

# Production 10 wrote its cat "a small shape high in the sky, moving away from the camera" over a crane_up,
# and the camera rising with it rendered the cat filling the frame.
RULING_LEAVING_SYSTEM = """\
You are the script supervisor of a live-action video. A clip's camera travels with a MEMBER ("follow" or "crane_up"), so that member stays near the camera and large in the frame. You read the clip's CLOSING, its last frame in words, as JSON.

RULES
- "away" is true when the CLOSING has the MEMBER going away from the camera: growing smaller in the frame, far off, a shape, a dot or a speck in the distance, disappearing into the background. It is false when the member stays near the camera, at whatever size it has while the camera keeps up with it.

Answer only JSON, in this shape:
{"away":false}"""

RULING_LEAVING_USER = """\
MEMBER: {member}
CAMERA: {move}
CLOSING: {closing}"""

# "moment": production 10's card said "closed eyes" from a sleeping photo, and the retention line kept them
# closed in every shot. "held": the take moves a held thing, and a card that keeps it contradicts the hand
# the opening names.
RULING_TRAITS_SYSTEM = """\
You are the script supervisor of a live-action video. A member's card lists its TRAITS: its lasting look, which every shot of the video restates. You check each trait, as JSON.

RULES
- "verdict", for each trait in order:
  - "own": a lasting part of this member's own look: its hair, face, body, coat and markings, its clothes and what it wears (a watch, glasses, a hair tie); for a place, its landmarks, furniture and surfaces.
  - "other": a part of another listed MEMBER, which has a card of its own (the hookah's hose on the man's card); "of" is that member's id.
  - "held": a thing held or carried for the moment (a glass in her hand, a hose between its paws).
  - "moment": a passing state of the photograph's moment, which the video changes: eyes closed, asleep, a mouth open, a yawn, a blink.
- "of" is "" unless "verdict" is "other".

Answer only JSON, in this shape:
{"traits":[{"verdict":"own","of":""}]}"""

RULING_TRAITS_USER = """\
MEMBER: {member}

OTHER MEMBERS
{others}

TRAITS, one "traits" entry for each, in this order:
{traits}"""

# --- composing: a first frame from words, over a reference of the room ---------------------------------------
COMPOSE_INSTRUCTION = """\
Keep this room exactly as it is: its walls, furniture, lighting and colour. Change nothing about the camera. Add the people and things described, placed as described, in photographic detail: {scene}."""
# With each person and animal handed over as its own picture. Said in words alone, production 13's man
# (2026-10-04) came back in a bright costume-green robe with gold medallions over his dark embroidered one,
# and a sect courtyard filled with monks no plan had asked for.
COMPOSE_LIKENESS = "Everyone taken from a picture looks exactly as in that picture: the same face, hair and build, and the same clothes in every colour, pattern and fold, except what is said to be different now."
# A full moon handed over under COMPOSE_LIKENESS alone came back as a woman in a fur suit, and a moon
# over the sea with a woman in a floral dress before it (2026-10-17): the face and clothes asked for a person.
COMPOSE_LIKENESS_THING = "Each thing taken from a picture looks exactly as in that picture: the same shape, colours and surface; it is no person and wears nothing."
COMPOSE_ALONE = "Nobody else is in the frame: no other people and no other animals."
COMPOSE_NOTE = "Also: {note}."
COMPOSE_STYLE = "Picture {picture} is only a reference for the look: match its colour grade, light and lens, and take nothing else from it."
COMPOSE_LOOK_WORDS = "The look of the whole picture: {look}."
# A look of a member drawn from the picture it starts from, changed only by the amendment; a person's base
# face rides along as Picture 2, so a chain of edits does not drift from the face it began with.
COMPOSE_LOOK = """\
Keep the {noun} in this picture exactly as it is: {kept}, and the same pose, framing, backdrop and light. Change only this, and nothing else: {amendment}. An unretouched real photograph, not a render or an illustration."""
COMPOSE_LOOK_KEEP = {
    "person": "the same face, hair and build, and every piece of clothing the change does not name, in every colour, pattern and fold",
    "animal": "the same face, coat, markings and build",
    "object": "the same size, colours and material, and its shape wherever the change does not reach",
    "environment": "the same walls, ground, furniture, colours and light",
}
# Measured 2026-10-17 on a cat that becomes the moon: "turn the cat into the moon, keeping the framing and
# backdrop" left the cat whole on its windowsill with a moon laid over it, and the judge passed it at 0.9.
COMPOSE_FORM = """\
Replace the {was} in this picture with the {form}, with {look}. No {was} is left anywhere in the picture: the {form} is all that remains of it, whole and centred, alone against a plain backdrop. An unretouched real photograph, not a render or an illustration."""
COMPOSE_LOOK_FACE = "The face is exactly the face in Picture 2: the same features, skin and hair."
COMPOSE_FIXES = "Also fix this: {fixes}."
COMPOSE_LOOK_GARMENT = (
    "The {name} is exactly the garment in Picture {picture}: the same cut, colour, pattern and trim, now worn."
)
COMPOSE_LOOK_ALLOWED = "These may differ as the change needs: {allowed}."
# The shape the old plates were written in (b1d50dc): the realism clause is positive because the editor
# runs at cfg 1 and drops the negative prompt.
COMPOSE_PLATE = """\
Replace everything behind the {noun} with a flat, even mid-grey studio backdrop. Keep the {noun} exactly as it is: {traits}, the same pose, framing and light. Keep it an unretouched real photograph with natural skin texture and visible pores."""
# The opening photo whole, fitted into the canvas over a blurred stretch of itself (`cut_crops.padded`).
COMPOSE_EXTEND_OPENING = """\
The sharp photograph in the middle is part of a wider scene. Replace the blurred bands at its {sides} with the rest of that scene, continuing its walls, ground, furniture, light and perspective out to the edges. Keep everything in the sharp photograph exactly as it is: every person, face, pose and object, in the same place and size. No borders, frames or seams. One unretouched real photograph."""
# Told in words, a robe's colour and embroidery overrode the picture it came with: production 13's retry
# (2026-10-04), noted "the dark green one with gold floral embroidery", came back with gold flowers a
# hand wide. A redo is pointed back at the picture instead.
COMPOSE_CLOTHES_NOTE = "the clothes, hair and face of {who} exactly as in their picture, adding nothing to them"
# Over a photo of one member, when the place has no picture of its own: the member stays, the room goes.
COMPOSE_KEEP_INSTRUCTION = """\
Keep the {noun} exactly as it is in this photo: its face, colours, markings, fur and build. Replace everything around it - the ground, the background and the light - with the place described, and place everyone as described, as one unretouched real photograph in natural light: {scene}."""
# A member no photograph shows, drawn alone over a blank grey plate as its reference picture. Named
# COMPOSE_ so the director's fingerprint leaves them out: a change to them redraws, it does not re-plan.
# Measured on production 4 (2026-09-27): without "not a render" the pigeon came out as a CG model; an
# object not said to lie stood up on its legs; a place without its camera "far from the ground" was
# shot from the asphalt, every pebble of it in the foreground, and at 1.6 m by number still was.
COMPOSE_REFERENCE_INSTRUCTION = {
    "person": """\
Replace this plain grey picture with an unretouched real photograph of one {noun} standing alone: {traits}. Show the whole {noun} from head to feet in the middle of the frame with space around, the body turned three-quarters toward the camera and the face looking at it, sharp and in focus, in soft even daylight against a plain seamless light-grey backdrop. No other people, animals or objects, and no text. {look}""",
    "animal": """\
Replace this plain grey picture with an unretouched real photograph of one ordinary live {noun} standing alone: {traits}. Show the whole {noun} from head to tail in the middle of the frame with space around it, its body turned three-quarters toward the camera, sharp and in focus, in soft even daylight on a plain seamless light-grey backdrop. No other animals, people or objects, and no text. {look}""",
    "object": """\
Replace this plain grey picture with an unretouched real photograph of one {noun} lying on its own: {traits}. Show the whole {noun} in the middle of the frame with space around it, seen three-quarters from slightly above, sharp and in focus, in soft even daylight on a plain seamless light-grey backdrop. Nothing else in the picture, and no text. {look}""",
    "environment": """\
Replace this plain grey picture with an unretouched real photograph of the {noun}, empty of people and animals, lit by {lights}: {traits}. A wide establishing view of the {noun} from a standing person's eye height, the camera level and far from the ground. Everything sharp from near to far. {look}""",
}
COMPOSE_REFERENCE_LOOK = (
    "A real phone photograph, not a render or an illustration: natural colours and soft highlight roll-off."
)

# --- the idea, rewritten into a story before any plan ---------------------------------------------------
# Named IDEA_ so the director's fingerprint leaves them out: a change to them rewrites ideas, it does not
# re-plan. Each rule is a failure of this pipeline on productions 4 and 5 (2026-09-26): the street taken
# from a bed photo, the pigeon alive again after it was roasted, the cat ending every shot facing the
# camera, "fires flame" rendered as a stream from its mouth.
IDEA_SYSTEM = """\
You are the story editor of a live-action video. A planner turns your STORY into consecutive shots, one or more per beat, each as long as its action needs, and a video model renders each shot from the reference photographs, labelled Picture 1, Picture 2 and so on. The planner reads your words literally: what they do not say, it fills in from the photographs or leaves out. You receive the user's rough IDEA and the photographs, and you rewrite the IDEA into a STORY the planner renders well, as JSON.

WHAT GOES WRONG, AND THE RULES THAT PREVENT IT
1. The place, first and concretely. Unless the story says where it happens, the planner takes the place from Picture 1: a street became a bed because the only photograph was of a cat on a bed. "place" is the opening paragraph, 35 to 80 words: the place the IDEA names, with the time of day and the light, the ground and every surface the action touches, and what stands in the background. When no picture shows that place, describe it entirely in words and take nothing of it from the pictures. The same paragraph ends on the first frame: where each subject is, its pose, and what it is doing. Picture 1 is a real photograph, so this paragraph says only what it shows, as it shows it: name no object it does not show, keep every door and window open or closed as it is, and give each subject its pose exactly (lying stretched on its side, paws out in front - not curled up).
2. One main event per beat. "beats" has one entry per main event of the IDEA, in order, as many as fit the LENGTH: the planner gives each beat a shot of its own, as long as that action needs. A beat is the few seconds of visible action one shot shows: one to three plain sentences, no more words than LENGTH allows. Never two big events in one beat - a door opening is one, the dog running out through it is the next - and never a beat with nothing new happening. An action and its immediate result belong to the same beat (a stone thrown and the window it breaks): a shot that starts in mid-flight renders it again from the start.
3. Transformations, spelled out. When something turns into something else (burned, roasted, frozen, broken, eaten), the beat describes what it is afterwards, and then says in a sentence of its own that the old thing no longer exists anywhere in the scene, naming it as it was ("No ice is left anywhere.", "No whole vase remains anywhere."). Later beats name only the new thing. Otherwise the planner keeps the old thing in the later shots. A subject that turns into something else and stays itself (the cat that becomes the moon) is the exception: it is still that subject, so later beats name it as both ("the cat from the photographs, now the moon"), the beat says what it looks like now and which of its own features show in it, and the sentence of its own says that what it replaces is gone ("No other moon is in the sky.").
4. How every subject ends. A subject that leaves turns its back to the camera and goes, and the beat says which way: out of a named side of the frame, or down the street getting smaller until it disappears; say what is left behind ("The yard is left empty."). A subject that stays ends in a pose that says which way it faces. Without this, every animal ends its shot turned to the camera.
5. The unusual, as the camera sees it. Describe anything that does not happen every day physically: its shape, size, colour, material and motion ("a thin straight beam of pale blue light, as wide as a pencil, that leaves a trail of frost", never "shoots ice"). Describe the result of an effect as an object you could photograph ("frozen solid inside a clear block of ice the size of a shoebox, white frost on its whiskers"), never as a verb alone.
6. Sound. "sound" is one line: the sound of the place, then the sound of each beat's key moment in order ("a sharp crack as the beam fires, a brittle crunch as the ice closes"). It ends with "No dialogue." unless the IDEA gives words to be spoken; those go in the beat where they are said, in double quotes.
7. The user's story, kept. Keep the IDEA's intent, its subjects, its events and their order. Add no character, animal, object or event the IDEA does not have, and soften nothing: what happens in the IDEA happens in the story, as bluntly. Call a subject a photograph shows by what it is and where it comes from ("the cat from the photographs", "the man from the photographs"), and leave its colours, markings and clothes to the photographs: a word that disagrees with them fights them. Describe the look only of a subject no picture shows. Look at every picture: when no picture shows the place, or a picture disagrees with the IDEA, say so in "changes".
8. Plain prose. Present tense, concrete nouns and verbs, one subject doing one visible thing at a time. Camera directions are not needed. No quality words (cinematic, stunning, 4K), no metaphors, no markdown, and no numbers in front of the beats: the program numbers them.

ALSO FOR THE USER
- "changes": 2 to 6 short sentences, each one thing you changed from the IDEA and why ("Described the street, since no photograph shows it.").
- "questions": at most 3 short questions about real choices the IDEA left open, each naming what you chose; never one your story or the IDEA already answers. Most ideas need none: then [].

Answer only JSON, in this shape:
{"place":"...","beats":["...","..."],"sound":"...","changes":["..."],"questions":[]}"""

IDEA_USER = """\
IDEA (the user's own words):
{idea}

LENGTH: the video lasts at most {total:g} seconds, in shots of {shortest:g} to {longest:g} seconds each. "beats" has {least} to {most} entries, each at most {beat_words} words.
PICTURES: {pictures}. Picture 1 is where the video opens: the planner takes the place from it unless the story names another."""

# A round with feedback revises the version it was given; a whole new rewrite loses what the user liked.
IDEA_REVISE_USER = """\
IDEA (the user's own words, as first given):
{idea}

YOUR PREVIOUS VERSION:
{previous}

THE USER'S FEEDBACK ON IT:
{feedback}

Revise YOUR PREVIOUS VERSION: change what the feedback asks, and whatever that change forces elsewhere (a later beat or the sound line that mentions the same thing). Keep every other sentence word for word. The rules still hold, and the feedback wins over the IDEA where they disagree. In "changes", list only what this revision changed, and why.

LENGTH: the video lasts at most {total:g} seconds, in shots of {shortest:g} to {longest:g} seconds each. "beats" has {least} to {most} entries, each at most {beat_words} words.
PICTURES: {pictures}. Picture 1 is where the video opens: the planner takes the place from it unless the story names another."""

# --- manga: pages read into panels, panels redrawn as anime keyframes, shots filled ---------------------
# MANGA_, not DIRECTOR_: the manga stages key themselves by these words, and a change to them re-plans no
# production made from a story.
MANGA_CONFIRM_SYSTEM = """\
You check where the panels of a comic page are. You see the page with numbered boxes drawn over it in the order a reader reads them. A panel is one drawn picture with its own frame or edge; lettering in the gutter between panels is not a panel. Say where every panel really is, as JSON.

RULES
- "panels": every panel of the page in reading order ({order}), each as "box": [left, top, right, bottom] on a 0-1000 scale of the page's width and height.
- Keep a numbered box exactly as drawn when it holds exactly one panel. Split a box that holds two or more panels; join boxes that cut one panel apart; drop a box around lettering alone.
- A picture that bleeds off the page edge reaches that edge. Panels may overlap only where the art itself overlaps.

Answer only JSON, in this shape:
{{"panels":[{{"box":[0,0,1000,480]}},{{"box":[0,500,1000,1000]}}]}}"""
MANGA_CONFIRM_USER = "There are {count} numbered boxes. Where are the panels?"
MANGA_BAND_SYSTEM = """\
You see one band cut from a vertical comic strip. It may hold a drawn picture, lettering (speech balloons, captions, sound effects) or both. Say where the drawn picture is, without the blank background and the balloons around it, as JSON.

Answer only JSON, in this shape:
{"picture":true,"box":[0,120,1000,880]}  with "box" as [left, top, right, bottom] on a 0-1000 scale of the band, or {"picture":false,"box":[]} when the band holds lettering alone."""
MANGA_BAND_USER = "Where is the drawn picture in this band?"

MANGA_READ_SYSTEM = """\
You read one panel of an English comic, word for word, for the people adapting it into an anime. Picture 1 is the panel. {context}You write down what is lettered in it and what happens in it, as JSON.

RULES
- "balloons": every speech balloon, thought bubble, shouted line and narration box of Picture 1, in the order they are read ({order}). "kind" is "speech", "thought", "shout" (a jagged or burst balloon) or "narration" (a caption box, no speaker). "text" is exactly what is lettered, every word in its order, nothing added, dropped or reworded; write ordinary sentence case even when the lettering is all capitals. A balloon that holds only dots or punctuation ("...", "!?") is still a balloon, written as lettered.
- "speaker": who says or thinks the balloon, by name; "" for narration. A NAME is the character's own name when the comic gives it (in a balloon, a caption, or the KNOWN NAMES below), else two to four plain lowercase words for what they are ("bald man", "crying girl", "horned monster"). Use the same name for the same character every time{known_rule}.
- "sfx": every sound effect lettered in the art, as written (translated notes in the gutter count); [] when there is none.
- "who": the NAME of every character drawn in Picture 1, plus every speaker of a speech, thought or shout balloon even when only their voice is in it.
- "drawn": false when Picture 1 holds only lettering on a plain white or black ground, with nothing drawn; true otherwise. "who" is then the speakers alone.
- "action": one sentence of at most 30 words: what visibly happens in Picture 1, naming the characters by NAME.
- "into_next": {into_next_rule}
- "why": one short phrase for "into_next".

Answer only JSON, in this shape:
{{"balloons":[{{"kind":"speech","speaker":"bald man","text":"..."}}],"sfx":["..."],"who":["bald man"],"drawn":true,"action":"...","into_next":"cut","why":"..."}}"""
MANGA_READ_CONTEXT_PAGE = "Picture 2 is the whole page it is on, for context only. "
MANGA_READ_CONTEXT_NEXT = "Picture {picture} is the panel read after it. "
MANGA_READ_LAND = '"land" when the panel after it (Picture {picture}) continues the same action in roughly the same framing, so one movement can carry Picture 1 into it; "cut" when it is a new angle, a reaction close-up, another place or a jump in time.'
MANGA_READ_LAST = '"cut": nothing is read after this panel.'
MANGA_READ_KNOWN = ", and a name from KNOWN NAMES whenever it is that character"
MANGA_READ_USER = """\
{known}Read Picture 1."""
MANGA_READ_NAMES = "KNOWN NAMES, from the panels read before this one: {names}.\n\n"

MANGA_CAST_SYSTEM = """\
You are the casting director of an anime adapted from a comic. A reader went through the comic panel by panel and named every character it saw, but it may have given one character several names (its own name in one panel, "bald man" in another). You decide who the characters are, as JSON.

RULES
- "characters": one entry per character. "id": one lowercase English word or two joined by "_", the character's own name when the comic gives one ("saitama", "vaccine_man"), else what it is ("crying_girl"). "kind": "person" or "animal". "names": every NAME of the list that means this character.
- Every NAME of the list is in exactly one character's "names". Two NAMES are one character only when the panels make it clear: the same look, the same voice, or the comic naming them.

Answer only JSON, in this shape:
{"characters":[{"id":"saitama","kind":"person","names":["saitama","bald man"]}]}"""
MANGA_CAST_USER = """\
NAMES: {names}

THE PANELS, in order (who is drawn | what happens | who speaks):
{panels}"""

# A panel cleaned of its lettering before anything is drawn from it. The editor runs at cfg 1, so what it is
# told to keep is said positively.
COMPOSE_MANGA_CLEAN = """\
Remove every speech balloon, thought bubble, caption box and every piece of lettering and sound-effect text from this comic panel, and fill each place with what was behind it, continuing the drawing, its lines, tones and background. Keep every character, face, pose, line and the framing exactly as they are."""
MANGA_CLEAN_SYSTEM = """\
You check a comic panel that an editor cleaned of its lettering. Picture 1 is the panel as drawn; Picture 2 is the cleaned one. Answer as JSON.

- "lettering": true when Picture 2 still shows any balloon, caption box, letters or sound-effect text.
- "changed": true when a character in Picture 2 differs from Picture 1: a different face, pose, body or clothes, or a character added or gone.
- "why": at most 15 words.

Answer only JSON, in this shape:
{"lettering":false,"changed":false,"why":"..."}"""
MANGA_CLEAN_USER = "Is Picture 2 clean of lettering, with the characters unchanged?"

# One colour sheet per character, drawn once over its best crop and approved before any keyframe takes it.
COMPOSE_MANGA_SHEET = """\
Redraw this {noun} as a full-colour anime character design sheet: one front view of the whole {noun}, standing, in the middle of the frame on a plain white background, in clean anime line art with flat cel shading. Keep the same face, hair style, build and clothes design: {traits}.{colours} Choose one clear colour for every part and give no text."""
COMPOSE_MANGA_SHEET_COLOURS = " Colours: {colours}."
COMPOSE_MANGA_SHEET_KEEP = " Keep every colour exactly as this picture shows it."
# The panel redrawn as the shot's first frame, each character in its approved sheet's colours, said in words.
# Measured 2026-10-17/18 on One-Punch Man pp. 3-6: handed the sheets as pictures, the editor pasted the sheet's whole
# figure over a close-up of an arm and over antenna tips, five panels in a row coming out as one portrait, whether
# told to draw each "exactly as in" its sheet or to take "only the colours, never its pose, size or framing".
COMPOSE_MANGA_ANIME = """\
Colour this comic panel as one frame of a full-colour anime, in clean anime line art with flat cel shading and painted backgrounds. Picture 1 is the frame: keep its composition, camera angle and framing, and every figure, pose and object exactly where it is and exactly as large, a figure the frame cuts off staying cut off and a close-up staying a close-up; continue the scene out over any blurred band at the edges. Add no person, face or figure.{cast}{style}"""
COMPOSE_MANGA_COLOURS_SYSTEM = """\
You read the colours off an anime character design sheet, for a painter who colours the same character in other pictures and never sees the sheet.

Name the colour of every part that shows: hair, skin, eyes, each piece of clothing, shoes, and anything worn or growing on the body such as horns or antennae. Name colours plainly ("dark navy", "pale peach"), each with its part, in one phrase of at most {words} words, no sentence around it.

Answer only JSON, in this shape:
{{"colours":"short black hair, pale skin, brown eyes, a white headband, a dark grey t-shirt, blue jeans"}}"""
COMPOSE_MANGA_COLOURS_USER = "The colours of this {noun}?"
COMPOSE_MANGA_CAST = " The {who} has {colours}."
COMPOSE_MANGA_STYLE = " Picture {picture} is only a reference for the look: match its colour palette, line weight and shading, and take nothing else from it."

MANGA_STYLE = (
    "The target video is a full-colour 2D anime in clean line art with flat cel shading and painted backgrounds, "
    "and colors keep the look of {room} throughout."
)

MANGA_FILL_SYSTEM = """\
You direct an anime adapted panel by panel from a comic. Every panel is already one clip with its first frame drawn from the panel, its length and its cast fixed. MiniMax H3 animates each clip from its first frame. You plan what moves in every clip, as JSON. You do not add, drop, merge or reorder clips, and you do not write dialogue: the comic's words are given and the program puts them in.

WHAT YOU RECEIVE
- CAST: every member with an id, what it is and its visible traits.
- PLACES: the places the comic happens in, as members of kind "environment".
- CLIPS: per clip its length, the panel's action, who is drawn in it, its spoken balloons numbered from 1, and whether the clip ENDS ON THE NEXT PANEL ("lands") or the next clip cuts.

RULES
1. Each clip shows the panel's action coming to life: the motion the panel freezes, carried on for the clip's length. A clip that lands moves from its panel toward the next panel's picture, so its last frame is exactly the next panel; its "closing" describes that next panel's first frame, naming every character of it as {{id}}, where each is in the frame by its sides, the pose and which way each face points. A clip followed by a cut ends with its action complete and everyone still; its "closing" describes that last frame the same way.
2. "beats": 1 to 4 beats covering the clip exactly: the first starts at 0, each next one starts where the one before ended, and the last ends at the clip's "seconds"; seconds with at most two decimals, each beat at least {beat_min:g} second unless the clip is shorter. "phase" is one of carry, build, primary, reaction, settle. "who" lists the ids that move in the beat, only from the clip's cast. "action" is one or two plain sentences, at most 30 words, of visible motion.
3. "dialogue": one entry per spoken balloon of the clip, in the order given, each {{"balloon": its number, "start": seconds}}; a line lasts about {words_per_second:g} words a second and ends at least {dialogue_tail:g} seconds before the clip ends. Give "voice" (how that member sounds, the same words every time: "a flat, calm young male voice") on the first line a member speaks in the whole plan.
4. "mouths": every way a mouth opens in the clip, as short plain phrases with possessives and no braces ("his flat reply", "the shout of the monster"); each spoken line's speaker is listed, and a beat's action has that member speak ("{{saitama}} answers flatly"). A mouth that only listens stays closed.
5. "place": the id of the one place the clip happens in. "scene": in clip 1 and in every clip that follows a cut, the scene heading: INT or EXT, the place in two to four plain words, " · ", the time of day ("EXT ruined city · day"); "" in a clip that continues. "opening": in clip 1 and in every clip that follows a cut, one snapshot sentence of the clip's first frame, the panel, written like a closing; "" otherwise.
6. "camera": {{"move": one of the CAMERA MOVES, "toward": a cast id}}: the panel's framing decides it ("hold" for a still panel, "push_in" toward a face in close-up).
7. Write only what a camera can see: no feelings by name, no quality words, never mention panels, pages, clips or the comic in any text. Name members as {{id}} in "action", "closing" and "opening", never with a possessive after the braces.

FIELDS per clip: "seconds" (as given), "purpose" (one sentence, at most 25 words), "place", "scene", "opening", "beats", "camera", "mouths", "closing" (at most {closing_words} words), "irreversible_change" (at most 20 words, or "none"), "ambience" (one noun phrase of the place's sound), "dialogue".

Answer only JSON, in this shape (write every clip):
{{"ambience":"...","clips":[{{"seconds":3.04,"purpose":"...","place":"city","scene":"EXT ruined city · day","opening":"...","beats":[{{"start":0,"end":3.04,"phase":"primary","who":["saitama"],"action":"..."}}],"camera":{{"move":"hold","toward":"saitama"}},"mouths":[],"closing":"...","irreversible_change":"none","ambience":"","dialogue":[]}}]}}"""
MANGA_FILL_USER = """\
CAST (id | kind | what it is | visible traits):
{cast}

PLACES: {places}

CLIPS:
{clips}

CAMERA MOVES:
{moves}"""
MANGA_FILL_CLIP = """\
Clip {number}: {seconds:g} seconds; {joins}
  panel: {action}
  cast: {cast}
  spoken balloons: {balloons}"""
