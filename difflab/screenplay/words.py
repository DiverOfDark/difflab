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
- A member is one person, one animal or one object that the IDEA needs and a picture shows, or the place where the video happens.
- Exactly one member is the environment: the place shown in Picture 1. The whole video happens there. The other pictures are used only for the members they show, never for their places.
- Add an object only when the IDEA needs it and a picture shows it (the hookah that is smoked). Furniture, walls, tables and lights belong to the environment.
- Add no person or animal that the IDEA does not need, even when a picture shows one: people in the background are bystanders, not members.
- A person or animal the IDEA brings in that no picture shows is a member too: list it in no picture's "shows" and it is created from words alone ("a man enters the scene" where every picture is of the woman). Only a person or animal may be a member no picture shows; an object and the place are members only where a picture shows them.
- "id": one lowercase English word, or two joined by "_" ("man", "woman", "cat", "hookah", "lounge"); every id is different.
- "kind": "person", "animal", "object" or "environment".
- "noun": what the member is, in at most two words ("man", "woman", "cat", "hookah", "shisha lounge").
- "possessive": "his" or "her" for a person, "its" for anything else.
- "find": how to find the member in a photograph, in at most 14 words, by what is visible: position, clothing, colour, pose ("the man in the black t-shirt holding the hookah hose"). For a member no picture shows, what the IDEA asks of it instead ("the man who enters and stands behind her").
- "pictures": one entry per picture, in order, listing the id of every member that picture clearly shows. List the environment for Picture 1 only. A member no picture shows is listed in none of them.

Answer only JSON, in this shape:
{"members":[{"id":"man","kind":"person","noun":"man","possessive":"his","find":"..."},{"id":"lounge","kind":"environment","noun":"shisha lounge","possessive":"its","find":"..."}],"pictures":[{"picture":1,"shows":["man","lounge"]},{"picture":2,"shows":[]}]}"""

SURVEY_USER = """\
IDEA:
{idea}

There are {count} pictures. Picture 1 is the opening."""

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
- Invent an ordinary look that the IDEA allows and nothing in it contradicts. Keep it plain: hair, one or two garments, at most one accessory. An animal: coat, markings, tail, ears.
- Decide nothing the IDEA has already decided. Where it says what the member wears or looks like, that is the look.
- Say nothing about the place, the light, the camera or any other member: this member's own look only.
- Nothing held and nothing done: a thing carried belongs to the action, not to the look, and the take moves it.
- Never state age, ethnicity, body weight, brand names or anyone's name, and never describe the shape of the nose, eyes, lips or jaw.
- "phrase": the full noun phrase, at most 14 words, reading after "with" ("a charcoal-grey open-collar shirt").
- "short": the same thing in at most 5 words, keeping its colour, with "a" or "an" in front of a single countable thing.
- "name": what it is in one or two plain words, without colour ("shirt", "hair", "watch").
- "part": one of hair, coat, markings, face, top, bottom, footwear, accessory, tail, ears.
- 4 to 6 traits.

Answer only JSON, in this shape:
{"traits":[{"part":"hair","phrase":"short dark-brown hair","short":"short dark-brown hair","name":"hair"}]}"""

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
You are the director and script supervisor of ONE unbroken live-action take that MiniMax H3 renders as a chain of short clips. H3 renders at most 15 seconds at a time, so the take is split into consecutive clips of equal length. Each clip starts on the exact last frame of the clip before it and inherits its last moment of motion and sound. You plan what happens in every clip, as JSON. You do not write the final prompts: a program turns your plan into them, and a writer expands your notes into prose.

WHAT YOU RECEIVE
- IDEA: what should happen over the whole take.
- CAST: every member with an id, what it is and its visible traits, taken from reference photographs. Use only these members; add no people, animals or props. The environment member is the place, and it is in every clip.
- OPENING: the exact arrangement of the first frame, which is a real photograph. Clip 1 starts on it; a member it does not name is outside the frame at the start.
- CLIPS: how many clips there are and how long each one is.

HOW A CHAINED TAKE BREAKS, AND THE RULES THAT PREVENT IT
1. A contradiction at a join renders as a union: extra people, doubled props. The program copies each clip's "closing" word for word as the opening of the next clip, so write every closing as a precise, complete snapshot of the last frame: where every cast member on screen is, by the sides of the frame (left of frame, right of centre, lower foreground - never "to his left"), the pose, what each hand or paw holds and in which hand, where every object of the cast is and who holds it, named as its {{id}}, which way each face points, and any smoke still in the air. Everyone on screen at the end of a clip who is also in the next clip is named in its closing.
2. The opening is a moment in the middle of something. The first beat of clip 1 finishes whatever the OPENING shows under way (an exhale ends and its cloud thins, a raised hand comes to rest) before anything new starts.
3. Airlock. Every clip after the first begins with an "airlock" beat of {airlock:g} seconds (at least {airlock_min:g}) in which the closing arrangement of the clip before holds: the performers only breathe, blink, shift their weight and turn their eyes, and smoke drifts. Nobody speaks, enters, leaves or passes anything during an airlock.
4. Land settled, on the faces. Every clip ends with a "settle" beat of {settle:g} seconds (at least {settle_min:g}): the action is complete, everyone is still, and the face of every person and animal on screen is turned toward the camera, front or three-quarter, never hidden by a hand, a prop or smoke. The closing says so ("both faces turned three-quarters to the camera, smiling with closed lips"), and "faces_at_close" lists them.
5. One main change per clip. Each clip has exactly one beat of phase "primary" or "handoff" that carries its main event, and it starts between {early:.0f}% and {late:.0f}% of the clip, because the end of a clip gets squeezed. "irreversible_change" names the physical fact that is true at the end of the clip and was not at its start ("the hose is in her hands", "the cat sits on the table"). A mood or a camera move is not a change.
6. Hand-offs take time. Passing an object from one hand or paw to another is a "handoff" beat of at least {handoff_min:g} seconds in which the reach, the grip and the release are all visible.
7. The story comes into the frame. The framing stays close to the opening framing. A member the clip's opening does not name is outside the frame when the clip starts: it takes no part in the first beat, and appears in a later beat by entering from a named edge (left, right, bottom) or from the background, or when a camera move brings it into view; nothing happens off-screen. Someone the IDEA places at the same table or close by is already sitting there, just out of frame on the side where the IDEA puts them (across the table is on the table's side of the frame): a camera move ("reveal_left", "reveal_right") brings them in, seated. Walking in is for someone who arrives. At most {max_living} people and animals are on screen in a clip. Every person and animal in a clip's cast moves in at least one beat or is named in its opening or closing.
8. Physical truth. Bodies and objects keep their weight, size and contact with surfaces. Animals move like real animals: a cat jumps, walks, sits, sniffs, paws and meows, and holds a hose by clasping it between its front paws. An object in someone's left hand stays in that hand until a beat moves it.
9. Keep the IDEA's own verbs and counts. If the IDEA says a member jumps, it jumps (not walks); "rings" are two or more rings; a giggle that comes with a hand-over happens in that hand-over beat.
10. Camera. Each clip has exactly one camera move from the list, made during its main beat and aimed at one cast member ("toward"). Use "hold" while three characters must stay in frame.
11. Mouths and voices. A mouth opens only for what "mouths" lists, and each thing it lists happens in a beat's action by that member. A smile keeps the lips closed and is not listed. Nobody talks unless the IDEA gives exact words. Laughter, giggles, sighs, exhales and animal sounds need no words. If the IDEA gives words, put each line in "dialogue" inside one clip, after the airlock and ending at least {dialogue_tail:g} seconds before the clip ends, at about {words_per_second:g} words per second.
12. Write only what a camera can see or a microphone can hear: no mood words, no quality words, and never mention clips, segments, prompts or stitching in any text.

FIELDS
- "ambience": one noun phrase, starting in lowercase, of the room sound heard through the whole take ("quiet lounge room tone with a low ventilation hum and a faint clink of glassware from the bar").
- "story": every action of the IDEA in its order, each as its own event in the IDEA's own words ("blows smoke rings", "jumps onto the table", "giggles"), with the clip (counted from 1) in which a beat's "action" shows it with the same verb. An airlock or a settle beat shows no event.
- "clips": exactly as many as CLIPS says. In each clip:
  - "purpose": one sentence of at most 25 words.
  - "cast": the id of every member on screen at any moment of the clip, the environment included.
  - "beats": 3 to 5 beats. The first starts at 0, each next one starts where the one before ended, and the last ends exactly at the clip length; seconds with at most two decimals. "phase" is one of airlock, build, primary, handoff, reaction, settle. "who" lists the ids that move in the beat. "action" is one or two plain sentences, at most 35 words, of visible motion: who moves which body part, where the prop goes, what the smoke does.
  - "camera": {{"move": one of the CAMERA MOVES, "toward": a cast id}}.
  - "mouths": every way a mouth opens in this clip, as short plain phrases with possessives and no braces ("his slow draw on the mouthpiece", "her giggle", "the two meows of the cat"); [] when every mouth stays closed.
  - "closing": one snapshot sentence of at most {closing_words} words, as rule 1 says.
  - "faces_at_close": the id of every person and animal in the closing.
  - "irreversible_change": at most 20 words.
  - "dialogue": [] unless the IDEA gives exact words; then {{"who": id, "start": seconds, "language": "English", "line": "...", "voice": "a bright, amused young female voice"}} per line.
- In "action" and "closing", write members as their id in braces: {{man}}, {{woman}}, {{cat}}. Never put a possessive after the braces: write "the hand of {{woman}}" or "her hand", never "{{woman}}'s hand".

Answer only JSON, in this shape (one clip and two of its beats shown; write every clip and every beat):
{{"ambience":"...","story":[{{"event":"...","clip":1}}],"clips":[{{"purpose":"...","cast":["man","hookah","lounge"],"beats":[{{"start":0,"end":2,"phase":"build","who":["man"],"action":"..."}},{{"start":2,"end":6.5,"phase":"primary","who":["man"],"action":"..."}}],"camera":{{"move":"hold","toward":"man"}},"mouths":["his slow exhale"],"closing":"...","faces_at_close":["man"],"irreversible_change":"...","dialogue":[]}}]}}"""

PLAN_USER = """\
IDEA:
{idea}

CAST (id | kind | what it is | visible traits):
{cast}

OPENING (the first frame of clip 1, a real photograph):
{opening}

CLIPS: {count} clips of {seconds:.2f} seconds each; the take lasts {total:.1f} seconds. In every clip the main beat starts between {early:.1f} and {late:.1f} seconds, and every beat lasts at least {beat_min:g} second.
CAMERA MOVES:
{moves}"""

WRITE_SYSTEM = """\
You are the prose writer for ONE clip of an unbroken live-action take rendered by MiniMax H3. A program has already written everything that must be exact: the style, the opening arrangement, the place and its light, the camera move, the rule for mouths, the closing arrangement and every reference label. You write only the motion in between, as JSON.

WHAT YOU WRITE
- "beats": one string per listed beat, in the same order, each 35 to 60 words of present-tense prose that films exactly that beat's planned action. Show the physical mechanics: which hand or paw, how far the body leans, where the hose goes, how the smoke leaves the lips and spreads, what the eyes do, how fur, fabric and hair respond. Put the sound a motion makes in the same sentence when it matters (the water bubbles as she draws).
- "summary": one sentence of 20 to 40 words saying what happens in this clip, naming the members as {id}.
- "sounds": one or two sentences of this clip's physical and non-verbal sounds, with no words spoken.

RULES
1. Name a cast member as its id in braces, {man}, {woman}, {cat}, the first time it acts in a beat; afterwards use he, she, it or a plain noun such as "the cat". Every member listed as acting in a beat is named in braces in that beat. Never put a possessive after the braces: write "the hand of {woman}" or "her hand". Never write labels in angle brackets.
2. No timing words and no numbers of seconds: the program puts a timing phrase in front of every beat ("For the first two seconds the arrangement holds with small natural motion:", "Early in the clip,", "Over about four seconds in the middle of the clip,"). Start each beat with the motion itself, in lowercase unless it starts with braces.
3. An airlock beat is quiet: breath, blinking, weight shifts, drifting smoke, a glance. Nothing enters, leaves or changes hands.
4. The program writes the clip's final settle from the closing arrangement, so your last beat ends where that arrangement begins.
5. Stay inside the plan: no new people, animals, props, lights, cuts or camera moves, and no clothing, hair or markings beyond the listed traits. Keep hands and paws exactly as the plan has them: an object in a left hand stays in that left hand.
6. Mouths only do what the mouth rule allows. A sound made by a mouth - a giggle, a laugh, a meow, an exhale, a sigh - happens only when the mouth rule lists it, and each one the rule lists is shown in the beat where it happens and heard in "sounds". Sides are sides of the frame ("toward the left of the frame"), never "to his left".
7. A dialogue line is written as {say}, immediately after the speaker's own braces, with none of its words. The program puts the words in and names the speaker there.
  Right: {right_girl} {say}, her shoulders shaking.
  Wrong: {right_girl} laughs, saying {say}.  (the marker must touch the braces)
  Wrong: {right_girl} says "so there!"  (never write the words; they are not yours to choose)
  One {say} per line of dialogue the plan gives this beat, in the order the plan lists them. A beat with no dialogue has no {say}.
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

WRITE_BEAT = "{number}. [{phase}, about {length:.1f} seconds, acting: {who}] {action}"

# Asked of every JSON answer the director wants. A third of every token it decodes used to be
# indentation and code fences - 9,887 tokens of the measured run's answers against 6,535 minified -
# and at a flat ~17.7 tokens per second that whitespace is minutes, not bytes. The space after ":" and ","
# is not asked away: `exl3vl.ANY_JSON` requires it, since masking it out emptied every answer.
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
SENTENCES = {
    "person": "{subject} is the {noun} whose face comes from {faces} and whose {parts} come from {bodies}, taking only {possessive} appearance from those photos: {traits}.",
    "person_face": "{subject} is the {noun} whose face comes from {faces}, taking only {possessive} appearance from {that}: {traits}.",
    "member": "{subject} is the {noun} from {pictures}, taking only {what} from {that}: {traits}.",
    # A member no photograph shows: its traits are the whole of it, so it cites no picture.
    "unseen": "{subject} is the {noun}, who appears in no reference picture and is created from this description alone: {traits}.",
    "environment": "{subject} is the {noun} and its photographic look from {pictures}: {traits}, and the photo's natural color balance, contrast and skin tones.",
    "first_frame": "<Picture 1> is the first frame of [Shot 1]: {arrangement}.",
    "video": "<Video 1> is the source video that the target video continues from its final frame.",
    "audio": "<Audio 1> is the synchronized audio track of <Video 1>, whose room tone continues into the target video.",
    "lead": "The target video is one unbroken handheld shot that begins from <Picture 1> in {room}.",
    "lead_continued": "The target video continues <Video 1> in one unbroken handheld shot that begins from <Picture 1> in {room}.",
    "audio_role": "The room tone of <Audio 1> carries on throughout.",
    "retain": "{subject} (appears in [Shot 1]): fully_preserved - {traits} are retained.",
    "retain_frame": "<Picture 1> ([Shot 1] first frame): fully_preserved - the video starts exactly from <Picture 1>'s framing, subject state and lighting.",
    "retain_video": "<Video 1> (continuation source): fully_preserved - the settled poses, handheld drift and light of its final moment continue without a reset.",
    "retain_audio": "<Audio 1>: reference - its room tone continues in the same character without copying the original signal.",
    "begin": "[Shot 1] The shot begins from <Picture 1>: {arrangement}.",
    "begin_continued": "[Shot 1] The shot begins from <Picture 1>, continuing <Video 1> without a reset: {arrangement}.",
    "room": "The setting is {room}, with {traits}, lit by {lights}.",
    "present": "{subject} is the {noun} with {traits}.",  # a member the prose and the closing never name
    "mouths": "Every visible mouth stays closed apart from {actions}.",
    "mouths_none": "Every visible mouth stays closed.",
    "line": "says in {voice}, <d>[{language}] {words}</d> The lips close as the line ends.",
    "settle": "{timing} the arrangement settles: {arrangement}, as the camera holds on this settled {shot} through the end of the video.",
    "room_sound": "{ambience} continues throughout.",
    "room_sound_continued": "The room sound of <Audio 1> continues throughout, the same {ambience}.",
}
# One take is a lone [Shot 1], so beats are timed in words, the way MiniMax's own one-take Context-IR
# output does it; "At MM:SS.mmm" is reserved for a cut.
TIMING = {
    "first": "For the first {seconds},",
    "airlock": "For the first {seconds} the arrangement holds with small natural motion:",
    "handoff": "Over about {seconds} in the middle of the clip,",
    "settle": "In the final {seconds}",
}
TIMING_BY_POSITION = (  # the rest, by where the beat starts (fraction of the clip)
    (0.0, "Early in the clip,"),
    (0.3, "In the middle seconds,"),
    (0.55, "Later in the clip,"),
    (0.75, "Toward the end of the clip,"),
)
TIMING_REPEAT = "A moment later,"
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
    "hold": "The camera holds its framing, shaking slightly as if handheld.",
    "push_in": "The camera pushes in with small amplitude at slow speed toward {toward}.",
    "pull_out": "The camera pulls out with small amplitude at slow speed, keeping {toward} in frame.",
    "reveal_left": "The camera pulls out with large amplitude at slow speed while trucking left, bringing {toward} into the frame.",
    "reveal_right": "The camera pulls out with large amplitude at slow speed while trucking right, bringing {toward} into the frame.",
    "pan_left": "The camera pans left with small amplitude at slow speed to follow {toward}.",
    "pan_right": "The camera pans right with small amplitude at slow speed to follow {toward}.",
    "truck_left": "The camera trucks left with small amplitude at slow speed, keeping {toward} centred.",
    "truck_right": "The camera trucks right with small amplitude at slow speed, keeping {toward} centred.",
    "tilt_down": "The camera tilts down with small amplitude at slow speed toward {toward}.",
}
MUSIC = "N/A"  # a score regenerated per chunk cannot be continuous; add music after the join


# --- judging: a keyframe or a clip against the shot it was made for ---------------------------------------
JUDGE_SYSTEM = """\
You are a continuity supervisor on a film set. You are shown a still, or a few frames of a clip in order, and told what the shot was meant to show and who is in it, with how each of them looks. You judge only what you can see. Score 1 when the picture shows the shot as described with every listed member present and looking exactly as described; score 0 when it shows something else, someone missing, or a member unrecognisable. Score in between for a picture that mostly shows it: name every mismatch you see as an issue, in a few words each."""

JUDGE_USER = """\
This is a {kind}: {count} picture(s), in order.

WHAT THE SHOT SHOWS
Purpose: {purpose}
Arrangement: {arrangement}

WHO IS IN IT
{cast}

Answer JSON: {{"score": 0.0-1.0, "cast": [{{"id": "<id>", "present": true|false, "likeness": 0.0-1.0}}, ...], "issues": ["..."], "notes": "one sentence"}}. One "cast" entry per listed member."""

# --- composing: a first frame from words, over a reference of the room ---------------------------------------
COMPOSE_INSTRUCTION = """\
Keep this room exactly as it is: its walls, furniture, lighting and colour. Change nothing about the camera. Add the people and things described, placed as described, in photographic detail: {scene}."""
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
You are the story editor of a live-action video. A planner turns your STORY into consecutive shots of equal length, and a video model renders each shot from the reference photographs, labelled Picture 1, Picture 2 and so on. The planner reads your words literally: what they do not say, it fills in from the photographs or leaves out. You receive the user's rough IDEA and the photographs, and you rewrite the IDEA into a STORY the planner renders well, as JSON.

WHAT GOES WRONG, AND THE RULES THAT PREVENT IT
1. The place, first and concretely. Unless the story says where it happens, the planner takes the place from Picture 1: a street became a bed because the only photograph was of a cat on a bed. "place" is the opening paragraph, 35 to 80 words: the place the IDEA names, with the time of day and the light, the ground and every surface the action touches, and what stands in the background. When no picture shows that place, describe it entirely in words and take nothing of it from the pictures. The same paragraph ends on the first frame: where each subject is, its pose, and what it is doing.
2. One beat per shot. "beats" has exactly as many entries as SHOTS says, one per shot, in order. A beat is the few seconds of visible action one shot shows: one to three plain sentences, no more words than SHOTS allows. Spread the IDEA's events over the shots in its order so every beat carries about the same amount of action: never two big events in one beat, and never a beat with nothing new happening. An action and its immediate result belong to the same beat (a stone thrown and the window it breaks): a shot that starts in mid-flight renders it again from the start.
3. Transformations, spelled out. When something turns into something else (burned, roasted, frozen, broken, eaten), the beat describes what it is afterwards, and then says in a sentence of its own that the old thing no longer exists anywhere in the scene, naming it as it was ("No ice is left anywhere.", "No whole vase remains anywhere."). Later beats name only the new thing. Otherwise the planner keeps the old thing in the later shots.
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

SHOTS: {count} shots of {seconds:g} seconds each. "beats" has exactly {count} entries, each at most {beat_words} words.
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

SHOTS: {count} shots of {seconds:g} seconds each. "beats" has exactly {count} entries, each at most {beat_words} words.
PICTURES: {pictures}. Picture 1 is where the video opens: the planner takes the place from it unless the story names another."""
