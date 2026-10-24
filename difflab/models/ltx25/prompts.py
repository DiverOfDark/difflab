"""What the VLM is told when it writes an LTX prompt."""

from diffusers.pipelines.ltx2.utils import I2V_DEFAULT_SYSTEM_PROMPT, LTX2_5_I2V_DEFAULT_SYSTEM_PROMPT

# Lightricks' own I2V enhancer prompt: the caption style 2.5 was trained on.
VLM_SYSTEM_PROMPT = LTX2_5_I2V_DEFAULT_SYSTEM_PROMPT
# Lightricks' enhancer prompt from before 2.5, which 2.3's captions follow.
VLM_SYSTEM_PROMPT_23 = I2V_DEFAULT_SYSTEM_PROMPT

# Licon's 2.3 examples say "Figure N", one line per reference with the scene last; here the first frame is the
# scene, so it gets no line.
VLM_SYSTEM_PROMPT_23_REFERENCES = (
    I2V_DEFAULT_SYSTEM_PROMPT
    + """
REFERENCE PICTURES (this overrides any instruction to output only the prompt): after the first frame you are given more pictures, in order. The first frame is the scene and has no Figure number. The second picture you see is Figure 1, the third is Figure 2, and so on: each shows a person or an object that must appear in the video looking exactly as it does in its picture, even where the first frame shows it from far away or not at all.

Before the prompt, write one line per reference picture and none for the first frame, in order, starting "Figure N: " and describing what it shows as a short comma-separated list. A person: apparent gender and age group, skin tone, hair, face, every garment, footwear and accessory. An object: what it is, its material and colour. End every line with the picture's rendering style, such as "realistic natural texture". Keep each line concise: too much or too little description both weaken the likeness.

Then a blank line, then the prompt. In it, call each referenced subject "Figure N" at its first appearance and whenever it acts, and say where it stands in the frame and what it does, so the model knows which picture each subject comes from. A subject the first frame already shows is the same subject as its Figure: describe it as one.
"""
)

# The labelling follows LiconStudio's MSR examples: an "Image N:" line per reference, then the scene naming them.
VLM_SYSTEM_PROMPT_REFERENCES = (
    LTX2_5_I2V_DEFAULT_SYSTEM_PROMPT
    + """
REFERENCE PICTURES (this overrides "Output ONLY the caption text"): after the first frame you are given more pictures, in order. The first frame has no Image number. The second picture you see is Image 1, the third is Image 2, and so on: each shows a person, an object or a place that must appear in the video looking exactly as it does in its picture, even where the first frame shows it from far away or not at all.

Before the caption, write one line per reference picture and none for the first frame, in order, starting "Image N: " and describing what it shows as a short comma-separated list. A person: apparent gender and age group, skin tone, hair, face, every garment, footwear and accessory. An object: what it is, its material and colour. A place: "Scene, " then its layout, materials and light. End every line with the picture's rendering style, such as "photorealistic natural texture".

Then a blank line, then the caption. In the caption, call each referenced subject "Image N" at its first appearance and whenever it acts, with a few words of what it wears ("Image 1 in the red linen dress"), so the model knows which picture each subject comes from. A subject the first frame already shows is the same subject as its Image: describe it as one.
"""
)
