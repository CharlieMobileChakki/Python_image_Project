"""ChatGPT Vision prompt generator for Flow B (Model-Worn Garment Extraction).

Analyzes a model-worn photograph via OpenAI Vision (gpt-4o-mini / gpt-4o) and builds
forensic-grade, textile-accurate descriptions combined with fixed 3D ghost-mannequin studio
rules for the Qwen-Image-2.1-Turbo GPU worker.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
from typing import Any

import httpx
from PIL import Image

logger = logging.getLogger(__name__)

# Fixed backend studio rules (empty 3D invisible ghost-mannequin product shot, Myntra/Meesho style)
OUTERWEAR_STUDIO_RULES = (
    "Pristine e-commerce 3D ghost-mannequin catalog presentation (Zara / Myntra / Hugo Boss / SuitSupply fashion hero shot): "
    "EXCLUSIVELY THE STANDALONE BLAZER / SUIT JACKET / COAT: "
    "CRITICAL HOLLOW APERTURE (ABSOLUTELY ZERO INNER SHIRT / ZERO T-SHIRT / ZERO TIE): "
    "This is an empty 3D ghost mannequin of ONLY the outerwear jacket. The chest opening, neckline, and space between the lapels MUST BE A HOLLOW, OPEN 3D APERTURE showing only the jacket's inner silk/viscose lining and open interior with zero human body and zero inner clothing. "
    "CRITICAL NEGATIVE RULE: ABSOLUTELY DO NOT generate, paint, or add any inner shirt, t-shirt, collared dress shirt, crew-neck, tie, or chest fabric fused inside the jacket! "
    "DO NOT fill the chest aperture with the jacket's own color! The jacket must be completely empty and hollow inside with its authentic lapel roll and structured tailored shoulders. "
    "MANDATORY POSE STRAIGHTENING: Centered upright, perfectly straight 0-degree direct frontal shot facing the camera straight-on, level horizontal shoulders, both tailored sleeves hanging straight down along both sides at symmetrical matching angles, crisp pressed lapels, and level bottom hemline. "
    "MANDATORY FABRIC TEXTURE & WEAVE LOCK: Reconstruct the authentic suit wool, linen blend, cotton twill, or velvet weave with genuine tailored structure, structured shoulder pads, and authentic lapel roll. "
    "Pure solid white background (#FFFFFF), completely free of any person, model, face, hair, neck, skin, arms, hands, legs, bed, mattress, bedsheet, blanket, carpet, floor, furniture, or shadows."
)

UPPER_STUDIO_RULES = (
    "Pristine e-commerce 3D ghost-mannequin catalog presentation (Zara / Myntra / Amazon Fashion hero shot): "
    "Empty hollow 3D inner neckline aperture, empty open sleeve cuffs, and empty open bottom hem showing the hollow interior with zero human body inside. "
    "MANDATORY FABRIC TEXTURE & WEAVE LOCK: Reconstruct the authentic microscopic textile micro-texture: whether fine 100% combed cotton single jersey knit, breathable honeycomb piqué knit, waffle knit, 2x2 rib knit, crisp poplin, or heavy diagonal twill denim, preserve the exact textile tactile grain and authentic hand without plastic smoothing. "
    "Seamlessly reconstruct the continuous authentic fabric pattern and textile weave across the chest, stomach, and sleeves. "
    "For pullover t-shirts and jerseys, maintain a completely solid, seamless front panel with ZERO buttons or vertical seams down the torso. "
    "MANDATORY POSE STRAIGHTENING: Centered upright, perfectly straight 0-degree direct frontal shot facing the camera straight-on, perfectly level horizontal shoulders, both sleeves hanging straight down along both sides at symmetrical matching angles, and strictly level horizontal hemline. "
    "SPOTLESS PRISTINE FABRIC & ZERO SMUDGE RULE: The fabric across the chest, torso, and ESPECIALLY BOTH SLEEVES must be solid, pristine, completely clean, and uniform in its authentic base color with ZERO dark gradient smudges, ZERO dirty gray airbrush marks, ZERO shadow bleeding from the bedsheet or folds, and ZERO armpit burn patches. Both sleeves must have perfectly clean, unblemished, matching fabric surfaces with balanced, identical studio illumination. Any contrast trim (like collar or cuff bands) must remain strictly confined to the thin outer edge/cuff hem and NEVER bleed or smudge onto the sleeves or torso. "
    "CHEST EMBLEMS & BRAND LOGOS: Faithfully preserve and replicate ONLY the authentic logos, badges, embroidery, and crests that are genuinely visible on the garment in the reference photo. If logos are present on either or both sides of the chest, reconstruct each logo in its exact position, proportions, and colors with sharp crisp detail. STRICTLY ZERO ADDED LOGOS / UNBRANDED: If the garment is plain without logos, keep it completely plain solid fabric with ABSOLUTELY NO chest logo, no crest, no emblem, no brand mark, and no text anywhere. Do NOT add, invent, or hallucinate any sportswear tick, luxury badge, or brand symbol. "
    "CRITICAL ISOLATION: COMPLETELY EXCLUDE, REMOVE, AND DO NOT GENERATE any pants, jeans, trousers, belt, legs, or bottom garments! The bottom half of the image MUST BE 100% empty, clean pure solid white (#FFFFFF). "
    "Pure solid white background (#FFFFFF), completely free of any person, model, face, hair, neck, skin, arms, hands, legs, bed, mattress, bedsheet, blanket, carpet, floor, furniture, or shadows."
)

LOWER_STUDIO_RULES = (
    "Pristine e-commerce 3D ghost-mannequin catalog presentation (Zara / Myntra / Amazon Fashion hero shot): "
    "Empty hollow 3D open waistband aperture at top, and empty open bottom ankle leg cuffs showing the hollow interior with zero human body inside. "
    "MANDATORY FABRIC TEXTURE & WEAVE LOCK: Reconstruct the authentic microscopic textile weave: whether heavy diagonal 3x1 twill denim with authentic slub yarns and natural wash fading, smooth cotton twill chino, crisp linen slub weave, or tailored wool blend suiting, preserve the exact real-world textile texture, tactile hand, and drape without plastic smoothing. "
    "Seamlessly reconstruct the authentic fabric weave, textile texture, waistband, front fly button/zipper, pockets, and full trouser leg length. "
    "MANDATORY POSE STRAIGHTENING: Centered upright, perfectly straight 0-degree direct frontal shot facing the camera straight-on, horizontally level waistband across the top, both trouser legs hanging completely straight down vertically parallel to each other side-by-side with zero knee bend and zero walking tilt, both ankle leg cuffs aligned at the exact same horizontal height. "
    "BRAND MARKS & POCKET TABS: Reconstruct authentic pocket tabs, rivet details, and brand patches only if genuinely visible on the garment. If plain, STRICTLY ZERO added patches, logos, or graphics. "
    "CRITICAL ISOLATION: ABSOLUTELY DO NOT GENERATE ANY SHIRT, TOP, T-SHIRT, JACKET, COLLAR, SHOULDERS, SLEEVES, CHEST, TORSO, OR UPPER BODY! "
    "The top half of the canvas MUST BE 100% empty, clean, pure solid white (#FFFFFF). Generate EXCLUSIVELY the standalone pants/trousers/skirt floating by themselves with NO upper garment attached! "
    "Pure solid white background (#FFFFFF), completely free of any person, model, face, hair, neck, skin, arms, hands, legs, bed, mattress, bedsheet, blanket, carpet, floor, furniture, or shadows."
)

FULL_OUTFIT_STUDIO_RULES = (
    "Pristine e-commerce 3D ghost-mannequin catalog presentation (Zara / Myntra / Amazon Fashion hero shot): "
    "Empty hollow 3D inner neckline aperture, empty open sleeve cuffs, and empty open bottom hem showing hollow interior with zero human body inside. "
    "MANDATORY FABRIC TEXTURE & WEAVE LOCK: Reconstruct the authentic microscopic textile weave, fiber hand, drape weight, and natural folding physics of the dress/ensemble. "
    "Whether fluid silk/satin with liquid sheen, delicate sheer georgette with translucent drape, heavy brocade/jacquard weave with raised metallic zari threads, "
    "textured organic linen with horizontal slub grain, crisp cotton poplin, or plush velvet pile, reproduce the exact real-world textile micro-texture, yarn grain, and authentic fabric drape without smoothing into plastic or artificial gloss. "
    "MANDATORY LOGO / BRAND MARK & EMBELLISHMENT LOCK: Faithfully preserve authentic embroidery, zari borders, lace trims, prints, and brand crests genuinely visible on the reference photo at their exact placement, scale, and colors without altering or redesigning. "
    "If the garment is plain/solid without logos, keep it completely plain with STRICTLY ZERO added logos, crests, badges, emblems, or text anywhere. Do NOT add, invent, or hallucinate any brand mark or symbol. "
    "MANDATORY POSE STRAIGHTENING: Centered upright, perfectly straight 0-degree direct frontal shot facing the camera straight-on, perfectly level horizontal shoulders, symmetrical silhouette, and plumb vertical alignment. "
    "SPOTLESS PRISTINE FABRIC & ZERO SMUDGE RULE: The fabric across the entire garment must be solid, pristine, completely clean, and uniform in its authentic base dye with ZERO dark gradient smudges, ZERO dirty gray airbrush marks, and ZERO shadow bleeding from the bedsheet, floor, or fold crevices. Balanced, uniform studio illumination. "
    "Pure solid white background (#FFFFFF), completely free of any person, model, face, hair, neck, skin, arms, hands, legs, bed, mattress, bedsheet, blanket, carpet, floor, furniture, or shadows."
)

COMMON_STUDIO_RULES = UPPER_STUDIO_RULES

SYSTEM_VISION_PROMPT = """You are a master e-commerce fashion catalog director and senior textile analyst creating image-editing prompts for high-end 3D ghost-mannequin catalog photography (Myntra, Ajio, Meesho, Zara, Amazon Fashion).

Your job is to analyze the clothing in the reference photograph (whether worn by a person OR laid flat on a bed, table, floor, or hanger) and produce a forensic-grade, textile-accurate description of each REAL stitched garment so that an AI diffusion image editor can reconstruct the EXACT standalone 3D garment without any human body parts and without any background surfaces.

CRITICAL RULES:

1. BACKGROUND & SURFACE DISCRIMINATION (BEDSHEETS, FLOORS, RUGS, FURNITURE):
   - The photo may show clothing laid flat on a bed, mattress, floor, table, or sofa.
   - STRICT RULE: The bedsheet, bed cover, blanket, quilt, floor rug, carpet, or wooden bed frame is 100% BACKGROUND FURNITURE.
   - NEVER, UNDER ANY CIRCUMSTANCES, MISTAKE A BEDSHEET OR BED COVER PATTERN FOR A GARMENT!
     * Even if the bedsheet has intricate geometric patterns (e.g., honeycomb hexagons, mandalas, block prints, florals, or decorative borders), it is STRICTLY BACKGROUND.
     * DO NOT extract the bedsheet as a shirt, kurta, top, dupatta, or fabric!
     * DO NOT transfer the bedsheet's pattern, prints, or colors onto any garment description!
   - ONLY analyze genuine, stitched, wearable clothing items (e.g., an olive green kurta laid on the bed).
   - If only ONE garment is present in the photo (e.g., only a kurta lying on a bed), return ONLY THAT SINGLE GARMENT in the "garments" list. DO NOT hallucinate or invent a second garment out of the bedsheet!

2. Precise Color Tone, Shade, Fabric Texture & Material Forensic Audit (SAME-TO-SAME REPLICATION):
   - Never use broad or generic bright color names like just "green", "olive-green", or "blue".
   - You must forensic-analyze the exact real-world shade, darkness, undertone, and saturation level:
     * If the fabric has a dark, muted, dusty, desaturated, or earthy tone (e.g., dusty army-olive, muted slate-sage, dark khaki-green, charcoal navy, faded terracotta, dusty peach), explicitly specify the EXACT muted shade: "deep muted dusty army-olive with desaturated matte greyish undertones, exactly preserving the dark muted dye tone and saturation of the reference photo without any artificial brightening, neon tint, or oversaturation".
     * If the fabric has a specific weave (e.g., textured cotton slub, raw handloom khadi, matte linen, twill denim, crepe, georgette), describe the exact matte weave and thread texture.
     * If plain solid, explicitly state: "pure solid matte fabric with uniform dye, no prints, no patterns, and no shiny gloss".
   - FORENSIC TEXTILE & FABRIC TEXTURE AUDIT (DRESSES, SAREES, LEHENGAS, KURTAS, TOPS, PANTS):
     * Always inspect and declare the microscopic textile texture and drape physics:
       - Dresses, Sarees, Lehengas & Gowns:
         * Mulberry Silk / Satin: "lustrous mulberry silk with fluid liquid drape and microscopic satin sheen".
         * Georgette / Chiffon: "delicate sheer translucent georgette with fine grainy crepe twist and soft floating drape".
         * Brocade / Jacquard / Zari: "rich woven brocade with raised structural jacquard relief and metallic gold/silver zari threads".
         * Velvet: "plush dense upright velvet pile with rich deep luminous highlights".
         * Organic Linen / Khadi: "textured raw linen with visible horizontal slub yarn grain and crisp matte hand".
         * Cotton Cambric / Poplin: "smooth compact combed cotton poplin with fine plain weave and clean drape".
       - T-Shirts, Polos & Sportswear:
         * Combed Jersey: "fine 100% combed cotton single jersey knit with subtle horizontal micro-ribs".
         * Piqué Knit: "breathable honeycomb waffle piqué knit texture (classic polo)".
         * Athletic Micro-Mesh: "breathable polyester performance knit with microscopic moisture-wicking pores".
       - Trousers, Chinos & Denim Jeans:
         * Denim: "heavy 14oz rigid 3x1 diagonal twill denim with authentic slub yarn grain, authentic wash fading, and contrast topstitching".
         * Chinos: "smooth compact cotton twill weave with solid uniform dye and clean flat-front finish".
   - MANDATORY LOGO / BRAND MARK AUDIT & ANTI-HALLUCINATION LOCK:
     * Declare `has_logo`: true if authentic brand logo, club crest, or graphic emblem is genuinely visible; false if plain/solid/unbranded.
     * Declare `logo_and_graphics`: exact placement, design, and colors of genuine visible markings, or "none (plain unbranded solid fabric with strictly zero logos, emblems, or text)".
     * STRICT ANTI-HALLUCINATION RULE: If `has_logo` is false, explicitly state: "STRICTLY ZERO ADDED LOGOS: Fabric is completely plain and unbranded. Do NOT add, invent, or hallucinate any brand logo, Nike/Adidas tick, luxury badge, crest, emblem, or text anywhere."
     * If genuine logos exist (e.g. club crest, Adidas 3-bars, polo pony): describe exact placement and colors without altering or redesigning.
   - ATHLETIC WEAR, SPORTS KITS & BRANDED APPAREL (CRITICAL ACCENT COLOR & LOGO FIDELITY):
     * NEVER guess or hallucinate colors or logos from brand memory or other product variants!
     * Deconstruct and specify each visible element's exact optical pixel colors:
       - Base fabric color: Exact visible color and tone (e.g., solid optic white, heather grey, jet black, crimson red).
       - Collar & Neckline Trim: Identify exact visible hue without shifting (e.g. if navy/dark blue, do not call it green; if black, do not call it brown).
       - Accents, Stripes & Piping: Specify exact color, count, and placement of any stripes or contrast bands.
       - Sleeve Cuffs: Specify matching trim or rib-knit colors.
       - DUAL CHEST EMBLEMS & BRAND LOGO AUDIT (MANDATORY ZERO DROPPING):
         * Inspect BOTH sides of the chest independently!
         * RIGHT CHEST (viewer's left): Look for any sportswear brand logo (e.g., Adidas 3-bars, Nike swoosh, Puma cat, Under Armour, Jordan, etc.). If present, declare its exact brand, shape, and color.
         * LEFT CHEST (viewer's right): Look for any team crest, club emblem, badge, or pocket graphic.
         * If BOTH emblems are present, you MUST explicitly describe BOTH side-by-side in garment_description so the generator includes both and drops neither!
         * If the garment is a plain casual shirt with NO logos, explicitly state "clean plain chest with no logos".
       - STRICT FORBIDDEN DRIFT: Forbid the generator from changing any accent colors or omitting visible logos.
   - TROUSERS, CHINOS, JEANS & BOTTOM WEAR (CRITICAL REAL-WORLD COLOR & TEXTILE ACCURACY):
     * Forensic-analyze the exact real-world color, wash, and fabric:
       - If the pants are light beige, khaki, cream, or sand (like chino trousers): explicitly specify "men's flat-front straight-leg chino trousers in smooth light beige / warm khaki cotton twill fabric, with uniform solid matte light beige dye, slant front pockets, and clean straight hems".
       - If dark / raw indigo / black jeans: explicitly specify "men's classic straight-leg denim jeans in uniform solid deep dark raw indigo denim wash with zero artificial thigh fading, no bleached thigh highlights, and no whiskering".
       - If olive, grey, white, brown, or navy: describe the exact authentic dye shade and weave without altering it.
     * Always match the real-world fabric weave (cotton twill, chino, denim, linen, khadi, wool blend).
     * Explicitly specify: "solid uniform dye from waistband to ankle cuffs, perfectly straight symmetrical legs".

3. MANDATORY POSE RECTIFICATION & STRAIGHTENING (E-COMMERCE CATALOG GHOST-MANNEQUIN STANDARD):
   - In e-commerce catalog standards (Zara, Myntra, Amazon, Ajio), product hero photos MUST be strictly front-facing, straight, and symmetrical.
   - Reference photos often show human models standing casually, walking, with bent knees, legs apart, tilted waistbands, hands in pockets, or bent arms.
   - STRICT RULE: YOU MUST EXPLICITLY INSTRUCT THE RECONSTRUCTION TO STRAIGHTEN AND RE-ALIGN THE GARMENT INTO A PERFECT 0-DEGREE STRAIGHT E-COMMERCE STUDIO 3D GHOST MANNEQUIN SHOT.
   - FOR LOWER BODY (PANTS, CHINOS, JEANS, TROUSERS, SKIRTS):
     * Explicitly specify: "OVERRIDE HUMAN BODY POSE: Straighten both legs completely! Do NOT copy the model's bent knee, tilted leg, or walking stance from the reference photo. Reconstruct both trouser legs hanging perfectly straight down, plumb, vertical, and parallel to each other side-by-side with zero bend and zero tilt. The waistband must be 100% horizontal and level across the top with an open 3D hollow waistband aperture, and both bottom leg cuffs must align at the exact same horizontal bottom height."
   - FOR UPPER BODY (SHIRTS, TOPS, KURTAS, JACKETS):
     * Explicitly specify: "OVERRIDE HUMAN BODY POSE: Perfectly level horizontal shoulders, both sleeves hanging naturally and straight down along both sides at symmetrical matching angles, front buttons/placket running in a plumb vertical line down the center, and a level horizontal bottom hem."
   - STRICT AUTHENTICITY ("DRESS WAHI REHNA CHAHIYE"):
     * While straightening the pose into a perfect e-commerce studio display, the garment itself (color, dye hue, fabric weave, texture, pockets, buttons, fly, hems) MUST match the reference photo 100% identically!

4. Silhouette & Construction Anatomy (T-SHIRT / FOOTBALL JERSEY vs BUTTONED SHIRT DISCRIMINATION):
   - FIRST CLASSIFY CLOSURE & SILHOUETTE TYPE:
     * PULLOVER T-SHIRT / SPORTS JERSEY / POLO / HOODIE:
       If the garment is a pullover, the front torso across chest, stomach, and belly MUST BE a single, continuous, solid seamless fabric panel!
       STRICT RULE: ABSOLUTELY ZERO full-length buttons running down the front, ZERO vertical button plackets down the belly, ZERO front zipper, and ZERO full-front openings!
       If there is a collar closure (e.g. henley button tab, polo slit, or mandarin neck slit with single button), it is strictly a short 1-to-2 inch neck opening at the throat only. Below the neck opening, the entire torso is completely smooth, solid, seamless pullover fabric.
     * FORMAL BUTTON-DOWN SHIRT / JACKET:
       Only if the garment genuinely has full-length buttons or a zipper running from collar down to hem, declare a front button placket.
   - Neckline / Collar: Exact shape (e.g., Mandarin stand band collar with small front slit, polo collar, round crew neck, V-neck).
   - Sleeves: Full-length, three-quarter, or half/short sleeves; note any cuff trim bands.
   - Slits & Hemline: Straight hemline, athletic curved hem, etc.

5. Outfit Mode & Separates (STRICT REALITY RULE):
   - LAYERED APPAREL (BLAZER / COAT / JACKET OVER SHIRT / T-SHIRT):
     * If the person in the photo is wearing a LAYERED OUTFIT (e.g. an outerwear blazer, suit jacket, coat, or cardigan worn over an inner collared shirt, t-shirt, or top):
       YOU MUST EXTRACT THEM AS SEPARATE GARMENTS in the "garments" list!
       1. Standalone Outerwear Entry: Category "upper_clothes" for the blazer/jacket alone.
          - Set `recognized_label`: "Studio Blazer" or "Studio Jacket".
          - In `garment_description`, describe ONLY the outer jacket (lapels, collar, button stance, exterior pockets, outer fabric weave, and inner lining).
          - Explicitly specify: "STANDALONE OUTERWEAR: The interior between the lapels must be an open, empty 3D hollow aperture showing only the jacket's inner lining. ABSOLUTELY NO inner shirt, NO t-shirt, NO tie, and NO fused chest fabric inside the jacket."
       2. Standalone Inner Top Entry: Category "upper_clothes" for the shirt/t-shirt alone.
          - Set `recognized_label`: "Studio Shirt" or "Studio T-Shirt".
          - In `garment_description`, reconstruct the COMPLETE standalone shirt/t-shirt with both shoulders, full chest panel, neckline, and sleeves completely unoccluded as if worn by itself without any jacket.
       3. Lower garment (if present): Category "pants".
   - IF AND ONLY IF the person is actually wearing contrasting separates (e.g. shirt + jeans) OR two distinct garments are clearly laid out:
     Output separate entries for the upper garment ("upper_clothes") and lower garment ("pants").
   - IF ONLY ONE GARMENT IS VISIBLE OR LAID OUT (e.g., a single kurta, a single dress, a single shirt):
     Output ONLY that one garment! Do NOT invent a second garment.
   - IF THE OUTFIT IS A ONE-PIECE ENSEMBLE (dress, gown, jumpsuit, saree, lehenga):
     Output a single entry with category "dress", "saree", or "lehenga".

6. JEWELRY & PRECIOUS ORNAMENTS (FORENSIC GEMOLOGIST & OBJECT DECONSTRUCTION):
   - If the reference photograph shows JEWELRY (necklaces, earrings, rings, bangles, crystals, diamonds, gemstones):
     ABSOLUTELY NEVER classify it as clothing!
     Categorize each piece strictly as 'necklace', 'earrings', 'ring', or 'bangles'.

   - EXACT PIECE COUNT & UNIT ENUMERATION (CRITICAL ZERO-HALLUCINATION RULE):
     * Count the EXACT number of physical jewelry pieces visible in the image!
     * If the photo shows a pair of 2 kadas: specify "item_count": 2, and "piece_descriptor": "exactly 2 matching broad cuff kada bangles (a pair of 2)".
     * If 1 ring: specify "item_count": 1, and "piece_descriptor": "1 standalone ring".
     * If 1 necklace: specify "item_count": 1, and "piece_descriptor": "1 standalone necklace".
     * If 1 pair of earrings: specify "item_count": 2, and "piece_descriptor": "exactly 1 matching pair of 2 earrings".
     * If a stack of 4 bangles: specify "item_count": 4, and "piece_descriptor": "a set of exactly 4 matching bangles".
     * NEVER leave the count ambiguous! An AI image generator WILL hallucinate extra pieces unless you explicitly lock the piece count!

   - PHYSICAL FORM FACTOR & BAND PROFILE (DIMENSIONAL ACCURACY):
     * For Bangles / Kadas:
       - Is it a broad thick cuff kada (wide cylindrical band, ~20-35mm wide, thick rigid circular walls)?
       - Or thin round wire bangles (~2-4mm thin round loops)?
       - Or a flexible chain / link bracelet?
       - NEVER describe broad cuff kadas as generic "bangles", because thin wire bangles look completely different!
       - Specify "form_factor": "broad thick cuff kada with wide cylindrical band (approx 25-30mm wide) and thick rigid circular walls".
     * For Earrings:
       - Specify top stud (solid gold dome/button, floral bead rosette, solitaire crystal, plain ball post) + main body (chandelier drop, jhumka bell, hoop, drop pendant). Explicitly note if the top stud is solid metal with NO center stone!
     * For Rings:
       - Band profile (wide cigar band, thin shank, split shank, pave band) + crown setting (solitaire, halo, cluster, bezel).
     * For Necklaces:
       - Choker (tight neck), collar (broad V/U collar), princess drape, long chain/haar.

   - SURFACE TEXTURE & ACCENT FIDELITY (ANTI-HALLUCINATION):
     * Only describe surface embellishments that are ACTUALLY CLEARLY VISIBLE in the reference photo!
     * DO NOT hallucinate bright multicolor enamel (meenakari) unless bright enamel paint is unmistakably present!
     * If the piece is solid engraved gold with relief floral work or filigree, describe it as:
       "Intricate solid antique gold relief floral and foliate hand-carvings with beaded borders. No loud multicolor enamel."

7. DEDICATED JEWELRY SETS vs INCIDENTAL JEWELRY DISCRIMINATION (STRICT 85% PROMINENCE RULE):
   - STRICT INCIDENTAL JEWELRY REJECTION:
     * When the photograph shows a person, model, or mannequin wearing APPAREL (dress, kurta, top, shirt, suit, pants, saree, lehenga), the user's primary intent is EXCLUSIVELY the clothing/apparel!
     * Casual or incidental jewelry worn by the person (such as small rings on fingers, ear studs, thin chains, wrist threads, or casual watches) MUST BE COMPLETELY IGNORED AND EXCLUDED!
     * DO NOT create any entry in the 'garments' list for incidental jewelry worn by an apparel model!
     * ONLY extract jewelry if:
       1. The photograph is a dedicated standalone macro jewelry showcase (no dominant apparel model), OR
       2. The jewelry item is the unmistakable primary hero subject occupying >= 15% of the frame with >= 85% visual clarity and prominence.
   - DEDICATED JEWELRY SETS (STANDALONE JEWELRY SHOWCASE ONLY):
     * If and only if the photograph is a dedicated standalone jewelry showcase containing a JEWELRY SET (e.g. Necklace AND matching Earrings, or Necklace + Earrings + Ring on display):
       Return a separate entry for each distinct jewelry piece in the "garments" list:
       - Entry 1: category "necklace" for the standalone necklace.
       - Entry 2: category "earrings" for the matching pair of earrings.
       - Entry 3: category "ring" (if a ring is also part of the set).
       Both pieces belong to the set and must be recognized and generated!

8. MANDATORY BOUNDING BOX LOCALIZATION (box_2d) FOR MACRO ZOOM & CROPPING:
   - For every detected piece (especially earrings in a jewelry set, rings, bangles, or necklaces), provide:
     "box_2d": [ymin, xmin, ymax, xmax] normalized on a 0-1000 scale.
   - For earrings in a necklace set, box_2d MUST tightly enclose BOTH EARRINGS TOGETHER side-by-side (the entire matching pair spanning the upper-central region between the necklace strands). Do NOT box only one earring!

9. DYNAMIC PRECIOUS METAL ALLOY & FORENSIC GEMSTONE COLOR AUDIT:
   - Identify the EXACT metal alloy from the reference photo: "22K Warm Yellow Gold", "18K Rose Gold", "Rhodium-Plated Silver", "Platinum", "Antique Matte Temple Gold", etc.
   - CRITICAL GEMSTONE & STONELESS AUDIT (ANTI-HALLUCINATION):
     * If the piece has NO stones (pure solid metal with filigree, engraving, or granulation):
       Set "gemstone_type": "none (pure solid precious metal with zero stones or crystals)".
       Explicitly state in garment_description that the top stud, body, and drops are 100% solid metal with ZERO added stones!
     * If the piece HAS stones / pearls / crystals:
       Identify every stone type, cut, and color (e.g. "Natural lustrous white seed pearl fringe", "Faceted blush-pink crystal drops", "Emerald green & ruby red cabochons", "Brilliant-cut clear diamond solitaire").
       Detail the exact position of each stone and its color palette with 100% forensic fidelity!

11. FOLDED, CRUMPLED, OR SIDELONG/TILTED CLOTHING (FULL UNWRAPPING & UNFOLDING MANDATE):
   - Real-world users frequently upload photos where clothing is folded diagonally, crumpled, bunched up, half-hidden, or rotated sideways/horizontally on a bed, sofa, or floor.
   - MANDATORY DE-FOLDING & SYMMETRY RESTORATION:
     * You MUST mentally unfold, unwrinkle, and steam-press the garment into its complete, full-size, fully spread-out 3D form.
     * If the photo shows a shirt folded in half or folded into a triangle with only ONE sleeve visible: You MUST extrapolate and reconstruct the COMPLETE shirt with BOTH MATCHING SLEEVES, fully opened torso, complete chest, and level hemline.
     * DO NOT generate a folded shirt! Reconstruct it 100% unfolded, fully expanded, and symmetrical.
   - STRICT ANTI-SHADOW & ANTI-BLEED DIRECTIVE (CRITICAL):
     * When clothing is folded or placed on high-contrast surfaces (e.g., black-patterned bedsheets, dark carpets), deep shadows and background colors often touch the fabric.
     * YOU MUST EXPLICITLY INSTRUCT THE GENERATOR TO REJECT ALL SHADOWS: The fabric (especially on both sleeves and chest) must be completely clean, crisp, and uniform. FORBID any dark gray gradients, dirty shadows, armpit smudges, or pigment bleed from bedsheets or fold crevices.
   - ORIENTATION RECTIFICATION:
     * Regardless of whether the photo is rotated horizontally (90 degrees sideways), diagonally, or upside down, the output MUST be strictly upright facing forward (0-degree vertical alignment, neckline at top, hemline at bottom).

12. Allowed category values:
   - For Apparel: "upper_clothes", "pants", "dress", "skirt", "saree", "lehenga"
   - For Jewelry: "necklace", "ring", "earrings", "bangles"

Return strictly valid JSON format:
{
  "garments": [
    {
      "category": "dress",
      "recognized_label": "Emerald Green Mulberry Silk Anarkali Dress with Zari Embroidery",
      "fabric_texture": "lustrous mulberry silk with fluid liquid drape and microscopic satin sheen",
      "surface_finish": "subtle silk luster with radiant highlights",
      "has_logo": false,
      "logo_and_graphics": "none (plain unbranded bodice with hand-embroidered golden zari floral motifs along neckline and hem; strictly zero brand logos or emblems)",
      "box_2d": [100, 200, 900, 800],
      "garment_description": "An authentic emerald green mulberry silk Anarkali dress featuring a tailored bodice with delicate golden zari embroidery around the round neckline and full-length sleeves, flowing into a generous flared kali skirt with embroidered border hem."
    }
  ]
}"""


def _encode_image_for_vision(image_rgb: Image.Image, max_edge: int = 1024) -> str:
    """Resize and base64-encode PIL RGB image as JPEG for fast high-detail Vision API calls."""
    img = image_rgb.convert("RGB")
    w, h = img.size
    if max(w, h) > max_edge:
        scale = max_edge / float(max(w, h))
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _build_final_prompt(
    garment_description: str,
    category: str,
    metal_type: str | None = None,
    gemstone_type: str | None = None,
    recognized_label: str | None = None,
    item_count: int | None = None,
    piece_descriptor: str | None = None,
    form_factor: str | None = None,
    fabric_texture: str | None = None,
    surface_finish: str | None = None,
    has_logo: bool | None = None,
    logo_and_graphics: str | None = None,
) -> str:
    """Put category-specific 3D ghost-mannequin isolation, e-commerce pose straightening, and fabric lock FIRST."""
    cat = (category or "").strip().lower()

    # Route jewelry categories directly to luxury jewelry studio rules
    if is_jewelry_domain(cat):
        stand = (
            "beige_velvet_brass_t_bar_stand" if cat in ("earrings", "jhumkas")
            else ("beige_velvet_brass_ring_block" if cat == "ring"
            else ("beige_velvet_brass_bangle_bar" if cat in ("bangles", "kada", "bracelet")
            else "beige_velvet_brass_neck_bust"))
        )
        # Dynamic metal inference if not explicitly provided
        inferred_metal = (metal_type or "").strip()
        if not inferred_metal:
            desc_lower = garment_description.lower()
            if "rose gold" in desc_lower:
                inferred_metal = "Polished Rose Gold Alloy"
            elif "yellow gold" in desc_lower or "22k" in desc_lower or "24k" in desc_lower:
                inferred_metal = "22K Warm Yellow Gold"
            elif any(k in desc_lower for k in ("silver", "rhodium", "white gold", "platinum")):
                inferred_metal = "Rhodium-Plated Silver / Platinum"
            elif "gold" in desc_lower:
                inferred_metal = "Polished Gold"
            else:
                inferred_metal = "Precious Metal Alloy"

        cnt = item_count or (2 if cat in ("earrings", "jhumkas") else 1)
        p_desc = piece_descriptor or (f"exactly {cnt} matching" if cnt > 1 else f"exactly {cnt} standalone")

        return _build_final_jewelry_prompt({
            "category": cat,
            "recognized_label": recognized_label or garment_description[:60],
            "item_count": cnt,
            "piece_descriptor": p_desc,
            "form_factor": form_factor or "",
            "metal_type": inferred_metal,
            "gemstone_type": (gemstone_type or "").strip(),
            "stand_type": stand,
            "jewelry_description": garment_description,
        })

    surface_rejection = (
        "STRICT SURFACE & SHADOW REJECTION: If the reference photo shows clothing placed on a bedsheet, mattress, blanket, floor, rug, carpet, or table, "
        "COMPLETELY IGNORE the bedsheet and its background patterns! Do NOT apply any bedsheet print, geometric shapes, or bedding colors to the garment. "
        "DO NOT BLEED ANY FOLD SHADOWS OR BACKGROUND COLORS ONTO THE FABRIC OR SLEEVES! Both sleeves must be 100% clean and pristine with zero dark smudges. "
        "If the clothing is folded diagonally or crumpled, FULLY UNFOLD and expand it into a complete, symmetrical, unwrinkled 3D garment. "
        "Reconstruct ONLY the authentic stitched garment itself."
    )

    desc_clean = (garment_description or "").strip()
    label_clean = (recognized_label or "").strip()
    full_text = f"{desc_clean} {label_clean}".lower()

    # Dynamic fabric texture clause
    fabric_txt = (fabric_texture or "").strip()
    finish_txt = (surface_finish or "").strip()
    fabric_parts = []
    if fabric_txt:
        fabric_parts.append(f"Authentic fabric texture and weave: {fabric_txt}")
    if finish_txt:
        fabric_parts.append(f"Surface finish: {finish_txt}")
    fabric_clause = (
        f"EXACT FABRIC TEXTURE LOCK: {'. '.join(fabric_parts)}. "
        "Reconstruct authentic microscopic textile grain, fiber hand, weave structure, and natural drape weight without smoothing into plastic or artificial gloss. "
        if fabric_parts
        else ""
    )

    # Dynamic logo & brand mark anti-hallucination clause
    has_explicit_logo = has_logo is True or any(
        k in full_text
        for k in ("logo", "crest", "emblem", "graphic", "print", "badge", "patch", "nike", "adidas", "puma", "under armour", "jordan")
    )
    if has_logo is False or (has_logo is None and not has_explicit_logo):
        logo_clause = (
            "STRICTLY ZERO ADDED LOGOS / UNBRANDED: The fabric is completely solid and unbranded. "
            "ABSOLUTELY DO NOT add, invent, or hallucinate any brand logo, Nike/Adidas tick, luxury badge, crest, emblem, or text anywhere on this garment. "
            "Keep the fabric completely plain, pristine, and unbranded. "
        )
    elif logo_and_graphics and logo_and_graphics.strip() and not any(k in logo_and_graphics.lower() for k in ("none", "no logo", "unbranded", "plain")):
        logo_clause = (
            f"EXACT LOGO & GRAPHICS LOCK: Faithfully preserve and replicate the authentic visible logo/crest/print ({logo_and_graphics.strip()}) "
            "in its exact scale, position, and colors without alteration, mutation, or distortion. "
        )
    else:
        logo_clause = (
            "CHEST EMBLEMS & LOGOS: Faithfully preserve and replicate only the authentic logos, badges, embroidery, and crests genuinely visible on the reference photo in their exact relative positions. If plain without logos, do NOT invent or add any. "
        )

    if cat == "upper_clothes":
        is_outerwear = any(
            k in full_text
            for k in (
                "blazer", "suit jacket", "jacket", "coat", "cardigan", "tuxedo",
                "shrug", "overcoat", "waistcoat", "nehru jacket", "bomber"
            )
        )
        is_pullover = not is_outerwear and any(
            k in full_text
            for k in (
                "t-shirt", "tshirt", "t shirt", "tee", "jersey", "polo", "hoodie", "sweatshirt",
                "pullover", "football", "soccer", "cricket", "basketball", "athletic", "track top"
            )
        )

        if is_outerwear:
            isolation_header = (
                "Generate an empty 3D ghost-mannequin e-commerce studio catalog product photograph (Zara / Myntra / Hugo Boss fashion hero shot) showing EXCLUSIVELY the standalone "
                "outerwear blazer / suit jacket / coat floating upright in a direct 0-degree straight front-facing view on a pure solid white background (#FFFFFF). "
                "The bottom half of the image MUST BE 100% EMPTY, clean pure solid white. "
                "CRITICAL HOLLOW APERTURE MANDATE (ZERO FUSED INNER SHIRT / ZERO T-SHIRT / ZERO TIE): "
                "This is an empty 3D ghost mannequin of ONLY the outerwear jacket itself! "
                "The interior space between the lapels, collar, and chest MUST BE AN OPEN, EMPTY 3D HOLLOW APERTURE showing only the jacket's inner silk/viscose lining with zero human body and zero inner clothing! "
                "CRITICAL: ABSOLUTELY DO NOT generate, paint, or add any inner shirt, collared dress shirt, t-shirt, crew-neck, tie, or chest fabric fused inside the jacket! "
                "ABSOLUTELY DO NOT fill or color the chest opening in the jacket's own color! The jacket must look like an authentic standalone unfastened or single-button jacket on an invisible 3D ghost mannequin with an empty hollow interior! "
                "No human body, NO head, NO face, NO hair, NO neck, NO skin, NO arms, NO hands, NO legs, NO feet, and NO pants. "
            )
            closure_item = (
                "3. Tailored lapels (notch, peak, or shawl) pressing cleanly and symmetrically; "
                "jacket front closure cleanly presented (either open showing hollow interior or single button fastened) with plumb center vertical alignment. "
                "The chest opening MUST BE 100% EMPTY, hollow, and open showing the jacket's inner lining — ABSOLUTELY ZERO inner shirt, ZERO t-shirt, ZERO necktie, and ZERO fused chest fabric."
            )
            studio_rules_to_use = OUTERWEAR_STUDIO_RULES
        else:
            isolation_header = (
                "Generate an empty 3D ghost-mannequin e-commerce studio catalog product photograph (Zara / Myntra / Amazon fashion hero shot) showing EXCLUSIVELY the standalone "
                "upper-body shirt / top / kurta floating upright in a direct 0-degree straight front-facing view on a pure solid white background (#FFFFFF). "
                "The bottom half of the image MUST BE 100% EMPTY, clean pure solid white. "
                "CRITICAL ISOLATION RULE: COMPLETELY EXCLUDE, REMOVE, AND DO NOT GENERATE any pants, jeans, trousers, belt, waistline, or legs! "
                "No human body, NO head, NO face, NO hair, NO neck, NO skin, NO arms, NO hands, NO legs, NO feet. "
            )
            if is_pullover:
                closure_item = (
                    "3. PULLOVER SOLID FRONT PANEL (ZERO FULL-LENGTH BUTTONS): This is an athletic pullover T-shirt / sports jersey. "
                    "The front torso across the chest, stomach, and belly MUST BE a solid, continuous, seamless fabric panel. "
                    "ABSOLUTELY NO full-length front buttons, NO vertical button placket running down the front, NO front zipper, and NO button-down shirt opening! "
                    "If there is a collar closure at the throat (such as a henley slit, small polo tab, or mandarin collar button), it is strictly a short 1-to-2 inch neck opening only; "
                    "below the neck opening, the entire chest, stomach, and torso MUST be 100% solid, smooth, seamless pullover fabric."
                )
            else:
                closure_item = (
                    "3. Front placket, buttons, or zipper (if present on the original garment) must run in a plumb, perfectly vertical straight line down the center."
                )
            studio_rules_to_use = UPPER_STUDIO_RULES

        pose_rectification = (
            "MANDATORY E-COMMERCE POSE STRAIGHTENING: "
            "DO NOT COPY ANY BENT ARM, HAND-IN-POCKET, OR TILTED POSTURE FROM THE REFERENCE PHOTO! "
            "Realign into a pristine symmetrical catalog display: "
            "1. Both shoulders must be at the exact same horizontal height (perfectly level shoulders). "
            "2. Both sleeves must hang straight down symmetrically on both sides along the torso at equal matching angles. "
            f"{closure_item} "
            "4. Bottom hemline must be completely straight and horizontally level across. "
            "5. Empty hollow 3D inner neckline aperture, open sleeve cuffs, and open bottom hem showing the hollow interior with zero human body inside. "
        )
        garment_lock = (
            "STRICT AUTHENTICITY & DESIGN CLONING (DRESS WAHI REHNA CHAHIYE): "
            "While straightening into the e-commerce studio pose, you MUST preserve the EXACT garment from the reference photo without altering it! "
            f"{fabric_clause}"
            f"{logo_clause}"
            "Faithfully replicate the identical real-world dye color, hue, lightness, textile weave, pattern motifs, collar shape, buttons, placket, and stitching directly from the reference photo. "
            "STRICT COLOR FIDELITY: Retain the authentic real-world colors of all fabric panels, contrast trims, collar borders, and stripes directly from the photo without any artificial hue shift or color substitution."
        )
        return (
            f"{isolation_header} "
            f"{pose_rectification} "
            f"{garment_lock} "
            f"{surface_rejection} "
            f"Exact standalone upper garment to isolate and faithfully reconstruct from reference photo: {desc_clean}. "
            f"{studio_rules_to_use}"
        )

    elif cat in ("pants", "skirt"):
        isolation_header = (
            "Generate an empty 3D ghost-mannequin e-commerce studio catalog product photograph (Zara / Myntra / Amazon fashion hero shot) showing EXCLUSIVELY the standalone "
            "lower-body pants / trousers / chinos / jeans / shorts / skirt floating upright in a direct 0-degree straight front-facing view on a pure solid white background (#FFFFFF). "
            "The top half of the image MUST BE 100% EMPTY, clean pure solid white. "
            "CRITICAL ISOLATION RULE: ABSOLUTELY DO NOT GENERATE ANY SHIRT, TOP, T-SHIRT, JACKET, COLLAR, SHOULDERS, SLEEVES, CHEST, TORSO, OR UPPER BODY! "
            "Generate EXCLUSIVELY the standalone lower garment by itself with NO upper body garment attached! "
            "No human body, NO head, NO face, NO hair, NO neck, NO skin, NO arms, NO hands, NO legs, NO feet. "
        )
        pose_rectification = (
            "MANDATORY E-COMMERCE POSE STRAIGHTENING (OVERRIDE HUMAN BODY POSE): "
            "DO NOT COPY THE MODEL'S CROOKED WALKING POSE, BENT KNEE, BOWED LEGS, OR ASYMMETRIC STANCE FROM THE REFERENCE PHOTO! "
            "You MUST completely straighten and realign the trousers into a pristine e-commerce 3D ghost-mannequin studio catalog hero photograph: "
            "1. STRAIGHT PARALLEL LEGS: Both trouser legs MUST hang completely straight down vertically, perfectly parallel to each other side-by-side with zero knee bend, zero curve, and zero walking tilt. "
            "2. HORIZONTALLY LEVEL WAISTBAND: The waistband at top MUST be 100% horizontal, straight, and level across, with an open circular/oval 3D hollow waistband aperture showing the empty interior with zero human body inside. "
            "3. EVEN BOTTOM CUFFS: Both left and right bottom ankle leg cuffs MUST align at the exact same horizontal bottom baseline height (completely even, parallel hems). "
            "4. NATURAL 3D MANNEQUIN VOLUME: Filled out with smooth, natural cylindrical 3D depth as if fitted on an invisible 3D mannequin, smooth steam-pressed ironed fabric without person-walking wrinkles or bent knee folds. "
        )
        garment_lock = (
            "STRICT AUTHENTICITY & DESIGN CLONING (DRESS WAHI REHNA CHAHIYE): "
            "While straightening the pose into the e-commerce studio display, you MUST preserve the EXACT garment from the reference photo without altering it! "
            f"{fabric_clause}"
            f"{logo_clause}"
            "Faithfully replicate the identical real-world dye color, hue, lightness, smooth fabric weave, textile texture, seams, front fly, button closure, belt loops, and pockets directly from the reference photo without changing the garment."
        )
        return (
            f"{isolation_header} "
            f"{pose_rectification} "
            f"{garment_lock} "
            f"{surface_rejection} "
            f"Exact standalone lower garment to isolate and faithfully reconstruct from reference photo: {desc_clean}. "
            f"{LOWER_STUDIO_RULES}"
        )

    else:
        isolation_header = (
            "Generate a perfectly straight 0-degree direct front-facing empty 3D ghost-mannequin e-commerce studio product photograph (Zara / Myntra catalog hero shot) showing the standalone "
            "outfit floating upright on a pure solid white background (#FFFFFF), with completely NO person, NO model, NO skin, NO human body. "
        )
        pose_rectification = (
            "MANDATORY E-COMMERCE POSE STRAIGHTENING: "
            "DO NOT COPY ANY WALKING POSE, TILTED HIP, OR BENT ARM FROM THE REFERENCE PHOTO! "
            "Straighten the outfit into a perfectly plumb, symmetrical 0-degree direct frontal e-commerce studio catalog hero photograph with level horizontal shoulders, symmetrical silhouette, and smooth 3D hollow ghost mannequin volume. "
        )
        garment_lock = (
            "STRICT AUTHENTICITY & DESIGN CLONING (DRESS WAHI REHNA CHAHIYE): "
            "While straightening into the e-commerce studio pose, you MUST preserve the EXACT garment from the reference photo without altering it! "
            f"{fabric_clause}"
            f"{logo_clause}"
            "Faithfully replicate the identical real-world color, hue, fabric weave, textile texture, embroidery, prints, and construction directly from the reference photo without changing the garment."
        )
        return (
            f"{isolation_header} "
            f"{pose_rectification} "
            f"{garment_lock} "
            f"{surface_rejection} "
            f"Exact standalone garment to reconstruct from reference photo: {desc_clean}. "
            f"{FULL_OUTFIT_STUDIO_RULES}"
        )


def _fallback_garment_plan(target_garment: str = "all") -> list[dict[str, Any]]:
    """Deterministic fallback prompt if OpenAI API key is absent or unavailable."""
    t = (target_garment or "all").strip().lower()
    mapping = {
        "upper_clothes": (
            "upper_clothes",
            "Top / Shirt / Upper Garment",
            "the upper-body garment (top, shirt, kurta, jacket, or blouse) with its exact original colors, microscopic fabric weave, and print motifs",
            "fine combed cotton single jersey knit or crisp poplin weave",
            "matte natural finish",
            False,
            "none (plain solid unbranded fabric with zero added logos)",
        ),
        "pants": (
            "pants",
            "Pants / Jeans / Lower Garment",
            "the lower-body trousers, pants, or jeans with original color, authentic twill weave, and tailored fit",
            "heavy diagonal cotton twill or authentic denim slub weave",
            "matte natural finish",
            False,
            "none (plain unbranded waistband and legs with zero added logos)",
        ),
        "dress": (
            "dress",
            "Dress / Kurta / Full Outfit",
            "the full-length dress, kurta, or outfit preserving exact original prints, embroidery, fabric drape, and neckline",
            "fluid silk, georgette, or fine woven textile with authentic drape and micro-texture",
            "subtle natural luster",
            False,
            "none (authentic embroidery/prints only with zero brand logos)",
        ),
        "skirt": (
            "skirt",
            "Skirt",
            "the skirt with original pattern, fabric weave, and borders",
            "fine woven textile with natural pleats and drape",
            "matte natural finish",
            False,
            "none (plain fabric with zero brand logos)",
        ),
        "saree": (
            "saree",
            "Saree",
            "the complete draped saree with borders, pallu print, and authentic silk/georgette weave",
            "lustrous mulberry silk or fluid sheer georgette with metallic zari borders",
            "radiant silk sheen",
            False,
            "none (traditional zari motifs only with zero modern brand logos)",
        ),
        "lehenga": (
            "lehenga",
            "Lehenga / Ethnic Set",
            "the complete lehenga ensemble with intricate embroidery, borders, and authentic fabric drape",
            "rich brocade jacquard or silk-blend with raised zari threadwork",
            "luminous festive luster",
            False,
            "none (traditional embroidery only with zero brand logos)",
        ),
    }
    cat, label, desc, tex, finish, has_lg, lg_desc = mapping.get(
        t,
        (
            "dress",
            "Studio Garment",
            "the primary garment worn by the person preserving exact original colors, textile weave, and print motifs",
            "fine woven textile with authentic texture and drape",
            "natural matte finish",
            False,
            "none (plain unbranded fabric with zero added logos)",
        ),
    )
    return [
        {
            "category": cat,
            "recognized_label": label,
            "garment_description": desc,
            "fabric_texture": tex,
            "surface_finish": finish,
            "has_logo": has_lg,
            "logo_and_graphics": lg_desc,
            "prompt": _build_final_prompt(
                garment_description=desc,
                category=cat,
                recognized_label=label,
                fabric_texture=tex,
                surface_finish=finish,
                has_logo=has_lg,
                logo_and_graphics=lg_desc,
            ),
            "prompt_source": "default_fallback",
        }
    ]


def analyze_garment_and_build_prompts(
    image_rgb: Image.Image,
    target_garment: str = "all",
    custom_prompt: str | None = None,
) -> list[dict[str, str]]:
    """Analyze a model-worn photo using ChatGPT Vision and build Qwen generation prompts.

    Args:
        image_rgb: Original model-worn PIL RGB image.
        target_garment: Optional category filter ('all', 'upper_clothes', 'pants', 'dress', 'skirt', 'saree', 'lehenga').
        custom_prompt: Optional caller-supplied prompt override.

    Returns:
        List of garment prompt dicts: [{category, recognized_label, garment_description, prompt, prompt_source}].
    """
    if custom_prompt and custom_prompt.strip():
        cat = target_garment if target_garment and target_garment != "all" else "dress"
        return [
            {
                "category": cat,
                "recognized_label": "Custom Garment Extraction",
                "garment_description": custom_prompt.strip(),
                "prompt": f"{custom_prompt.strip()} {COMMON_STUDIO_RULES}",
                "prompt_source": "custom_prompt",
            }
        ]

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        logger.warning("OPENAI_API_KEY is not set; using default garment prompt fallback.")
        return _fallback_garment_plan(target_garment)

    model_name = os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini").strip()
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    b64_jpeg = _encode_image_for_vision(image_rgb)

    target_norm = (target_garment or "all").strip().lower()
    if target_norm == "upper_clothes":
        user_instruction = (
            "Analyze ONLY the UPPER-BODY garment (shirt, t-shirt, top, kurta, jacket) in this image. "
            "Completely IGNORE and exclude any pants, jeans, trousers, or bottom wear. "
            "STRICT BACKGROUND & FOLDING RULE: If the clothing is placed on a bedsheet, mattress, blanket, floor, or table, "
            "the bedsheet / floor is 100% BACKGROUND. Do NOT mistake bedsheet patterns for garment motifs! "
            "CRITICAL DE-FOLDING & UNWRAPPING: If the garment is folded (e.g. folded diagonally into a triangle, only one sleeve visible, or rotated sideways on a bed), "
            "you MUST detect that it is folded and instruct the full unfolding into a complete, symmetrical 3D ghost mannequin with BOTH matching sleeves, complete torso, and level hemline. "
            "CRITICAL COLOR & FABRIC FIDELITY: Identify the EXACT real-world dye shade, darkness, and saturation. If it is dark, muted, dusty, or desaturated (e.g., dusty army-olive, slate-khaki, deep moss), describe that exact deep muted matte shade without brightening or oversaturating. "
            "CRITICAL SILHOUETTE (PULLOVER T-SHIRT / JERSEY vs SHIRT): If this is a T-shirt, athletic jersey, or polo, it is a PULLOVER! The entire front torso must be a solid, seamless continuous panel of fabric with ZERO buttons running down the chest/stomach. "
            "MANDATORY CHEST EMBLEM & LOGO AUDIT: Inspect both sides of the chest independently! If brand logos (e.g., Nike, Adidas, Puma, etc.) or team/club crests are visible on either or both sides of the chest, you MUST describe each emblem's exact brand name, placement, and colors so no emblem is dropped. If the chest is plain, state 'plain chest with no logos'. "
            "CRITICAL ACCENT COLOR ACCURACY: Forensic-analyze the exact colors of collar trim, stripes, patterns, and sleeve cuffs directly from visible pixels without shifting or substituting hues. "
            "STRICT SPOTLESS FABRIC & SHADOW REJECTION: The fabric (especially on both sleeves) must be completely clean with ZERO dark smudges, ZERO dirty gray airbrush patches, and ZERO shadow bleed from bedsheets or fold crevices. Any dark collar/cuff trim must stay strictly at the outer edge hem. "
            "CRITICAL POSE STRAIGHTENING (E-COMMERCE CATALOG STANDARD): Reconstruct with level horizontal shoulders, symmetrical hanging sleeves along both sides at matching angles, and vertical plumb alignment, while preserving the exact original fabric, color, prints, and collar ('dress wahi rehna chahiye')."
        )
    elif target_norm in ("pants", "skirt"):
        user_instruction = (
            "Analyze ONLY the LOWER-BODY garment (pants, jeans, trousers, shorts, skirt) in this image. "
            "Completely IGNORE and exclude any upper-body shirt, top, or kurta. "
            "STRICT BACKGROUND RULE: If placed on a bedsheet, blanket, or floor, IGNORE the bedsheet completely. "
            "CRITICAL COLOR & FABRIC FIDELITY: Identify the EXACT real-world dye wash, darkness, and saturation level without altering the color. "
            "CRITICAL POSE STRAIGHTENING (E-COMMERCE CATALOG STANDARD): In e-commerce ghost mannequin catalog shots, the pant MUST NOT have bent knees or tilted legs! Explicitly command both legs to be straightened out completely to hang vertically parallel to each other side-by-side with zero knee bend and zero walking tilt, with horizontally level waistband across the top and level ankle cuffs, while preserving the exact original fabric, color, pockets, and stitching ('dress wahi rehna chahiye')."
        )
    elif target_norm in ("dress", "saree", "lehenga"):
        user_instruction = (
            f"Analyze the full {target_norm} in this image. "
            "STRICT BACKGROUND RULE: If placed on a bedsheet, blanket, or floor, IGNORE the background bedsheet completely. "
            "CRITICAL COLOR & FABRIC TEXTURE FIDELITY: Inspect microscopic textile weave (mulberry silk, fluid satin, georgette, chiffon, velvet, linen slubs, brocade zari, cotton cambric). Describe exact fabric texture, surface sheen, and drape weight. "
            "CRITICAL EMBROIDERY & LOGO AUDIT: Distinguish authentic traditional embroidery, zari borders, prints, and embellishments from modern logos. If plain, declare 'strictly zero logos or brand marks'. If embroidered, specify exact zari motifs and thread colors. "
            "CRITICAL POSE STRAIGHTENING: Reconstruct in a plumb, symmetrical 0-degree straight e-commerce studio catalog display while preserving all authentic garment details."
        )
    else:  # "all"
        user_instruction = (
            "Analyze the item(s) in this image for exact 3D studio catalog reconstruction. "
            "CRITICAL DOMAIN DISCRIMINATION & 85% PROMINENCE RULE (APPAREL vs JEWELRY): "
            "1. PRIMARY APPAREL LOCK: If the photo shows a model or person wearing APPAREL (dress, kurta, top, suit, shirt, pants, saree, lehenga), "
            "   extract EXCLUSIVELY the clothing/apparel items! "
            "   STRICTLY IGNORE and DO NOT extract any casual/incidental jewelry (such as small rings on fingers, ear studs, thin bracelets, wrist threads) worn by the model. "
            "   Only return genuine clothing categories ('dress', 'upper_clothes', 'pants', 'saree', 'lehenga', 'skirt'). "
            "2. STANDALONE JEWELRY SHOWCASE: ONLY if the photo is a dedicated macro jewelry showcase (NOT a casual accessory on a clothed person): "
            "   - If a dedicated jewelry set (e.g. Necklace AND matching Earrings on display): return multiple entries (necklace, earrings, ring). "
            "   - If single jewelry item: categorize strictly as 'necklace', 'earrings', 'ring', or 'bangles' with high-precision gemologist deconstruction. "
            "   - Enforce >= 85% prominence: only extract jewelry if it is the primary hero subject with >= 85% visual clarity and prominence. "
            "3. If the photo contains APPAREL: Categorize as 'upper_clothes', 'pants', 'dress', 'skirt', 'saree', or 'lehenga' with full textile weave, dye darkness, and pose straightening. "
            "   CRITICAL PULLOVER vs SHIRT & COLOR LOCK: If it is a T-shirt, athletic jersey, or polo, classify it strictly as a PULLOVER with a solid seamless front panel (ZERO buttons down torso). Inspect both sides of the chest for any authentic brand logos, crests, or emblems and describe all of them without omitting any. "
            "   CRITICAL DE-FOLDING & SPOTLESS FABRIC: If the garment is folded diagonally or crumpled on a bed/floor, command full de-folding into a complete symmetrical 3D garment with pristine unsmudged sleeves. "
            "4. Surface Rejection: If placed on a bedsheet, blanket, table, or floor, the surface is 100% BACKGROUND."
        )
    payload: dict[str, Any] = {
        "model": model_name,
        "temperature": 0.1,
        "max_tokens": 1000,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM_VISION_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_instruction},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64_jpeg}",
                            "detail": "high",
                        },
                    },
                ],
            },
        ],
    }

    try:
        with httpx.Client(timeout=35.0) as client:
            resp = client.post(
                f"{base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        raw_content = data["choices"][0]["message"]["content"]
        parsed = json.loads(raw_content)
        items = parsed.get("garments", [])
        if not isinstance(items, list) or not items:
            logger.warning("ChatGPT Vision returned empty garments list; falling back.")
            return _fallback_garment_plan(target_garment)

        target_norm = (target_garment or "all").strip().lower()
        results: list[dict[str, Any]] = []

        APPAREL_CATS = {"dress", "upper_clothes", "pants", "skirt", "saree", "lehenga"}
        JEWELRY_CATS = {"necklace", "ring", "earrings", "bangles"}
        has_apparel = any(str(it.get("category", "")).strip().lower() in APPAREL_CATS for it in items)

        for item in items[:4]:
            cat = str(item.get("category", "dress")).strip().lower()
            label = str(item.get("recognized_label", "Studio Garment")).strip()
            desc = str(item.get("garment_description", "")).strip()
            metal = str(item.get("metal_type", "")).strip() or None
            gemstone = str(item.get("gemstone_type", "")).strip() or None
            item_cnt = int(item.get("item_count", 0) or 0) or None
            piece_desc = str(item.get("piece_descriptor", "")).strip() or None
            form_fact = str(item.get("form_factor", "")).strip() or None
            fabric_tex = str(item.get("fabric_texture", "")).strip() or None
            surf_finish = str(item.get("surface_finish", "")).strip() or None
            has_logo_val = bool(item.get("has_logo", False)) if "has_logo" in item else None
            logo_gfx = str(item.get("logo_and_graphics", "")).strip() or None
            box_2d = item.get("box_2d")

            # Incidental jewelry filter: when apparel is present, drop minor/incidental jewelry (<85% prominence)
            if cat in JEWELRY_CATS and has_apparel and target_norm not in JEWELRY_CATS:
                prominence = int(item.get("prominence_score") or item.get("clarity_percentage") or 0)
                is_primary = bool(item.get("is_primary_subject", False))
                box_area = 1.0
                if isinstance(box_2d, list) and len(box_2d) == 4:
                    ymin, xmin, ymax, xmax = box_2d
                    box_area = ((ymax - ymin) / 1000.0) * ((xmax - xmin) / 1000.0)
                if prominence < 85 or box_area < 0.15 or not is_primary:
                    logger.info(
                        "Suppressed incidental jewelry item '%s' (category=%s, prominence=%d, box_area=%.3f) in favor of primary apparel.",
                        label,
                        cat,
                        prominence,
                        box_area,
                    )
                    continue

            if not desc:
                continue
            if target_norm != "all" and cat != target_norm:
                continue
            results.append(
                {
                    "category": cat,
                    "recognized_label": label,
                    "garment_description": desc,
                    "metal_type": metal or "",
                    "gemstone_type": gemstone or "",
                    "item_count": item_cnt,
                    "piece_descriptor": piece_desc,
                    "form_factor": form_fact,
                    "fabric_texture": fabric_tex or "",
                    "surface_finish": surf_finish or "",
                    "has_logo": has_logo_val,
                    "logo_and_graphics": logo_gfx or "",
                    "box_2d": box_2d,
                    "prompt": _build_final_prompt(
                        garment_description=desc,
                        category=cat,
                        metal_type=metal,
                        gemstone_type=gemstone,
                        recognized_label=label,
                        item_count=item_cnt,
                        piece_descriptor=piece_desc,
                        form_factor=form_fact,
                        fabric_texture=fabric_tex,
                        surface_finish=surf_finish,
                        has_logo=has_logo_val,
                        logo_and_graphics=logo_gfx,
                    ),
                    "prompt_source": f"openai:{model_name}",
                }
            )

        # If target filter eliminated all items, keep the first detected garment rather than failing
        if not results and items:
            first = items[0]
            cat = str(first.get("category", "dress")).strip().lower()
            label = str(first.get("recognized_label", "Studio Garment")).strip()
            desc = str(first.get("garment_description", "the primary garment")).strip()
            metal = str(first.get("metal_type", "")).strip() or None
            gemstone = str(first.get("gemstone_type", "")).strip() or None
            fabric_tex = str(first.get("fabric_texture", "")).strip() or None
            surf_finish = str(first.get("surface_finish", "")).strip() or None
            has_logo_val = bool(first.get("has_logo", False)) if "has_logo" in first else None
            logo_gfx = str(first.get("logo_and_graphics", "")).strip() or None
            box_2d = first.get("box_2d")
            results.append(
                {
                    "category": cat,
                    "recognized_label": label,
                    "garment_description": desc,
                    "metal_type": metal or "",
                    "gemstone_type": gemstone or "",
                    "fabric_texture": fabric_tex or "",
                    "surface_finish": surf_finish or "",
                    "has_logo": has_logo_val,
                    "logo_and_graphics": logo_gfx or "",
                    "box_2d": box_2d,
                    "prompt": _build_final_prompt(
                        garment_description=desc,
                        category=cat,
                        metal_type=metal,
                        gemstone_type=gemstone,
                        recognized_label=label,
                        fabric_texture=fabric_tex,
                        surface_finish=surf_finish,
                        has_logo=has_logo_val,
                        logo_and_graphics=logo_gfx,
                    ),
                    "prompt_source": f"openai:{model_name}",
                }
            )

        return results or _fallback_garment_plan(target_garment)

    except Exception as exc:
        logger.warning("ChatGPT Vision prompt generation failed (%s); using fallback prompt.", exc)
        return _fallback_garment_plan(target_garment)


# ==============================================================================
# JEWELRY & PRECIOUS ORNAMENTS 3D STUDIO PIPELINE (MULTI-ANGLE AWARE)
# ------------------------------------------------------------------------------
# Specialized 3D display stand generation for Necklaces, Rings, Earrings, Bangles
# Supports single or multi-angle input fusion (up to 4 reference photos).
# Mounts authentic gold, diamond, and gemstone ornaments on luxury catalog stands:
# ==============================================================================
# JEWELRY & PRECIOUS ORNAMENTS 3D LUXURY SHOWCASE STAND RULES
# ------------------------------------------------------------------------------
# Mounts authentic gold, diamond, and gemstone ornaments on luxury boutique stands:
# - Earrings/Jhumkas: Sculpted Warm Beige Velvet Wishbone Wings + Polished Gold Brass Stem & Base (User Reference Standard)
# - Necklaces/Chokers: Sculpted Warm Beige Velvet Neck Bust Mannequin + Polished Gold Brass Pedestal
# - Rings/Bands: Angled 3D Warm Beige Velvet / Polished Brass Slotted Ring Display Block
# - Bangles/Kadas: Horizontal Cylindrical Warm Beige Velvet Bangle Display Bar on Polished Brass Uprights
# ==============================================================================

JEWELRY_NECKLACE_STUDIO_RULES = (
    "Pristine luxury jewelry boutique showcase catalog photograph (Tanishq / Kalyan Jewellers / Cartier luxury hero shot). "
    "LUXURY SHOWCASE STAND & CONTRAST (BEIGE VELVET & POLISHED GOLD BRASS): The standalone authentic necklace is mounted and draped elegantly around the neck of a sculpted luxury boutique neck bust mannequin upholstered in premium soft warm beige / champagne micro-suede velvet, anchored on a weighted circular pedestal rimmed in gleaming polished gold brass. "
    "The warm beige velvet texture and gleaming polished brass accents provide rich three-dimensional tactile depth and soft contact shadows that stand out with breathtaking elegance against the pure solid white studio background (#FFFFFF), ensuring the stand structure never blends or washes into the background! "
    "MANDATORY E-COMMERCE DRAPE SYMMETRY (OVERRIDE REFERENCE POSE): Regardless of whether the reference photo shows the necklace laid flat, coiled, or tangled on a surface: "
    "1. Centered upright, direct 0-degree straight frontal studio camera view. "
    "2. The necklace loops naturally and symmetrically around both sides of the contoured warm beige velvet neck bust with equal curve, drape, and tension. "
    "3. The central medallion / pendant rests completely flat, centered, and plumb along the vertical center axis of the bust chest. "
    "4. Completely untangled, zero twists, zero asymmetry. "
    "PRECIOUS METAL & POLISH LUSTER: Faithfully preserve the authentic precious metal alloy (rose gold, yellow gold, platinum, or rhodium-plated silver), gemstone facet cuts, crystal colors, and prong settings with 100% forensic accuracy directly from the reference photo(s). DO NOT change the metal alloy or gemstone colors! "
    "8K ULTRA-HD MACRO RESOLUTION (EXTREME ZOOM CLARITY): Razor-sharp microscopic jewelry studio illumination. Every individual filigree thread, openwork jaali wire, embossed leaf vein, polished gold bead drop, and metal reflection must be rendered in ultra-crisp, tack-sharp microscopic focus with ZERO blur and ZERO pixelation even when deeply zoomed in. "
    "SOLID STUDIO WHITE BACKGROUND (STAND INTACT): Pure solid pristine studio white background (#FFFFFF). The entire luxury beige velvet and brass neck bust display stand MUST remain 100% intact, fully visible, and seamlessly grounded on the pure solid white background with subtle realistic contact shadows. DO NOT make transparent and DO NOT cut or remove the display stand! Completely free of any human model, skin, neck, face, hands, clothes, table clutter, or props."
)

JEWELRY_RING_STUDIO_RULES = (
    "Pristine luxury jewelry boutique showcase catalog photograph (Tiffany & Co. / BlueStone / CaratLane luxury hero shot). "
    "LUXURY SHOWCASE STAND & CONTRAST (BEIGE VELVET & POLISHED GOLD BRASS): The standalone ring is mounted firmly upright inside the cushioned slot of a luxury angled warm beige / champagne velvet ring display block with polished gold brass metallic side trim. "
    "The block has a distinct, refined warm beige velvet finish with gleaming gold metallic edges clearly defined against the pure solid white background (#FFFFFF) so the stand never washes into the background! "
    "MANDATORY E-COMMERCE RING MOUNTING (OVERRIDE REFERENCE POSE): Regardless of whether the reference photo shows the ring lying flat on its side: "
    "1. OPTIMAL 3D HERO PERSPECTIVE: Direct 30-degree to 45-degree three-quarter isometric 3D beauty angle. "
    "2. CRITICAL SHANK & CROWN VISIBILITY: BOTH the top crown / center gemstone / head motif AND the intricate side shank carvings, side filigree, gallery jali, and band engravings must be SIMULTANEOUSLY prominent and clearly visible in this single photograph! "
    "3. Upright and untilted inside the display slot. "
    "PRECIOUS METAL & GEMSTONE FIDELITY: Reconstruct the exact precious metal alloy (22K/18K warm yellow gold, platinum, or rose gold), micro-pave prong settings, diamond facets, and side band engravings with 100% forensic precision from all reference angles. "
    "8K ULTRA-HD MACRO RESOLUTION (EXTREME ZOOM CLARITY): Razor-sharp macro jewelry studio illumination with crisp facet reflections and tack-sharp microscopic focus. Every prong, facet, and carved metal grain must be crystal clear with zero blur when zoomed in. "
    "SOLID STUDIO WHITE BACKGROUND (STAND INTACT): Pure solid clean studio white background (#FFFFFF). The beige velvet and brass ring display block must remain 100% intact and visible on the solid white background. DO NOT make transparent. Zero human fingers, NO hands, NO skin, and NO table props."
)

JEWELRY_EARRINGS_STUDIO_RULES = (
    "Pristine luxury jewelry boutique showcase catalog photograph (Amazon Luxury / Tanishq / Kalyan Jewellers luxury hero shot). "
    "LUXURY SHOWCASE STAND (BEIGE VELVET WISHBONE WINGS + POLISHED GOLD BRASS STEM & BASE): "
    "The pair of matching earrings is displayed mounted on a luxury custom jewelry boutique showcase display stand: "
    "A sculpted, contoured wishbone / dual-wing earring display cross-mount upholstered in premium soft warm beige / champagne micro-suede velvet, supported by a polished gleaming gold brass metallic vertical post/stem, anchored into a weighted circular base with matching warm beige velvet top inlay rimmed in polished gleaming gold brass. "
    "The warm beige velvet wings and gleaming gold brass hardware provide rich three-dimensional tactile depth and soft contact shadows that stand out with breathtaking elegance against the pure solid white studio background (#FFFFFF), ensuring the stand structure never blends or washes into the white background! "
    "MANDATORY E-COMMERCE DUAL-WING MOUNTING & SYMMETRICAL AIRBORNE SUSPENSION (OVERRIDE REFERENCE POSE): "
    "DO NOT COPY ANY ASYMMETRIC, TILTED, LEANING, OR LYING-FLAT POSE FROM THE REFERENCE PHOTO! "
    "Even if the reference photo shows the earrings lying flat on a surface/cloth, tilted at an angle, or overlapping: "
    "1. WING-MOUNTED TOP STUDS / POSTS: Both earrings are mounted symmetrically onto the stand's sculpted beige velvet wings. Faithfully replicate the EXACT top ear-stud / post motif directly from the reference photo (whether a solid gold domed sphere, floral bead rosette, engraved button, or gemstone solitaire)—pinned cleanly on the front vertical faces of the left and right velvet wings. If the reference top stud is solid metal with NO stone in the center, replicate pure solid metal with ZERO added stones! If it has a gemstone or crystal, replicate that exact stone! "
    "2. AIRBORNE FREE-HANGING BODIES: The main earring bodies (chandeliers, crystal clusters, dangle pendants, drops, or bell domes) and all bottom fringe drops hang freely suspended in mid-air below the wings. ABSOLUTELY NEVER let any earring rest, touch, lean, or lie on the bottom base, table, or pole! Both earrings must be 100% airborne and hanging straight down with natural gravitational drape. "
    "3. PERFECT HORIZONTAL ALIGNMENT: Both earrings hang at the EXACT SAME vertical height, perfectly level and parallel to each other. Both must face directly forward straight to camera (0-degree direct plumb frontal view). "
    "4. AMPLE HORIZONTAL SEPARATION (ZERO OVERLAP): The left and right earrings are widely and evenly separated by the stand's wishbone wings. There MUST be wide, clean, uniform spacing between the two earrings (like Amazon Luxury and Tanishq catalog hero shots). Absolutely ZERO overlap, zero touching, zero crowding! Every single gemstone facet, crystal prong, metal link, and drop is clearly visible with zero occlusion. "
    "5. IDENTICAL UNTWISTED PAIR: Replicate the authentic matching pair in pristine upright hang, faithfully reproducing the exact top stud, articulation links, chandelier/drop bodies, and cascading crystal/pearl/bead drops in crisp, untangled vertical symmetry! "
    "PRECIOUS CRAFTSMANSHIP, DYNAMIC MATERIAL & COLOR FIDELITY: Exact replication of authentic precious metal alloy (rich warm yellow gold, antique temple gold, rose gold, platinum, or rhodium-plated silver), surface luster, claw prong settings, and gemstone cuts directly from the reference photo! MANDATORY MATERIAL LOCK: Faithfully clone the authentic color palette! If the earrings are pure gold or gold with pearls, replicate the rich 22K yellow gold gleam and pearl luster with ZERO hallucinated diamonds or colored stones! If the earrings feature colored gemstones (such as rubies, emeralds, sapphires, or colored crystals), replicate all those colored gemstones across the stud, body, and drops with 100% color accuracy! "
    "8K ULTRA-HD MACRO RESOLUTION (EXTREME ZOOM CLARITY): Ultra-sharp microscopic macro studio focus on every light-refracting crystal facet edge, prong claw, and metal reflection with zero blur and zero pixelation even when deeply zoomed in. "
    "SOLID STUDIO WHITE BACKGROUND (STAND INTACT): Pure solid clean studio white background (#FFFFFF) with the beige velvet and polished brass display stand fully intact. DO NOT make transparent. Completely free of any human ears, hair, face, neck, skin, or hands."
)

JEWELRY_BANGLES_STUDIO_RULES = (
    "Pristine luxury jewelry boutique showcase catalog photograph (Kalyan / Joyalukkas / Tanishq luxury hero shot). "
    "LUXURY SHOWCASE STAND & CONTRAST (BEIGE VELVET & POLISHED GOLD BRASS): The bangles / kada / bracelet are mounted vertically upright around a horizontal cylindrical warm beige / champagne velvet jewelry bangle display bar supported on polished golden brass upright pillars and weighted circular base. "
    "The stand has a distinct warm beige velvet tone and polished gold brass accents that stand out sharply against the pure solid white background (#FFFFFF)! "
    "MANDATORY E-COMMERCE BANGLE DISPLAY (OVERRIDE REFERENCE POSE): Regardless of whether the reference photo shows bangles lying flat on a table: "
    "1. 3D CIRCULAR PERSPECTIVE: Elegant three-quarter perspective highlighting the circular 3D wrist profile, outer circumference carvings, side filigree work, screw/clasp fastenings, and hollow inner curvature. "
    "2. EXACT PIECE COUNT & UNIFORM SPACING: Mount the EXACT number of pieces from the reference photo! If there are 2 broad kadas in the reference photo, mount EXACTLY 2 kadas side-by-side with an elegant clean gap between them. ABSOLUTELY NEVER add 3, 4, or 5 extra bangles! "
    "3. BROAD CUFF PROFILE PRESERVATION: If the reference shows broad thick cuff kadas with wide cylindrical walls, replicate the identical wide band width and thick cylindrical walls. DO NOT convert broad kadas into thin skinny round wire bangles! "
    "PRECIOUS METAL FIDELITY: Exact replication of authentic precious metal alloy (yellow gold, rose gold, white gold, or silver), surface finish, and gemstone settings directly from the reference photo(s). "
    "8K ULTRA-HD MACRO RESOLUTION (EXTREME ZOOM CLARITY): Hyper-detailed macro studio lighting with crisp specular reflections and tack-sharp detail across the entire circumference. "
    "SOLID STUDIO WHITE BACKGROUND (STAND INTACT): Pure solid clean studio white background (#FFFFFF) with the beige velvet and brass cylindrical display bar fully intact. DO NOT make transparent. Completely free of any human wrist, hands, arms, or skin."
)

JEWELRY_STAND_RULES_MAP = {
    "necklace": JEWELRY_NECKLACE_STUDIO_RULES,
    "pendant": JEWELRY_NECKLACE_STUDIO_RULES,
    "choker": JEWELRY_NECKLACE_STUDIO_RULES,
    "haar": JEWELRY_NECKLACE_STUDIO_RULES,
    "mangalsutra": JEWELRY_NECKLACE_STUDIO_RULES,
    "ring": JEWELRY_RING_STUDIO_RULES,
    "earrings": JEWELRY_EARRINGS_STUDIO_RULES,
    "jhumkas": JEWELRY_EARRINGS_STUDIO_RULES,
    "bangles": JEWELRY_BANGLES_STUDIO_RULES,
    "kada": JEWELRY_BANGLES_STUDIO_RULES,
    "bracelet": JEWELRY_BANGLES_STUDIO_RULES,
}

SYSTEM_JEWELRY_VISION_PROMPT = """You are a master jewelry gemologist, high-jewelry e-commerce catalog director, and 3D precious ornament analyst (Tanishq, Cartier, Tiffany & Co., CaratLane).

Your mission is to analyze 1 to 4 photographs of a jewelry piece and synthesize a unified, forensic-grade 3D catalog description.

CRITICAL PHYSICAL OBJECT COUNT & QUANTITY DISCRIMINATION:
- Count the EXACT number of physical jewelry pieces visible in the image!
- If the photo shows ONE SINGLE kada / bangle / bracelet / ring / necklace / earring:
  Set "item_count": 1, and "piece_descriptor": "exactly 1 standalone single kada/bangle".
  ABSOLUTELY NEVER default or duplicate a single kada into a pair of 2! ONE kada must remain ONE kada!
- If the photo shows a pair of 2 matching kadas / bangles:
  Set "item_count": 2, and "piece_descriptor": "a pair of exactly 2 matching kadas/bangles".
- If the photo shows 1 pair of matching earrings:
  Set "item_count": 2, and "piece_descriptor": "exactly 1 matching pair of 2 earrings".
- If multiple photos (multi-angle) show the SAME single kada from different perspectives:
  Set "item_count": 1, because all photos show the SAME single item!
- Distinguish a single thick broad kada from multiple bangles: a wide cylindrical band with continuous rigid circular wall is ONE kada.

CRITICAL JEWELRY DECONSTRUCTION RULES:
1. FORENSIC COMPONENT-BY-COMPONENT DECONSTRUCTION:
   Do NOT provide a short generic title! You must deconstruct the jewelry into its exact anatomical components:
   - Top Element / Stud / Head Setting: Describe the exact shape (e.g. hemispherical domed center sphere with mirror polish, surrounded by a concentric floral halo of polished gold spherical granulation beads).
   - Articulation / Connection: Describe the heavy polished jump ring or link connecting the stud to the body.
   - Main Body / Dome / Medallion / Band: Describe the dome architecture (stepped conical bell, horizontal embossed bands, rows of micro granulation spheres, openwork jaali lattice filigree, engraved motifs).
   - Hanging Fringe / Drops / Gemstones: Detail the exact dangling accents (e.g. dense cascading fringe of lustrous round natural white seed pearls interlinked with tiny hanging polished gold droplet beads).
   - Precious Metal & Luster: Authentic metal alloy (22K yellow gold, antique temple gold, rose gold, silver, platinum) with realistic surface texture.

2. MULTI-ANGLE DETAIL FUSION (DO NOT MISS SIDE DESIGNS):
   - When multiple images are provided, cross-examine all angles:
     * Image 1 might show the top face/crown (e.g. center gemstone or medallion).
     * Image 2 & 3 might reveal the side shank, gallery openwork, side jali filigree, or band carvings.
   - Synthesize ALL angles into a unified description so that NO side detail or hidden engraving is lost!

3. CATEGORY & LUXURY BOUTIQUE STAND ASSIGNMENT:
   - "earrings": Luxury warm beige velvet dual-wing wishbone display stand on polished gold brass stem and circular base.
   - "necklace": Sculpted luxury warm beige velvet neck bust mannequin stand on polished gold brass circular pedestal.
   - "ring": Angled 3D warm beige velvet ring display block with polished gold brass trim (30-45 degree isometric hero angle).
   - "bangles": Horizontal cylindrical warm beige velvet bangle display bar with polished gold brass uprights.

Return strictly valid JSON:
{
  "jewelry_item": {
    "category": "necklace" | "ring" | "earrings" | "bangles",
    "subcategory": "kada" | "bangle" | "bracelet" | "ring" | "necklace" | "earrings" | "jhumkas" | "pendant",
    "item_count": 1,
    "piece_descriptor": "exactly 1 standalone single broad cuff kada",
    "form_factor": "broad thick cuff kada with wide cylindrical band (approx 25-30mm wide) and thick rigid circular walls",
    "recognized_label": "22K Antique Gold Hand-Carved Broad Cuff Kada",
    "metal_type": "22K Antique Warm Yellow Gold with Matte Relief Luster",
    "gemstone_type": "none (pure solid gold floral relief carvings, no multicolor enamel)",
    "stand_type": "beige_velvet_brass_t_bar_stand" | "beige_velvet_brass_neck_bust" | "beige_velvet_brass_ring_block" | "beige_velvet_brass_bangle_bar",
    "optimal_angle": "0-degree frontal" | "30-45 degree three-quarter isometric",
    "jewelry_description": "Forensic, multi-component gemologist description detailing the cylindrical band width, floral relief carvings, micro-granulation tiers, and clasp droplets."
  }
}"""


def is_jewelry_domain(text: str | None) -> bool:
    """Check whether a given label, category, or query refers to jewelry rather than garments."""
    if not text:
        return False
    t = text.strip().lower()
    jewelry_keywords = {
        "jewelry", "jewellery", "gold", "silver", "diamond", "necklace", "ring",
        "earring", "earrings", "jhumka", "jhumkas", "bangle", "bangles", "kada",
        "bracelet", "pendant", "choker", "mangalsutra", "haar", "chain", "anklet",
        "payal", "polki", "kundan", "solitaire", "gemstone", "ornament",
    }
    tokens = set(t.replace("_", " ").replace("-", " ").split())
    if tokens & jewelry_keywords:
        return True
    return any(kw in t for kw in jewelry_keywords)


def _build_final_jewelry_prompt(jewelry_data: dict[str, Any]) -> str:
    """Build high-fidelity Qwen prompt for 3D jewelry catalog generation on luxury display stands."""
    category = str(jewelry_data.get("category", "necklace")).strip().lower()
    label = str(jewelry_data.get("recognized_label", "Fine Gold Jewelry")).strip()
    metal = str(jewelry_data.get("metal_type", "22K Yellow Gold")).strip()
    desc = str(jewelry_data.get("jewelry_description", "")).strip()
    stand_type = str(jewelry_data.get("stand_type", "beige_velvet_brass_neck_bust")).strip()

    studio_rules = JEWELRY_STAND_RULES_MAP.get(category, JEWELRY_NECKLACE_STUDIO_RULES)

    if category == "ring":
        header = (
            "Generate an empty luxury jewelry e-commerce studio catalog product photograph showing the standalone "
            f"precious ring mounted upright inside the cushioned slot of a luxury angled warm beige velvet ring display block with polished gold brass trim. "
            "OPTIMAL 3D HERO PERSPECTIVE: 30-degree to 45-degree three-quarter isometric angle. "
            "CRITICAL MULTI-ANGLE DETAIL RECONSTRUCTION: Reconstruct BOTH the top crown / center gemstone AND the intricate "
            "side shank carvings, side filigree, gallery openwork, and band engravings simultaneously in one single photograph! "
        )
    elif category in ("necklace", "pendant", "choker", "haar", "mangalsutra"):
        header = (
            "Generate an empty luxury jewelry e-commerce studio catalog product photograph showing the standalone "
            f"precious {category} mounted and draped symmetrically on a sculpted luxury boutique warm beige velvet neck bust mannequin stand on polished gold brass pedestal. "
            "0-degree direct frontal upright camera shot, with the necklace looping naturally around the contoured neck bust and "
            "the central pendant resting flat, centered, and plumb on the front chest of the bust. "
        )
    elif category in ("earrings", "jhumkas"):
        header = (
            "Generate an empty luxury jewelry boutique showcase catalog product photograph (Amazon Luxury / Tanishq hero shot) showing the matching pair of "
            f"standalone {category} mounted symmetrically on a luxury sculpted warm beige velvet wishbone dual-wing earring display stand on polished gold brass stem and circular base. "
            "The top ear-studs / posts are pinned cleanly on the front faces of the beige velvet wings, and BOTH earring bodies (chandeliers, dangles, drops, or domes) hang fully suspended in mid-air below the wings with wide, clean horizontal spacing between them! "
            "ABSOLUTELY NO EARRING LYING ON THE BASE, TABLE, OR POLE! ZERO OVERLAP! "
        )
    else:  # bangles, kada, bracelet
        item_cnt = int(jewelry_data.get("item_count", 1) or 1)
        subcat = str(jewelry_data.get("subcategory", category)).strip()
        if item_cnt == 1:
            header = (
                "Generate an empty luxury jewelry boutique showcase catalog product photograph (Kalyan / Tanishq / Joyalukkas luxury hero shot) showing "
                f"the standalone single {subcat} mounted upright around a horizontal cylindrical warm beige velvet jewelry display bar on polished golden brass upright pillars and weighted base. "
                f"There must be EXACTLY ONE (1) {subcat} mounted on the bar, showing its wide cylindrical 3D band width and hollow inner circular aperture! "
                "CRITICAL ZERO-DUPLICATION RULE: ABSOLUTELY DO NOT DUPLICATE OR CREATE A PAIR! EXACTLY 1 SINGLE KADA, NOT 2!"
            )
        else:
            header = (
                "Generate an empty luxury jewelry boutique showcase catalog product photograph (Kalyan / Tanishq / Joyalukkas luxury hero shot) showing "
                f"the pair of exactly {item_cnt} matching {category} mounted upright side-by-side around a horizontal cylindrical warm beige velvet jewelry display bar on polished golden brass upright pillars and weighted base. "
                f"There must be EXACTLY {item_cnt} {category} mounted on the bar, neatly spaced with an elegant clean gap between them, showing their wide cylindrical 3D band width and hollow inner circular apertures! "
                f"ZERO EXTRA BANGLES! DO NOT generate 3, 4, 5, or more bangles! EXACTLY {item_cnt}!"
            )

    gemstone = str(jewelry_data.get("gemstone_type", "")).strip()
    gem_lower = gemstone.lower()
    has_stones = bool(gemstone) and not any(k in gem_lower for k in ("none", "no stone", "pure solid", "zero stone", "metal only", "no gem"))

    # Dynamic precious metal lock with neutral balanced studio lighting (no artificial yellow tinting)
    metal_lock = (
        f"STRICT PRECIOUS METAL & NATURAL LIGHTING FIDELITY: Faithfully preserve the authentic real-world {metal} "
        "color, tone, saturation, and finish directly from the reference photo! "
        "DO NOT artificially oversaturate, yellow-tint, or alter the metal color. "
        "If the metal is antique gold, pale gold, rose gold, silver, or platinum, replicate the exact authentic alloy hue, "
        "matte finish, or polished reflection with balanced, neutral 5000K-5500K daylight-balanced studio lighting."
    )

    if has_stones:
        gem_lock = (
            f"STRICT GEMSTONE & COLOR FIDELITY: Replicate the authentic gemstone materials and colors ({gemstone}) "
            "directly from the reference photo! Preserve the exact stone colors, faceted cuts, bezel/prong settings, and placements. "
            "DO NOT substitute or change stone colors!"
        )
    else:
        gem_lock = (
            "PURE SOLID METAL FIDELITY (ZERO HALLUCINATED GEMS): The jewelry piece is crafted from pure solid precious metal! "
            "ABSOLUTELY DO NOT add, embed, or hallucinate white diamonds, clear crystals, or random colored stones into the studs, body, or borders! "
            "If the reference top stud or body is solid engraved/domed gold, it MUST remain 100% solid gold with zero stones!"
        )

    if category in ("earrings", "jhumkas"):
        stud_solid_rule = (
            "If the reference top stud is a solid gold dome, bead rosette, floral motif, or plate with no center stone, IT MUST BE 100% SOLID GOLD WITH ZERO CENTER DIAMOND OR STONE! "
            if not has_stones or "pearl" in gem_lower else ""
        )
        earring_anatomy_lock = (
            "STRICT EARRING ANATOMY & TOP STUD FIDELITY: "
            "1. Top Ear-Stud: Replicate the EXACT top stud motif directly from the reference photo! "
            f"{stud_solid_rule}"
            "2. Main Body & Drop: Replicate the authentic silhouette, bells, chandeliers, or drops with 100% fidelity. "
            "3. Zero jhumka bell distortion: keep the authentic hanging silhouette with pristine plumb hang and 0-degree frontal view."
        )
    else:
        earring_anatomy_lock = ""

    ultra_hd_zoom_lock = (
        "CRITICAL 8K ULTRA-HD MASTER MACRO RESOLUTION (ZERO BLUR UNDER EXTREME ZOOM): "
        "The final image MUST be rendered in tack-sharp microscopic macro catalog focus! "
        "Every single millimeter of the jewelry piece—every individual filigree wire, embossed jaali texture, prong claw, "
        "faceted diamond edge, gold reflection, and surface engraving—must be razor sharp, crisp, and hyper-detailed. "
        "Zero blur, zero pixelation, zero digital noise, and zero compression fuzziness. Under extreme deep zoom, "
        "every single pixel must remain tack-sharp, clean, and perfectly defined!"
    )

    white_background_lock = (
        "CRITICAL SOLID STUDIO WHITE BACKGROUND (LUXURY BEIGE VELVET & BRASS STAND 100% INTACT, NOT TRANSPARENT): "
        "The entire scene MUST be a finished, complete e-commerce studio catalog photograph on a pure solid white background (#FFFFFF). "
        "DO NOT remove the background! DO NOT make transparent! The luxury display stand (sculpted beige velvet wishbone wings / neck bust / "
        "ring block / bangle bar with polished gold brass stem and circular base) has a distinct, refined warm beige velvet finish with polished gold brass accents that stands out sharply and crisply against the pure solid white background (#FFFFFF) with subtle realistic contact shadows. "
        "The stand MUST remain 100% fully intact, visible, and grounded on the pure solid white studio background."
    )

    item_cnt = int(jewelry_data.get("item_count", 1) or 1)
    piece_desc = str(jewelry_data.get("piece_descriptor", f"exactly {item_cnt} piece(s)")).strip()
    form_factor = str(jewelry_data.get("form_factor", "")).strip()

    unit_count_lock = (
        f"MANDATORY EXACT PIECE COUNT LOCK: The catalog photograph MUST display {piece_desc.upper()}! "
        f"ABSOLUTELY DO NOT generate extra pieces! If the reference photo shows {item_cnt} piece(s), "
        f"you MUST generate EXACTLY {item_cnt} piece(s) and NO MORE. "
        f"Zero hallucinated extra bangles, zero extra rings, zero extra necklaces!"
    )

    form_factor_lock = (
        f"MANDATORY FORM FACTOR & BAND PROFILE FIDELITY: Replicate the EXACT physical dimensions and silhouette: {form_factor}. "
        "If the piece is a broad thick cuff kada, preserve the wide band width and thick cylindrical walls! "
        "ABSOLUTELY DO NOT reduce broad cuff kadas into thin skinny round wire bangles!"
    ) if form_factor else ""

    surface_lock = (
        "STRICT SURFACE TEXTURE & ACCENT FIDELITY: Replicate ONLY the authentic surface carvings, engravings, "
        "and colors visible in the reference photo! DO NOT invent or hallucinate loud multicolor enamel (meenakari) "
        "or random colored stones if the reference piece is solid gold with engraved metalwork!"
    )

    macro_framing_lock = (
        "CRITICAL MACRO PRODUCT FRAMING (MAXIMUM CANVAS UTILIZATION): "
        "The jewelry piece must fill 75% to 82% of the canvas frame! "
        "Professional close-up macro studio crop focusing tightly and intimately on the precious jewelry piece and its luxury showcase mount. "
        "Eliminate excessive empty dead space so that every micro-facet, gemstone prong, and filigree wire occupies maximum pixel density with razor-sharp zoom clarity!"
    )

    return (
        f"{header} "
        f"{macro_framing_lock} "
        f"{ultra_hd_zoom_lock} "
        f"{white_background_lock} "
        f"{unit_count_lock} "
        f"{form_factor_lock} "
        f"{surface_lock} "
        f"{metal_lock} "
        f"{gem_lock} "
        f"{earring_anatomy_lock} "
        f"Exact jewelry piece to faithfully reconstruct from all reference angles: {desc or label}. "
        f"{studio_rules}"
    )


def _fallback_jewelry_plan(jewelry_type: str = "auto") -> dict[str, Any]:
    """Deterministic fallback for jewelry prompt if OpenAI API key is unavailable."""
    jt = (jewelry_type or "necklace").strip().lower()
    if jt not in JEWELRY_STAND_RULES_MAP:
        jt = "necklace"

    labels = {
        "necklace": ("22K Yellow Gold Filigree Medallion Necklace", "beige_velvet_brass_neck_bust", JEWELRY_NECKLACE_STUDIO_RULES),
        "ring": ("Gold Diamond Solitaire Ring with Filigree Side Shank", "beige_velvet_brass_ring_block", JEWELRY_RING_STUDIO_RULES),
        "earrings": ("22K Gold Traditional Jhumka Earrings Pair", "beige_velvet_brass_t_bar_stand", JEWELRY_EARRINGS_STUDIO_RULES),
        "bangles": ("22K Gold Handcrafted Kada Bangle", "beige_velvet_brass_bangle_bar", JEWELRY_BANGLES_STUDIO_RULES),
    }
    label, stand, rules = labels.get(jt, labels["necklace"])
    item_cnt = 2 if jt in ("earrings", "jhumkas") else 1

    jewelry_data = {
        "category": jt,
        "subcategory": "kada" if jt in ("bangles", "kada") else jt,
        "item_count": item_cnt,
        "piece_descriptor": f"exactly {item_cnt} standalone piece" if item_cnt == 1 else f"a pair of {item_cnt} pieces",
        "recognized_label": label,
        "metal_type": "22K Warm Yellow Gold",
        "stand_type": stand,
        "optimal_angle": "0-degree frontal" if jt != "ring" else "30-45 degree isometric",
        "jewelry_description": f"Authentic {label} with intricate traditional hand-carved openwork filigree and high-polish luster.",
    }
    return {
        **jewelry_data,
        "prompt": _build_final_jewelry_prompt(jewelry_data),
        "prompt_source": "default_fallback",
    }


def create_multi_angle_composite(images: list[Image.Image], target_size: int = 1024) -> Image.Image:
    """Combine 1 to 4 multi-angle jewelry photos into a clean composite image for Qwen visual conditioning.

    - 1 image: Resized to fit within target_size square.
    - 2 images: Arranged side-by-side horizontally.
    - 3 or 4 images: Arranged in a clean 2x2 grid.
    """
    if not images:
        return Image.new("RGB", (target_size, target_size), (255, 255, 255))
    if len(images) == 1:
        img = images[0].convert("RGB")
        img.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (target_size, target_size), (255, 255, 255))
        paste_x = (target_size - img.width) // 2
        paste_y = (target_size - img.height) // 2
        canvas.paste(img, (paste_x, paste_y))
        return canvas

    if len(images) == 2:
        cell_w, cell_h = target_size // 2, target_size
        canvas = Image.new("RGB", (target_size, target_size), (255, 255, 255))
        for idx, src in enumerate(images[:2]):
            im = src.convert("RGB")
            im.thumbnail((cell_w - 10, cell_h - 10), Image.Resampling.LANCZOS)
            px = idx * cell_w + (cell_w - im.width) // 2
            py = (cell_h - im.height) // 2
            canvas.paste(im, (px, py))
        return canvas

    # 3 or 4 images: 2x2 grid
    cell_w, cell_h = target_size // 2, target_size // 2
    canvas = Image.new("RGB", (target_size, target_size), (255, 255, 255))
    positions = [
        (0, 0),
        (cell_w, 0),
        (0, cell_h),
        (cell_w, cell_h),
    ]
    for idx, src in enumerate(images[:4]):
        im = src.convert("RGB")
        im.thumbnail((cell_w - 10, cell_h - 10), Image.Resampling.LANCZOS)
        base_x, base_y = positions[idx]
        px = base_x + (cell_w - im.width) // 2
        py = base_y + (cell_h - im.height) // 2
        canvas.paste(im, (px, py))
    return canvas


def analyze_jewelry_and_build_prompts(
    images: list[Image.Image] | Image.Image,
    jewelry_type: str = "auto",
    custom_prompt: str | None = None,
) -> dict[str, Any]:
    """Analyze 1 to 4 multi-angle photographs of a jewelry piece and build a unified 3D studio prompt.

    Args:
        images: A single PIL Image or a list of 1 to 4 PIL Images (different angles of the same jewelry item).
        jewelry_type: Hint category ('auto', 'necklace', 'ring', 'earrings', 'bangles').
        custom_prompt: Optional user-supplied prompt override.

    Returns:
        Dictionary containing:
        {
            "category": "necklace" | "ring" | "earrings" | "bangles",
            "recognized_label": str,
            "metal_type": str,
            "stand_type": str,
            "optimal_angle": str,
            "jewelry_description": str,
            "prompt": str,
            "prompt_source": str,
            "views_analyzed": int,
        }
    """
    if isinstance(images, Image.Image):
        img_list = [images]
    else:
        img_list = list(images) if images else []

    if not img_list:
        return _fallback_jewelry_plan(jewelry_type)

    if custom_prompt and custom_prompt.strip():
        cat = jewelry_type if jewelry_type != "auto" else "necklace"
        rules = JEWELRY_STAND_RULES_MAP.get(cat, JEWELRY_NECKLACE_STUDIO_RULES)
        return {
            "category": cat,
            "recognized_label": "Custom Luxury Jewelry Piece",
            "metal_type": "Precious Metal",
            "stand_type": "white_neck_bust" if cat == "necklace" else "slotted_ring_block",
            "optimal_angle": "0-degree frontal",
            "jewelry_description": custom_prompt.strip(),
            "prompt": f"{custom_prompt.strip()} {rules}",
            "prompt_source": "custom_prompt",
            "views_analyzed": len(img_list),
        }

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        logger.warning("OPENAI_API_KEY is not set; using default jewelry fallback plan.")
        plan = _fallback_jewelry_plan(jewelry_type)
        plan["views_analyzed"] = len(img_list)
        return plan

    model_name = os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini").strip()
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")

    # Encode all uploaded angles
    encoded_images = [_encode_image_for_vision(im) for im in img_list[:4]]

    user_text = (
        f"Analyze the {len(encoded_images)} photograph(s) provided. All photos show the SAME piece of jewelry from different angles "
        "(e.g. top view, side profile, perspective angle, or close-up). "
        "CRITICAL MULTI-ANGLE FUSION: Synthesize all views into ONE unified forensic jewelry description: "
        "1. Identify the jewelry category ('necklace', 'ring', 'earrings', 'bangles'). "
        "2. Exact metal alloy, karat purity, and polish (22K yellow gold, platinum, etc.). "
        "3. Top / Crown details: center motif, gemstone, setting style. "
        "4. Side profile & Shank details: side filigree, openwork jali, engravings, or accent stones visible in the side photos. "
        "5. Select the appropriate luxury e-commerce catalog stand: "
        "   - Necklaces -> 'slate_grey_neck_bust' "
        "   - Rings -> 'slate_grey_slotted_ring_block' (30-45 degree hero angle showing both top crown and side shank) "
        "   - Earrings -> 'slate_grey_t_bar_earring_stand' (both earrings airborne suspended from arms, wide clean spacing, zero overlap, zero on base) "
        "   - Bangles -> 'slate_grey_cylindrical_bangle_bar' "
        "Strictly preserve the authentic precious metal and craftsmanship without hallucinating different colors."
    )

    user_content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
    for b64_str in encoded_images:
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{b64_str}",
                "detail": "high",
            },
        })

    payload: dict[str, Any] = {
        "model": model_name,
        "temperature": 0.1,
        "max_tokens": 1000,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM_JEWELRY_VISION_PROMPT},
            {"role": "user", "content": user_content},
        ],
    }

    try:
        with httpx.Client(timeout=35.0) as client:
            resp = client.post(
                f"{base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        raw_content = data["choices"][0]["message"]["content"]
        parsed = json.loads(raw_content)
        item = parsed.get("jewelry_item") or parsed

        category = str(item.get("category", "necklace")).strip().lower()
        if category not in JEWELRY_STAND_RULES_MAP:
            category = "necklace"
        subcat = str(item.get("subcategory", category)).strip().lower()
        item_cnt = int(item.get("item_count", 1) or 1)
        piece_desc = str(item.get("piece_descriptor", f"exactly {item_cnt} standalone piece")).strip()
        form_fact = str(item.get("form_factor", "")).strip()
        label = str(item.get("recognized_label", "Fine Gold Jewelry")).strip()
        metal = str(item.get("metal_type", "22K Warm Yellow Gold")).strip()
        gemstone = str(item.get("gemstone_type", "")).strip()
        stand = str(item.get("stand_type", "white_neck_bust")).strip()
        angle = str(item.get("optimal_angle", "0-degree frontal")).strip()
        desc = str(item.get("jewelry_description", label)).strip()

        jewelry_data = {
            "category": category,
            "subcategory": subcat,
            "item_count": item_cnt,
            "piece_descriptor": piece_desc,
            "form_factor": form_fact,
            "recognized_label": label,
            "metal_type": metal,
            "gemstone_type": gemstone,
            "stand_type": stand,
            "optimal_angle": angle,
            "jewelry_description": desc,
            "views_analyzed": len(encoded_images),
        }
        jewelry_data["prompt"] = _build_final_jewelry_prompt(jewelry_data)
        jewelry_data["prompt_source"] = f"openai:{model_name}"
        return jewelry_data

    except Exception as exc:
        logger.warning("ChatGPT Vision jewelry multi-angle analysis failed (%s); using fallback.", exc)
        plan = _fallback_jewelry_plan(jewelry_type)
        plan["views_analyzed"] = len(img_list)
        return plan
