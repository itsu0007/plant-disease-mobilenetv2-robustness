"""
Plant Disease Taxonomy & Domain Knowledge Mapping
Defines the standard PlantVillage disease classes, metadata, symptoms, and treatment guidelines.
"""

from typing import Dict, List, Optional

# Standard 38 PlantVillage Classes with detailed agronomic metadata
PLANT_DISEASE_METADATA: Dict[str, Dict[str, str]] = {
    "Apple___Apple_scab": {
        "crop": "Apple",
        "condition": "Apple Scab",
        "status": "Diseased",
        "symptoms": "Olive-green to black velvety spots on leaves; distorted leaves and dark scabby spots on fruit.",
        "treatment": "Apply fungicides such as captan or myclobutanil during early spring. Rake and destroy fallen leaves.",
        "prevention": "Plant scab-resistant cultivars (e.g., Honeycrisp, Enterprise). Ensure proper tree pruning for air circulation."
    },
    "Apple___Black_rot": {
        "crop": "Apple",
        "condition": "Black Rot",
        "status": "Diseased",
        "symptoms": "Purple circular leaf spots expanding to 'frog-eye' pattern; rotten, mummified black fruit; trunk cankers.",
        "treatment": "Prune infected twigs and mummified fruit. Apply copper-based or sulfur-based fungicides during bloom.",
        "prevention": "Prune out dead wood and disinfect pruning tools between cuts. Avoid wounding bark."
    },
    "Apple___Cedar_apple_rust": {
        "crop": "Apple",
        "condition": "Cedar Apple Rust",
        "status": "Diseased",
        "symptoms": "Bright yellow-orange leaf lesions with tiny black fungal bodies, later developing tubular projections beneath.",
        "treatment": "Apply sterol-inhibiting or protective fungicides (e.g., myclobutanil) as buds begin to break.",
        "prevention": "Remove eastern red cedar trees in the immediate vicinity (500m radius) if feasible."
    },
    "Apple___healthy": {
        "crop": "Apple",
        "condition": "Healthy",
        "status": "Healthy",
        "symptoms": "Vibrant green leaves, uniform texture, no discoloration or lesion spotting.",
        "treatment": "No treatment required. Maintain regular watering, balanced soil nutrition, and pest scouting.",
        "prevention": "Conduct seasonal preventative pruning and soil health testing."
    },
    "Blueberry___healthy": {
        "crop": "Blueberry",
        "condition": "Healthy",
        "status": "Healthy",
        "symptoms": "Glossy deep green foliage with uniform venation and healthy shoot tips.",
        "treatment": "No treatment required.",
        "prevention": "Maintain acidic soil pH (4.5 - 5.5) and keep root zone mulched with pine bark."
    },
    "Cherry___Powdery_mildew": {
        "crop": "Cherry",
        "condition": "Powdery Mildew",
        "status": "Diseased",
        "symptoms": "White powdery fungal coating on leaves and young shoots; leaves may curl upward and drop prematurely.",
        "treatment": "Apply horticultural oils, potassium bicarbonate, or sulfur fungicides at first sign of powdery growth.",
        "prevention": "Avoid excess nitrogen fertilization; ensure canopy allows ample sunlight penetration."
    },
    "Cherry___healthy": {
        "crop": "Cherry",
        "condition": "Healthy",
        "status": "Healthy",
        "symptoms": "Smooth green leaves, sturdy petiole attachment, no fungal patches.",
        "treatment": "No treatment required.",
        "prevention": "Annual thinning cuts to improve aeration across branches."
    },
    "Corn___Cercospora_leaf_spot_Gray_leaf_spot": {
        "crop": "Corn (Maize)",
        "condition": "Gray Leaf Spot",
        "status": "Diseased",
        "symptoms": "Rectangular, tan-to-gray lesions restricted between leaf veins, causing extensive blighting.",
        "treatment": "Apply triazole or strobilurin-based foliar fungicides if disease appears prior to tasseling.",
        "prevention": "Rotate crops with non-grasses (soybeans, legumes) and select resistant hybrids."
    },
    "Corn___Common_rust": {
        "crop": "Corn (Maize)",
        "condition": "Common Rust",
        "status": "Diseased",
        "symptoms": "Oval to elongated reddish-brown pustules scattered across both leaf surfaces.",
        "treatment": "Fungicide sprays (strobilurins/triazoles) when weather is cool and moist and lesions reach economic thresholds.",
        "prevention": "Plant resistant hybrids; plant early to avoid peak spore migration windows."
    },
    "Corn___Northern_Leaf_Blight": {
        "crop": "Corn (Maize)",
        "condition": "Northern Leaf Blight",
        "status": "Diseased",
        "symptoms": "Large, elongated cigar-shaped grayish-green lesions that turn tan as tissue necroses.",
        "treatment": "Apply registered fungicides at tassel stage if lesions appear on the ear leaf or higher.",
        "prevention": "Tillage to bury crop debris; rotation with soybeans; use resistant varieties."
    },
    "Corn___healthy": {
        "crop": "Corn (Maize)",
        "condition": "Healthy",
        "status": "Healthy",
        "symptoms": "Robust deep-green parallel venation without pustules or lesions.",
        "treatment": "No treatment required.",
        "prevention": "Maintain nitrogen-phosphorus-potassium balance and proper weed control."
    },
    "Grape___Black_rot": {
        "crop": "Grape",
        "condition": "Black Rot",
        "status": "Diseased",
        "symptoms": "Reddish-brown circular leaf spots with tiny black fruiting bodies; shrivelled, black, hard berries.",
        "treatment": "Apply protective fungicides (e.g., mancozeb, captan, or kresoxim-methyl) from bud break through veraison.",
        "prevention": "Sanitation: prune and destroy all mummified grape clusters and infected canes."
    },
    "Grape___Esca_Black_Measles": {
        "crop": "Grape",
        "condition": "Esca (Black Measles)",
        "status": "Diseased",
        "symptoms": "'Tiger-stripe' chlorosis and necrosis between leaf veins; dark speckled spots on berries; wood decay.",
        "treatment": "No cure once systemic; prune diseased trunks down to healthy wood. Seal large wounds with pruning paint.",
        "prevention": "Avoid pruning during damp weather; apply Trichoderma protective sprays to pruning cuts."
    },
    "Grape___Leaf_blight_Isariopsis": {
        "crop": "Grape",
        "condition": "Leaf Blight",
        "status": "Diseased",
        "symptoms": "Irregular dark brown patches on leaf margins and blades, causing premature defoliation.",
        "treatment": "Copper oxychloride or systemic fungicide sprays in mid-to-late season.",
        "prevention": "Ensure good canopy airflow and avoid overhead sprinkler irrigation."
    },
    "Grape___healthy": {
        "crop": "Grape",
        "condition": "Healthy",
        "status": "Healthy",
        "symptoms": "Broad, vibrant green palmate leaves free of lesions or necrotic zones.",
        "treatment": "No treatment required.",
        "prevention": "Routine canopy management and shoot positioning."
    },
    "Orange___Haunglongbing_Citrus_greening": {
        "crop": "Citrus (Orange)",
        "condition": "Citrus Greening (HLB)",
        "status": "Diseased",
        "symptoms": "Asymmetrical blotchy mottle chlorosis across leaf veins; small, lopsided, bitter fruit with aborted seeds.",
        "treatment": "No cure; manage Asian Citrus Psyllid insect vectors using targeted insecticides. Nutritional foliage sprays.",
        "prevention": "Plant certified disease-free nursery stock; immediately eradicate confirmed infected trees to protect groves."
    },
    "Peach___Bacterial_spot": {
        "crop": "Peach",
        "condition": "Bacterial Spot",
        "status": "Diseased",
        "symptoms": "Angular purple-brown lesions that drop out leaving 'shot holes'; sunken cracked fruit spots.",
        "treatment": "Apply copper bactericides during dormant season and oxytetracycline during growing season.",
        "prevention": "Plant resistant cultivars; avoid planting in extremely sandy, windblown soils without windbreaks."
    },
    "Peach___healthy": {
        "crop": "Peach",
        "condition": "Healthy",
        "status": "Healthy",
        "symptoms": "Lanceolate green leaves with clean margins and no perforated shot holes.",
        "treatment": "No treatment required.",
        "prevention": "Balanced fertilization avoiding excessive nitrogen that invites pests."
    },
    "Pepper_bell___Bacterial_spot": {
        "crop": "Bell Pepper",
        "condition": "Bacterial Spot",
        "status": "Diseased",
        "symptoms": "Water-soaked, circular-to-irregular dark brown spots with chlorotic yellow halos; leaf drop.",
        "treatment": "Copper-based sprays combined with mancozeb applied preventatively during warm rainy weather.",
        "prevention": "Use certified pathogen-free seeds; avoid overhead irrigation; implement 2-year crop rotation."
    },
    "Pepper_bell___healthy": {
        "crop": "Bell Pepper",
        "condition": "Healthy",
        "status": "Healthy",
        "symptoms": "Dark glossy green foliage, robust structure, no water-soaked lesions.",
        "treatment": "No treatment required.",
        "prevention": "Maintain consistent moisture and calcium levels to avoid blossom end rot."
    },
    "Potato___Early_blight": {
        "crop": "Potato",
        "condition": "Early Blight",
        "status": "Diseased",
        "symptoms": "Dark brown-to-black concentric rings ('target board' pattern) on older leaves, surrounded by yellow halos.",
        "treatment": "Apply protective fungicides (chlorothalonil, mancozeb) or systemic strobilurins when symptoms emerge.",
        "prevention": "Maintain adequate plant nutrition (especially nitrogen and potassium); practice crop rotation."
    },
    "Potato___Late_blight": {
        "crop": "Potato",
        "condition": "Late Blight",
        "status": "Diseased",
        "symptoms": "Water-soaked pale green-to-brown lesions on leaf tips/margins, white fuzzy mildew beneath in high humidity.",
        "treatment": "Apply systemic fungicides (mefenoxam, cymoxanil, cyazofamid) promptly upon detection.",
        "prevention": "Destroy cull piles; plant certified blight-free seed tubers; monitor blight forecast warnings."
    },
    "Potato___healthy": {
        "crop": "Potato",
        "condition": "Healthy",
        "status": "Healthy",
        "symptoms": "Lush compound leaves with crisp edges and uniform green color.",
        "treatment": "No treatment required.",
        "prevention": "Proper hill mounding and drip irrigation to keep foliage dry."
    },
    "Strawberry___Leaf_scorch": {
        "crop": "Strawberry",
        "condition": "Leaf Scorch",
        "status": "Diseased",
        "symptoms": "Irregular purple or dark brownish blotches that enlarge, turning leaf margins brown and scorched.",
        "treatment": "Fungicide sprays (captan or thiophanate-methyl) applied in early spring and post-harvest.",
        "prevention": "Remove dead foliage after harvest; renovate strawberry beds annually; avoid wet foliage."
    },
    "Strawberry___healthy": {
        "crop": "Strawberry",
        "condition": "Healthy",
        "status": "Healthy",
        "symptoms": "Trifoliate bright green serrated leaves with no purple spots.",
        "treatment": "No treatment required.",
        "prevention": "Mulch with clean straw to keep leaves and fruit off bare soil."
    },
    "Tomato___Bacterial_spot": {
        "crop": "Tomato",
        "condition": "Bacterial Spot",
        "status": "Diseased",
        "symptoms": "Small (2-3mm) circular dark brown water-soaked lesions, often surrounded by yellow halos; greasy appearance.",
        "treatment": "Apply fixed copper mixed with mancozeb sprays at 7-10 day intervals during humid weather.",
        "prevention": "Use hot-water treated or certified disease-free seeds; sterilize garden stakes and tools."
    },
    "Tomato___Early_blight": {
        "crop": "Tomato",
        "condition": "Early Blight",
        "status": "Diseased",
        "symptoms": "Concentric rings ('target rings') on lower leaves, surrounded by yellowing; leaf defoliation from base up.",
        "treatment": "Prune infected lower foliage; apply copper fungicide or chlorothalonil weekly.",
        "prevention": "Mulch soil surface to prevent soil splash onto lower leaves; stake plants for upright aeration."
    },
    "Tomato___Late_blight": {
        "crop": "Tomato",
        "condition": "Late Blight",
        "status": "Diseased",
        "symptoms": "Dark oily water-soaked patches rapidly spreading over leaves, white spore fuzz under leaves; stem rot.",
        "treatment": "Immediate application of copper or systemic oomycete fungicides; bag and discard severely infected plants.",
        "prevention": "Keep foliage dry; maximize spacing; avoid planting close to potatoes."
    },
    "Tomato___Leaf_Mold": {
        "crop": "Tomato",
        "condition": "Leaf Mold",
        "status": "Diseased",
        "symptoms": "Pale yellow chlorotic patches on leaf upper surface; olive-green to brown velvety mold on underside.",
        "treatment": "Ventilate greenhouses; apply copper-based or biofungicide sprays (Bacillus subtilis).",
        "prevention": "Maintain relative humidity below 85% in protected cultivation; use resistant varieties."
    },
    "Tomato___Septoria_leaf_spot": {
        "crop": "Tomato",
        "condition": "Septoria Leaf Spot",
        "status": "Diseased",
        "symptoms": "Numerous small circular spots with dark brown margins and sunken grayish centers studded with tiny black dots.",
        "treatment": "Remove diseased lower leaves; apply chlorothalonil, copper, or mancozeb at regular intervals.",
        "prevention": "Rotate crops every 2-3 years; use drip irrigation; avoid working in the field when plants are wet."
    },
    "Tomato___Spider_mites_Two_spotted_spider_mite": {
        "crop": "Tomato",
        "condition": "Two-Spotted Spider Mite",
        "status": "Diseased",
        "symptoms": "Fine yellow stippling or bronzing on leaves; fine webbing visible on undersides; leaves dry and drop.",
        "treatment": "Apply insecticidal soap, neem oil, or miticides (abamectin); release predatory mites (Phytoseiulus persimilis).",
        "prevention": "Avoid drought stress and dusty conditions which favor spider mite population spikes."
    },
    "Tomato___Target_Spot": {
        "crop": "Tomato",
        "condition": "Target Spot",
        "status": "Diseased",
        "symptoms": "Pinpoint brown spots enlarging to circular lesions with light brown centers and dark concentric rings.",
        "treatment": "Fungicide treatments including azoxystrobin, chlorothalonil, or famoxadone.",
        "prevention": "Ensure wide plant spacing; remove bottom leaves to prevent moisture buildup near soil."
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "crop": "Tomato",
        "condition": "Yellow Leaf Curl Virus (TYLCV)",
        "status": "Diseased",
        "symptoms": "Severe upward leaf cupping, yellowing along margins, stunted bushy plant growth, aborted flower buds.",
        "treatment": "No chemical cure for viral infection; control whitefly insect vectors using reflective mulch and systemic insecticides.",
        "prevention": "Use virus-resistant hybrid varieties; install fine insect netting in nursery and greenhouse beds."
    },
    "Tomato___Tomato_mosaic_virus": {
        "crop": "Tomato",
        "condition": "Mosaic Virus (ToMV)",
        "status": "Diseased",
        "symptoms": "Mottling with alternating light and dark green areas; distorted 'fern-leaf' or strap-like leaves; stunted growth.",
        "treatment": "Eradicate infected plants immediately; wash hands with soap after handling tobacco products.",
        "prevention": "Disinfect tools in 20% nonfat dry milk or trisodium phosphate; use resistant cultivars."
    },
    "Tomato___healthy": {
        "crop": "Tomato",
        "condition": "Healthy",
        "status": "Healthy",
        "symptoms": "Vibrant emerald green foliage, uniform growth, healthy blossom development without lesions.",
        "treatment": "No treatment required. Maintain balanced nitrogen-potassium supply and consistent moisture.",
        "prevention": "Proactive staking, mulching, and pest monitoring."
    }
}

CLASS_NAMES: List[str] = sorted(list(PLANT_DISEASE_METADATA.keys()))
NUM_CLASSES: int = len(CLASS_NAMES)
CLASS_TO_IDX: Dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS: Dict[int, str] = {i: name for i, name in enumerate(CLASS_NAMES)}

def get_class_name(idx: int) -> str:
    """Return the raw class key string for a given index."""
    return IDX_TO_CLASS.get(idx, f"Unknown_Class_{idx}")

def get_clean_name(class_key: str) -> str:
    """Return a clean human-readable title from the class key."""
    meta = PLANT_DISEASE_METADATA.get(class_key)
    if meta:
        return f"{meta['crop']} - {meta['condition']}"
    # Fallback parsing
    parts = class_key.replace("___", " - ").replace("_", " ")
    return parts

def get_crop_and_disease(class_key: str) -> tuple:
    """Return (crop, disease, status) tuple."""
    meta = PLANT_DISEASE_METADATA.get(class_key)
    if meta:
        return meta["crop"], meta["condition"], meta["status"]
    return "Plant", class_key, "Unknown"

def get_treatment_info(class_key: str) -> Dict[str, str]:
    """Retrieve full diagnosis and agronomic recommendations."""
    return PLANT_DISEASE_METADATA.get(class_key, {
        "crop": "Unknown",
        "condition": class_key,
        "status": "Unknown",
        "symptoms": "No specific symptoms recorded for this class.",
        "treatment": "Consult a local agricultural extension specialist for diagnosis.",
        "prevention": "Ensure good hygiene, sanitization, and soil drainage."
    })
