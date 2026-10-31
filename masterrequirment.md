# MASTER REQUIREMENTS
## AI-Based Garment Extraction, Background Removal, Repair & Image Generation Engine

### 1. Project Objective

Humein ek intelligent AI-based Garment Processing Engine build karna hai jo kisi bhi input image se garment (clothing item) ko automatically identify, extract, clean, repair aur standardize kar sake.

System ka primary objective hai:

- Input image mein person, hands, face, body aur background ko remove karna.
- Sirf garment/clothing item ko extract karna.
- Garment ke missing ya hidden portions ko AI Inpainting aur Generative AI se repair karna.
- Garment ka original color, design, pattern, texture aur shape maximum possible preserve karna.
- Final garment image ko transparent background ke saath 1024 × 1024 resolution mein standardize karna.
- Processed image ko API response ke through return karna.

**Important:** Abhi development aur testing ke liye hum manually images provide karenge. Lekin final architecture API-based honi chahiye, jahan external application ya service image send karegi aur hamara engine processed garment image return karega.

---

### 2. Future API-Based Workflow

Future mein system ka workflow kuch is tarah hoga:

1. External application hamari API ko image send karegi.
2. Hamara AI engine image receive karega.
3. System automatically image analysis karega.
4. Garment identify aur segment karega.
5. Required processing perform karega.
6. Person, body, hands aur unwanted background remove karega.
7. Garment ko repair ya reconstruct karega, agar required ho.
8. Garment ko straighten, align aur normalize karega.
9. Final output ko 1024 × 1024 transparent PNG format mein convert karega.
10. Processed image API response ke through return karega.

Manual upload sirf development/testing interface hoga. Core processing logic API-independent aur reusable honi chahiye.

---

### 3. Scenario 1 – Single Garment Extraction

**Example:** API se ek image receive hoti hai jisme sirf ek garment hai, jaise:

- Shirt
- T-Shirt
- Kurta
- Jeans
- Trouser
- Saree
- Dress
- Jacket
- Other clothing items

#### Expected Processing

1. Input image receive karna.
2. Garment detection aur segmentation karna.
3. Garment ko background se separate karna.
4. Agar person ya body image mein present hai, toh usko remove karna.
5. Garment ke unnecessary surrounding pixels remove karna.
6. Garment ki original shape, color, design aur texture preserve karna.
7. Garment ko properly straighten aur align karna.
8. Garment ko 1024 × 1024 transparent canvas par fit karna.
9. Final image return karna.

#### Expected Output

- Only garment visible.
- No person, face, hands or body.
- No unwanted background.
- Transparent background.
- Properly aligned garment.
- Standard 1024 × 1024 PNG.

**Example:** Agar input mein ek shirt hai, toh output mein sirf shirt honi chahiye, bina person aur background ke.

---

### 4. Scenario 2 – Garment Extraction with Hand/Body Occlusion Removal

**Example:** Ek person ne shirt ya trouser pehna hua hai aur uska hand garment ke upar hai.

Jaise:
- Person ka hand trouser ki pocket mein hai.
- Hand shirt ke front portion ko cover kar raha hai.
- Arms garment ke kuch portions ko hide kar rahe hain.
- Person ka face, neck aur body garment ke saath visible hain.

Is situation mein normal background removal sufficient nahi hoga, kyunki hand remove karne ke baad garment ka missing portion bhi repair karna padega.

#### Expected Processing

**Step 1 – Garment Segmentation**

- Garment ko identify karna.
- Person, face, hands, arms aur other body parts identify karna.
- Garment ka visible portion extract karna.

**Step 2 – Occlusion Detection**

- Identify karna ki garment ka kaunsa portion hand ya body se covered hai.
- Hand/arm aur garment ke overlapping region ko detect karna.
- Missing garment region ka approximate area identify karna.

**Step 3 – Hand/Body Removal**

- Hand, arms aur unwanted body portions ko garment se separate karna.
- Person ke unwanted portions remove karna.
- Garment ke visible pixels preserve karna.

**Step 4 – AI Inpainting and Garment Repair**

Hand remove karne ke baad jo garment portion missing hai, usko AI Inpainting se reconstruct karna.

Repair ke waqt system ko following information consider karni hogi:

- Nearby fabric color.
- Original garment texture.
- Fabric pattern and print.
- Garment design.
- Stitching and seams.
- Shape and structure.
- Nearby visible garment regions.

System ka objective missing portion ko is tarah reconstruct karna hai ki repaired area original garment ke saath naturally blend ho.

**Step 5 – Generative Repair Fallback**

Agar missing region bada hai ya LaMa-based inpainting se satisfactory result nahi milta, toh generative AI model use kiya ja sakta hai.

Generative model ko garment ke existing color, design, print aur texture se guidance milni chahiye.

**Important:** AI ko original garment ke visible portions unnecessarily modify nahi karne chahiye. Reconstructed area sirf missing region tak restricted hona chahiye.

**Step 6 – Final Garment Extraction**

- Person completely remove karna.
- Background remove karna.
- Repaired garment ko clean karna.
- Garment ko straighten aur align karna.
- 1024 × 1024 transparent PNG generate karna.

#### Expected Output

Ek complete garment image jisme:
- Hand visible nahi hona chahiye.
- Person ya body visible nahi honi chahiye.
- Missing garment portion naturally repaired hona chahiye.
- Original garment ka design aur color maximum preserve hona chahiye.
- Background completely transparent hona chahiye.

---

### 5. Scenario 3 – Garment Reconstruction / Missing Design Generation

**Example:** Input image mein garment partially damaged, cut, incomplete ya heavily occluded hai.

System ko existing garment ke visual information se missing portions generate karne honge.

#### Expected Processing

1. Garment identify karna.
2. Original visible fabric ka analysis karna.
3. Color palette extract karna.
4. Fabric texture aur print identify karna.
5. Pattern aur design analyze karna.
6. Garment ki overall shape aur structure understand karna.
7. Missing portions identify karna.
8. AI generative model ki help se missing portions reconstruct karna.
9. Generated portions ko existing garment ke saath blend karna.
10. Final garment ko clean aur standardize karna.

#### Generation Guidelines

- Original color preserve karna.
- Original print aur pattern follow karna.
- Fabric texture match karna.
- Existing garment ki shape follow karna.
- Unnecessary design elements add nahi karne.
- Visible original portions ko preserve karna.

**Note:** System ko visual consistency maximize karni hai. Lekin completely hidden design details ko original jaisa exactly recover karne ki guarantee nahi di ja sakti.

---

### 6. Common Final Image Processing Pipeline

Chahe image Scenario 1 se aaye, Scenario 2 se ya Scenario 3 se, final output ke liye common processing pipeline honi chahiye.

| Processing | Requirement |
|---|---|
| Garment Extraction | Only garment should remain |
| Person Removal | Face, body, hands and arms removed |
| Background Removal | Transparent background |
| Garment Repair | Missing/occluded portions reconstructed |
| Image Cleanup | Remove unwanted pixels and artifacts |
| Straightening | Proper garment orientation |
| Alignment | Center garment on canvas |
| Resizing | 1024 × 1024 |
| Output Format | PNG with transparency |

### 7. AI Models and Processing Components

System modular hona chahiye. Har task ke liye appropriate model use karna hai.

| Component | Responsibility |
|---|---|
| SegFormer B2 Clothes | Garment semantic segmentation |
| BiRefNet | Foreground/background separation |
| LaMa Inpainting | Local missing-region repair |
| Stable Diffusion Inpainting | Complex generative repair |
| IP-Adapter | Fabric reference and visual guidance |
| ControlNet | Shape and structural guidance |
| OpenCV | Image alignment, mask processing and cleanup |
| Pillow / NumPy | Image normalization and output generation |

Models ko task ke according route karna hai. Har image par unnecessarily sabhi models execute nahi karne hain.

### 8. Intelligent Task Router

System ko automatically decide karna chahiye ki image ke liye kaunsi processing required hai.

**Case A – Complete Garment**

Garment complete hai aur koi obstruction nahi hai.

Processing:
Segmentation → Background Removal → Alignment → Resize → Output

**Case B – Hand or Body Occlusion**

Garment ka portion hand/body se covered hai.

Processing:
Segmentation → Occlusion Detection → Hand/Body Removal → LaMa Inpainting → Quality Check → Output

**Case C – Large Missing Region**

Garment ka significant portion missing hai.

Processing:
Segmentation → Missing Region Detection → Generative Repair → Quality Check → Output

**Case D – Complex or Uncertain Image**

Agar system confidently repair nahi kar sakta, toh image ko uncertain status ke saath return karna hai. Incorrect reconstruction ko silently successful output nahi banana hai.

### 9. Image Preservation and Quality Requirements

Ye requirements bahut important hain:

1. Original visible garment pixels ko maximum preserve karna hai.
2. Repair mask sirf approved missing garment region cover kare.
3. Transparent background ko automatically opaque nahi banana.
4. Garment ke outside generated pixels allow nahi hone chahiye.
5. Original fabric color aur pattern mein unnecessary changes nahi hone chahiye.
6. Repaired region ki boundaries natural honi chahiye.
7. Garment ki shape distort nahi honi chahiye.
8. Final image mein grey edges, unwanted shadows, artifacts aur background fragments nahi hone chahiye.
9. Low-confidence cases ko flag karna hai.
10. Original input aur intermediate processing artifacts debugging ke liye optionally retain kiye ja sakte hain.

### 10. API Architecture Requirements

Final system FastAPI-based Python service hona chahiye.

Proposed endpoints:

**POST `/api/v1/garment/process`**

Main endpoint. Image receive karke automatic task routing aur processing karega.

**POST `/api/v1/garment/extract`**

Garment segmentation aur background removal.

**POST `/api/v1/garment/repair`**

Hand/body occlusion removal aur missing garment repair.

**POST `/api/v1/garment/reconstruct`**

Complex missing portions ke generative reconstruction ke liye.

**GET `/health`**

Service health check.

Final architecture mein main process endpoint common entry point hona chahiye. Baaki endpoints specialized processing/testing ke liye honge.

### 11. Expected API Response

Successful processing par response mein following information honi chahiye:

- Processing status.
- Final image URL ya image response.
- Image dimensions.
- Output format.
- Transparency status.
- Detected garment category.
- Processing operations performed.
- Repair status.
- Confidence/quality indicators.
- Processing time.

Agar processing fail hoti hai toh meaningful error response return hona chahiye.

### 12. Development Strategy

Development incremental phases mein karna hai:

**Phase 1 – Single Garment Extraction**
- Multiple garment categories test karna.
- Background removal.
- Person removal.
- Transparent 1024 × 1024 output.

**Phase 2 – Hand/Body Occlusion Detection**
- Hand, arm aur body overlap detection.
- Garment-specific missing region masks.
- Occlusion removal.

**Phase 3 – LaMa-Based Garment Repair**
- Small and medium missing portions repair karna.
- Multiple garment types par validation.
- Fabric consistency aur boundary quality test karna.

**Phase 4 – Generative AI Repair**
- Large missing portions reconstruct karna.
- Stable Diffusion Inpainting.
- IP-Adapter and ControlNet integration.
- Existing fabric design preserve karne ka evaluation.

**Phase 5 – Intelligent Task Router**
- Automatic scenario detection.
- Appropriate model selection.
- Confidence-based fallback.

**Phase 6 – Final Image Standardization**
- Background cleanup.
- Garment alignment.
- Transparent 1024 × 1024 output.
- Quality validation.

**Phase 7 – API Integration**
- FastAPI endpoints.
- Image input/output handling.
- Model lifecycle management.
- Error handling.
- Performance and concurrency testing.

### 13. Current Development Constraints

- Existing Flow A aur Flow B production implementations ko bina explicit approval modify nahi karna hai.
- New experimental modules isolated rakhne hain.
- Existing working environments aur dependencies ko unnecessarily upgrade nahi karna hai.
- CPU compatibility maintain karni hai.
- Heavy generative models ko optional fallback engine ke roop mein evaluate karna hai.
- Har phase ke liye independent test cases, benchmark results aur visual output save karne hain.
- Commercial use se pehle har model, checkpoint aur adapter ki license verify karni hai.

### 14. Final Project Goal

Hamara final goal ek fully automated **AI Garment Extraction and Reconstruction API** build karna hai.

External application sirf ek image send karegi.

Hamara system automatically:

1. Image understand karega.
2. Garment identify karega.
3. Person, face, hands, arms aur unwanted background remove karega.
4. Garment ka missing portion detect karega.
5. Required hone par AI Inpainting ya Generative AI se garment repair karega.
6. Original fabric design, color aur texture preserve karne ki koshish karega.
7. Garment ko clean, straighten aur align karega.
8. 1024 × 1024 transparent PNG generate karega.
9. Final processed garment image API ke through return karega.

**Final Output:**

Input: Person wearing a garment / Single garment image.

Output: Only one clean, properly aligned, repaired (if required), background-free garment image in 1024 × 1024 transparent PNG format.

**Core Vision: One Input Image → Automatic AI Processing → One Clean Garment Image.**