# Toronto Restaurant Recommendation System

![Toronto Skyline at Sunset](https://upload.wikimedia.org/wikipedia/commons/3/3c/Sunset_Toronto_Skyline_Panorama_Crop_from_Snake_Island.jpg)

<sub>Photo by Jchmrt, licensed under CC BY‑SA 4.0 on Wikimedia Commons</sub>

---

## 📌 Problem Statement

Toronto has a huge and diverse restaurant scene. Choosing a place to eat can be overwhelming without the right tools.  
**Goal:** Build a restaurant recommendation system that suggests similar restaurants based on **cuisine category**, **price range**, and **location** (latitude/longitude).

---

## 📁 Dataset

**Source:** [Kaggle – Toronto Restaurants](https://www.kaggle.com/datasets/kevinbi/toronto-restaurants)  
**File:** `trt_rest.csv`  
**Shape:** **15,821 rows × 9 columns**

**Columns**

- `Category`
- `Restaurant Name`
- `Restaurant Address`
- `Restaurant Phone`
- `Restaurant Price Range`
- `Restaurant Website`
- `Restaurant Yelp URL`
- `Restaurant Latitude`
- `Restaurant Longitude`

**Missing values (raw):**

- Restaurant Price Range: **4,280**
- Restaurant Website: **4,845**
- Restaurant Phone: **705**
- Restaurant Address: **48**
- Restaurant Latitude: **48**
- Restaurant Longitude: **48**
- Category, Restaurant Name, Restaurant Yelp URL: **0**

> Note: This dataset does **not** contain user ratings or textual reviews.

---

## 🧠 Project Workflow

1. **EDA**

   - Inspected schema, value distributions, and missingness.
   - Identified 48 rows missing essential location fields (address + lat/lon).

2. **Data Cleaning**

   - Chose to **flag** rows with missing location for non-geospatial tasks and **exclude** them for location-based similarity.
   - Standardized `Restaurant Price Range` values.

3. **Feature Engineering**

   - **Price Range → Numeric scale** (e.g., `Under $10`, `$11-30`, `$31-60`, `Above $61`).
   - **Category → One-hot encoding** (e.g., `cat_Ramen`, `cat_Pizza`, …).
   - **Latitude/Longitude → Min–Max scaling** to [0, 1] for distance-aware similarity.

4. **Modeling (Content-Based Recommender)**

   - Built a **feature matrix** from: `Price_Encoded` + one-hot `Category` + scaled `Latitude/Longitude`.
   - Used **cosine similarity** to compute similarity between restaurants.
   - Implemented a function to return **Top-10** similar restaurants given an input.

5. **(Optional) Weighting**
   - Allowed configurable weights for price vs. category vs. location to tune similarity behavior.

---

## 📊 How the Recommender Works

- Input: A restaurant name (e.g., “Kinton Ramen”).
- Process: Find the row → compute cosine similarity to all others → exclude itself → return **Top-10** most similar.
- Similarity considers:
  - **Cuisine match** (shared categories),
  - **Comparable price level**, and
  - **Geographic proximity** (via scaled lat/lon).

> **Cosine similarity** measures how aligned two feature vectors are (ignores absolute magnitude), making it well-suited for one-hot + scaled numeric features.

---

## 🧪 Results (Qualitative)

- Produces sensible “you might also like” lists that align in cuisine and price.
- Nearby alternatives can be emphasized by increasing the weight of location features.
- Works fully offline—no user ratings required.

---

## 🧰 Setup & Run

```bash
# Clone the repo
git clone https://github.com/OkayJeff5/toronto-restaurant-recommender.git
cd toronto-restaurant-recommender

# Create and activate a virtual environment with uv
uv venv
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt

# (Option A) Work in Jupyter
jupyter notebook

# (Option B) Run a script (if you have a recommend.py)
python recommend.py
```

**Suggested dependencies (`requirements.txt`):**

```
pandas
numpy
scikit-learn
matplotlib
seaborn
jupyter
```

---

## 🚀 Future Improvements

- Add **Haversine distance** for true geographic distance (km) filtering.
- Introduce **neighborhood filters** and **radius constraints** (e.g., within 2 km).
- Incorporate **user interactions** (ratings/favorites) to build a **hybrid** recommender.
- Deploy an interactive **Streamlit** app (search → recommend → show on map).

---

## 👤 Author

**Siyu (Jeff) Liu** — Toronto, Canada

- GitHub: [@OkayJeff5](https://github.com/OkayJeff5)
- LinkedIn: [linkedin.com/in/okayjeff5](https://www.linkedin.com/in/okayjeff5)

---
