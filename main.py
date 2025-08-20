import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches

DATA_PATH = "trt_rest.csv"

# Define price order and mapping
PRICE_ORDER = ["Under $10", "$11-30", "$31-60", "Above $61"]
PRICE_MAP = {p: i for i, p in enumerate(PRICE_ORDER)}

def preprocess_data(filepath: str):
    df = pd.read_csv(filepath)

    # 1) Remove duplicate rows
    before = len(df)
    df = df.drop_duplicates().copy()
    # print(f"Removed {before - len(df)} duplicate rows")

    # 2) Drop rows with missing location (address/lat/lon)
    before = len(df)
    df = df.dropna(subset=["Restaurant Address", "Restaurant Latitude", "Restaurant Longitude"]).copy()
    # print(f"Dropped {before - len(df)} rows with missing location")

    # 3) Fill missing price range with mode
    if df["Restaurant Price Range"].isnull().any():
        mode_val = df["Restaurant Price Range"].mode(dropna=True)[0]
        df["Restaurant Price Range"] = df["Restaurant Price Range"].fillna(mode_val)

    # Normalize spacing and map symbols if present
    df["Restaurant Price Range"] = df["Restaurant Price Range"].astype(str).str.strip()
    sym_map = {"$": "Under $10", "$$": "$11-30", "$$$": "$31-60", "$$$$": "Above $61"}
    df["Restaurant Price Range"] = df["Restaurant Price Range"].replace(sym_map)

    # Encode price
    df["Price_Encoded"] = df["Restaurant Price Range"].map(PRICE_MAP)
    if df["Price_Encoded"].isnull().any():
        price_mode = df["Restaurant Price Range"].mode()[0]
        df["Price_Encoded"] = df["Price_Encoded"].fillna(PRICE_MAP.get(price_mode, 1))

    # One-hot encode category
    cat_dummies = pd.get_dummies(df["Category"].astype(str).str.strip(), prefix="cat")

    # Scale latitude/longitude
    scaler = MinMaxScaler()
    df[["Lat_Scaled", "Lon_Scaled"]] = scaler.fit_transform(df[["Restaurant Latitude", "Restaurant Longitude"]])

    # Build feature matrix
    feature_cols = ["Price_Encoded", "Lat_Scaled", "Lon_Scaled"] + list(cat_dummies.columns)
    feature_matrix = pd.concat([df[["Price_Encoded", "Lat_Scaled", "Lon_Scaled"]], cat_dummies], axis=1).values

    # print(f"Remaining rows: {len(df)}, Categories encoded: {cat_dummies.shape[1]}")

    return df.reset_index(drop=True), feature_matrix

def recommend_similar(restaurant_name: str, df: pd.DataFrame, feature_matrix: np.ndarray, top_n: int = 10):
    # Find restaurant index
    matches = df.index[df["Restaurant Name"].str.lower() == restaurant_name.lower()].tolist()
    if not matches:
        suggestions = get_close_matches(restaurant_name, df["Restaurant Name"].unique(), n=5, cutoff=0.6)
        raise ValueError(f"No match for '{restaurant_name}'. Suggestions: {', '.join(suggestions) if suggestions else 'None'}")

    idx = matches[0]

    # Compute cosine similarity
    sim = cosine_similarity(feature_matrix[idx:idx+1], feature_matrix).ravel()
    sim[idx] = -1.0

    # Get top-N
    top_idx = np.argsort(sim)[::-1][:top_n]
    cols_to_show = ["Restaurant Name", "Category", "Restaurant Price Range", "Restaurant Address"]
    result = df.iloc[top_idx][cols_to_show].copy()
    result.insert(1, "Similarity", sim[top_idx].round(3))

    return result.reset_index(drop=True)

if __name__ == "__main__":
    df, feature_matrix = preprocess_data(DATA_PATH)

    query = input("Enter a restaurant name: ").strip()
    try:
        recs = recommend_similar(query, df, feature_matrix, top_n=10)
        print("\nTop 10 similar restaurants:\n")
        print(recs.to_string(index=False))
    except ValueError as e:
        print(e)
