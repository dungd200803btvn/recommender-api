from flask import Flask, request, jsonify
import pandas as pd
import joblib
import numpy as np
from lightfm import LightFM
from sklearn.metrics.pairwise import cosine_similarity

# Load model và dữ liệu
bundle = joblib.load("lightfm_model4_bundle.pkl")
model = bundle["model"]
user_encoder = bundle["user_encoder"]
item_encoder = bundle["item_encoder"]
df_products = pd.read_csv("products.csv")

app = Flask(__name__)

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json()
    user_id = data.get("user_id")
    product_id = data.get("product_id")
    top_k = data.get("top_k", 10)
    alpha = 0.2

    if user_id not in user_encoder.classes_:
        return jsonify({"error": "User không tồn tại"}), 400

    user_idx = user_encoder.transform([user_id])[0]
    item_vectors = model.item_embeddings

    if not product_id:
        all_item_indices = np.arange(len(item_encoder.classes_))
        scores = model.predict(user_ids=np.full_like(all_item_indices, user_idx), item_ids=all_item_indices)
        df_result = pd.DataFrame({
            'product_id': item_encoder.inverse_transform(all_item_indices),
            'score': scores,
            'similarity': [0.0] * len(all_item_indices),
        })
        df_result = df_result[df_result["score"] > 0]
        df_result = df_result.sort_values(by="score", ascending=False).head(top_k)
        return jsonify(df_result.to_dict(orient="records"))

    if product_id not in item_encoder.classes_:
        return jsonify({"error": "Product không tồn tại"}), 400

    item_idx = item_encoder.transform([product_id])[0]
    target_vector = item_vectors[item_idx].reshape(1, -1)
    similarity_scores = cosine_similarity(target_vector, item_vectors)[0]
    top_indices = similarity_scores.argsort()[::-1]
    top_indices = [i for i in top_indices if i != item_idx]

    top_indices = np.array(top_indices)
    scores = model.predict(user_ids=np.full_like(top_indices, user_idx), item_ids=top_indices)

    df_result = pd.DataFrame({
        'product_id': item_encoder.inverse_transform(top_indices),
        'score': scores,
        'similarity': similarity_scores[top_indices],
    })
    df_result["combined_score"] = alpha * df_result["score"] + (1 - alpha) * df_result["similarity"]
    df_result = df_result.sort_values(by="combined_score", ascending=False).head(top_k)
    return jsonify(df_result.to_dict(orient="records"))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
