import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import joblib
import sys
import json
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.join(script_dir, ".."))

m2 = joblib.load(os.path.join(root_path, "labeled2.pkl"))
m1_dir = os.path.join(root_path, "old")
m1 = joblib.load(os.path.join(m1_dir, "deals_model.pkl"))
m1_scaler = joblib.load(os.path.join(m1_dir, "scaler.pkl"))
m1_encoder = joblib.load(os.path.join(m1_dir, "encoder.pkl"))

m1_cols = list(m1.feature_names_in_)
features_m2 = list(getattr(m2, "feature_names_in_", ["price", "median", "min_180", "flash_factor"]))

category_avg_discount = {
    1:45, 2:35, 3:55, 4:30, 5:30, 6:20, 7:25, 8:45, 9:45,
    10:20, 11:40, 12:25, 13:25, 14:45, 15:28, 16:35, 17:25
}

category_tolerance = {
    1:3, 2:2, 3:4, 4:2, 5:3, 6:2, 7:2, 8:3, 9:3,
    10:2, 11:4, 12:1, 13:1, 14:3, 15:3, 16:3, 17:2
}

m1_num_cols = [
    'price','rating','rating_count','last_lowest_price','ma_15','ma_3',
    'ma_30','ma_7','median','day_percent_30','day_percent_90',
    'drop_median','drop_p20'
]

print("_READY", flush=True)

while True:
    try:
        raw = sys.stdin.readline()
        if not raw:
            break
        if not raw.strip():
            continue

        data = json.loads(raw)
        job_id = str(data.pop("job_id", ""))
        frequency = float(data.get("frequency") or 0)

        if frequency > 0.04:
            print(json.dumps({"job_id": job_id, "score": 0}), flush=True)
            continue

        df = pd.DataFrame([data])
        df["category"] = df["category"].astype(int)

        df_m1 = df.copy()
        encoded = m1_encoder.transform(df_m1[["category"]])
        encoded_df = pd.DataFrame(encoded, columns=m1_encoder.get_feature_names_out(["category"]), index=df_m1.index)
        df_m1[m1_num_cols] = m1_scaler.transform(df_m1[m1_num_cols])
        X_m1 = pd.concat([df_m1.drop(columns="category"), encoded_df], axis=1)[m1_cols]
        score_m1 = int(round(float(m1.predict_proba(X_m1)[0, 1]) * 1000))

        cat = df["category"]
        avg = cat.map(category_avg_discount)
        tolerance = cat.map(category_tolerance)
        extra = pd.DataFrame({
            "avg_category_discount": avg,
            "category_tolerance_val": tolerance,
            "cat_target_pct": (avg - tolerance) / 100
        }, index=df.index)

        X_m2 = pd.concat([df.drop(columns="category"), extra], axis=1)[features_m2]
        score_m2 = int(round(float(m2.predict_proba(X_m2)[0, 1]) * 1000))

        result = {
            "job_id": job_id,
            "score": round((score_m1 + score_m2) / 2),
            "m1" : score_m1,
            "m2" : score_m2
        }

        print(json.dumps(result), flush=True)

    except Exception as e:
        print(json.dumps({"error": str(e)}), flush=True)