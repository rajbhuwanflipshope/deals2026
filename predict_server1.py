import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import joblib
import sys
import json
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
# labeled.pkl, deals_model.pkl, scaler.pkl, and encoder.pkl all live directly
# alongside this script (no parent "old" directory exists on disk) — the
# previous script_dir/".." lookup pointed one level too high, e.g. /root
# instead of /root/deals2026, which is why joblib.load kept failing with
# FileNotFoundError for labeled.pkl.
root_path = script_dir

m2 = joblib.load(os.path.join(root_path, "labeled.pkl"))
m1_dir = root_path
m1 = joblib.load(os.path.join(m1_dir, "deals_model.pkl"))
m1_scaler = joblib.load(os.path.join(m1_dir, "scaler.pkl"))
m1_encoder = joblib.load(os.path.join(m1_dir, "encoder.pkl"))

m1_cols = list(m1.feature_names_in_)
num_cols = ['price','rating','rating_count','last_lowest_price','ma_15','ma_3','ma_30','ma_7','median','day_percent_30','day_percent_90','drop_median','drop_p20']

print("_READY", flush=True)

while True:
    try:
        raw = sys.stdin.readline()
        if not raw: break
        if not raw.strip(): continue

        data = json.loads(raw)
        job_id = str(data.pop("job_id", ""))
        if float(data.get("frequency") or 0) > 0.04:
            print(json.dumps({"job_id": job_id, "score": 0}), flush=True)
            continue

        df = pd.DataFrame([data])
        df["category"] = df["category"].astype(int)

        # M1
        d1 = df.copy()
        enc = encoder.transform(d1[["category"]])
        enc = pd.DataFrame(enc, columns=encoder.get_feature_names_out(["category"]), index=d1.index)
        d1[num_cols] = scaler.transform(d1[num_cols])
        X1 = pd.concat([d1.drop(columns="category"), enc], axis=1)[m1_cols]
        s1 = int(round(float(m1.predict_proba(X1)[0,1]) * 1000))

        # M2: median -> median_180, min -> min_180
        X2 = pd.DataFrame([{
            "price": data["price"],
            "median_180": data["median"],
            "min_180": data["min"],
            "flash_factor": data["flash_factor"]
        }])
        s2 = int(round(float(m2.predict_proba(X2)[0,1]) * 1000))

        print(json.dumps({
            "job_id": job_id,
            "score": round((s1 + s2) / 2),
            #"m1": s1,
            #"m2": s2
        }), flush=True)

    except Exception as e:
        print(json.dumps({"error": str(e)}), flush=True)