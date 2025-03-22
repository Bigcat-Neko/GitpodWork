import tensorflow as tf
import xgboost as xgb
import pickle
import joblib
import stable_baselines3 as sb3

def load_models():
    models = {}

    # Load CNN Model
    try:
        models["cnn"] = tf.keras.models.load_model("models/cnn_model.keras")
        print("[✅] CNN model loaded successfully!")
    except Exception as e:
        print(f"[❌] Error loading CNN model: {e}")

    # Load GRU Model
    try:
        models["gru"] = tf.keras.models.load_model("models/gru_model.h5")
        print("[✅] GRU model loaded successfully!")
    except Exception as e:
        print(f"[❌] Error loading GRU model: {e}")

    # Load XGBoost Model
    try:
        models["xgboost"] = xgb.XGBRegressor()
        models["xgboost"].load_model("models/xgboost_model.json")
        print("[✅] XGBoost model loaded successfully!")
    except Exception as e:
        print(f"[❌] Error loading XGBoost model: {e}")

    # Load LightGBM Model
    try:
        models["lightgbm"] = joblib.load("models/lightgbm_model.pkl")
        print("[✅] LightGBM model loaded successfully!")
    except Exception as e:
        print(f"[❌] Error loading LightGBM model: {e}")

    # Load Stacking Model
    try:
        models["stacking"] = joblib.load("models/stacking_model.pkl")
        print("[✅] Stacking model loaded successfully!")
    except Exception as e:
        print(f"[❌] Error loading Stacking model: {e}")

    # Load Sentiment Model
    try:
        with open("models/sentiment_model.pkl", "rb") as f:
            models["sentiment"] = pickle.load(f)
        print("[✅] Sentiment model loaded successfully!")
    except Exception as e:
        print(f"[❌] Error loading Sentiment model: {e}")

    # Load Anomaly Detection Model
    try:
        with open("models/anomaly_model.pkl", "rb") as f:
            models["anomaly"] = pickle.load(f)
        print("[✅] Anomaly Detection model loaded successfully!")
    except Exception as e:
        print(f"[❌] Error loading Anomaly Detection model: {e}")

    # Load Price Scaler
    try:
        with open("models/price_scaler.pkl", "rb") as f:
            models["scaler"] = pickle.load(f)
        print("[✅] Price Scaler loaded successfully!")
    except Exception as e:
        print(f"[❌] Error loading Price Scaler: {e}")

    # Load Trade Signal Model
    try:
        models["trade_signal"] = tf.keras.models.load_model("models/trade_signal_model.h5")
        print("[✅] Trade Signal model loaded successfully!")
    except Exception as e:
        print(f"[❌] Error loading Trade Signal model: {e}")

    # Load PPO Trading Model
    try:
        models["ppo"] = sb3.PPO.load("models/trading_ppo.zip")
        print("[✅] PPO Trading model loaded successfully!")
    except Exception as e:
        print(f"[❌] Error loading PPO Trading model: {e}")

    # Load DQN Trading Model
    try:
        models["dqn"] = sb3.DQN.load("models/trading_dqn.zip")
        print("[✅] DQN Trading model loaded successfully!")
    except Exception as e:
        print(f"[❌] Error loading DQN Trading model: {e}")

    return models

models = load_models()
print("\n📌 **Final Model Loading Process Complete!**")
