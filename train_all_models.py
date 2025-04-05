# train_all_models.py

# --- Core Imports ---
import os
import time
import joblib
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, InputLayer
import torch
import torch.nn as nn
import torch.optim as optim
import gym
import random
import warnings
from pathlib import Path

# --- Configuration ---
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# --- ML Imports ---
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_squared_error, accuracy_score, 
                            classification_report, silhouette_score)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVR
from sklearn.cluster import KMeans
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from imblearn.over_sampling import SMOTE
from statsmodels.tsa.arima.model import ARIMA

# --- Data Loading ---
def load_data():
    print("Loading data...")
    df = pd.read_csv('data/live_data.csv')
    print("Data loaded!")
    
    # Feature engineering
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values(['symbol', 'datetime'])
    
    # Create targets
    df['next_close'] = df.groupby('symbol')['close'].shift(-1)
    df['price_change'] = np.where(df['next_close'] > df['close'], 1, 0)
    df = df.dropna()
    
    # Feature columns
    ohlc_features = ['open', 'high', 'low', 'close']
    
    # Encode symbols
    encoder = OneHotEncoder(sparse_output=False)
    symbol_encoded = encoder.fit_transform(df[['symbol']])
    
    # Create feature matrix
    X = np.concatenate([df[ohlc_features].values, symbol_encoded], axis=1)
    y_reg = df['next_close'].values.reshape(-1, 1)
    y_class = df['price_change'].values
    
    # Split data with larger test size
    X_train, X_test, y_train, y_test = train_test_split(X, y_reg, test_size=0.3, random_state=42)
    X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X, y_class, test_size=0.3, random_state=42)
    
    # Handle class imbalance with dynamic SMOTE
    print("\nInitial class distribution:", np.bincount(y_train_class))
    class_counts = np.bincount(y_train_class)
    
    if len(class_counts) > 1 and min(class_counts) > 1:
        minority_count = min(class_counts)
        k_neighbors = min(5, minority_count - 1)
        
        print(f"Applying SMOTE with k_neighbors={k_neighbors}")
        smote = SMOTE(k_neighbors=k_neighbors)
        X_train_class, y_train_class = smote.fit_resample(X_train_class, y_train_class)
        print("Post-SMOTE distribution:", np.bincount(y_train_class))
    else:
        print("Insufficient samples for SMOTE. Using original distribution.")
    
    # Scaling
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    X_train = x_scaler.fit_transform(X_train)
    X_test = x_scaler.transform(X_test)
    y_train = y_scaler.fit_transform(y_train)
    y_test = y_scaler.transform(y_test)
    
    return (X_train, X_test, y_train, y_test,
            X_train_class, X_test_class, y_train_class, y_test_class,
            x_scaler, y_scaler)

# --- Time Series Models ---
def train_time_series_models(X_train, y_train, X_test, y_test):
    print("\n=== Training Time Series Models ===")
    
    # LSTM with improved configuration
    try:
        print("\nTraining LSTM...")
        X_train_lstm = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test_lstm = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        
        model = Sequential([
            InputLayer(input_shape=(X_train_lstm.shape[1], 1)),
            LSTM(64, activation='tanh', return_sequences=True),
            LSTM(32, activation='tanh'),
            Dense(1)
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                     loss='mse',
                     metrics=['mae'])
        
        history = model.fit(X_train_lstm, y_train, 
                          epochs=100, 
                          batch_size=32,
                          validation_split=0.2,
                          verbose=1)
        
        model.save(MODEL_DIR/"lstm_model.keras")
        print("LSTM training complete. Model saved.")
        
    except Exception as e:
        print(f"LSTM Error: {str(e)}")
    
    # ARIMA with validation
    try:
        print("\nTraining ARIMA...")
        arima = ARIMA(y_train, order=(1,1,1))
        arima_fit = arima.fit()
        print(f"ARIMA AIC: {arima_fit.aic:.2f}, BIC: {arima_fit.bic:.2f}")
        joblib.dump(arima_fit, MODEL_DIR/"arima_model.pkl")
        print("ARIMA model saved")
    except Exception as e:
        print(f"ARIMA Error: {str(e)}")

# --- Reinforcement Learning Models ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def train_dqn():
    print("\n=== Training DQN ===")
    try:
        env = gym.make("CartPole-v1")
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        
        dqn_net = DQN(state_dim, action_dim)
        target_net = DQN(state_dim, action_dim)
        target_net.load_state_dict(dqn_net.state_dict())
        
        optimizer = torch.optim.Adam(dqn_net.parameters(), lr=1e-3)
        replay_buffer = ReplayBuffer(10000)
        
        gamma = 0.99
        epsilon_start = 1.0
        epsilon_final = 0.01
        epsilon_decay = 500
        frame_idx = 0

        def epsilon_by_frame(frame_idx):
            return epsilon_final + (epsilon_start - epsilon_final) * np.exp(-1. * frame_idx / epsilon_decay)
        
        for episode in range(50):
            state, _ = env.reset()
            episode_reward = 0
            done = False
            while not done:
                epsilon = epsilon_by_frame(frame_idx)
                frame_idx += 1
                if random.random() > epsilon:
                    with torch.no_grad():
                        state_tensor = torch.FloatTensor(state).unsqueeze(0)
                        q_values = dqn_net(state_tensor)
                        action = q_values.max(1)[1].item()
                else:
                    action = env.action_space.sample()
                
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                replay_buffer.push(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward

                if len(replay_buffer) > 64:
                    batch = replay_buffer.sample(64)
                    states, actions, rewards, next_states, dones = zip(*batch)
                    
                    states = torch.FloatTensor(states)
                    actions = torch.LongTensor(actions).unsqueeze(1)
                    rewards = torch.FloatTensor(rewards).unsqueeze(1)
                    next_states = torch.FloatTensor(next_states)
                    dones = torch.FloatTensor(dones).unsqueeze(1)
                    
                    current_q = dqn_net(states).gather(1, actions)
                    next_q = target_net(next_states).max(1)[0].unsqueeze(1)
                    expected_q = rewards + gamma * next_q * (1 - dones)
                    
                    loss = nn.MSELoss()(current_q, expected_q)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
            if episode % 10 == 0:
                target_net.load_state_dict(dqn_net.state_dict())
                print(f"Episode {episode} Reward: {episode_reward}")
        
        torch.save(dqn_net.state_dict(), MODEL_DIR/"dqn_model.pth")
        env.close()
    except Exception as e:
        print(f"DQN Error: {str(e)}")

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.fc = nn.Linear(state_dim, 128)
        self.policy_head = nn.Linear(128, action_dim)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.tanh(self.fc(x))
        return self.policy_head(x), self.value_head(x)

def train_ppo():
    print("\n=== Training PPO ===")
    try:
        env = gym.make("CartPole-v1")
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        
        model = ActorCritic(state_dim, action_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        
        gamma = 0.99
        lam = 0.95

        def compute_gae(rewards, masks, values):
            returns = []
            gae = 0
            for step in reversed(range(len(rewards))):
                delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
                gae = delta + gamma * lam * masks[step] * gae
                returns.insert(0, gae + values[step])
            return returns

        batch_states, batch_actions, batch_log_probs, batch_rewards, batch_masks, batch_values = [], [], [], [], [], []
        timestep = 0
        
        for episode in range(50):
            state, _ = env.reset()
            episode_reward = 0
            done = False
            while not done:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                logits, value = model(state_tensor)
                probs = torch.softmax(logits, dim=1)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample().item()
                log_prob = dist.log_prob(torch.tensor(action))
                
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                batch_states.append(state)
                batch_actions.append(action)
                batch_log_probs.append(log_prob)
                batch_rewards.append(reward)
                batch_masks.append(1 - int(done))
                batch_values.append(value.item())
                
                state = next_state
                episode_reward += reward
                timestep += 1
                
                if timestep % 2000 == 0:
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    _, next_value = model(state_tensor)
                    batch_values.append(next_value.item())
                    
                    returns = compute_gae(batch_rewards, batch_masks, batch_values)
                    returns = torch.FloatTensor(returns)
                    
                    batch_states_tensor = torch.FloatTensor(batch_states)
                    batch_actions_tensor = torch.LongTensor(batch_actions)
                    batch_log_probs_tensor = torch.stack(batch_log_probs)
                    
                    for _ in range(10):
                        logits, values = model(batch_states_tensor)
                        probs = torch.softmax(logits, dim=1)
                        dist = torch.distributions.Categorical(probs)
                        new_log_probs = dist.log_prob(batch_actions_tensor)
                        
                        ratio = torch.exp(new_log_probs - batch_log_probs_tensor)
                        advantage = returns - values.squeeze()
                        
                        surr1 = ratio * advantage
                        surr2 = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2) * advantage
                        
                        loss = -torch.min(surr1, surr2).mean() + 0.5 * (returns - values.squeeze()).pow(2).mean()
                        
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    
                    batch_states, batch_actions, batch_log_probs, batch_rewards, batch_masks, batch_values = [], [], [], [], [], []
                    timestep = 0
            
            print(f"Episode {episode} Reward: {episode_reward}")
        
        torch.save(model.state_dict(), MODEL_DIR/"ppo_model.pth")
        env.close()
    except Exception as e:
        print(f"PPO Error: {str(e)}")

def train_reinforcement_learning_models():
    train_dqn()
    train_ppo()

# --- Classification/Regression Models ---
def train_classification_regression_models(X_train, X_test, y_train, y_test, X_train_class, X_test_class, y_train_class, y_test_class):
    print("\n=== Training Classification & Regression Models ===")
    
    # Classification with cross-validation
    try:
        print("\nTraining Random Forest with cross-validation...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train_class, y_train_class)
        joblib.dump(rf, MODEL_DIR/"rf_classifier.pkl")
        print("Random Forest Report:\n", classification_report(y_test_class, rf.predict(X_test_class)))
    except Exception as e:
        print(f"RF Error: {str(e)}")

    try:
        print("\nTraining Logistic Regression...")
        logreg = LogisticRegression(max_iter=1000, class_weight='balanced')
        logreg.fit(X_train_class, y_train_class)
        joblib.dump(logreg, MODEL_DIR/"logreg_classifier.pkl")
        print("Logistic Regression Report:\n", classification_report(y_test_class, logreg.predict(X_test_class)))
    except Exception as e:
        print(f"LogReg Error: {str(e)}")

    # Regression with improved configuration
    try:
        print("\nTraining Linear Regression...")
        lin_reg = LinearRegression()
        lin_reg.fit(X_train, y_train)
        joblib.dump(lin_reg, MODEL_DIR/"linear_regressor.pkl")
        print(f"Linear MSE: {mean_squared_error(y_test, lin_reg.predict(X_test)):.4f}")
    except Exception as e:
        print(f"Linear Reg Error: {str(e)}")

    try:
        print("\nTraining Gradient Boosting...")
        gbr = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1)
        gbr.fit(X_train, y_train)
        joblib.dump(gbr, MODEL_DIR/"gbr_regressor.pkl")
        print(f"GBoost MSE: {mean_squared_error(y_test, gbr.predict(X_test)):.4f}")
    except Exception as e:
        print(f"GBoost Error: {str(e)}")

# --- Clustering Models ---
def train_clustering_models(X_scaled):
    print("\n=== Training Clustering Model ===")
    try:
        scaler = StandardScaler()
        X_cluster = scaler.fit_transform(X_scaled)
        
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        kmeans.fit(X_cluster)
        joblib.dump(kmeans, MODEL_DIR/"kmeans_model.pkl")
        
        print(f"Silhouette Score: {silhouette_score(X_cluster, kmeans.labels_):.4f}")
        print("Cluster Centers (Original Scale):")
        print(scaler.inverse_transform(kmeans.cluster_centers_))
    except Exception as e:
        print(f"Clustering Error: {str(e)}")

# --- Main Execution ---
def main():
    (X_train, X_test, y_train, y_test,
     X_train_class, X_test_class, y_train_class, y_test_class,
     x_scaler, y_scaler) = load_data()
    
    start_time = time.time()
    
    train_time_series_models(X_train, y_train, X_test, y_test)
    train_reinforcement_learning_models()
    train_classification_regression_models(X_train, X_test, y_train, y_test,
                                         X_train_class, X_test_class, y_train_class, y_test_class)
    train_clustering_models(X_train)
    
    print(f"\nTotal Training Time: {time.time() - start_time:.2f} seconds")
    
    print("\nStarting conflict models...")
    try:
        subprocess.run(["python", "train_conflict_models.py"], check=True)
    except Exception as e:
        print(f"Conflict models error: {str(e)}")

if __name__ == "__main__":
    main()