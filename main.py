import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np

def print_separator(title=""):
    print("\n" + "=" * 30)
    if title:
        print(title)
    print("=" * 30)

def load_data(file_name):
    data = pd.read_csv(file_name)
    pd.set_option('display.max_columns', 100)
    print_separator("Data Head")
    print(data.head())
    print_separator("All column names:")
    print(f' {data.columns}')
    return data.dropna(axis=0)

def select_features(data, feature_list):
    X = data[feature_list]
    y = data.AveragePrice
    return X, y

def train_decision_tree(X, y, random_state=1):
    model = DecisionTreeRegressor(random_state=random_state)
    model.fit(X, y)
    return model

def evaluate_model(model, X, y, message="Predictions"):
    predictions = model.predict(X)
    print_separator(message)
    # for true, pred in zip(y, predictions):
        # print(f"Actual: {true}, Predicted: {pred}")

def split_data(X, y, random_state=1):
    return train_test_split(X, y, random_state=random_state)

def calculate_mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def get_optimal_leaf_nodes(X_train, X_val, y_train, y_val, leaf_nodes_list):
    thelist = {}
    print_separator("Optimal Leaf Nodes Search")
    for max_leaf_nodes in leaf_nodes_list:
        model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
        model.fit(X_train, y_train)
        preds_val = model.predict(X_val)
        mae = calculate_mae(y_val, preds_val)
        thelist[max_leaf_nodes] = mae
        print(f"Max leaf nodes: {max_leaf_nodes} \t Mean Absolute Error: {mae}")
    optimal_nodes = min(thelist, key=thelist.get)
    return optimal_nodes, thelist 

def train_random_forest(X_train, y_train, n_estimators=70, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features='auto', random_state=1):
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=None,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model

# Main script
file = 'data/avocado.csv'
data = load_data(file)
features = ['Total Volume', 'S_hass_avocados', 'Total Bags']
X, y = select_features(data, features)
train_X, val_X, train_y, val_y = split_data(X, y)

# Decision Tree with full data
dt_model_full = train_decision_tree(X, y)
evaluate_model(dt_model_full, X.head(10), y.head(10), "Initial Decision Tree Predictions")

# Decision Tree with train/validation split
dt_model_split = train_decision_tree(train_X, train_y)
evaluate_model(dt_model_split, val_X, val_y, "Decision Tree Validation Predictions")

# Find optimal leaf nodes
leaf_nodes_list = [5, 25, 50, 100, 250, 500, 750]
optimal_nodes, leaf_nodes_results = get_optimal_leaf_nodes(train_X, val_X, train_y, val_y, leaf_nodes_list)
print_separator(f"Optimal number of leaf nodes: {optimal_nodes}")

# Retrain with optimal leaf nodes
final_dt_model = DecisionTreeRegressor(max_leaf_nodes=optimal_nodes, random_state=1)
final_dt_model.fit(X, y)
evaluate_model(final_dt_model, X.head(20), y.head(20), "Optimal Leaf Nodes Predictions")

# Random Forest Model
rf_model = train_random_forest(train_X, train_y)
rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = calculate_mae(val_y, rf_val_predictions)
print_separator("Random Forest Model Evaluation")
print(f'Random Forest Model Mean Absolute Error: {rf_val_mae}')




# Function to create synthetic data
def create_synthetic_data(original_data, n_samples=10):
    np.random.seed(0)
    
    # Randomly sample values from specified columns
    Total_Volume = np.random.choice(original_data['Total Volume'], size=n_samples)
    S_hass_avocados = np.random.choice(original_data['S_hass_avocados'], size=n_samples)
    Total_Bags = np.random.choice(original_data['Total Bags'], size=n_samples)

    synthetic_data = pd.DataFrame({
        'Total Volume': Total_Volume,
        'S_hass_avocados': S_hass_avocados,
        'Total Bags': Total_Bags
    })

    return synthetic_data


# Create synthetic data
new_data = create_synthetic_data(data, 10)


# Predict prices using the Random Forest model
new_predictions = rf_model.predict(new_data)

print("Synthetic Data:")
print(new_data)

# Print predictions
print("\nPredicted Average Prices:")
print(new_predictions)