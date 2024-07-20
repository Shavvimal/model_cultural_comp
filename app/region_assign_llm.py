import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

results = pd.read_pickle("../data/res_country_scores_pca.pkl")

non_llm_data = results[results["llm"] == False]
llm_data = results[results["llm"] == True]


# Create Training Data
vis_data = non_llm_data.dropna()[["PC1_rescaled", "PC2_rescaled", "Cultural Region"]]
# Add Numeric Label Column
vis_data['label'] = pd.Categorical(vis_data['Cultural Region']).codes

x = vis_data['PC1_rescaled']
y = vis_data['PC2_rescaled']
train_data = np.column_stack((x, y)).astype(float)
labels = np.array(vis_data['label']).astype(int)

# Define the parameter grid
param_grid_fine = {
    'C': [500, 1000, 1500, 2000],
    'gamma': [0.05, 0.1, 0.15, 0.2],
    'kernel': ['rbf']
}

# Create a SVM model
svm = SVC()
# Create a GridSearchCV object
grid_search = GridSearchCV(svm, param_grid_fine, refit=True, verbose=2, cv=5)
# Fit the model
grid_search.fit(train_data, labels)
# Print the best parameters
print("Best parameters found: ", grid_search.best_params_)
# Use the best parameters to train the SVM
best_svm = grid_search.best_estimator_
# Fit the best model
best_svm.fit(train_data, labels)


# Predict cultural region for LLM data
x_llm = llm_data['PC1_rescaled']
y_llm = llm_data['PC2_rescaled']
llm_data_points = np.column_stack((x_llm, y_llm)).astype(float)

# Predict using the best SVM model
llm_predictions = best_svm.predict(llm_data_points)

# Map numeric labels back to cultural regions
label_to_region = dict(enumerate(pd.Categorical(vis_data['Cultural Region']).categories))
llm_data['Predicted Cultural Region'] = [label_to_region[label] for label in llm_predictions]