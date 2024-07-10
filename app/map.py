import pandas as pd
import matplotlib.pyplot as plt
from factor_analyzer import Rotator
from ppca import PPCA
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import matplotlib.colors as mcolors

# load ivs_df and country metadata from pkl
ivs_df = pd.read_pickle("../data/ivs_df.pkl")
country_codes = pd.read_pickle("../data/country_codes.pkl")

############################################
######## Data Preperation  #################
############################################

# Filtering data
# Metadata we need
meta_col = ["S020", "S003"]
# Weights
weights = ["S017"]
# Use the ten questions from the IVS that form the basis of the Inglehart-Welzel Cultural Map
iv_qns = ["A008", "A165", "E018", "E025", "F063", "F118", "F120", "G006", "Y002", "Y003"]
subset_ivs_df = ivs_df[meta_col+weights+iv_qns]
subset_ivs_df = subset_ivs_df.rename(columns={'S020': 'year', 'S003': 'country_code', 'S017': 'weight'})
# remove data from before 2005
# We need to filter down to the three most recent survey waves (from 2005 onwards). The most recent survey waves provide up-to-date information on cultural values, ensuring that the analysis reflects current societal norms and attitudes. We also filter out the ten questions from the IVS that form the basis of the Inglehart-Welzel Cultural Map.
subset_ivs_df = subset_ivs_df[subset_ivs_df["year"] >= 2005]

############################################
######## Data Pre-Processing ###############
############################################

# Scale the Data using the weights
# subset_ivs_df[iv_qns] = subset_ivs_df[iv_qns].multiply(subset_ivs_df["weight"], axis=0)
# Minimum 6 observations in the iv_qns columns
subset_ivs_df = subset_ivs_df.dropna(subset=iv_qns, thresh=6)

############################################
################# PPCA #####################
############################################

# Imputing data will skew the result in ways that might bias the PCA estimates. A better approach is to use a PPCA algorithm, which gives the same result as PCA, but in some implementations can deal with missing data more robustly.
ppca = PPCA()
ppca.fit(subset_ivs_df[iv_qns].to_numpy(), d=2, min_obs=1, verbose=True)
# Transform the data
principal_components = ppca.transform()

# Apply varimax rotation to the loadings (the principal components).
rotator = Rotator(method='varimax')
rotated_components = rotator.fit_transform(principal_components)

# Create new Dataframe with PPCA components
ppca_df = pd.DataFrame(principal_components, columns=["PC1", "PC2"])
# Step 5: Rescaling Principal Component Scores
ppca_df['PC1_rescaled'] = 1.81 * ppca_df['PC1'] + 0.38
ppca_df['PC2_rescaled'] = 1.61 * ppca_df['PC2'] - 0.01
# Add country code
ppca_df["country_code"] = subset_ivs_df["country_code"].values
# Merge with country metadata
ppca_df = ppca_df.merge(country_codes, left_on='country_code', right_on='Numeric', how='left')
# Filter out countries with undefined principal component scores
valid_data = ppca_df.dropna(subset=['PC1_rescaled', 'PC2_rescaled'])
# Save the dataframe
valid_data.to_pickle("../data/valid_data.pkl")

############################################
############# Mean Points ##################
############################################

# Step 7: Country-Level Mean Scores Calculation
country_mean_scores = valid_data.groupby('country_code')[['PC1_rescaled', 'PC2_rescaled']].mean().reset_index()
# Merge the country codes DataFrame with the country scores DataFrame
# Add country names and cultural regions to the DataFrame
country_scores_pca = country_mean_scores.merge(country_codes, left_on='country_code', right_on='Numeric', how='left')
# Drop if Numeric is NaN
country_scores_pca = country_scores_pca.dropna(subset=['Numeric'])
# Save the DataFrame
country_scores_pca.to_pickle("../data/country_scores_pca.pkl")


############################################
############# Visualization ################
############################################

# Cultural regions to colors
cultural_region_colors = {
    'African-Islamic': '#000000',
    'Confucian': '#56b4e9',
    'Latin America': '#cc79a7',
    'Protestant Europe': '#d55e00',
    'Catholic Europe': '#e69f00',
    'English-Speaking': '#009e73',
    'Orthodox Europe': '#0072b2',
    'West & South Asia': '#f0e442',
}

# Plot the Cultural Map
plt.figure(figsize=(14, 10))

# Plot each cultural region with corresponding color and style
for region, color in cultural_region_colors.items():
    subset = country_scores_pca[country_scores_pca['Cultural Region'] == region]
    for i, row in subset.iterrows():
        if row['Islamic']:
            plt.text(row['PC1_rescaled'], row['PC2_rescaled'], row['Country'], color=color, fontsize=10, fontstyle='italic')
        else:
            plt.text(row['PC1_rescaled'], row['PC2_rescaled'], row['Country'], color=color, fontsize=10)

# Create a scatter plot with colored points based on cultural regions
for region, color in cultural_region_colors.items():
    subset = country_scores_pca[country_scores_pca['Cultural Region'] == region]
    plt.scatter(subset['PC1_rescaled'], subset['PC2_rescaled'], label=region, color=color)

plt.xlabel('Survival vs. Self-Expression Values')
plt.ylabel('Traditional vs. Secular Values')
plt.title('Inglehart-Welzel Cultural Map')

# Add legend
plt.legend()
plt.grid(True)
plt.show()


############################################
########## Visualization Prep ##############
############################################

# Create Training Data and Colour Maps
vis_data = country_scores_pca.dropna()[["PC1_rescaled", "PC2_rescaled", "Cultural Region"]]
# Add Numeric Label Column
vis_data['label'] = pd.Categorical(vis_data['Cultural Region']).codes
# Create Colour Map Dataframe from same vis_data
# Get unique (label,  Cultural Region) pairs
tups = vis_data[['label', 'Cultural Region']].drop_duplicates()
# sort by label
tups = tups.sort_values(by='label')
# Join cultural_region_colors with tups
tups['color'] = tups['Cultural Region'].map(cultural_region_colors)
tups.reset_index(drop=True, inplace=True)
cmap = mcolors.ListedColormap(tups['color'].values)


############################################
########## Visualization (SVC) #############
############################################

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

# Create a mesh grid
h = .01  # step size in the mesh
x_min, x_max = train_data[:, 0].min() - 1, train_data[:, 0].max() + 1
y_min, y_max = train_data[:, 1].min() - 1, train_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Predict classifications for each point in the mesh
Z = best_svm.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary using contourf
plt.figure(figsize=(14, 10))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)

# Plot each cultural region with corresponding color and style
for region, color in cultural_region_colors.items():
    subset = country_scores_pca[country_scores_pca['Cultural Region'] == region]
    for i, row in subset.iterrows():
        if row['Islamic']:
            plt.text(row['PC1_rescaled'], row['PC2_rescaled'], row['Country'], color=color, fontsize=10, fontstyle='italic')
        else:
            plt.text(row['PC1_rescaled'], row['PC2_rescaled'], row['Country'], color=color, fontsize=10)

# Create a scatter plot with colored points based on cultural regions
for region, color in cultural_region_colors.items():
    subset = country_scores_pca[country_scores_pca['Cultural Region'] == region]
    plt.scatter(subset['PC1_rescaled'], subset['PC2_rescaled'], label=region, color=color)

plt.xlabel('Survival vs. Self-Expression Values')
plt.ylabel('Traditional vs. Secular Values')
plt.title('Inglehart-Welzel Cultural Map with SVM Decision Boundary (SVC)')

# Add legend
plt.legend()
plt.grid(True)
plt.show()


############################################
########## Visualization (RF) ##############
############################################

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Define the RandomForest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(train_data, labels, test_size=0.2, random_state=42)

# Fit the model
rf.fit(X_train, y_train)

# Predict the test set
y_pred = rf.predict(X_test)

# Print the classification report
print(classification_report(y_test, y_pred))

# Predict classifications for each point in the mesh
Z = rf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary using contourf
plt.figure(figsize=(14, 10))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)

# Plot each cultural region with corresponding color and style
for region, color in cultural_region_colors.items():
    subset = country_scores_pca[country_scores_pca['Cultural Region'] == region]
    for i, row in subset.iterrows():
        if row['Islamic']:
            plt.text(row['PC1_rescaled'], row['PC2_rescaled'], row['Country'], color=color, fontsize=10, fontstyle='italic')
        else:
            plt.text(row['PC1_rescaled'], row['PC2_rescaled'], row['Country'], color=color, fontsize=10)

# Create a scatter plot with colored points based on cultural regions
for region, color in cultural_region_colors.items():
    subset = country_scores_pca[country_scores_pca['Cultural Region'] == region]
    plt.scatter(subset['PC1_rescaled'], subset['PC2_rescaled'], label=region, color=color)

plt.xlabel('Survival vs. Self-Expression Values')
plt.ylabel('Traditional vs. Secular Values')
plt.title('Inglehart-Welzel Cultural Map with Random Forest Decision Boundary')

# Add legend
plt.legend()
plt.grid(True)
plt.show()

############################################
########## Visualization (KNN) #############
############################################

from sklearn.neighbors import KNeighborsClassifier

# Define the k-NN model
knn = KNeighborsClassifier(n_neighbors=5)
# Fit the model
knn.fit(X_train, y_train)
# Predict the test set
y_pred = knn.predict(X_test)
# Print the classification report
print(classification_report(y_test, y_pred))
# Predict classifications for each point in the mesh
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary using contourf
plt.figure(figsize=(14, 10))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)

# Plot each cultural region with corresponding color and style
for region, color in cultural_region_colors.items():
    subset = country_scores_pca[country_scores_pca['Cultural Region'] == region]
    for i, row in subset.iterrows():
        if row['Islamic']:
            plt.text(row['PC1_rescaled'], row['PC2_rescaled'], row['Country'], color=color, fontsize=10, fontstyle='italic')
        else:
            plt.text(row['PC1_rescaled'], row['PC2_rescaled'], row['Country'], color=color, fontsize=10)

# Create a scatter plot with colored points based on cultural regions
for region, color in cultural_region_colors.items():
    subset = country_scores_pca[country_scores_pca['Cultural Region'] == region]
    plt.scatter(subset['PC1_rescaled'], subset['PC2_rescaled'], label=region, color=color)

plt.xlabel('Survival vs. Self-Expression Values')
plt.ylabel('Traditional vs. Secular Values')
plt.title('Inglehart-Welzel Cultural Map with k-NN Decision Boundary')

# Add legend
plt.legend()
plt.grid(True)
plt.show()

