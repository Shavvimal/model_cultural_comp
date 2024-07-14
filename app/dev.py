import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from factor_analyzer import Rotator
from ppca import PPCA
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import matplotlib.colors as mcolors
from scipy.ndimage import center_of_mass
from matplotlib import colormaps

def load_data():
    ivs_df = pd.read_pickle("../data/ivs_df.pkl")
    country_codes = pd.read_pickle("../data/country_codes.pkl")
    return ivs_df, country_codes

def preprocess_data(ivs_df):
    meta_col = ["S020", "S003"]
    weights = ["S017"]
    iv_qns = ["A008", "A165", "E018", "E025", "F063", "F118", "F120", "G006", "Y002", "Y003"]
    subset_ivs_df = ivs_df[meta_col + weights + iv_qns]
    subset_ivs_df = subset_ivs_df.rename(columns={'S020': 'year', 'S003': 'country_code', 'S017': 'weight'})
    subset_ivs_df = subset_ivs_df[subset_ivs_df["year"] >= 2005]
    subset_ivs_df = subset_ivs_df.dropna(subset=iv_qns, thresh=6)
    return subset_ivs_df, iv_qns

def perform_ppca(subset_ivs_df, iv_qns):
    ppca = PPCA()
    ppca.fit(subset_ivs_df[iv_qns].to_numpy(), d=2, min_obs=1, verbose=True)
    principal_components = ppca.transform()
    rotator = Rotator(method='varimax')
    rotated_components = rotator.fit_transform(principal_components)
    ppca_df = pd.DataFrame(principal_components, columns=["PC1", "PC2"])
    ppca_df['PC1_rescaled'] = 1.81 * ppca_df['PC1'] + 0.38
    ppca_df['PC2_rescaled'] = 1.61 * ppca_df['PC2'] - 0.01
    return ppca_df

def merge_with_metadata(ppca_df, subset_ivs_df, country_codes):
    ppca_df["country_code"] = subset_ivs_df["country_code"].values
    ppca_df = ppca_df.merge(country_codes, left_on='country_code', right_on='Numeric', how='left')
    valid_data = ppca_df.dropna(subset=['PC1_rescaled', 'PC2_rescaled'])
    return valid_data

def calculate_mean_scores(valid_data, country_codes):
    country_mean_scores = valid_data.groupby('country_code')[['PC1_rescaled', 'PC2_rescaled']].mean().reset_index()
    country_scores_pca = country_mean_scores.merge(country_codes, left_on='country_code', right_on='Numeric', how='left')
    country_scores_pca = country_scores_pca.dropna(subset=['Numeric'])
    return country_scores_pca

def create_color_map():
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
    return cultural_region_colors

def visualize_cultural_map(country_scores_pca, cultural_region_colors, title):
    plt.figure(figsize=(14, 10))
    for region, color in cultural_region_colors.items():
        subset = country_scores_pca[country_scores_pca['Cultural Region'] == region]
        for i, row in subset.iterrows():
            if row['Islamic']:
                plt.text(row['PC1_rescaled'], row['PC2_rescaled'], row['Country'], color=color, fontsize=10, fontstyle='italic')
            else:
                plt.text(row['PC1_rescaled'], row['PC2_rescaled'], row['Country'], color=color, fontsize=10)
        plt.scatter(subset['PC1_rescaled'], subset['PC2_rescaled'], label=region, color=color)
    plt.xlabel('Survival vs. Self-Expression Values')
    plt.ylabel('Traditional vs. Secular Values')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def prepare_visualization_data(country_scores_pca, cultural_region_colors):
    vis_data = country_scores_pca.dropna()[["PC1_rescaled", "PC2_rescaled", "Cultural Region"]]
    vis_data['label'] = pd.Categorical(vis_data['Cultural Region']).codes
    tups = vis_data[['label', 'Cultural Region']].drop_duplicates()
    tups = tups.sort_values(by='label')
    tups['color'] = tups['Cultural Region'].map(cultural_region_colors)
    tups.reset_index(drop=True, inplace=True)
    cmap = mcolors.ListedColormap(tups['color'].values)
    return vis_data, cmap, tups

def create_meshgrid(data, step_size=0.01):
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step_size),
                         np.arange(y_min, y_max, step_size))
    return xx, yy

def plot_contours(ax, model, xx, yy, **params):
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

def train_svm(train_data, labels, param_grid):
    grid_search = GridSearchCV(SVC(), param_grid, refit=True, verbose=2, cv=5)
    grid_search.fit(train_data, labels)
    best_svm = grid_search.best_estimator_
    best_svm.fit(train_data, labels)
    return best_svm, grid_search.best_params_

def train_random_forest(train_data, labels):
    X_train, X_test, y_train, y_test = train_test_split(train_data, labels, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    print(classification_report(y_test, y_pred))
    return rf

def train_knn(train_data, labels):
    X_train, X_test, y_train, y_test = train_test_split(train_data, labels, test_size=0.2, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print(classification_report(y_test, y_pred))
    return knn

def main():
    ivs_df, country_codes = load_data()
    subset_ivs_df, iv_qns = preprocess_data(ivs_df)
    ppca_df = perform_ppca(subset_ivs_df, iv_qns)
    valid_data = merge_with_metadata(ppca_df, subset_ivs_df, country_codes)
    country_scores_pca = calculate_mean_scores(valid_data, country_codes)
    cultural_region_colors = create_color_map()
    visualize_cultural_map(country_scores_pca, cultural_region_colors, 'Inglehart-Welzel Cultural Map')

if __name__ == "__main__":
    main()
