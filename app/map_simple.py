import pandas as pd
import matplotlib.pyplot as plt
from factor_analyzer import Rotator
from ppca import PPCA


class CulturalMap:

    def __init__(self, ivs_df_path, country_codes_path):
        self.ivs_df = pd.read_pickle(ivs_df_path)
        self.country_codes = pd.read_pickle(country_codes_path)
        self.subset_ivs_df = None
        self.ppca_df = None
        self.valid_data = None
        self.country_scores_pca = None
        self.cultural_region_colors = {
            'African-Islamic': '#000000',
            'Confucian': '#56b4e9',
            'Latin America': '#cc79a7',
            'Protestant Europe': '#d55e00',
            'Catholic Europe': '#e69f00',
            'English-Speaking': '#009e73',
            'Orthodox Europe': '#0072b2',
            'West & South Asia': '#f0e442',
        }
        # Metadata we need
        self.meta_col = ["S020", "S003"]
        # Weights
        self.weights = ["S017"]
        # Use the ten questions from the IVS that form the basis of the Inglehart-Welzel Cultural Map
        self.iv_qns = ["A008", "A165", "E018", "E025", "F063", "F118", "F120", "G006", "Y002", "Y003"]
        self.ppca = PPCA()
        self.rotator = Rotator(method='varimax')
        self.pc_rescale_params = {'PC1': (1.81, 0.38), 'PC2': (1.61, -0.01)}

    def prepare_data(self):
        """
        Data Preparation
        """
        # Filtering data
        self.subset_ivs_df = self.ivs_df[self.meta_col + self.weights + self.iv_qns]
        self.subset_ivs_df = self.subset_ivs_df.rename(
            columns={'S020': 'year', 'S003': 'country_code', 'S017': 'weight'})
        # Remove data from before 2005
        # We need to filter down to the three most recent survey waves (from 2005 onwards).
        # The most recent survey waves provide up-to-date information on cultural values,
        # ensuring that the analysis reflects current societal norms and attitudes.
        # We also filter out the ten questions from the IVS that form the basis of the Inglehart-Welzel Cultural Map.
        self.subset_ivs_df = self.subset_ivs_df[self.subset_ivs_df["year"] >= 2005]
        # Scale the Data using the weights
        # self.subset_ivs_df[self.iv_qns] = self.subset_ivs_df[self.iv_qns].multiply(self.subset_ivs_df["weight"], axis=0)
        # Minimum 6 observations in the iv_qns columns
        self.subset_ivs_df = self.subset_ivs_df.dropna(subset=self.subset_ivs_df.columns[3:], thresh=6)

    def perform_ppca(self):
        """
        PPCA
        """
        # Imputing data will skew the result in ways that might bias the PCA estimates.
        # A better approach is to use a PPCA algorithm, which gives the same result as PCA,
        # but in some implementations can deal with missing data more robustly.
        self.ppca.fit(self.subset_ivs_df[self.iv_qns].to_numpy(), d=2, min_obs=1, verbose=True)
        # Transform the data
        principal_components = self.ppca.transform()
        # Apply varimax rotation to the loadings (the principal components).
        rotated_components = self.rotator.fit_transform(principal_components)
        # Create new Dataframe with PPCA components
        self.ppca_df = pd.DataFrame(rotated_components, columns=["PC1", "PC2"])
        # Step 5: Rescaling Principal Component Scores
        self.ppca_df['PC1_rescaled'] = self.pc_rescale_params['PC1'][0] * self.ppca_df['PC1'] + \
                                       self.pc_rescale_params['PC1'][1]
        self.ppca_df['PC2_rescaled'] = self.pc_rescale_params['PC2'][0] * self.ppca_df['PC2'] + \
                                       self.pc_rescale_params['PC2'][1]
        # Add country code
        self.ppca_df["country_code"] = self.subset_ivs_df["country_code"].values
        # Merge with country metadata
        self.ppca_df = self.ppca_df.merge(self.country_codes, left_on='country_code', right_on='Numeric', how='left')
        # Filter out countries with undefined principal component scores
        self.valid_data = self.ppca_df.dropna(subset=['PC1_rescaled', 'PC2_rescaled'])
        # Save the dataframe
        self.valid_data.to_pickle("../data/valid_data.pkl")

    def calculate_mean_scores(self):
        """
        Mean Points
        """
        # Step 7: Country-Level Mean Scores Calculation
        country_mean_scores = self.valid_data.groupby('country_code')[
            ['PC1_rescaled', 'PC2_rescaled']].mean().reset_index()
        # Merge the country codes DataFrame with the country scores DataFrame
        # Add country names and cultural regions to the DataFrame
        self.country_scores_pca = country_mean_scores.merge(self.country_codes, left_on='country_code',
                                                            right_on='Numeric', how='left')
        # Drop if Numeric is NaN
        self.country_scores_pca = self.country_scores_pca.dropna(subset=['Numeric'])
        # Save the DataFrame
        self.country_scores_pca.to_pickle("../data/country_scores_pca.pkl")

    def visualize_cultural_map(self, title='Inglehart-Welzel Cultural Map'):
        """
        Visualization
        """
        plt.figure(figsize=(14, 10))
        # Plot each cultural region with corresponding color and style
        for region, color in self.cultural_region_colors.items():
            subset = self.country_scores_pca[self.country_scores_pca['Cultural Region'] == region]
            for i, row in subset.iterrows():
                if row['Islamic']:
                    plt.text(row['PC1_rescaled'], row['PC2_rescaled'], row['Country'], color=color, fontsize=10,
                             fontstyle='italic')
                else:
                    plt.text(row['PC1_rescaled'], row['PC2_rescaled'], row['Country'], color=color, fontsize=10)
            # Create a scatter plot with colored points based on cultural regions
            plt.scatter(subset['PC1_rescaled'], subset['PC2_rescaled'], label=region, color=color)
        plt.xlabel('Survival vs. Self-Expression Values')
        plt.ylabel('Traditional vs. Secular Values')
        plt.title(title)
        # Add legend
        plt.legend()
        plt.grid(True)
        plt.show()

    def save_ppca_model(self, fpath):
        """
        Save the PPCA model parameters to a file.
        """
        self.ppca.save(fpath)

    def load_ppca_model(self, fpath):
        """
        Load the PPCA model parameters from a file.
        """
        self.ppca.load(fpath)

    def project_new_data(self, new_data):
        """
        Project new data onto the existing cultural map using the saved PPCA model.
        """
        # New data should be 10 x n

        # Ensure the new data is in a DataFrame and has the same columns as the original IVS questions
        new_data_df = pd.DataFrame(new_data, columns=self.iv_qns)
        # Apply the PPCA transformation using the loaded model
        principal_components = self.ppca.transform(new_data_df.to_numpy())
        # Apply the varimax rotation
        rotated_components = self.rotator.transform(principal_components)
        # Create a DataFrame for the new principal components
        new_ppca_df = pd.DataFrame(rotated_components, columns=["PC1", "PC2"])

        # Rescale the principal component scores using the same parameters
        new_ppca_df['PC1_rescaled'] = self.pc_rescale_params['PC1'][0] * new_ppca_df['PC1'] + \
                                      self.pc_rescale_params['PC1'][1]
        new_ppca_df['PC2_rescaled'] = self.pc_rescale_params['PC2'][0] * new_ppca_df['PC2'] + \
                                      self.pc_rescale_params['PC2'][1]

        return new_ppca_df


# Example usage
if __name__ == "__main__":
    cultural_map = CulturalMap("../data/ivs_df.pkl", "../data/country_codes.pkl")
    cultural_map.prepare_data()
    cultural_map.perform_ppca()
    cultural_map.calculate_mean_scores()
    cultural_map.visualize_cultural_map()


    # GPT-4
    gpt_4 = [
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    ]

    # LLama
    llama_3 = [
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    ]

    # Qwen
    qwen = [
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    ]

    # New dataframe, "model" column included
    new_data = pd.DataFrame(gpt_4, columns=cultural_map.iv_qns)
    new_data["model"] = "GPT-4"
    new_data = new_data.append(pd.DataFrame(llama_3, columns=cultural_map.iv_qns))
    new_data["model"] = "LLama"
    new_data = new_data.append(pd.DataFrame(qwen, columns=cultural_map.iv_qns))
    new_data["model"] = "Qwen"






    # Project new data
    new_projection = cultural_map.project_new_data(new_data)
    print(new_projection)
