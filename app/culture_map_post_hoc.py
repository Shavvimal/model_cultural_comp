import pandas as pd
import matplotlib.pyplot as plt
from factor_analyzer import Rotator
from ppca import PPCA
import os
import glob
from typing import List



class CulturalMap:

    def __init__(self, ivs_df_path, country_codes_path):
        self.ivs_df = pd.read_pickle(ivs_df_path)
        self.country_codes = pd.read_pickle(country_codes_path)
        self.subset_ivs_df = None
        self.ppca_df = None
        self.valid_data = None
        self.country_scores_pca = None
        self.llm_scores_pca = None

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

    def visualize_cultural_map(self, title='Inglehart-Welzel Cultural Map', with_llms = False):
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

        if with_llms:
            # Plot each LLM from self.llm_scores_pca
            for i, row in self.llm_scores_pca.iterrows():
                if row['Chinese']:
                    plt.text(row['PC1_rescaled'], row['PC2_rescaled'], row['llm'], color="#bada55", fontsize=10,
                             fontstyle='italic')
                else:
                    plt.text(row['PC1_rescaled'], row['PC2_rescaled'], row['llm'], color="#bada55", fontsize=10)

                # Create a scatter plot with colored points based on cultural regions
                plt.scatter(row['PC1_rescaled'], row['PC2_rescaled'], label="AI Model", color="#bada55")


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


    ##############################################
    ############### LLM Plotting #################
    ##############################################
    @staticmethod
    def Y002_transform(ans: (int, int)):
        q_154 = ans[0]
        q_155 = ans[1]

        if q_154 < 0 or q_155 < 0:
            return -5
        if (q_154 == 1 and q_155 == 3) or (q_154 == 3 and q_155 == 1):
            return 1
        if (q_154 == 2 and q_155 == 4) or (q_154 == 4 and q_155 == 2):
            return 3

        return 2

    @staticmethod
    def Y003_transform(ans: List[int]):

        # Inputs are like this [6, 7, 8, 9, 10]
        # Return a list of true or fale from 0 through 10 based on if the number appears in the input
        boolList = [i in ans for i in range(1, 12)]
        # Map True to 1 and False to 2
        scores = [1 if i else 2 for i in boolList]
        qn_ans_dict = {
            "q7": scores[0],
            "q8": scores[1],
            "q9": scores[2],
            "q10": scores[3],
            "q11": scores[4],
            "q12": scores[5],
            "q13": scores[6],
            "q14": scores[7],
            "q15": scores[8],
            "q16": scores[9],
            "q17": scores[10],
        }

        # Compute Y003=-5.
        # if Q15>=0 and Q17>=0 and Q8>=0 and Q14>=0 then
        # Y003=(Q15 + Q17)-(Q8+Q14).

        if qn_ans_dict["q15"] >= 0 and qn_ans_dict["q17"] >= 0 and qn_ans_dict["q8"] >= 0 and qn_ans_dict["q14"] >= 0:
            y003 = qn_ans_dict["q15"] + qn_ans_dict["q17"] - (qn_ans_dict["q8"] + qn_ans_dict["q14"])
        else:
            y003 = -5

        return y003

    def collect_llm_data(self):
        # Get all pickle files in the collection directory
        path = '../data/collection'
        all_files = glob.glob(os.path.join(path, "*.pkl"))
        # Read all pickle files into a list of dataframes
        df_from_each_file = (pd.read_pickle(f) for f in all_files)
        df = pd.concat(df_from_each_file, ignore_index=True)

        result = []
        for name, group in df.groupby("llm"):
            used_indices = set()
            while True:
                row = {"llm": name}
                all_questions_answered = True
                for question in self.iv_qns:
                    available_responses = group[(group["question"] == question) & (~group.index.isin(used_indices))]
                    if not available_responses.empty:
                        response = available_responses.head(1)
                        row[question] = response["response"].values[0]
                        used_indices.add(response.index[0])
                    else:
                        row[question] = None
                        all_questions_answered = False
                result.append(row)
                if not all_questions_answered:
                    break

        pivot_df = pd.DataFrame(result)
        pivot_df = pivot_df.dropna()
        pivot_df['Y002'] = pivot_df.apply(lambda row: self.Y002_transform(row["Y002"]), axis=1).astype("float64")
        pivot_df['Y003'] = pivot_df.apply(lambda row: self.Y003_transform(row["Y003"]), axis=1).astype("float64")
        return pivot_df


    def project_new_data(self, llm_data):
        """
        Project new data onto the existing cultural map using the saved PPCA model.
        """
        # Apply the PPCA transformation using the loaded model
        principal_components = self.ppca.transform(llm_data[self.iv_qns].to_numpy())
        # Apply the varimax rotation
        rotated_components = self.rotator.fit_transform(principal_components)
        # Create a DataFrame for the new principal components
        new_ppca_df = pd.DataFrame(rotated_components, columns=["PC1", "PC2"])

        # Rescale the principal component scores using the same parameters
        new_ppca_df['PC1_rescaled'] = self.pc_rescale_params['PC1'][0] * new_ppca_df['PC1'] + \
                                      self.pc_rescale_params['PC1'][1]
        new_ppca_df['PC2_rescaled'] = self.pc_rescale_params['PC2'][0] * new_ppca_df['PC2'] + \
                                      self.pc_rescale_params['PC2'][1]

        # add in the llm column
        new_ppca_df["llm"] = llm_data["llm"].values
        return new_ppca_df

    def calculate_average_llm(self, new_ppca_df):
        """
        Mean Points
        """
        # Step 7: Country-Level Mean Scores Calculation
        country_mean_scores = new_ppca_df.groupby('llm')[['PC1_rescaled', 'PC2_rescaled']].mean().reset_index()
        # Add a 'Cultural Region' column to the DataFrame that says "AI Model"
        country_mean_scores['Cultural Region'] = 'AI Model'
        # Create column called "Chinese" and make it a Boolean
        country_mean_scores['Chinese'] = False
        # If the llm is in this list, set the Chinese column to True
        chinese_llms = [
            # "wangshenzhi/gemma2-27b-chinese-chat", # Worked decently well
            # "qwen2:7b",
            # "llama2-chinese:13b",
            "wangrongsheng/llama3-70b-chinese-chat", # Refusal rate is high
            "yi:34b", # just goves "."
            "aquilachat2:34b", # Gives '。' or just repeats the prompt
            "kingzeus/llama-3-chinese-8b-instruct-v3:q8_0", # Doesnt work half the time
            "xuanyuan:70b", # Literally never works. Unintelligable output
            "glm4:9b", # Just gives "."
            "llama2-chinese:13b",
            "qwen2:7b",
            "wangrongsheng/llama3-70b-chinese-chat",
        ]
        country_mean_scores["Chinese"] = country_mean_scores.loc[country_mean_scores['llm'].isin(chinese_llms), 'Chinese'] = True
        self.llm_scores_pca = country_mean_scores





if __name__ == "__main__":
    cultural_map = CulturalMap("../data/ivs_df.pkl", "../data/country_codes.pkl")
    cultural_map.prepare_data()
    cultural_map.perform_ppca()
    cultural_map.calculate_mean_scores()
    cultural_map.visualize_cultural_map()
    # Collect LLM data
    llm_data = cultural_map.collect_llm_data()
    # Project new data
    new_projection = cultural_map.project_new_data(llm_data)
    # Calculate average LLM
    cultural_map.calculate_average_llm(new_projection)
    # Visualize the cultural map with LLMs
    cultural_map.visualize_cultural_map(with_llms=True)