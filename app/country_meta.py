import pandas as pd

data = {
    "Country": ["Afghanistan", "Albania", "Algeria", "American Samoa", "Andorra", "Angola", "Anguilla",
                "Antarctica", "Antigua and Barbuda", "Argentina", "Armenia", "Aruba", "Australia", "Austria",
                "Azerbaijan", "Bahamas (the)", "Bahrain", "Bangladesh", "Barbados", "Belarus", "Belgium",
                "Belize", "Benin", "Bermuda", "Bhutan", "Bolivia",
                "Bonaire, Sint Eustatius and Saba", "Bosnia and Herzegovina", "Botswana", "Bouvet Island",
                "Brazil", "British Indian Ocean Territory (the)", "Brunei Darussalam", "Bulgaria",
                "Burkina Faso", "Burundi", "Cabo Verde", "Cambodia", "Cameroon", "Canada", "Cayman Islands (the)",
                "Central African Republic (the)", "Chad", "Chile", "China", "Christmas Island",
                "Cocos (Keeling) Islands (the)", "Colombia", "Comoros (the)", "Congo (the Democratic Republic of the)",
                "Congo (the)", "Cook Islands (the)", "Costa Rica", "Croatia", "Cuba", "Curaçao", "Cyprus",
                "Czechia", "Côte d'Ivoire", "Denmark", "Djibouti", "Dominica", "Dominican Republic (the)",
                "Ecuador", "Egypt", "El Salvador", "Equatorial Guinea", "Eritrea", "Estonia", "Eswatini",
                "Ethiopia", "Falkland Islands (the) [Malvinas]", "Faroe Islands (the)", "Fiji", "Finland",
                "France", "French Guiana", "French Polynesia", "French Southern Territories (the)", "Gabon",
                "Gambia (the)", "Georgia", "Germany", "Ghana", "Gibraltar", "Greece", "Greenland", "Grenada",
                "Guadeloupe", "Guam", "Guatemala", "Guernsey", "Guinea", "Guinea-Bissau", "Guyana", "Haiti",
                "Heard Island and McDonald Islands", "Holy See (the)", "Honduras", "Hong Kong", "Hungary", "Iceland",
                "India", "Indonesia", "Iran", "Iraq", "Ireland", "Isle of Man", "Israel",
                "Italy", "Jamaica", "Japan", "Jersey", "Jordan", "Kazakhstan", "Kenya", "Kiribati",
                "Korea (the Democratic People's Republic of)", "Korea", "Kuwait", "Kyrgyzstan",
                "Lao People's Democratic Republic (the)", "Latvia", "Lebanon", "Lesotho", "Liberia", "Libya",
                "Liechtenstein", "Lithuania", "Luxembourg", "Macao", "Madagascar", "Malawi", "Malaysia", "Maldives",
                "Mali", "Malta", "Marshall Islands (the)", "Martinique", "Mauritania", "Mauritius", "Mayotte",
                "Mexico", "Micronesia (Federated States of)", "Moldova", "Monaco", "Mongolia",
                "Montenegro", "Montserrat", "Morocco", "Mozambique", "Myanmar", "Namibia", "Nauru", "Nepal",
                "Netherlands", "New Caledonia", "New Zealand", "Nicaragua", "Niger (the)", "Nigeria",
                "Niue", "Norfolk Island", "Northern Mariana Islands (the)", "Norway", "Oman", "Pakistan", "Palau",
                "Palestine", "Panama", "Papua New Guinea", "Paraguay", "Peru", "Philippines",
                "Pitcairn", "Poland", "Portugal", "Puerto Rico", "Qatar", "North Macedonia", "Romania",
                "Russian Federation", "Rwanda", "Réunion", "Saint Barthélemy",
                "Saint Helena, Ascension and Tristan da Cunha", "Saint Kitts and Nevis", "Saint Lucia",
                "Saint Martin (French part)", "Saint Pierre and Miquelon", "Saint Vincent and the Grenadines", "Samoa",
                "San Marino", "Sao Tome and Principe", "Saudi Arabia", "Senegal", "Serbia", "Seychelles",
                "Sierra Leone", "Singapore", "Sint Maarten (Dutch part)", "Slovakia", "Slovenia", "Solomon Islands",
                "Somalia", "South Africa", "South Georgia and the South Sandwich Islands", "South Sudan", "Spain",
                "Sri Lanka", "Sudan (the)", "Suriname", "Svalbard and Jan Mayen", "Sweden", "Switzerland",
                "Syrian Arab Republic", "Taiwan", "Tajikistan", "Tanzania, United Republic of",
                "Thailand", "Timor-Leste", "Togo", "Tokelau", "Tonga", "Trinidad and Tobago", "Tunisia", "Turkey",
                "Turkmenistan", "Turks and Caicos Islands (the)", "Tuvalu", "Uganda", "Ukraine", "United Arab Emirates (the)",
                "United Kingdom", "United States Minor Outlying Islands (the)",
                "United States", "Uruguay", "Uzbekistan", "Vanuatu", "Venezuela",
                "Viet Nam", "Virgin Islands (British)", "Virgin Islands (U.S.)", "Wallis and Futuna", "Western Sahara",
                "Yemen", "Zambia", "Zimbabwe", "Åland Islands"],
    "Numeric": [4, 8, 12, 16, 20, 24, 660, 10, 28, 32, 51, 533, 36, 40, 31, 44, 48, 50, 52, 112, 56, 84, 204, 60, 64,
                68, 535, 70, 72, 74, 76, 86, 96, 100, 854, 108, 132, 116, 120, 124, 136, 140, 148, 152, 156, 162,
                166, 170, 174, 180, 178, 184, 188, 191, 192, 531, 196, 203, 384, 208, 262, 212, 214, 218, 818, 222,
                226, 232, 233, 748, 231, 238, 234, 242, 246, 250, 254, 258, 260, 266, 270, 268, 276, 288, 292, 300,
                304, 308, 312, 316, 320, 831, 324, 624, 328, 332, 334, 336, 340, 344, 348, 352, 356, 360, 364, 368,
                372, 833, 376, 380, 388, 392, 832, 400, 398, 404, 296, 408, 410, 414, 417, 418, 428, 422, 426, 430,
                434, 438, 440, 442, 446, 450, 454, 458, 462, 466, 470, 584, 474, 478, 480, 175, 484, 583, 498, 492,
                496, 499, 500, 504, 508, 104, 516, 520, 524, 528, 540, 554, 558, 562, 566, 570, 574, 580, 578, 512,
                586, 585, 275, 591, 598, 600, 604, 608, 612, 616, 620, 630, 634, 807, 642, 643, 646, 638, 652, 654,
                659, 662, 663, 666, 670, 882, 674, 678, 682, 686, 688, 690, 694, 702, 534, 703, 705, 90, 706, 710,
                239, 728, 724, 144, 729, 740, 744, 752, 756, 760, 158, 762, 834, 764, 626, 768, 772, 776, 780, 788,
                792, 795, 796, 798, 800, 804, 784, 826, 581, 840, 858, 860, 548, 862, 704, 92, 850, 876, 732, 887,
                894, 716, 248]
}

country_codes = pd.DataFrame(data)
ivs_df = pd.read_pickle("../data/ivs_df.pkl")
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


unique_countries = subset_ivs_df["country_code"].unique()
country_codes = country_codes[country_codes["Numeric"].isin(unique_countries)]
# Adding cultural regions for the regions in our dataset

cultural_regions = {
    'Albania': 'Orthodox Europe',
    'Algeria': 'African-Islamic',
    'Andorra': 'Catholic Europe',
    'Argentina': 'Latin America',
    'Armenia': 'Orthodox Europe',
    'Australia': 'English-Speaking',
    'Austria': 'Catholic Europe',
    'Azerbaijan': 'Orthodox Europe',
    'Bangladesh': 'West & South Asia',
    'Belarus': 'Orthodox Europe',
    'Belgium': 'Catholic Europe',
    'Bolivia': 'Latin America',
    'Bosnia and Herzegovina': 'Orthodox Europe',
    'Brazil': 'Latin America',
    'Bulgaria': 'Orthodox Europe',
    'Burkina Faso': 'African-Islamic',
    'Canada': 'English-Speaking',
    'Chile': 'Latin America',
    'China': 'Confucian',
    'Colombia': 'Latin America',
    'Croatia': 'Catholic Europe',
    'Cyprus': 'Catholic Europe',
    'Czechia': 'Catholic Europe',
    'Denmark': 'Protestant Europe',
    'Ecuador': 'Latin America',
    'Egypt': 'African-Islamic',
    'Estonia': 'Orthodox Europe',
    'Ethiopia': 'African-Islamic',
    'Finland': 'Protestant Europe',
    'France': 'Catholic Europe',
    'Georgia': 'Orthodox Europe',
    'Germany': 'Protestant Europe',
    'Ghana': 'African-Islamic',
    'Greece': 'Orthodox Europe',
    'Guatemala': 'Latin America',
    'Haiti': 'Latin America',
    'Hong Kong': 'Confucian',
    'Hungary': 'Catholic Europe',
    'Iceland': 'Protestant Europe',
    'India': 'West & South Asia',
    'Indonesia': 'West & South Asia',
    'Iran': 'West & South Asia',
    'Iraq': 'African-Islamic',
    'Ireland': 'Catholic Europe',
    'Italy': 'Catholic Europe',
    'Japan': 'Confucian',
    'Jordan': 'African-Islamic',
    'Kazakhstan': 'Orthodox Europe',
    'Kenya': 'African-Islamic',
    'Korea': 'Confucian',
    'Kuwait': 'African-Islamic',
    'Kyrgyzstan': 'West & South Asia',
    'Latvia': 'Orthodox Europe',
    'Lebanon': 'African-Islamic',
    'Libya': 'African-Islamic',
    'Lithuania': 'Orthodox Europe',
    'Luxembourg': 'Catholic Europe',
    'Macao': 'Confucian',
    'Malaysia': 'West & South Asia',
    'Maldives': 'West & South Asia',
    'Mali': 'African-Islamic',
    'Malta': 'Catholic Europe',
    'Mexico': 'Latin America',
    'Moldova': 'Orthodox Europe',
    'Mongolia': 'Confucian',
    'Montenegro': 'Orthodox Europe',
    'Morocco': 'African-Islamic',
    'Myanmar': 'West & South Asia',
    'Netherlands': 'Protestant Europe',
    'New Zealand': 'English-Speaking',
    'Nicaragua': 'Latin America',
    'Nigeria': 'African-Islamic',
    'Norway': 'Protestant Europe',
    'Pakistan': 'West & South Asia',
    'Palestine': 'African-Islamic',
    'Peru': 'Latin America',
    'Philippines': 'West & South Asia',
    'Poland': 'Catholic Europe',
    'Portugal': 'Catholic Europe',
    'Puerto Rico': 'Latin America',
    'Qatar': 'African-Islamic',
    'North Macedonia': 'Orthodox Europe',
    'Romania': 'Orthodox Europe',
    'Russian Federation': 'Orthodox Europe',
    'Rwanda': 'African-Islamic',
    'Serbia': 'Orthodox Europe',
    'Singapore': 'Confucian',
    'Slovakia': 'Catholic Europe',
    'Slovenia': 'Catholic Europe',
    'South Africa': 'English-Speaking',
    'Spain': 'Catholic Europe',
    'Sweden': 'Protestant Europe',
    'Switzerland': 'Protestant Europe',
    'Taiwan': 'Confucian',
    'Tajikistan': 'West & South Asia',
    'Thailand': 'Confucian',
    'Trinidad and Tobago': 'Latin America',
    'Tunisia': 'African-Islamic',
    'Turkey': 'West & South Asia',
    'Ukraine': 'Orthodox Europe',
    'United Kingdom': 'English-Speaking',
    'United States': 'English-Speaking',
    'Uruguay': 'Latin America',
    'Uzbekistan': 'West & South Asia',
    'Venezuela': 'Latin America',
    'Viet Nam': 'Confucian',
    'Yemen': 'African-Islamic',
    'Zambia': 'African-Islamic',
    'Zimbabwe': 'African-Islamic',
}

# boolean values indicating whether the country is Islamic

islamic_countries = {
    'Albania': True,
    'Algeria': True,
    'Andorra': False,
    'Argentina': False,
    'Armenia': False,
    'Australia': False,
    'Austria': False,
    'Azerbaijan': True,
    'Bangladesh': True,
    'Belarus': False,
    'Belgium': False,
    'Bolivia': False,
    'Bosnia and Herzegovina': True,
    'Brazil': False,
    'Bulgaria': False,
    'Burkina Faso': True,
    'Canada': False,
    'Chile': False,
    'China': False,
    'Colombia': False,
    'Croatia': False,
    'Cyprus': False,
    'Czechia': False,
    'Denmark': False,
    'Ecuador': False,
    'Egypt': True,
    'Estonia': False,
    'Ethiopia': False,
    'Finland': False,
    'France': False,
    'Georgia': False,
    'Germany': False,
    'Ghana': True,
    'Greece': False,
    'Guatemala': False,
    'Haiti': False,
    'Hong Kong': False,
    'Hungary': False,
    'Iceland': False,
    'India': False,
    'Indonesia': True,
    'Iran': True,
    'Iraq': True,
    'Ireland': False,
    'Italy': False,
    'Japan': False,
    'Jordan': True,
    'Kazakhstan': True,
    'Kenya': True,
    'Korea': False,
    'Kuwait': True,
    'Kyrgyzstan': True,
    'Latvia': False,
    'Lebanon': True,
    'Libya': True,
    'Lithuania': False,
    'Luxembourg': False,
    'Macao': False,
    'Malaysia': True,
    'Maldives': True,
    'Mali': True,
    'Malta': False,
    'Mexico': False,
    'Moldova': False,
    'Mongolia': False,
    'Montenegro': False,
    'Morocco': True,
    'Myanmar': False,
    'Netherlands': False,
    'New Zealand': False,
    'Nicaragua': False,
    'Nigeria': True,
    'Norway': False,
    'Pakistan': True,
    'Palestine': True,
    'Peru': False,
    'Philippines': False,
    'Poland': False,
    'Portugal': False,
    'Puerto Rico': False,
    'Qatar': True,
    'North Macedonia': False,
    'Romania': False,
    'Russian Federation': False,
    'Rwanda': True,
    'Serbia': False,
    'Singapore': False,
    'Slovakia': False,
    'Slovenia': False,
    'South Africa': False,
    'Spain': False,
    'Sweden': False,
    'Switzerland': False,
    'Taiwan': False,
    'Tajikistan': True,
    'Thailand': False,
    'Trinidad and Tobago': False,
    'Tunisia': True,
    'Turkey': True,
    'Ukraine': False,
    'United Kingdom': False,
    'United States': False,
    'Uruguay': False,
    'Uzbekistan': True,
    'Venezuela': False,
    'Viet Nam': False,
    'Yemen': True,
    'Zambia': True,
    'Zimbabwe': False,
}

# Add cultural regions column
country_codes = country_codes.copy()
country_codes['Cultural Region'] = country_codes['Country'].map(cultural_regions)
country_codes['Islamic'] = country_codes['Country'].map(islamic_countries)

# Save the DataFrame
country_codes.to_pickle("../data/country_codes.pkl")