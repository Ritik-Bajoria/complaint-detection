import pandas as pd
from sklearn.utils import resample


df = pd.read_csv('C:/Users/Legion/Ritik/Desktop/Programming/Intern work/07-Intern/complaint detector/Database/mydata_imbalanced.csv')
# Assuming your dataset is in a DataFrame called 'df'
# Separate the majority and minority classes
complaint = df[df['label'] == 'complaint']
non_complaint = df[df['label'] == 'non-complaint']

# Downsample the majority class
complaint_downsampled = resample(complaint,
                                 replace=False,  # sample without replacement
                                 n_samples=len(non_complaint),  # to match minority class
                                 random_state=42)

# Combine the balanced dataset
df_balanced = pd.concat([complaint_downsampled, non_complaint])

# Shuffle the dataset
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print(df_balanced['label'].value_counts())
df_balanced.to_csv("./Database/mydata_balanced.csv", index=False)


