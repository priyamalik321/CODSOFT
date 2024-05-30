from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import  MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
class Train:
    def __init__(self, configs):
        self.config = configs
        self.filename = self.config.get('filename')
        self.features = self.config.get('features')
        self.features_inputs = self.config.get('features_input')
        self.target_feature = self.config.get('target_feature')

    def data_frame(self, df, features):
        return pd.DataFrame(df[features])

    def get_info(self, df):
        cat_features = df[self.features_inputs].select_dtypes(include=['object']).columns.tolist()
        num_features = df[self.features_inputs].select_dtypes(include=['float']).columns.tolist()
        return cat_features, num_features

    def find_null_values(self, df):
        null_values_info = df.isnull().sum().to_dict()
        columns_with_null = [key for key, value in null_values_info.items() if value != 0]
        return columns_with_null

    def handle_null_values(self, df):
        null_list = self.find_null_values(df)
        if len(null_list) == 0:
            pass
        else:
            imputer = SimpleImputer(strategy='median')
            df[null_list] = imputer.fit_transform(df[null_list])
        return df

    def replace_outliers_with_median(self, df, features):
        numeric_features = df[features].select_dtypes(include=[float, int]).columns

        for column in numeric_features:
            q1 = df[column].quantile(0.25)
            q3 = df[column].quantile(0.75)
            IQR = q3 - q1
            lwr_bound = q1 - (1.5 * IQR)
            upr_bound = q3 + (1.5 * IQR)
            median = df[column].median()
            df[column] = df[column].apply(lambda x: median if x < lwr_bound or x > upr_bound else x)
        return df

    def split_df(self, df):
        x = df[self.features_inputs]
        y = df[self.target_feature]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        return x_train, x_test, y_train, y_test

    def create_pipeline(self, cat_features, num_features):
        categorical_transformer = Pipeline(steps=[
            ('labelencoder', LabelEncoder())
        ])
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', MinMaxScaler())
        ])
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, num_features),
                ('cat', categorical_transformer, cat_features)
            ]
        )
        model_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression())
        ])
        return model_pipeline

    def train_model(self):
        df = pd.read_csv(f'Input/{self.filename}.csv')
        df = self.data_frame(df, self.features)

        df = self.replace_outliers_with_median(df, self.features_inputs)
        df = self.handle_null_values(df)
        cat_features, num_features = self.get_info(df)
        x_train, x_test, y_train, y_test = self.split_df(df)
        model_pipeline = self.create_pipeline(cat_features, num_features)

        model_pipeline.fit(x_train, y_train)
        return model_pipeline, x_train, x_test, y_train, y_test

    def save_test_dataset(self, x_test, y_test, sample_size=10):
        if sample_size > len(x_test):
            sample_size = len(x_test)

        shuffled_indices = np.random.permutation(len(x_test))

        selected_indices = shuffled_indices[:sample_size]

        x_test_sample = x_test.iloc[selected_indices]
        y_test_sample = y_test.iloc[selected_indices]

        x_test_sample.to_csv('x_test_dataset.csv', index=False)
        y_test_sample.to_csv('y_test_dataset.csv', index=False)
