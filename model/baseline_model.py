import sklearn.neighbors
import pandas as pd
import numpy as np

class BaselineModel:
    def __init__(self):
        self.default_bandwidth = 1
        pass

    ############
    # Training #
    ############

    def _compute_default_values_over_return_status(self, df, return_status_column):
        default_values_df = df.value_counts(return_status_column)
        self.default_values = default_values_df.div(default_values_df.sum())
        self.default_values['predicted_return_status'] = self.default_values.idxmax()

    def _compute_number_of_sales_per_article_and_return(self, df, return_status_column):
        dummies = pd.get_dummies(df[return_status_column])
        df_encoded = pd.concat([df, dummies], axis=1)
        sales_by_fit  = df_encoded[["item_id"]+list(dummies.columns)].groupby(["item_id"]).sum() + 1
        self.sales_probability_by_fit = sales_by_fit.div(sales_by_fit.sum(axis=1), axis=0)
        self.sales_probability_by_fit['predicted_return_status'] = self.sales_probability_by_fit.idxmax(axis=1)
        self.sales_probability_by_fit['predicted_return_status_prob'] = self.sales_probability_by_fit[dummies.columns].max(axis=1)
        self.sales_probability_by_fit['predicted_return_status_logprob'] = np.log(self.sales_probability_by_fit['predicted_return_status_prob'])


    def train_model_return_status(self, df, return_status_column ='result'):
        self._compute_default_values_over_return_status(df, return_status_column)
        self._compute_number_of_sales_per_article_and_return(df, return_status_column)
    
    def _create_size_density(self, df, size_column = 'size'): # takes about 20s
        self.size_density_per_user = df.groupby("user_id")[size_column].agg(
            lambda X:
            sklearn.neighbors.KernelDensity(kernel='gaussian', bandwidth=self.default_bandwidth)
            .fit(X.values.reshape(-1, 1))
        ).rename("size_density")

    def _create_default_size_density(self, df, size_column ='size'):
        self.default_size_density = (sklearn.neighbors.KernelDensity(kernel='gaussian', bandwidth=self.default_bandwidth)
                                    .fit(df[size_column].values.reshape(-1,1)))

    def _compute_default_size_prediction(self):
        sizes = np.arange(0,58).reshape(-1, 1)
        self.default_size_scores = self.default_size_density.score_samples(sizes)
        self.default_best_size = np.argmax(self.default_size_scores)
        self.default_best_size_log_prob = self.default_size_scores.max()


    def _compute_size_likelihoods_per_user(self): # takes about 40s
        sizes = np.arange(0, 58).reshape(-1, 1)
        self.size_scores = self.size_density_per_user.apply(
            lambda kd:
            kd.score_samples(sizes)
        )
    def _find_best_size_per_user(self):
        self.best_size = pd.DataFrame({#"user_id": self.size_scores.index, 
                                       "predicted_size": self.size_scores.apply(np.argmax).values, 
                                       "predicted_size_logprob": self.size_scores.apply(max)}).reset_index()

    def train_model_size(self, df, size_column = 'size'): # takes ~ 1min
        # perhaps use skleran.neighbors.KernelDensity
        # they want to use MISE to select bandwidth
        # as a start we can use bandwidth = 1 ?
        self._create_default_size_density(df, size_column)
        self._create_size_density(df, size_column)
        self._compute_size_likelihoods_per_user()
        self._compute_default_size_prediction()
        self._find_best_size_per_user()

    def train_model(self, df, return_status_column = 'result', size_column='size'):
        self.train_model_return_status(df, return_status_column)
        self.train_model_size(df, size_column)

    ###############
    # Predictions #
    ###############

    def predict_return_status(self, df):
        prediction = df.merge(self.sales_probability_by_fit, how='left', on='item_id')
        prediction = prediction.fillna(self.default_values)
        return prediction

    def predict_size(self, df):
        prediction = df.merge(self.best_size, how='left', on='user_id')
        prediction["predicted_size"] = prediction["predicted_size"].fillna(self.default_best_size)
        prediction["predicted_size_logprob"] = prediction["predicted_size_logprob"].fillna(self.default_best_size_log_prob)
        prediction["predicted_size_prob"] = np.exp(prediction["predicted_size_logprob"])
        return prediction

    def _return_status_prob_from_prediction(self, predictions, return_status_column = 'result'):
        prob = predictions.apply(lambda row: row[row[return_status_column]], axis=1)
        return prob

    def return_status_prob(self, df, return_status_column = 'result'):
        predictions = self.predict_return_status(df)
        return self._return_status_prob_from_prediction(self, predictions, return_status_column)

    def return_status_logprob(self, df, return_status_column = 'result'):
        prob = self.return_status_prob(self, df, return_status_column)
        return np.log(prob)

    def size_logprob(self, df, size_column = 'size'):
        logprobs = df.merge(self.size_density_per_user, how='left', on='user_id')
        logprobs['size_density'] = logprobs['size_density'].fillna(self.default_size_density)
        #self.logprobs_size = logprobs
        logprobs = logprobs.apply(
            lambda row: 
            row['size_density'].score_samples(np.ones((1,1))*row[size_column]), axis=1).apply(lambda row: row[0])
        return logprobs

    def size_prob(self, df):
        logprobs = self.size_logprob(df)
        return np.exp(logprobs)
    
    def full_predict_and_logprob_return_status(self, df, return_status_column='result'):
        prediction = self.predict_return_status(df)
        prediction['return_status_prob'] = self._return_status_prob_from_prediction(prediction, return_status_column)
        prediction['return_status_logprob'] = np.log(prediction['return_status_prob'])
        return prediction

    def full_predict_and_logprob_size(self, df, size_column='size'): # takes ~ 1min
        prediction = self.predict_size(df)
        prediction["size_logprob"] = self.size_logprob(df, size_column)
        prediction["size_prob"] = np.exp(prediction["size_logprob"])
        #print(prediction.columns)
        return prediction

    def full_predict_and_logprob(self, df, size_column = 'size', return_status_column ='result'):
        size_prediction = self.full_predict_and_logprob_size(df, size_column)
        print("Predicted_size")
        full_prediction = self.full_predict_and_logprob_return_status(size_prediction, return_status_column)
        print("Predicted return status")
        full_prediction["full_predicted_prob"] = full_prediction["predicted_return_status_prob"] * full_prediction["predicted_size_prob"]
        full_prediction["full_predicted_logprob"] = full_prediction["predicted_return_status_logprob"] + full_prediction["predicted_size_logprob"]
        full_prediction["full_prob"] = full_prediction["return_status_prob"] * full_prediction["size_prob"]
        full_prediction["full_logprob"] = full_prediction["return_status_logprob"] + full_prediction["size_logprob"]
        return full_prediction

        # 2 ideas for functions
        # 1st: Write a function that for every costumer, size in df finds density in point size - do it every time a function is called
        # 2nd: Write a function that for each costumer selects best size (check all available sizes 0-58 and select one with max probablity) - do it once and safe lookup table
        
