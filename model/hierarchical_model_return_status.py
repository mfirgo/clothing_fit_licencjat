import numpy as np
import pandas as pd


class HierarchicalStatus:
    def __init__(self, category_treshold=100, alpha=0.1, beta=1, OTHER_CAT="other", init_w="random"):
        self.OTHER_CAT = OTHER_CAT
        self.beta = beta
        self.alpha = alpha
        self.treshold = category_treshold
        self.w = self._init_w(init_w)
        self.best_w = self.w
        self.best_prob = -1
        self.initialized = False
        self.train = None
        self.iterations = 0
        self.history = []

    def _init_w(self, init_w):
        if init_w == "random":
            return np.random.beta(self.alpha, self.beta)
        elif init_w == "mean":
            return self.alpha/(1+self.alpha)
        else:
            return init_w

    def train_model(self,  df=None, lr=0.01, T=100, log_every = 10, return_status_column = "result"):
        if not self.initialized and df is not None:
            print("Initializing model...", end=" ")
            self.initialize_model(df, return_status_column)
            print("Model initiated")
            self.prepare_trainset(df, return_status_column) 
            print("Trainset prepared")
        for iteration in range(T):
            self._training_step(lr)
            if (iteration+1)%log_every==0:
                print(iteration, self.history[-1])
        self.w=self.best_w

    def initialize_model(self, df, return_status_column = 'result'):
        dummies = pd.get_dummies(df[return_status_column])
        labels = dummies.columns
        self.labels = labels

        default_df = df.value_counts(return_status_column)
        self.default = default_df#.div(coldstart_df.sum())
        self.default['all'] = self.default.sum()
        self.default['predicted_return_status'] = self.default[labels].idxmax()

        df_encoded = pd.concat([df, dummies], axis=1)
        self.sales_at_article_level  = df_encoded[["item_id"]+list(labels)].groupby(["item_id"]).sum().rename(columns={l: str(l)+"_by_article" for l in labels}) #+ 1
        self.sales_at_article_level["all_by_article"] = self.sales_at_article_level.sum(axis=1)
        self.sales_at_category_level  = df_encoded[["category"]+list(labels)].groupby(["category"]).sum().rename(columns={l:str(l)+'_by_category' for l in labels}) #+ 1
        self.sales_at_category_level["all_by_category"] = self.sales_at_category_level.sum(axis=1)
        self.sales_at_category_level.loc[self.OTHER_CAT] = self.sales_at_category_level.sum()

        self.kept_categories = (self.sales_at_category_level.index)[self.sales_at_category_level["all_by_category"]>=self.treshold]
        self.initialized = True
        # self.sales_probability_by_fit = sales_by_fit.div(sales_by_fit.sum(axis=1), axis=0)
        # self.sales_probability_by_fit['predicted_result'] = self.sales_probability_by_fit.idxmax(axis=1)
    
    def prepare_trainset(self, df, return_status_column ='result'):
        self.train = df.copy()
        self.train["original_category"] = self.train["category"]
        self.train["count_category"] = (self.sales_at_category_level["all_by_category"])[self.train["category"]].values
        self.train.loc[~self.train["category"].isin(self.kept_categories), "category"] = self.OTHER_CAT
        self.train["count_category"] = (self.sales_at_category_level["all_by_category"])[self.train["category"]].values
        self.train["count_return_status_category"] = self.train.apply(
            lambda row: self.sales_at_category_level[str(row[return_status_column])+"_by_category"][row["category"]], axis=1
        )
        self.train["count_article"] = self.sales_at_article_level["all_by_article"][self.train["item_id"]].values
        self.train["count_return_status_article"] = self.train.apply(
            lambda row: self.sales_at_article_level[str(row[return_status_column])+"_by_article"][row["item_id"]], axis=1
        )
        #self.train["new_category"] = self.train["category"]    
    
    def _training_step(self, lr):
        derivative, mean_prob = self._compute_w_mean_derivative_and_probability() # mean or sum - but i think mean is better since adjusting learning rate will not be dependent on training set size
        self.history.append({"iter": self.iterations, "w": self.w, "mean_derivative": derivative, "mean_prob": mean_prob})
        if self.best_prob < mean_prob:
            self.best_prob = mean_prob
            self.best_w = self.w
        new_w = self.w + lr*derivative
        if 0 <= new_w <= 1:
            self.w += lr * derivative
        else:
            self.w = np.random.beta(self.alpha, self.beta)
        self.iterations +=1

    def _compute_w_mean_derivative_and_probability(self):
        probabilities = ((self.train["count_return_status_article"]
                        +self.w * self.train["count_return_status_category"])/
                        (self.train["count_article"]+self.w *self.train["count_category"]))
        logderivatives = (self.train["count_return_status_category"]/
                            np.abs(self.train["count_return_status_article"] + 
                             self.w * self.train["count_return_status_category"])
                       - self.train["count_category"]/
                            np.abs(self.train["count_article"] + 
                             self.w * self.train["count_category"]) 
                       )
        derivatives = probabilities*logderivatives
        return derivatives.mean(), probabilities.mean()
        
    def predict(self, df):
        prediction = df.copy()
        prediction["original_category"] = prediction["category"]
        prediction.loc[~prediction["category"].isin(self.kept_categories), "category"] = self.OTHER_CAT
        prediction = df.merge(self.sales_at_article_level, how='left', on='item_id')
        prediction = prediction.merge(self.sales_at_category_level, how='left', on='category')
        prediction = prediction.fillna(self.default.rename({l: str(l)+"_by_article" for l in list(self.labels)+['all']}))
        for status in self.labels:
            prediction[status] = ((prediction[str(status)+"_by_article"] + self.w *prediction[str(status)+'_by_category'])
                                / (prediction["all_by_article"] + self.w * prediction["all_by_category"]))
        prediction['predicted_return_status'] = prediction[self.labels].idxmax(axis=1)
        return prediction
