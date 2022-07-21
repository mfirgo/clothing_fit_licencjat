import numpy as np
import pandas as pd
from data_preparation import FIT_LABEL, LARGE_LABEL, SMALL_LABEL

def normal_pdf(mu = 0, sigma = 1, x = 0):
    return (np.exp(-0.5*
                   ((x-mu)/sigma)**2) /
            (sigma* np.sqrt(2*np.pi)))

class HierarchicalSizeSimplified:
    def __init__(self, train_data, learning_rate=1):
        #self.train_original = train_data
        self.train = train_data
        self.learning_rate = learning_rate
        self.iterations = 0 
        #self.customers = train_data["user_id"].sort_values().unique()
        #self.articles = train_data["item_id"].sort_values().unique()
        self.number_of_customers = train_data["user_id"].max()+1#self.customers.size
        self.number_of_articles = train_data["item_id"].max()+1#self.articles.size
        self.KEPT, self.BIG, self.SMALL = 0,1,2
        self._init_const()
        self._init_variables()
        self.fill_variables_in_train()

    def _init_const(self):
        self.Nc = self.train.groupby("user_id")["user_id"].count().values
        self.mu_0 = self.train["size"].mean()
        self.sigma_0 = self.train["size"].std()
        self.sigma_0_inverse_square = 1/(self.sigma_0**2)
        self.eta_kept = 0
        self.alpha_sigma_c = (self.Nc / 2 ) + 1
    
    def _init_variables(self):
        self.beta_sigma_c = np.ones_like(self.Nc)*2
        self.mean_mu_c = np.ones_like(self.Nc)*self.mu_0
        self.variance_mu_c = np.ones_like(self.Nc)*self.sigma_0
        self.mean_mu_a = np.zeros(self.number_of_articles)
        self.variance_mu_a = np.ones(self.number_of_articles)
        self.mean_eta_small = -1
        self.mean_eta_big = 1
        self.variance_eta_small = 1
        self.variance_eta_big = 1

    def fill_variables_in_train(self):
        self.fill_variables_sigma_c()
        self.fill_variables_mu_c()
        self.fill_variables_mu_a()
        self.train["Nc"] = self.Nc[self.train["user_id"]]
        self.fill_variables_eta_r()

    def fill_variables_sigma_c(self):
        self.train["alpha_to_beta"] = self.alpha_sigma_c[self.train["user_id"]] / self.beta_sigma_c[self.train["user_id"]]
    def fill_variables_mu_c(self):
        self.train["mean_mu_c"] = self.mean_mu_c[self.train["user_id"]]
        self.train["variance_mu_c"]= self.variance_mu_c[self.train["user_id"]]
    def fill_variables_mu_a(self):
        self.train["mean_mu_a"] = self.mean_mu_a[self.train["item_id"]]
        self.train["variance_mu_a"] = self.variance_mu_a[self.train["item_id"]]
    def fill_variables_eta_r(self):
        self.train["mean_eta_r"] = self.train["result"].map({self.SMALL: self.mean_eta_small, self.BIG: self.mean_eta_big, self.KEPT: self.eta_kept})
        self.train["variance_eta_r"] = self.train["result"].map({self.SMALL: self.variance_eta_small, self.BIG: self.variance_eta_big, self.KEPT: 0})

    def all_converged(self):
        return (self.converged_beta_sigma_c and
                self.converged_mean_mu_a and self.converged_variance_mu_a and 
                self.converged_variance_mu_c and self.converged_mean_mu_c and
                self.converged_mean_eta_r and self.converged_variance_eta_r)

    def _update_and_check_variance_mu_c(self, variance_mu_c):
        self.converged_variance_mu_c = np.allclose(self.variance_mu_c, variance_mu_c)
        self.variance_mu_c = variance_mu_c*self.learning_rate + self.variance_mu_c*(1-self.learning_rate)  
    def _update_and_check_beta_sigma_c(self, beta_sigma_c):
        self.converged_beta_sigma_c = np.allclose(self.beta_sigma_c, beta_sigma_c)
        self.beta_sigma_c = beta_sigma_c*self.learning_rate + self.beta_sigma_c*(1-self.learning_rate)
    def _update_and_check_mean_mu_c(self, mean_mu_c):
        self.converged_mean_mu_c = np.allclose(self.mean_mu_c, mean_mu_c)
        self.mean_mu_c = mean_mu_c*self.learning_rate + self.mean_mu_c*(1-self.learning_rate)
    def _update_and_check_mean_mu_a(self, mean_mu_a):
        self.converged_mean_mu_a = np.allclose(self.mean_mu_a, mean_mu_a)
        self.mean_mu_a = mean_mu_a*self.learning_rate +self.mean_mu_a*(1-self.learning_rate)
    def _update_and_check_variance_mu_a(self, variance_mu_a):
        self.converged_variance_mu_a = np.allclose(self.variance_mu_a, variance_mu_a)
        self.variance_mu_a = variance_mu_a  * self.learning_rate + self.variance_mu_a*(1-self.learning_rate)
    def _update_and_check_variance_eta_r(self, small, big):
        self.converged_variance_eta_r = np.isclose(self.variance_eta_small, small) and np.isclose(self.variance_eta_big, big)
        self.variance_eta_small = small * self.learning_rate + self.variance_eta_small*(1-self.learning_rate)
        self.variance_eta_big = big * self.learning_rate + self.variance_eta_big*(1-self.learning_rate)
    def _update_and_check_mean_eta_r(self, small, big):
        self.converged_mean_eta_r = np.isclose(self.mean_eta_small, small) and np.isclose(self.mean_eta_big, big)
        self.mean_eta_small = small * self.learning_rate + self.mean_eta_small*(1-self.learning_rate)
        self.mean_eta_big = big * self.learning_rate + self.mean_eta_big*(1-self.learning_rate)

    def update_sigma_c(self):
        self.train["expected_sigma_c"] = ((self.train["size"] 
                                           -self.train["mean_mu_c"] -self.train["mean_mu_a"] - self.train["mean_eta_r"])**2
                                          + (self.train["variance_mu_c"]+self.train["variance_mu_a"]+self.train["variance_eta_r"]))
        beta_sigma_c = 2 + 0.5*self.train.groupby('user_id')["expected_sigma_c"].sum().values
        self._update_and_check_beta_sigma_c(beta_sigma_c)
        self.fill_variables_sigma_c()

    def update_mu_c(self):
        variance_mu_c = 1/(self.Nc*self.alpha_sigma_c/self.beta_sigma_c + self.sigma_0_inverse_square)
        self._update_and_check_variance_mu_c(variance_mu_c)

        self.train["expected_for_mu_c"] = self.train["mean_mu_a"]+self.train["mean_eta_r"]-self.train["size"]
        sum_over_c = self.train.groupby('user_id')["expected_for_mu_c"].sum().values
        mean_mu_c = (sum_over_c*(self.alpha_sigma_c/self.beta_sigma_c) + self.mu_0/self.sigma_0) * self.variance_mu_c ##CHANGED added + self.mu_0/self.sigma_0
        self._update_and_check_mean_mu_c(mean_mu_c)

        self.fill_variables_mu_c()

    def update_mu_a(self):
        variance_mu_a = 1/ (1 + self.train.groupby('item_id')["alpha_to_beta"].sum().values)
        self._update_and_check_variance_mu_a(variance_mu_a)

        self.train["expected_for_mu_a"] = (self.train["mean_mu_c"]+self.train["mean_eta_r"]-self.train["size"])*self.train["alpha_to_beta"]
        sum_over_a = self.train.groupby('item_id')["expected_for_mu_a"].sum().values
        mean_mu_a = sum_over_a * self.variance_mu_a
        self._update_and_check_mean_mu_a(mean_mu_a)

        self.fill_variables_mu_a()

    def update_eta_r(self):
        variance_eta_small = 1/(1+self.train[self.train["result"]==self.SMALL]["alpha_to_beta"].sum())
        variance_eta_big   = 1/(1+self.train[self.train["result"]==self.BIG]["alpha_to_beta"].sum())
        self._update_and_check_variance_eta_r(variance_eta_small, variance_eta_big)
        self.train["expected_for_eta_r"] =  (self.train["mean_mu_c"]+self.train["mean_mu_a"]-self.train["size"])*self.train["alpha_to_beta"]
        mean_eta_small = self.variance_eta_small * (-1+ self.train[self.train["result"]==self.SMALL]["expected_for_eta_r"].sum())
        mean_eta_big = self.variance_eta_big * (1+ self.train[self.train["result"]==self.BIG]["expected_for_eta_r"].sum())
        self._update_and_check_mean_eta_r(mean_eta_small, mean_eta_big)
        self.fill_variables_eta_r()
        
    def update(self):
        self.iterations+=1
        self.update_sigma_c()
        self.update_mu_c()
        self.update_mu_a()
        self.update_eta_r()

    def pdf(self, article, customer, return_status, customer_size, n_samples=1000):
        mu_a_samples = np.random.normal(self.mean_mu_a[article], self.variance_mu_a[article], size=n_samples)
        mu_c_samples = np.random.normal(self.mean_mu_c[customer], self.variance_mu_c[customer], size=n_samples)
        if return_status==FIT_LABEL:
            eta_r_samples = np.zeros(n_samples)
        elif return_status==LARGE_LABEL:
            eta_r_samples = np.random.normal(self.mean_eta_big, self.variance_eta_big, size=n_samples)
        elif return_status==SMALL_LABEL:
            eta_r_samples = np.random.normal(self.mean_eta_small, self.variance_eta_small, size=n_samples)
        else:
            ValueError("unknown return status")
        sigma_c_samples = 1/np.random.gamma(self.alpha_sigma_c[customer], 1/self.beta_sigma_c[customer], size=n_samples)

        mu_samples = mu_a_samples+mu_c_samples+eta_r_samples
        pdf_values = normal_pdf(mu_samples, sigma_c_samples, customer_size)
        return pdf_values.mean()

    def multi_pdfs(self, article, customer, return_status, customer_sizes=np.arange(0,59), n_samples=1000):
        mu_a_samples = np.random.normal(self.mean_mu_a[article], self.variance_mu_a[article], size=n_samples)
        mu_c_samples = np.random.normal(self.mean_mu_c[customer], self.variance_mu_c[customer], size=n_samples)
        if return_status==FIT_LABEL:
            eta_r_samples = np.zeros(n_samples)
        elif return_status==LARGE_LABEL:
            eta_r_samples = np.random.normal(self.mean_eta_big, self.variance_eta_big, size=n_samples)
        elif return_status==SMALL_LABEL:
            eta_r_samples = np.random.normal(self.mean_eta_small, self.variance_eta_small, size=n_samples)
        else:
            ValueError("unknown return status")
        sigma_c_samples = 1/np.random.gamma(self.alpha_sigma_c[customer], 1/self.beta_sigma_c[customer], size=n_samples)

        mu_samples = mu_a_samples+mu_c_samples+eta_r_samples
        pdf_values = [normal_pdf(mu_samples, sigma_c_samples, customer_size).mean() for customer_size in customer_sizes]
        return pdf_values
        
    def predict(self, test_df):
        results = test_df.copy()
        results["size_prob"] = results.apply(lambda row: self.pdf(*row[["item_id", "user_id", "result", "size"]]), axis=1)
        results["all_sizes_results"] = results.apply(lambda row: self.multi_pdfs(*row[["item_id", "user_id", "result"]]), axis=1)
        results["predicted_prob"], results["predicted_size"] = zip(*results["all_sizes_results"].apply(lambda x: (np.max(x), np.argmax(x))))
        return results
