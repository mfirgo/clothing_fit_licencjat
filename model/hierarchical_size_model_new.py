import numpy as np
import pandas as pd
from utils.data_preparation import FIT_LABEL, LARGE_LABEL, SMALL_LABEL
from tqdm import tqdm
from utils.model_loading import *

def normal_pdf(mu = 0, sigma = 1, x = 0):
    return (np.exp(-0.5*
                   ((x-mu)/sigma)**2) /
            (sigma* np.sqrt(2*np.pi)))

class Parameter:
    def __init__(self, name, value, learning_rate, isconstant):
        self.name = name
        self.value = value
        self.learning_rate = learning_rate
        self.isconverged = isconstant
        self.isconstant = isconstant
        self.init = "default"

    def __str__(self):
        result =  (f"Parameter {self.name}, " +
                ("converged" if self.isconverged else "not converged") + ", " + 
                ("constant" if self.isconstant else f"variable, lr={self.learning_rate}") + ", " + 
                f"value: {self.value}"
               )
        return result

    def parameter_dict(self):
        if isinstance(self.value, (np.ndarray, pd.Series)):
            return {f"{self.name}_parameter_mean":self.value.mean(),
                    f"{self.name}_parameter_max" :self.value.max(),
                    f"{self.name}_parameter_min" :self.value.min(),
                    f"{self.name}_parameter_var" :self.value.var(),
                    f"{self.name}_parameter_median":np.median(self.value),
                    f"{self.name}_parameter_converged":self.isconverged
                    }
        else:
            return {self.name: self.value,
                    f"{self.name}_parameter_converged":self.isconverged
                    }
                    
    def hiperparameters(self):
        hiperparams_dict = {f"{self.name}_isconstant": self.isconstant,
                            f"{self.name}_init": self.init,
                            f"{self.name}_value": self.value}
        if not self.isconstant:
            hiperparams_dict[f"{self.name}_learning_rate"]=self.learning_rate
        return hiperparams_dict

    def initialize(self, init_config):
        if init_config=="default" or init_config=="custom_value":
            pass
        if isinstance(init_config, dict):
            init_type = init_config["type"]
            if init_type == "noise":
                init_config["previous_init"]=self.init
                scale = init_config["scale"] if "scale" in init_config else 0.1
                init_config["scale"] = scale
                noise = np.random.normal(0, scale, self.value.shape if isinstance(self.value, (np.ndarray, pd.Series)) else None)
                self.value = self.value + noise
            elif init_type == "uniform":
                low = init_config["low"] if "low" in init_config else 0
                init_config["low"] = low
                high = init_config["high"] if "high" in init_config else 58
                init_config["high"] = high
                self.value = np.random.uniform(low, high, self.value.shape if isinstance(self.value, (np.ndarray, pd.Series)) else None)
            elif init_type == "normal":
                loc = init_config["loc"] if "loc" in init_config else 12
                init_config["loc"] = loc
                scale = init_config["scale"] if "scale" in init_config else 2
                init_config["scale"] = scale
                self.value = np.random.normal(loc, scale, self.value.shape if isinstance(self.value, (np.ndarray, pd.Series)) else None)
            else:
                raise ValueError("unknown init type {init_type}")
        else:
            raise ValueError("unknown init config {init_config}")
        self.init = init_config
    
    def change_config(self, config):
        if f"{self.name}_isconstant" in config:
            self.isconstant = config[f"{self.name}_isconstant"]
            self.isconverged = self.isconstant
        if f"{self.name}_learning_rate" in config:
            self.learning_rate = config[f"{self.name}_learning_rate"]
        if f"{self.name}_value" in config:
            self.value = config[f"{self.name}_value"]
            self.init = "custom_value"
        if f"{self.name}_init" in config:
            self.initialize(config[f"{self.name}_init"])

    def update(self, new_value):
        if not self.isconstant:
            self.isconverged = np.allclose(self.value, new_value)
            self.value = new_value*self.learning_rate + self.value*(1-self.learning_rate) 
    
class HierarchicalSize:
    def __init__(self, train_data, default_learning_rate=1, max_user_id=None, max_item_id=None, model_name = 'h_size_with_update_order', config=None):
        self.creation_date = current_timestamp()
        self.model_name = model_name + "_" +self.creation_date+"/"+model_name
        self.default_learning_rate = default_learning_rate
        self.update_order = ["sigma_c", "mu_c", "mu_a", "eta_r"]
        self.KEPT, self.BIG, self.SMALL = FIT_LABEL, LARGE_LABEL, SMALL_LABEL
        self.history = []
        self.iterations = 0 
        self._create_parameters()
        self.set_train_data(train_data, max_user_id=max_user_id, max_item_id=max_item_id)
        self._init_parameters()
        if config is not None:
            self.change_config(config)
        else:
            self.fill_variables_in_train()


    #######################################
    ## Data and parameter initialization ##
    #######################################

    def set_train_data(self, train_data, max_user_id=None, max_item_id=None, user_col="user_id", item_col="item_id", status_col="result", size_col="size"):
        self.train = train_data.copy()
        self.user_col, self.item_col, self.status_col, self.size_col = user_col, item_col, status_col, size_col
        self.number_of_customers = train_data[user_col].max()+1 if max_user_id is None else max_user_id+1 
        self.number_of_articles = train_data[item_col].max()+1 if max_item_id is None else max_item_id+1 

    def _create_parameters(self):
        print("creating parameters")
        self.beta_sigma_c   = Parameter("beta_sigma_c", None, self.default_learning_rate, False)
        self.mean_mu_c      = Parameter("mean_mu_c", None, self.default_learning_rate, False)
        self.variance_mu_c  = Parameter("variance_mu_c", None, self.default_learning_rate, False)
        self.mean_mu_a      = Parameter("mean_mu_a", None, self.default_learning_rate, False)
        self.variance_mu_a  = Parameter("variance_mu_a", None, self.default_learning_rate, False)
        self.mean_eta_small = Parameter("mean_eta_small", None, self.default_learning_rate, False)
        self.mean_eta_big   = Parameter("mean_eta_big", None, self.default_learning_rate, False)
        self.mean_eta_kept  = Parameter("mean_eta_kept", None, self.default_learning_rate, True)
        self.variance_eta_small = Parameter("variance_eta_small", None, self.default_learning_rate, False)
        self.variance_eta_big   = Parameter("variance_eta_big", None, self.default_learning_rate, False)
        self.variance_eta_kept  = Parameter("variance_eta_kept", None, self.default_learning_rate, True)
        print("creating constant parameters")
        self.Nc = Parameter("Nc", None, None, True)
        self.mu_0 = Parameter("mu_0", None, None, True)
        self.sigma_0 = Parameter("sigma_0", None, None, True)
        self.sigma_0_inverse_square = Parameter("sigma_0_inverse_square", None, None, True)
        self.eta_kept = Parameter("eta_kept", None, None, True)
        self.alpha_sigma_c = Parameter("alpha_sigma_c", None, None, True)

    def _init_parameters(self):
        # This function must be changed to deal with coldstart
        self._init_constants()
        self._init_variables()

    def _init_constants(self):
        self.Nc.value = self.train.groupby("user_id")["user_id"].count()#.values
        self.mu_0.value = self.train["size"].mean()
        self.sigma_0.value = self.train["size"].std()
        self.sigma_0_inverse_square.value = 1/(self.sigma_0.value**2)
        self.eta_kept.value = 0
        self.alpha_sigma_c.value = (self.Nc.value/ 2 ) + 1
    
    def _init_variables(self):
        self.beta_sigma_c.value = np.ones_like(self.Nc.value)*2
        self.mean_mu_c.value = np.ones_like(self.Nc.value)*self.mu_0.value
        self.variance_mu_c.value = np.ones_like(self.Nc.value)*self.sigma_0.value
        self.mean_mu_a.value = np.zeros(self.number_of_articles)
        self.variance_mu_a.value = np.ones(self.number_of_articles)
        self.mean_eta_small.value = -1
        self.mean_eta_big.value = 1
        self.variance_eta_small.value = 1
        self.variance_eta_big.value = 1
        self.mean_eta_kept.value = 0
        self.variance_eta_kept.value = 0

    def fill_variables_in_train(self):
        self.fill_variables_sigma_c()
        self.fill_variables_mu_c()
        self.fill_variables_mu_a()
        self.train["Nc"] = self.Nc.value[self.train["user_id"]].values
        self.fill_variables_eta_r()

    def fill_variables_sigma_c(self):
        self.train["alpha_to_beta"] = (self.alpha_sigma_c.value[self.train["user_id"]] / self.beta_sigma_c.value[self.train["user_id"]]).values
    def fill_variables_mu_c(self):
        self.train["mean_mu_c"] = self.mean_mu_c.value[self.train["user_id"]]
        self.train["variance_mu_c"]= self.variance_mu_c.value[self.train["user_id"]]
    def fill_variables_mu_a(self):
        self.train["mean_mu_a"] = self.mean_mu_a.value[self.train["item_id"]]
        self.train["variance_mu_a"] = self.variance_mu_a.value[self.train["item_id"]]
    def fill_variables_eta_r(self):
        self.train["mean_eta_r"] = self.train["result"].map({self.SMALL: self.mean_eta_small.value, self.BIG: self.mean_eta_big.value, self.KEPT: self.mean_eta_kept.value})
        self.train["variance_eta_r"] = self.train["result"].map({self.SMALL: self.variance_eta_small.value, self.BIG: self.variance_eta_big.value, self.KEPT: self.variance_eta_kept.value})

    ##############
    ## Training ##
    ##############

    def all_converged(self):
        for param in self.get_parameters().values():
            if not param.isconstant:
                if not param.isconverged:
                    return False
        return True

    def update_sigma_c(self):
        self.train["expected_sigma_c"] = ((self.train["size"] 
                                           -self.train["mean_mu_c"] -self.train["mean_mu_a"] - self.train["mean_eta_r"])**2
                                          + (self.train["variance_mu_c"]+self.train["variance_mu_a"]+self.train["variance_eta_r"]))
        beta_sigma_c = 2 + 0.5*self.train.groupby('user_id')["expected_sigma_c"].sum().values
        self.beta_sigma_c.update(beta_sigma_c)
        self.fill_variables_sigma_c()

    def update_mu_c(self):
        variance_mu_c = 1/(self.Nc.value.values*self.alpha_sigma_c.value.values/self.beta_sigma_c.value + self.sigma_0_inverse_square.value)
        self.variance_mu_c.update(variance_mu_c)

        self.train["expected_for_mu_c"] = self.train["mean_mu_a"]+self.train["mean_eta_r"]-self.train["size"]
        sum_over_c = self.train.groupby('user_id')["expected_for_mu_c"].sum().values
        mean_mu_c = (sum_over_c*(self.alpha_sigma_c.value.values/self.beta_sigma_c.value) + self.mu_0.value/self.sigma_0.value) * self.variance_mu_c.value ##CHANGED added + self.mu_0/self.sigma_0
        self.mean_mu_c.update(mean_mu_c)

        self.fill_variables_mu_c()

    def update_mu_a(self):
        variance_mu_a = 1/ (1 + self.train.groupby('item_id')["alpha_to_beta"].sum().values)
        self.variance_mu_a.update(variance_mu_a)

        self.train["expected_for_mu_a"] = (self.train["mean_mu_c"]+self.train["mean_eta_r"]-self.train["size"])*self.train["alpha_to_beta"]
        sum_over_a = self.train.groupby('item_id')["expected_for_mu_a"].sum().values
        mean_mu_a = sum_over_a * self.variance_mu_a.value
        self.mean_mu_a.update(mean_mu_a)

        self.fill_variables_mu_a()

    def update_eta_r(self):
        variance_eta_small = 1/(1+self.train[self.train["result"]==self.SMALL]["alpha_to_beta"].sum())
        variance_eta_big   = 1/(1+self.train[self.train["result"]==self.BIG]["alpha_to_beta"].sum())
        variance_eta_kept  = 1/(1+self.train[self.train["result"]==self.KEPT]["alpha_to_beta"].sum())
        self.variance_eta_small.update(variance_eta_small)
        self.variance_eta_big.update(variance_eta_big)
        self.variance_eta_kept.update(variance_eta_kept)
        self.train["expected_for_eta_r"] =  (self.train["mean_mu_c"]+self.train["mean_mu_a"]-self.train["size"])*self.train["alpha_to_beta"]
        mean_eta_small = self.variance_eta_small.value * (-1+ self.train[self.train["result"]==self.SMALL]["expected_for_eta_r"].sum())
        mean_eta_big = self.variance_eta_big.value * (1+ self.train[self.train["result"]==self.BIG]["expected_for_eta_r"].sum())
        mean_eta_kept = self.variance_eta_kept.value * (self.train[self.train["result"]==self.KEPT]["expected_for_eta_r"].sum())

        self.mean_eta_small.update(mean_eta_small)
        self.mean_eta_big.update(mean_eta_big)
        self.mean_eta_kept.update(mean_eta_kept)
        self.fill_variables_eta_r()
        
    def _update_param(self, param):
        if param == "sigma_c":
            self.update_sigma_c()
        elif param == "mu_c":
            self.update_mu_c()
        elif param == "mu_a":
            self.update_mu_a()
        elif param == "eta_r":
            self.update_eta_r()
        else:
            raise ValueError(f"parameter {param} is not recognized")

    def update(self):
        self.iterations+=1
        for param in self.update_order:
            self._update_param(param)

    def fit(self, max_iterations=10000, save_every=100):
        for i in tqdm(range(max_iterations)):
            self.update()
            history = self.get_variable_parameters_values()
            history["iteration"] = self.iterations
            self.history.append(history)
            if i%save_every == save_every-1:
                self.save_parameters_to_file()
                self.save_history_to_file()
            if self.all_converged():
                print("Model converged")
                break

    ##########################
    ## Saving/loading model ##
    ##########################
    def change_config(self, config):
        if "update_order" in config:
            self.update_order = config["update_order"]
        for param in self.get_parameters().values():
            param.change_config(config)
        self.fill_variables_in_train()

    def get_model_config(self):
        model_config = self.get_parameters_config()
        model_config["model_name"] = self.model_name
        model_config["update_order"] = self.update_order
        return model_config
    
    def get_parameters_config(self):
        parameter_config = {}
        for param in self.get_parameters().values():
            parameter_config.update(param.hiperparameters())
        return parameter_config

    def get_parameters(self):
        return {key: param for key, param in self.__dict__.items() if isinstance(param, Parameter)}

    def get_parameters_values(self):
        parameters = {key: param.value for key, param in self.get_parameters().items()}
        return parameters

    def get_variable_parameters_values(self):
        parameters = {key: param.value for key, param in self.get_parameters().items() if not param.isconstant}
        return parameters

    def get_parameters_stats(self):
        parameters_stats = {}
        for param in self.get_parameters().values():
            if not param.isconstant:
                parameters_stats.update(param.parameter_dict())
        return parameters_stats

    def load_parameters_values(self, parameters):
        for key, param in parameters:
            if key in self.__dict__:
                self.__dict__[key].value = param
        self.fill_variables_in_train()

    def load_parameters(self, parameters):
        for key, param in parameters:
            if not isinstance(param, Parameter):
                raise ValueError(f"{param} is not parameter")
            if key in self.__dict__:
                self.__dict__[key] = param
        self.fill_variables_in_train()

    def save_parameters_to_file(self, dir = "models/"):
        filename = self.model_name
        save_model(self.get_parameters(), filename, extension="params", dir=dir)

    def save_history_to_file(self, dir="models/"):
        filename = self.model_name + f"H{self.iterations}"
        save_model(self.history, filename, dir, add_date=False, extension="history", prevent_overwrite=False)
        self.history=[]

    def get_trainset(self):
        return self.train

    def load_trainset(self, trainset):
        self.train = trainset.copy()


    ####################
    ###  Prediction  ###
    ####################

    def pdf(self, article, customer, return_status, customer_size, n_samples=1000):
        mu_a_samples = np.random.normal(self.mean_mu_a.value[article], self.variance_mu_a.value[article], size=n_samples)
        mu_c_samples = np.random.normal(self.mean_mu_c.value[customer], self.variance_mu_c.value[customer], size=n_samples)
        if return_status==FIT_LABEL:
            eta_r_samples = np.random.normal(self.mean_eta_kept.value, self.variance_eta_kept.value, size=n_samples)
        elif return_status==LARGE_LABEL:
            eta_r_samples = np.random.normal(self.mean_eta_big.value, self.variance_eta_big.value, size=n_samples)
        elif return_status==SMALL_LABEL:
            eta_r_samples = np.random.normal(self.mean_eta_small.value, self.variance_eta_small.value, size=n_samples)
        else:
            ValueError("unknown return status")
        sigma_c_samples = 1/np.random.gamma(self.alpha_sigma_c.value[customer], 1/self.beta_sigma_c.value[customer], size=n_samples)

        mu_samples = mu_a_samples+mu_c_samples+eta_r_samples
        pdf_values = normal_pdf(mu_samples, sigma_c_samples, customer_size)
        return pdf_values.mean()

    def multi_pdfs(self, article, customer, return_status, customer_sizes=np.arange(0,59), n_samples=1000):
        mu_a_samples = np.random.normal(self.mean_mu_a.value[article], self.variance_mu_a.value[article], size=n_samples)
        mu_c_samples = np.random.normal(self.mean_mu_c.value[customer], self.variance_mu_c.value[customer], size=n_samples)
        if return_status==self.KEPT:
            eta_r_samples = np.random.normal(self.mean_eta_kept.value, self.variance_eta_kept.value, size=n_samples)
        elif return_status==self.BIG:
            eta_r_samples = np.random.normal(self.mean_eta_big.value, self.variance_eta_big.value, size=n_samples)
        elif return_status==self.SMALL:
            eta_r_samples = np.random.normal(self.mean_eta_small.value, self.variance_eta_small.value, size=n_samples)
        else:
            ValueError("unknown return status")
        sigma_c_samples = 1/np.random.gamma(self.alpha_sigma_c.value[customer], 1/self.beta_sigma_c.value[customer], size=n_samples)

        mu_samples = mu_a_samples+mu_c_samples+eta_r_samples
        pdf_values = [normal_pdf(mu_samples, sigma_c_samples, customer_size).mean() for customer_size in customer_sizes]
        return pdf_values
        
    def predict(self, test_df):
        results = test_df.copy()
        results["size_prob"] = results.apply(lambda row: self.pdf(*row[["item_id", "user_id", "result", "size"]]), axis=1)
        results["all_sizes_results"] = results.apply(lambda row: self.multi_pdfs(*row[["item_id", "user_id", "result"]]), axis=1)
        results["predicted_prob"], results["predicted_size"] = zip(*results["all_sizes_results"].apply(lambda x: (np.max(x), np.argmax(x))))
        return results
