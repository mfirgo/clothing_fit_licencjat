from model.hierarchical_size_model_new import *
from model.hierarchical_model_return_status import *

class HierarchicalFullModel:
    def __init__(self, size_model, return_status_model):
        self.size_model = size_model
        self.return_status_model = return_status_model

    def predict(self, test, predict_type="expected"):
        return_status_result = self.return_status_model.predict(test)
        for result_description, result_label in [ ("fit", FIT_LABEL), ("large", LARGE_LABEL), ("small", SMALL_LABEL)]:
            all_same = test.copy()
            all_same["result"] = result_label
            all_same_result = self.size_model.predict(all_same, predict_type=predict_type)
            return_status_result[f"predicted_prob_{result_description}"] = all_same_result["predicted_prob"]
            return_status_result[f"predicted_size_{result_description}"] = all_same_result["predicted_size"]
            return_status_result[f"all_sizes_results_{result_description}"] = all_same_result["all_sizes_results"]
            return_status_result[f"predicted_full_prob_{result_description}"] = return_status_result[result_label] * return_status_result[f"predicted_prob_{result_description}"]
        full_prob_columns = [f"predicted_full_prob_{result_desc}" for result_desc in ["fit", "small", "large"]]
        full_prob_columns_to_return_status_desc = {f"predicted_full_prob_{result_desc}":result_desc for result_desc in ["fit", "small", "large"]}
        return_status_result["predicted_pair_prob"] = return_status_result[full_prob_columns].max(axis=1)
        return_status_result["predicted_pair_return_status"] = return_status_result[full_prob_columns].idxmax(axis=1).replace(full_prob_columns_to_return_status_desc)
        return_status_result["predicted_pair_size"] = return_status_result.apply(lambda row: row[f"predicted_size_{row['predicted_pair_return_status']}"], axis=1)
        return return_status_result

    def get_target_prob(self, test, predict_type="expected"):
        return_status_result = self.return_status_model.predict(test)
        full_result = self.size_model.size_prob(return_status_result, predict_type=predict_type)
        full_result["target_prob"] = full_result["size_prob"] * full_result["return_status_prob"]
        return full_result