from benchopt import BaseObjective
import copy


class Objective(BaseObjective):
    name = "Linear Regression"
    min_benchopt_version = "1.8"

    parameters = {
        "device": ["cpu"],
    }

    def set_data(self, dataloader, model):
        self.dataloader = dataloader
        self.model = model

    def get_one_result(self):
        return dict()

    def evaluate_result(self, logs={}):
        total_logs = {
            k: sum(v) for k, v in logs.items()
        }
        if "comm_time" in total_logs:
            total_logs["comm_ratio"] = total_logs["comm_time"] / total_logs["run_time"]
        return total_logs

    def get_objective(self):
        return dict(
            dataloader=copy.deepcopy(self.dataloader),
            model=copy.deepcopy(self.model),
            device=self.device
        )
