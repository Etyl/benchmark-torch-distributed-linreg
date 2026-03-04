from benchopt import BasePlot


class Plot(BasePlot):
    name = "Metrics"
    type = "boxplot"
    options = {
        "objective": ...,
        "dataset": ...,
        "metric": ["comm_time", "comm_time_cpu", "run_time"],
    }

    def plot(self, df, objective, dataset, metric):
        df = df.query(f"objective_name == '{objective}' and dataset_name == '{dataset}'")
        objective_column = f"objective_{metric}"
        plot_data = []
        for solver, df_filtered in df.groupby('solver_name'):
            if objective_column not in df_filtered.columns:
                continue

            y = [df_filtered[objective_column].values.tolist()]
            x = [solver]

            plot_data.append({
                "x": x,
                "y": y,
                "label": solver,
                "color": self.get_style(solver)["color"],
            })

        return plot_data

    def get_metadata(self, df, objective, dataset, metric):
        return {
            "title": f"Communication Time\n{objective}\nData: {dataset}",
        }