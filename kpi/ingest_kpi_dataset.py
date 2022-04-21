import argparse
import logging
import os

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO)

sns.set_theme(style="whitegrid")
sns.set(font_scale=0.9)
plt.tight_layout()

pd.set_option("display.max_columns", 20)
pd.set_option("display.max_rows", 20)
pd.set_option("display.width", 800)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PlotKPIDataset')
    parser.add_argument('--csv-input-file', type=str, required=True, help='KPI Dataset CSV file path')
    parser.add_argument('--generate-data-and-plots', action='store_true', help='Generate individual TS files and plots')
    args = parser.parse_args()

    input_abs_path = os.path.abspath(args.csv_input_file)
    input_dir = os.path.dirname(input_abs_path)
    input_filename = os.path.splitext(os.path.basename(input_abs_path))[0]

    kpi_dataset = (pd.read_csv(input_abs_path, sep=",",
                               dtype={'timestamp': 'int64', 'value': 'float64', 'label': 'int64', 'KPI ID': 'object'})
                   .sort_values(["KPI ID", "timestamp"]))
    kpi_dataset["timestamp"] = pd.to_datetime(kpi_dataset["timestamp"], unit='s')

    if args.generate_data_and_plots:
        output_ts_data_dir = os.path.join(input_dir, f"{input_filename}_ts_data")
        output_ts_plot_dir = os.path.join(input_dir, f"{input_filename}_ts_plot")

        os.makedirs(output_ts_data_dir, exist_ok=True)
        os.makedirs(output_ts_plot_dir, exist_ok=True)

        for kpi_id, kpi_df in list(kpi_dataset.groupby("KPI ID")):
            logging.info(f"Save data {kpi_id}")
            data_path = os.path.join(output_ts_data_dir, f"{input_filename}_{kpi_id}.csv")
            kpi_df.to_csv(data_path, index_label=False, index=False)

            logging.info(f"Save plot {kpi_id}")
            plot_path = os.path.join(output_ts_plot_dir, f"{input_filename}_{kpi_id}.png")

            fig, ax = plt.subplots(figsize=(20, 5.625))
            p1 = sns.scatterplot(data=kpi_df.loc[kpi_df.label.astype("bool"), :], x='timestamp', y='value',
                                 color='red', marker=7, ax=ax)
            p2 = sns.lineplot(data=kpi_df, x='timestamp', y='value', lw=0.5, ax=ax)
            ax.tick_params(axis='x', rotation=45)
            plt.title(f"{input_filename}_{kpi_id}")
            # plt.show(block=True)
            fig.savefig(plot_path, dpi=192, bbox_inches='tight')
            plt.close(fig)

    kpi_dataset