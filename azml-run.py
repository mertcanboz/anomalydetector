# run-pytorch-data.py
from azureml.core import Workspace
from azureml.core import Experiment
from azureml.core import Environment
from azureml.core import ScriptRunConfig
from azureml.core import Dataset

if __name__ == "__main__":
    ws = Workspace.from_config()
    datastore = ws.get_default_datastore()
    # dataset = Dataset.File.from_files(path=(datastore, 'datasets/cifar10'))

    experiment = Experiment(workspace=ws, name='msad-kpi-train-all-gpu')

    config = ScriptRunConfig(
        source_directory='./srcnn',
        script='train.py',
        compute_target='gpubox',
        arguments=[
            '--data', "kpi\kpi_train_ts_data",
            '--window', 1440,
            '--epoch', 200,
            '--use-gpu'],
    )

    # set up pytorch environment
    env = Environment.from_conda_specification(
        name='devenv-anomalydetector',
        file_path='conda.yml'
    )
    config.run_config.environment = env

    run = experiment.submit(config)
    aml_url = run.get_portal_url()
    print("Submitted to compute cluster. Click link below")
    print("")
    print(aml_url)
