import yaml


def read_params(config_path="params.yaml"):
    method_name = read_params.__name__

    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)

        return config

    except Exception as e:
        raise Exception(
            f"Exception occured in {__file__}, Method : {method_name}, Error : {str(e)}"
        )
