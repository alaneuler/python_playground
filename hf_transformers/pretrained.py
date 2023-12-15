from transformers.configuration_utils import PretrainedConfig

config = PretrainedConfig.from_pretrained("./hf_transformers/models/my_model")
config.save_pretrained("./hf_transformers/models/my_model2")
