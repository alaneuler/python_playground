from transformers import BertPreTrainedModel


class NERModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)


model = NERModel.from_pretrained("hfl/chinese-roberta-wwm-ext")
print(model)
