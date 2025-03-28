from peft import get_peft_model,LoraConfig

class lora_custom():
    def __init__(self,type):
        super(lora_custom, self).__init__()
        # 定义 LoRA 配置
        self.type=type
        if self.type!='esm3':
            self.lora_config = LoraConfig(
                r=16,  # 低秩适配器的秩
                lora_alpha=32,  # 权重缩放
                lora_dropout=0.1,  # Dropout
                target_modules=["plm_encoder.embeddings.word_embeddings",
                                "plm_encoder.embeddings.position_embeddings",
                                "plm_encoder.embeddings.token_type_embeddings",
                                "plm_encoder.encoder.layer.*.attention.self.query",
                                "plm_encoder.encoder.layer.*.attention.self.key",
                                "plm_encoder.encoder.layer.*.attention.self.value",
                                "plm_encoder.encoder.layer.*.intermediate.dense",
                                "plm_encoder.encoder.layer.*.output.dense",
                                "plm_encoder.pooler.dense",
                                "classifier.model.embeddings.word_embeddings",
                                "classifier.model.embeddings.position_embeddings",
                                "classifier.model.embeddings.token_type_embeddings",
                                "classifier.model.encoder.layer.*.attention.self.query",
                                "classifier.model.encoder.layer.*.attention.self.key",
                                "classifier.model.encoder.layer.*.attention.self.value",
                                "classifier.encoder.layer.*.attention.self.query",
                                "classifier.encoder.layer.*.attention.self.key",
                                "classifier.encoder.layer.*.attention.self.value",
                                "classifier.encoder.layer.*.intermediate.dense",
                                "classifier.encoder.layer.*.output.dense",
                                "classifier.fc.linear",])
        else:
            self.lora_config=LoraConfig(
                r=16,  # 低秩适配器的秩
                lora_alpha=32,  # 权重缩放
                lora_dropout=0.1,  # Dropout
                target_modules=[
                                "model.encoder.sequence_embed",
                                "model.encoder.plddt_projection",
                                "model.encoder.structure_per_res_plddt_projection",
                                "model.encoder.structure_tokens_embed",
                                "model.encoder.ss8_embed",
                                "model.encoder.sasa_embed",
                                "model.encoder.function_embed.*",
                                "model.transformer.blocks.0.attn.layernorm_qkv.1",    #
                                "model.transformer.blocks.0.attn.out_proj",
                                "model.transformer.blocks.0.geom_attn.proj",
                                "model.transformer.blocks.0.geom_attn.out_proj",
                                "model.transformer.blocks.0.ffn.1",
                                "model.transformer.blocks.0.ffn.3",
                                "model.transformer.blocks.*.attn.layernorm_qkv.1",
                                # "model.transformer.blocks.*.attn.out_proj",
                                # "model.transformer.blocks.*.geom_attn.proj",
                                # "model.transformer.blocks.*.geom_attn.out_proj",
                                # "model.transformer.blocks.*.ffn.1",
                                # "model.transformer.blocks.*.ffn.3",
                                # "model.output_heads.*.0",
                                # "model.output_heads.*.3",
                                "classification.encoder.layer.*.attention.self.query",
                                "classification.encoder.layer.*.attention.self.key",
                                "classification.encoder.layer.*.attention.self.value",
                                "classification.encoder.layer.*.key_conv_attn_layer.depthwise",
                                "classification.encoder.layer.*.key_conv_attn_layer.pointwise",
                                "classification.encoder.layer.*.conv_kernel_layer",
                                "classification.encoder.layer.*.conv_out_layer",
                                "classification.classifier.fc.2",
                                "classification.classifier.fc.5"
                                ]
            )
        
    def forward(self,model):
        # 获取模型
        model = get_peft_model(model,self.lora_config)
        return model