agri-perceiver/
│
├── configs/                     # Experiment configs only
│   ├── stage1_pretrain.yaml
│   ├── stage2_finetune.yaml
│   └── inference.yaml
│
├── models/                      # Neural network modules
│   ├── vision/
│   │   └── siglip_wrapper.py
│   │
│   ├── spatial/
│   │   └── tile_embeddings.py
│   │
│   ├── perceiver/
│   │   └── perceiver_resampler.py
│   │
│   ├── projector/
│   │   └── vision_projector.py
│   │
│   ├── llm/
│   │   └── phi3_lora.py
│   │
│   └── agri_vlm.py              # FULL architecture assembly
│
├── data/                        # Dataset interface ONLY
│   └── dataset_loader.py        # Reads tiles + JSONL (no labeling!)
│
├── training/
│   ├── train_stage1.py
│   ├── train_stage2.py
│   ├── losses.py
│   └── trainer_utils.py
│
├── inference/
│   ├── generate_json.py
│   ├── outlines_schema.py
│   └── api_server.py
│
├── monitoring/
│   ├── gpu_logger.py
│   └── system_logger.py
│
├── scripts/
│   ├── download_models.py
│   └── run_training.sh
│
├── tests/
│   └── test_forward_pass.py
│
├── .gitignore
├── requirements.txt
└── README.md
