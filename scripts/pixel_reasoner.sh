huggingface-cli download TIGER-Lab/PixelReasoner-SFT-Data images.zip --local-dir ./data/pixel_reasoner_sft_data --local-dir-use-symlinks False --repo-type dataset
huggingface-cli download TIGER-Lab/PixelReasoner-SFT-Data videos.zip --local-dir ./data/pixel_reasoner_sft_data --local-dir-use-symlinks False --repo-type dataset
cd data/pixel_reasoner_sft_data
unzip images.zip
unzip videos.zip