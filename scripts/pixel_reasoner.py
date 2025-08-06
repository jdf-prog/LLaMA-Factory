import fire
import datasets
import json
from pathlib import Path

IMAGE_TOKEN = "<image>"
VIDEO_TOKEN = "<video>"

def process_item(item, data_dir="./data/pixel_reasoner_sft_data"):
    messages = item["message_list"]
    new_messages = []
    images = []
    for message in messages:
        role = message["role"]
        content_list = message["content"]
        new_message = {
            "role": message["role"],
            "content": ""
        }
        for content in content_list:
            image_content = content.get("image")
            text_content = content.get("text")
            video_content = content.get("video")
            if image_content is not None:
                new_message["content"] += IMAGE_TOKEN
                images.append(image_content)
            if video_content is not None:
                new_message["content"] += IMAGE_TOKEN * len(video_content)
                images.extend(video_content)
            if text_content is not None:
                new_message["content"] += text_content
        new_messages.append(new_message)
    item["messages"] = new_messages
    if len(images) > 0:
        item["images"] = [
            str((Path(data_dir) / image).absolute())
            for image in images
        ]
        assert all(Path(image).exists() for image in item["images"]), "Some images do not exist."
    return item


def main(
    dataset_path="TIGER-Lab/PixelReasoner-SFT-Data",
    data_dir="./data/pixel_reasoner_sft_data",
):
    dataset = datasets.load_dataset(dataset_path)
    processed_dataset = dataset.map(
        process_item,
        num_proc=8,
        remove_columns=dataset["train"].column_names,
        desc="Processing PixelReasoner dataset",
    )
    with open(Path(data_dir) / "processed_data.json", "w") as f:
        json.dump(processed_dataset["train"].to_list(), f, indent=2)
        print(f"Processed data saved to {Path(data_dir) / 'processed_data.json'}")

if __name__ == "__main__":
    fire.Fire(main)
"""
Step 1
```bash
huggingface-cli download TIGER-Lab/PixelReasoner-SFT-Data images.zip --local-dir ./data/pixel_reasoner_sft_data --local-dir-use-symlinks False --repo-type dataset
huggingface-cli download TIGER-Lab/PixelReasoner-SFT-Data videos.zip --local-dir ./data/pixel_reasoner_sft_data --local-dir-use-symlinks False --repo-type dataset
cd data/pixel_reasoner_sft_data
unzip images.zip
unzip videos.zip
```
Step 2
```bash
python scripts/pixel_reasoner.py --dataset_path TIGER-Lab/PixelReasoner-SFT-Data --data_dir ./data/pixel_reasoner_sft_data
```
"""
