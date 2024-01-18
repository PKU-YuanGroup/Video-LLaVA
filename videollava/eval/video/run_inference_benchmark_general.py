import os
import argparse
import json
from tqdm import tqdm
# from video_chatgpt.eval.model_utils import initialize_model, load_video
# from video_chatgpt.inference import video_chatgpt_infer

from videollava.eval.video.run_inference_video_qa import get_model_output
from videollava.mm_utils import get_model_name_from_path
from videollava.model.builder import load_pretrained_model


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument('--model_path', help='', required=True)
    parser.add_argument('--cache_dir', help='', required=True)
    parser.add_argument('--video_dir', help='Directory containing video files.', required=True)
    parser.add_argument('--gt_file', help='Path to the ground truth file.', required=True)
    parser.add_argument('--output_dir', help='Directory to save the model results JSON.', required=True)
    parser.add_argument('--output_name', help='Name of the file for storing results JSON.', required=True)
    # parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--device", type=str, required=False, default='cuda:0')
    parser.add_argument('--model_base', help='', default=None, type=str, required=False)
    parser.add_argument("--model_max_length", type=int, required=False, default=2048)
    # parser.add_argument("--conv-mode", type=str, required=False, default='video-chatgpt_v1')
    # parser.add_argument("--projection_path", type=str, required=True)

    return parser.parse_args()


def run_inference(args):
    """
    Run inference on a set of video files using the provided model.

    Args:
        args: Command-line arguments.
    """# Initialize the model
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name)
    model = model.to(args.device)

    # Load the ground truth file
    with open(args.gt_file) as file:
        gt_contents = json.load(file)

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_list = []  # List to store the output results
    # conv_mode = args.conv_mode

    video_formats = ['.mp4', '.avi', '.mov', '.mkv']

    # Iterate over each sample in the ground truth file
    for sample in tqdm(gt_contents):
        video_name = sample['video_name']
        sample_set = sample
        question = sample['Q']

        try:
            # Load the video file
            for fmt in video_formats:  # Added this line
                temp_path = os.path.join(args.video_dir, f"{video_name}{fmt}")
                if os.path.exists(temp_path):
                    video_path = temp_path
                    output = get_model_output(model, processor['video'], tokenizer, video_path, question, args)
                    sample_set['pred'] = output
                    output_list.append(sample_set)
                    break

        except Exception as e:
            print(f"Error processing video file '{video_name}': {e}")

    # Save the output list to a JSON file
    with open(os.path.join(args.output_dir, f"{args.output_name}.json"), 'w') as file:
        json.dump(output_list, file)


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
