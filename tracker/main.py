import argparse
import os
import cv2
import tqdm

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger

from predictor import Visualizer

def setup_cfg(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)  #  FIXME  cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
    cfg.merge_from_list(args.opts)         #  FIXME  cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")

    # Set score thresholds.
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold  #  FIXME  Remove this confidence threshold.
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold  #  FIXME  Remove this confidence threshold.
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold

    cfg.freeze()
    return cfg

def get_parser():
    parser = argparse.ArgumentParser(description="Using Detectron2 to detect vehicles passing through a video")
    parser.add_argument(
        "--config-file",
        default="configs/",
        metavar="FILE",
        help="Path to config file"
    )
    parser.add_argument(
        "--input",
        default="input/video.mp4",
        help="Path to a video file"
    )
    parser.add_argument(  #  FIXME  Potentially unnecessary as user doesn't supply output filename.
        "--output",
        default="output/video.mkv",
        help="A file to save output visualizations. "
             "If not given, output will be shown in an OpenCV window."
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.8,
        help="Minimum score for instance predictions to be shown"
    )
    parser.add_argument(
        "--opts",
        default=[],
        nargs=argparse.REMAINDER,
        help="Modify config options using the command-line 'KEY VALUE' pairs"
    )
    return parser

if __name__ == "__main__":
    args = get_parser().parse_args()

    #  FIXME  Potentially unnecessary in the final build.
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    visualizer = Visualizer(cfg)

    # Read video and get properties.
    video = cv2.VideoCapture(args.input)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_per_second = video.get(cv2.CAP_PROP_FPS)
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Read filename of input video.
    basename = os.path.basename(args.input)

    #  FIXME  Potentially unnecessary since user doesn't provide output
    # name in final build.
    if args.output:
        # Creates a file of the same name as the input file if the given
        # output argument is a directory.
        if os.path.isdir(args.output):
            output_fname = os.path.join(args.output, basename)
            output_fname = os.path.splitext(output_fname)[0] + ".mkv"
        else:
            output_fname = args.output

        # Asserts that the file doesn't exist already.
        assert not os.path.isfile(output_fname), output_fname

        output_file = cv2.VideoWriter(
            filename=output_fname,
            #  NOTE  If this format doesn't work, try another one that is
            # available at http://www.fourcc.org/codecs.php
            fourcc=cv2.VideoWriter_fourcc(*"x264"),
            fps=float(frames_per_second),
            frameSize=(width, height),
            isColor=True
        )

    assert os.path.isfile(args.input)  # Asserts that the input file exists.

    # Writes the output.
    for vis_frame in tqdm.tqdm(visualizer.run_on_video(video), total=num_frames):
        #  FIXME  Potentially unnecessary. Reason above! Should update
        # this to work in a website.
        if args.output:
            output_file.write(vis_frame)
        else:
            cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
            cv2.imshow(basename, vis_frame)
            if cv2.waitKey(1) == 27:
                break  # Press 'Esc' to quit.

    # Release allocated resources.
    video.release()
    if args.output:
        output_file.release()
    else:
        cv2.destroyAllWindows()
