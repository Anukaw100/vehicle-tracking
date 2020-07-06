# Inspired by "https://stackoverflow.com/a/62446532" and the Detectron2 demo at
# "https://github.com/facebookresearch/detectron2/tree/master/demo".
import cv2
import torch

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode

class Visualizer(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode): Default is SEGMENTATION for similar
            colors in the same instances. Change this to IMAGE if random colors
            suffice.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode
        self.predictor = DefaultPredictor(cfg)

    def _frame_from_video(self, video):
        while video.isOpened():
            has_frame, frame = video.read()
            if has_frame:
                yield frame
            else:
                break

    def run_on_video(self, video):
        """
        Visualizes predictions on frames of the input video.
        Args:
            video (cv2.VideoCapture): a :class:`VideoCapture` object, whose source can be
                either a webcam or a video file.
        Yields:
            ndarray: BGR visualizations of each video frame.
        """
        video_visualizer = VideoVisualizer(self.metadata, self.instance_mode)

        #  TODO  Our main area of visualisation lies here.
        def process_predictions(frame, predictions):
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if "panoptic_seg" in predictions:
                panoptic_seg, segments_info = predictions["panoptic_seg"]
                vis_frame = video_visualizer.draw_panoptic_seg_predictions(
                    frame, panoptic_seg.to(self.cpu_device), segments_info
                )
            elif "instances" in predictions:
                predictions = predictions["instances"].to(self.cpu_device)
                vis_frame = video_visualizer.draw_instance_predictions(frame, predictions)
            elif "sem_seg" in predictions:
                vis_frame = video_visualizer.draw_sem_seg(
                    frame, predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )

            #  FIXME  Complete the prediction filtering and insertion.
            instance_classes = predictions["instances"].pred_classes
            # Vehicle IDs are between 2--8 inclusive: 2=car, 3=motorcycle, 4=airplane, 5=bus, 6=train, 7=truck, 8=boat.
            # Found this out from the self.metadata.
            mask = (instance_classes >= 2) & (instance_classes <= 8)  # Returns a tensor of 1s (true) and 0s (false) based on the value satisfying the condition.
            filtered_instance_classes = instance_classes[mask]        # Returns the values at the 1s.
            indices = torch.nonzero(mask, as_tuple=True)              # Returns the indices of the 1s.
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            print(panoptic_seg.to(self.cpu_device), segments_info, filtered_instance_classes, indices, video.get(cv2.CAP_PROP_POS_MSEC) / 1000, sep='\n')

            # Converts Matplotlib RGB format to OpenCV BGR format
            vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
            return vis_frame

        frame_gen = self._frame_from_video(video)
        for frame in frame_gen:
            yield process_predictions(frame, self.predictor(frame))
