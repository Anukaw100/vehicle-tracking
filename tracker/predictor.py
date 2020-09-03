# Inspired by "https://stackoverflow.com/a/62446532" and the Detectron2 demo at
# "https://github.com/facebookresearch/detectron2/tree/master/demo".
import cv2
import torch
import numpy

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from visualizer import CustomVisualizer
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode

#from sort import Sort
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort.nn_matching import NearestNeighborDistanceMetric
from deep_sort.deep_sort.tracker import Tracker

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
        video_visualizer = CustomVisualizer(self.metadata, self.instance_mode)
        
        # Instantiate the object tracker and create variables to store the IDs
        # and times.
        mot_tracker = Tracker(NearestNeighborDistanceMetric("euclidean", 50.0))
        vehicle_arrival_times = {}

        # Read as ID'd (eye-dee-id) instances.
        def record_time_of_arrival(ided_instances):
            for instance in ided_instances:
                instance_id = instance.track_id
                if not instance_id in vehicle_arrival_times:
                    vehicle_arrival_times[instance_id] = video.get(cv2.CAP_PROP_POS_MSEC)

        def generate_vehicle_indices(instances):
            # Vehicle IDs are between 2--8 inclusive: 2=car, 3=motorcycle,
            # 4=airplane, 5=bus, 6=train, 7=truck, 8=boat. Found this out from
            # self.metadata.

            # Returns a tensor of 1s (true) and 0s (false) based on the value
            # satisfying the condition.
            mask = (instances.pred_classes >= 2) & (instances.pred_classes <= 8)

            # Returns the indices of the 'True' values in the mask.
            return torch.nonzero(mask, as_tuple=True)

        def generate_object_id(instances):
            boxes = instances.pred_boxes.tensor.numpy()
            scores = instances.scores.numpy()
            detections = [Detection(boxes[index], scores[index], NoneType) for index in range(len(boxes))]
            mot_tracker.predict()
            mot_tracker.update(detections)
            return mot_tracker.tracks

        def process_predictions(frame, predictions):
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            #if "panoptic_seg" in predictions:
            #    panoptic_seg, segments_info = predictions["panoptic_seg"]
            #    vis_frame = video_visualizer.draw_panoptic_seg_predictions(
            #        frame, panoptic_seg.to(self.cpu_device), segments_info
            #    )
            #el
            if "instances" in predictions:
                predictions = predictions["instances"].to(self.cpu_device)
                filtered_predictions = predictions[generate_vehicle_indices(predictions)]
                ided_instances = generate_object_id(filtered_predictions)
                record_time_of_arrival(ided_instances)
                vis_frame = video_visualizer.draw_instance_predictions(frame, filtered_predictions, ided_instances, vehicle_arrival_times, video.get(cv2.CAP_PROP_POS_MSEC))
                #vis_frame = video_visualizer.draw_instance_predictions(frame, filtered_predictions)
            elif "sem_seg" in predictions:
                vis_frame = video_visualizer.draw_sem_seg(
                    frame, predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )

            # Converts Matplotlib RGB format to OpenCV BGR format
            vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
            return vis_frame

        frame_gen = self._frame_from_video(video)
        for frame in frame_gen:
            yield process_predictions(frame, self.predictor(frame))
