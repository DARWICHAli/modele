# High level API for image captionning
from feature_extractor import FeatureExtractor
import encoderCNN_pretrained

def get_segmentation_model():
    """
    Construct and return segmentation model.
    :return: A model compatible with :py:func:`segment_picture`.
    """
    modelClass = encoderCNN_pretrained.EncoderCNN()
    return modelClass

def segment_picture(model, picture, treshhold):
    """
    Compute and returns segmentation boxes from 'picture' using 'model'.
    :param model: Model returned by :py:func:`get_segmentation_model`.
    :param picture: Raw picture **without any kind of preprocessing **.
    :param treshhold: the minimum score of the proposed boxes to select.
    :type picture: torchvision.io.image (read_image function).
    :type treshhold: float.
    :return: A tuple containing (boxes, features, proba_cls). 'boxes' is a box compatible with torchvision.utils.draw_bounding_boxes, features is the embedding vector for each box, proba_cls is the probability for each class for each box.
    :rtype: (np.array(N, 2, 2), np.array(N, 2048), np.array(N, 1601))
    """
    imgT, boxes, pred_score, weights = model.get_prediction(picture, treshhold)  # Get predictions 
    resnet_features = FeatureExtractor(model.model, layers=['backbone.body.layer4', 'backbone.fpn', 'rpn', 'roi_heads.box_roi_pool', 'backbone.body.layer3', 'roi_heads', 'roi_heads.box_roi_pool', 'backbone'])
    features = resnet_features([imgT])
    return (boxes, features, pred_score)

def get_captionning_model():
    """
    Construct and return captionning model.
    :return: A model to be used with :py:func:`caption_embeddings`.
    """
    raise NotImplementedError

def caption_segmentation(model, features, proba_cls, max_detections=100, beam_size=5, max_len=20):
    """
    Compute and return sentences from segmentation.
    :param model: Model returned by :py:func:`get_captionning_model`
    :param features: Features given by :py:func:`segment_picture`.
    :type features: np.array(N, 2048)
    :param proba_cls: Probabilities of each feature given by :py:func:`segment_picture`.
    :type proba_cls: np.array(N, 1601)
    :param max_detections: Maximum number of detections to use.
    :param beam_size: Width of the beam search.
    :param max_len: Maximum length of a sentence.
    :return: An array of sentences and their respective probabilities.
    :rtype: list((sentence, probability))
    """
    raise NotImplementedError
    return [(sentence, proba)]



