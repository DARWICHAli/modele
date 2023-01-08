# High level API for image captionning

## Segmentation
from detectron2.data import MetadataCatalog
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.structures.instances import Instances
from detectron2.structures.boxes import Boxes
from detectron2.checkpoint import DetectionCheckpointer

from utils.extract_utils import get_image_blob
from bua.caffe import add_bottom_up_attention_config
from bua.caffe.modeling.layers.nms import nms

import torch
import numpy as np

## M2
import pickle
from data import TextField, UserImageDetectionsField, RawField, DataLoader
from data.dataset import PairedDataset
from models import model_factory
from evaluation import PTBTokenizer
import itertools

from argparse import Namespace
import os

def get_segmentation_model(config = 'weights/bottom-up/test-caffe-r101.yaml', weights_file=None, vocab_directory='evaluation'):
    """
    Construct and return segmentation model custom class .
    :param config: Path to configuration
    :type config: str
    :return: A model class containing intern API for object detection tasks :py:func:`segment_picture`.
    """
    classes = ['__background__']
    with open(os.path.join(vocab_directory, 'objects_vocab.txt')) as f:
        for object in f.readlines():
            classes.append(object.split(',')[0].lower().strip())

    # Load attributes
    attributes = ['__no_attribute__']
    with open(os.path.join(vocab_directory, 'attributes_vocab.txt')) as f:
        for att in f.readlines():
            attributes.append(att.split(',')[0].lower().strip())

    MetadataCatalog.get("vg").thing_classes = classes
    MetadataCatalog.get("vg").attr_classes = attributes

    cfg = get_cfg()
    add_bottom_up_attention_config(cfg, True)
    cfg.merge_from_file(config)
    cfg.merge_from_list(['MODEL.BUA.EXTRACT_FEATS',True])

    if not torch.cuda.is_available():
        cfg.MODEL.DEVICE = 'cpu'

    cfg.freeze()

    model = DefaultTrainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(weights_file, resume=True)
    model.eval()

    return {'model': model, 'cfg': cfg}

def segment_picture(dictionnary, picture, MIN_BOXES = 10, MAX_BOXES = 100, CONF_THRESH = 0.4):
    """
    Compute and returns segmentation boxes from 'picture' using 'model'.
    :param dictionnary: Model returned by :py:func:`get_segmentation_model`.
    :param picture: Raw picture **without any kind of preprocessing **.
    :param MIN_BOXES: Number of boxes to detect (min), default is 10.
    :param MAX_BOXES: Number of boxes to detect (max), default is 100.
    :param CONF_THRESH: Threshold required to keep a detection in. between 0 and 1.0
    :type picture: image from opencv.imread.
    :type MIN_BOXES: integer
    :type MAX_BOXES: integer
    :type CONF_THRESH: float
    :return: A tuple containing (boxes, features, proba_cls, instances). 'boxes' is a box compatible with torchvision.utils.draw_bounding_boxes, features is the embedding vector for each box, proba_cls is the probability for each class for each box. instances is an Instances object which can be used directly with detectron2.utils.visualizer.Visualizer.
    :rtype: (np.array(N, 4), np.array(N, 2048), np.array(N, 1601), Instances)
    """
    classes = MetadataCatalog.get("vg").thing_classes
    attributes = MetadataCatalog.get("vg").attr_classes

    model = dictionnary['model']
    cfg = dictionnary['cfg']
    dataset_dict = get_image_blob(picture, cfg.MODEL.PIXEL_MEAN)

    with torch.no_grad():
        boxes, scores, features_pooled, attr_scores = model([dataset_dict])

    dets = boxes[0].tensor.cpu() / dataset_dict['im_scale']
    scores = scores[0].cpu()
    feats = features_pooled[0].cpu()
    attr_scores = attr_scores[0].cpu()

    max_conf = torch.zeros((scores.shape[0])).to(scores.device)
    for cls_ind in range(1, scores.shape[1]):
            cls_scores = scores[:, cls_ind]
            keep = nms(dets, cls_scores, 0.3)
            max_conf[keep] = torch.where(cls_scores[keep] > max_conf[keep],
                                        cls_scores[keep],
                                        max_conf[keep])

    keep_boxes = torch.nonzero(max_conf >= CONF_THRESH).flatten()
    if len(keep_boxes) < MIN_BOXES:
        keep_boxes = torch.argsort(max_conf, descending=True)[:MIN_BOXES]
    elif len(keep_boxes) > MAX_BOXES:
        keep_boxes = torch.argsort(max_conf, descending=True)[:MAX_BOXES]

    boxes = dets[keep_boxes].numpy()
    objects = np.argmax(scores[keep_boxes].numpy()[:,1:], axis=1)
    sc = scores[keep_boxes]
    ft = feats[keep_boxes]
    attr_thresh = 0.1
    attr = np.argmax(attr_scores[keep_boxes].numpy()[:,1:], axis=1)
    attr_conf = np.max(attr_scores[keep_boxes].numpy()[:,1:], axis=1)
    for i in range(len(keep_boxes)):
        bbox = boxes[i]
        if bbox[0] == 0:
            bbox[0] = 1
        if bbox[1] == 0:
            bbox[1] = 1
        cls = classes[objects[i]+1]  # caffe +2
        if attr_conf[i] > attr_thresh:
            cls = attributes[attr[i]+1] + " " + cls   #  caffe +2

    instances = Instances(
            image_size=dataset_dict['image'].shape[:2],
            pred_boxes=boxes,
            scores=sc.max(-1)[0],
            pred_classes=sc.max(-1)[1],
            attr_scores=[attr_conf[i] for i in range(len(keep_boxes))],
            attr_classes=[attr[i]+1 for i in range(len(keep_boxes))]
    )

    return boxes, ft, sc, instances

def get_captionning_model(pth="meshed_memory_transformer.pth", vocab_file="./weights/m2/vocab.pkl", mtype='transformer_m2_origin', memory=40):
    """
    Construct and return captionning model.
    :param device: torch.device to load model into.
    :type device: torch.device
    :param directory: Directory containing vocabulary and weights of the model.
    :param weights_file: Name of the file containg weights of the model.
    :param vocab_file: Name of the file containing pickled vocabulary.
    :return: A model to be used with :py:func:`caption_embeddings`.
    """
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy', remove_punctuation=True, nopoints=False)
    text_field.vocab = pickle.load(open(vocab_file, 'rb'))
    
    args = Namespace(grid_on=False, max_detections=50, dim_feats=2048, d_k=64, d_v=64, head=8, d_m=memory, model=mtype)
    Transformer, TransformerEncoder, TransformerDecoderLayer, ScaledDotProductAttention = model_factory(args)
    encoder = TransformerEncoder(3, 0, attention_module=ScaledDotProductAttention,
                                 d_in=args.dim_feats,
                                 d_k=args.d_k,
                                 d_v=args.d_v,
                                 h=args.head,
                                 attention_module_kwargs={'m': args.d_m} if args.d_m > 0 else None
                                 )
    decoder = TransformerDecoderLayer(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'],
                                      d_k=args.d_k,
                                      d_v=args.d_v,
                                      h=args.head
                                      )
    model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder)

    if not torch.cuda.is_available():
        data = torch.load(pth, map_location=torch.device('cpu'))
    else:
        data = torch.load(pth)

    model.load_state_dict(data['state_dict'])
    model.eval()
    return {'model': model, 'text_field': text_field}

def caption_segmentation(model, features, proba_cls, max_detections=50, beam_size=5, max_len=20, out_size=1):
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
    :param out_size: Number of sequences to output.
    :return: An array of sentences and their respective probabilities.
    :rtype: list((sentence, probability))
    """
    image_field = UserImageDetectionsField(features, proba_cls, max_detections)
    t = np.expand_dims(image_field.preprocess(None), axis=0)
    tensor = torch.tensor(t)

    gen = {}
    probas = {}
    with torch.no_grad():
        out, log_probs = model['model'].beam_search(tensor, max_len, model['text_field'].vocab.stoi['<eos>'], beam_size, out_size=out_size, sort_by_prob=True)
    if out_size == 1:
        out = np.expand_dims(out, axis=0)
        log_probs = np.expand_dims(log_probs, axis=0)

    for j, (o, l) in enumerate(zip(out, log_probs)):
        caps_gen = model['text_field'].decode(o, join_words=False)
        for i, (gen_i, probs) in enumerate(zip(caps_gen, l)):
                    gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                    gen[f"{j}_{i}"] = gen_i.strip()
                    probas[f"{j}_{i}"] = probs

    return list(zip(gen.values(), probas.values()))

def get_trained_models():
    """Returns a dict of all trained combinations which can be passed to get_XXX_model.
    :return: A dictionnary where one key is one trained combination.
    :rtype: dict
    """
    return {
            'R101-M2-original': {
                'segmentation': {
                    'weights': 'bua-caffe-frcn-r101-k10-100.pth',
                    'config': 'test-caffe-r101.yaml',
                    'size': 100,
                    'MIN_BOXES': 10,
                    'MAX_BOXES': 100,
                    'THRESHOLD': 0.4
                    }, 
                'captioning': {
                    'pth': 'meshed_memory_transformer.pth',
                    'mtype': 'transformer_m2_origin',
                    'memory': 40,
                    'vocab': 'vocab.pkl'
                }
            },
            'R152-M2-256m+spider': {
                'segmentation': {
                    'weights': 'bua-caffe-frcn-r152.pth',
                    'config': 'test-caffe-r152.yaml',
                    'size': 152,
                    'MIN_BOXES': 10,
                    'MAX_BOXES': 100,
                    'THRESHOLD': 0.4
                    }, 
                'captioning': {
                    'pth': 'spider_test_m256_topg_best.pth',
                    'mtype': 'transformer',
                    'memory': 256,
                    'vocab': 'spider_test_m256_topg_best.vocab.pkl'
                }
            },
            'R152-M2-0m+ciderbugged': {
                'segmentation': {
                    'weights': 'bua-caffe-frcn-r152.pth',
                    'config': 'test-caffe-r152.yaml',
                    'size': 152,
                    'MIN_BOXES': 10,
                    'MAX_BOXES': 100,
                    'THRESHOLD': 0.4
                    }, 
                'captioning': {
                    'pth': '2017_152_best.pth',
                    'mtype': 'transformer',
                    'memory': 0,
                    'vocab': '2017_152_m0_best.vocab.pkl'
                }
            },
            'R152-M2-256m+cider': {
                'segmentation': {
                    'weights': 'bua-caffe-frcn-r152.pth',
                    'config': 'test-caffe-r152.yaml',
                    'size': 152,
                    'MIN_BOXES': 10,
                    'MAX_BOXES': 100,
                    'THRESHOLD': 0.4
                    }, 
                'captioning': {
                    'pth': '2017_152_m256_best.pth',
                    'mtype': 'transformer',
                    'memory': 256,
                    'vocab': '2017_152_m256_best.vocab.pkl'
                }
            }
        }

def load_models_from_dict(entry, model_directory='trained_models/'):
    """Load both models for segmentation and captionning from :py:func:get_trained_models.
    :param entry: entry from :py:func:get_trained_models. ex: get_captionning_model()['R101-M2-original'].
    :type entry: dict
    :param model_directory: Directory where models and configs are located.
    :type model_directory: str
    :return: (segmentation_model, captioning_model) to be used individually with :py:func:segment_picture and :py:func:caption_segmentation.
    :rtype: model, model
    """
    if not os.path.exists(model_directory):
        raise RuntimeError(f"Error: Model directory \"{model_directory}\" doesn't exist!")
    smodel = get_segmentation_model(config=os.path.join(model_directory, entry['segmentation']['config']), weights_file=os.path.join(model_directory, entry['segmentation']['weights']), vocab_directory=model_directory)

    cmodel = get_captionning_model(pth=os.path.join(model_directory, entry['captioning']['pth']), vocab_file=os.path.join(model_directory, entry['captioning']['vocab']), mtype=entry['captioning']['mtype'], memory=entry['captioning']['memory'])

    return smodel, cmodel

if __name__ == '__main__':
    import cv2
    import sys
    from detectron2.utils.visualizer import Visualizer
    from matplotlib import pyplot as plt

    available = get_trained_models()

    #for k, v in available.items():
    #    l, r = load_models_from_dict(v, 'models_data')

    seg_model, cap_model = load_models_from_dict(available['R152-M2-256m+spider'], model_directory='models_data')

#    seg_model = get_segmentation_model()
    path_to_img = sys.argv[1] if len(sys.argv) > 1 else 'demo/004545.jpg'
    image = cv2.imread(path_to_img)
    boxes, features, proba_cls, instances = segment_picture(seg_model, image)
    pred = instances.to('cpu')
    v = Visualizer(image[:, :, :], MetadataCatalog.get("vg"), 1)
    v = v.draw_instance_predictions(instances)
    plt.imshow(v.get_image()[:, :, ::-1])

    plt.show()
    cap_model = get_captionning_model()
    ret = caption_segmentation(cap_model, features, proba_cls,max_detections=21, beam_size=5, max_len=10,out_size=1)

    for s, p in ret:
        print(f"P({s}) = {p}")

