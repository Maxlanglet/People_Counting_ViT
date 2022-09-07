import tensorflow as tf
import numpy as np
import cv2
import torch


CLASS_COLOR_MAP = np.random.randint(0, 255, (100, 3))

mean = [103.939, 116.779, 123.68]

from detr_tf import bbox

def boxes_output(image, bbox_list, labels=None, scores=None, class_name=[], config=None):
    """ Numpy function used to display the bbox (target or prediction)
    """
    assert(image.dtype == np.float32 and image.dtype == np.float32 and len(image.shape) == 3)

    if config is not None and config.normalized_method == "torch_resnet":
        channel_avg = np.array([0.485, 0.456, 0.406])
        channel_std = np.array([0.229, 0.224, 0.225])
        image = (image * channel_std) + channel_avg
        image = (image*255).astype(np.uint8)
    elif config is not None and config.normalized_method == "tf_resnet":
        image = image + mean
        image = image[..., ::-1]
        image = image  / 255
        
    bbox_xcycwh = bbox.np_rescale_bbox_xcycwh(bbox_list, (image.shape[0], image.shape[1])) 
    print(image.shape, bbox_list.shape)
    bbox_x1y1x2y2 = bbox.np_xcycwh_to_xy_min_xy_max(bbox_xcycwh)

    if scores is not None:
        keep, count = nms(bbox_x1y1x2y2, scores)
        if count!=0:
            bbox_x1y1x2y2 = bbox_x1y1x2y2[keep]

    # Set the labels if not defined
    if labels is None: labels = np.zeros((bbox_x1y1x2y2.shape[0]))

    return bbox_x1y1x2y2

def nms(boxes, scores, overlap=0.4, top_k=100):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """

    scores = np.array(scores)
    scores = torch.tensor(scores)
    #boxes = boxes.numpy()
    boxes = torch.tensor(boxes)
    print(scores.numpy().shape)
    keep = scores.new(scores.size(0)).zero_().long()
    #keep = tf.fill(scores.shape, 0.0)
    #keep = tf.fill(tf.shape(Y), 0.0)
    #print(tf.size(boxes).numpy())
    #if tf.size(boxes).numpy() == 0:
    count = 0
    if scores.numpy().shape[0] == 0:
        return keep, count
    if boxes.numel() ==0:
        return keep, count
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    #area = (x2-x1)*(y2-y1)
    v, idx = scores.sort(0)  # sort in ascending order
    #idx = np.argsort(scores)
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    # keep = torch.Tensor()
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w * h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter / union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    return keep.numpy(), count

def numpy_bbox_to_image(image, bbox_list, labels=None, scores=None, class_name=[], config=None):
    """ Numpy function used to display the bbox (target or prediction)
    """
    assert(image.dtype == np.float32 and image.dtype == np.float32 and len(image.shape) == 3)

    if config is not None and config.normalized_method == "torch_resnet":
        channel_avg = np.array([0.485, 0.456, 0.406])
        channel_std = np.array([0.229, 0.224, 0.225])
        image = (image * channel_std) + channel_avg
        image = (image*255).astype(np.uint8)
    elif config is not None and config.normalized_method == "tf_resnet":
        image = image + mean
        image = image[..., ::-1]
        image = image  / 255
        
    bbox_xcycwh = bbox.np_rescale_bbox_xcycwh(bbox_list, (image.shape[0], image.shape[1])) 
    #print(image.shape, bbox_list.shape)
    bbox_x1y1x2y2 = bbox.np_xcycwh_to_xy_min_xy_max(bbox_xcycwh)

    if scores is not None:
        keep, count = nms(bbox_x1y1x2y2, scores)
        if count!=0:
            bbox_x1y1x2y2 = bbox_x1y1x2y2[keep]

    # Set the labels if not defined
    if labels is None: labels = np.zeros((bbox_x1y1x2y2.shape[0]))

    bbox_area = []
    # Go through each bbox
    for b in range(0, bbox_x1y1x2y2.shape[0]):
        x1, y1, x2, y2 = bbox_x1y1x2y2[b]
        bbox_area.append((x2-x1)*(y2-y1))

    # Go through each bbox
    for b in np.argsort(bbox_area)[::-1]:
        # Take a new color at reandon for this instance
        instance_color = np.random.randint(0, 255, (3))
        

        x1, y1, x2, y2 = bbox_x1y1x2y2[b]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(image.shape[1], x2), min(image.shape[0], y2)

        # Select the class associated with this bbox
        class_id = labels[int(b)]

        if scores is not None and len(scores) > 0:
            label_name = class_name[int(class_id)]   
            label_name = "%s:%.2f" % (label_name, scores[b])
        else:
            label_name = class_name[int(class_id)]    

        class_color = CLASS_COLOR_MAP[int(class_id)]
    
        color = instance_color
        
        multiplier = image.shape[0] / 500
        #cv2.rectangle(image, (x1, y1), (x1 + int(multiplier*15)*len(label_name), y1 + 20), class_color.tolist(), -10)
        #cv2.putText(image, label_name, (x1+2, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * multiplier, (0, 0, 0), 1)
        cv2.rectangle(image, (x1, y1), (x2, y2), tuple(class_color.tolist()), 2)

    return image


def get_model_inference(m_outputs: dict, background_class, bbox_format="xy_center"):

    predicted_bbox = m_outputs["pred_boxes"][0]
    predicted_labels = m_outputs["pred_logits"][0]

    softmax = tf.nn.softmax(predicted_labels)
    predicted_scores = tf.reduce_max(softmax, axis=-1)
    predicted_labels = tf.argmax(softmax, axis=-1)


    indices = tf.where(predicted_labels != background_class)
    indices = tf.squeeze(indices, axis=-1)

    predicted_scores = tf.gather(predicted_scores, indices)
    predicted_labels = tf.gather(predicted_labels, indices)
    predicted_bbox = tf.gather(predicted_bbox, indices)


    if bbox_format == "xy_center":
        predicted_bbox = predicted_bbox
    elif bbox_format == "xyxy":
        predicted_bbox = bbox.xcycwh_to_xy_min_xy_max(predicted_bbox)
    elif bbox_format == "yxyx":
        predicted_bbox = bbox.xcycwh_to_yx_min_yx_max(predicted_bbox)
    else:
        raise NotImplementedError()

    return predicted_bbox, predicted_labels, predicted_scores
