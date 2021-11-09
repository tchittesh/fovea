import cv2
from os.path import join, basename
import numpy as np
from mmcv import imdenormalize, imwrite, imshow_bboxes


def overlay_grid(imgs):
    # overlays a red grid on an NxCxHxW batch of imgs
    spacing = 100
    imgs[0, 0, 2::spacing, :] = 255
    imgs[0, 0, 3::spacing, :] = 255
    imgs[0, 0, 4::spacing, :] = 255
    imgs[0, 0, 5::spacing, :] = 255
    imgs[0, 0, :, 2::spacing] = 255
    imgs[0, 0, :, 3::spacing] = 255
    imgs[0, 0, :, 4::spacing] = 255
    imgs[0, 0, :, 5::spacing] = 255
    imgs[0, 1:3, 2::spacing, :] = 0
    imgs[0, 1:3, 3::spacing, :] = 0
    imgs[0, 1:3, 4::spacing, :] = 0
    imgs[0, 1:3, 5::spacing, :] = 0
    imgs[0, 1:3, :, 2::spacing] = 0
    imgs[0, 1:3, :, 3::spacing] = 0
    imgs[0, 1:3, :, 4::spacing] = 0
    imgs[0, 1:3, :, 5::spacing] = 0
    return imgs


def vis_obj_fancy(img, bboxes, labels, scores=None,
                  score_th=0.5, out_file=None):
    thickness = 2
    alpha = 0.2
    color_palette = {
        0: (196, 48, 22),   # person
        1: (63, 199, 10),   # bicycle
        2: (29, 211, 224),  # car
        3: (163, 207, 52),  # motorcycle
        4: (40, 23, 227),   # bus
        5: (29, 91, 224),   # truck
        6: (235, 197, 9),   # traffic_light
        7: (144, 39, 196),  # fire_hydrant
        8: (196, 0, 0),     # stop_sign
    }

    bboxes = np.asarray(bboxes)
    labels = np.asarray(labels)

    empty = len(bboxes) == 0
    if not empty and scores is not None and score_th > 0:
        sel = scores >= score_th
        bboxes = bboxes[sel]
        labels = labels[sel]
        empty = len(bboxes) == 0

    if empty:
        if out_file is not None:
            imwrite(img, out_file)
        return img

    bboxes = bboxes.round().astype(np.int32)

    # draw filled rectangles
    img_filled = img.copy()
    for bbox, label in zip(bboxes, labels):
        color = color_palette[label]
        color_rgb = (color[2], color[1], color[0])
        cv2.rectangle(
            img,
            (bbox[0], bbox[1]), (bbox[2], bbox[3]),
            color_rgb, thickness=-1,
        )
    img = cv2.addWeighted(img_filled, (1 - alpha), img, alpha, 0)
    # draw box contours
    for bbox, label in zip(bboxes, labels):
        color = color_palette[label]
        color_rgb = (color[2], color[1], color[0])
        cv2.rectangle(
            img,
            (bbox[0], bbox[1]), (bbox[2], bbox[3]),
            color_rgb, thickness=thickness,
        )

    if out_file is not None:
        imwrite(img, out_file)
    return img


def vis_batched_imgs(out_dir, imgs, img_metas, bboxes=None, denorm=True,
                     index=None, labels=None):
    # BxCxHxW -> BxHxWxC
    vis_imgs = imgs.permute(0, 2, 3, 1)
    vis_imgs = np.ascontiguousarray(vis_imgs.cpu().detach().numpy())

    for b in range(len(vis_imgs)):
        if denorm:
            mean = np.array(img_metas[b]['img_norm_cfg']['mean'])
            std = np.array(img_metas[b]['img_norm_cfg']['std'])
            to_rgb = img_metas[b]['img_norm_cfg']['to_rgb']
            vis_imgs[b] = imdenormalize(
                vis_imgs[b], mean, std, to_bgr=to_rgb)
        # Note that the 'filename' field can sometimes be the full path
        if index is not None:
            out_path = join(
                out_dir,
                f"{index:09}_{b:03}_{basename(img_metas[b]['filename'])}"
            )
        else:
            out_path = join(out_dir, basename(img_metas[b]['filename']))
        if labels is not None and labels[b] is not None and bboxes is not None:
            bbox = bboxes[b][0] if len(bboxes[b].shape) == 3 else bboxes[b]
            vis_obj_fancy(vis_imgs[b], bbox.cpu().detach().numpy(),
                          labels=labels[b], out_file=out_path)
        elif bboxes is not None:
            bbox = bboxes[b][0] if len(bboxes[b].shape) == 3 else bboxes[b]
            imshow_bboxes(vis_imgs[b], bbox.cpu().detach().numpy(), show=False,
                          out_file=out_path)
        else:
            imwrite(vis_imgs[b], out_path)
