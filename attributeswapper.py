from architectures import BigRegressor, BigGenerator
import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw
from torch.nn import functional as F

def r(x):
    return int(round(x))

def recalc_keypoints(keypoints, box, image_size):
    keypoints = np.array(keypoints.copy(), dtype=np.float)
    size = box[3] - box[1]
    keypoints[:, 0] -= box[0]
    keypoints[:, 1] -= box[1]
    keypoints /= size
    keypoints *= image_size
    return np.array(np.round(keypoints), dtype=np.int)

def expand(pts, k=1):
    pts = np.array(pts.copy(), dtype=np.float)
    center = pts.mean(axis=0)
    pts -= center
    pts *= k
    pts += center
    return np.array(np.round(pts), dtype=np.int)

def face_polygon(keypoints, e=1):
    keypoints = np.array([(x, y) for x, y in keypoints[0:17]] +\
                         [(keypoints[i][0], keypoints[i][1]) for i in [26, 25, 24, 23, 22]] +\
                         [(keypoints[i][0], keypoints[i][1]) for i in [21, 20, 19, 18, 17]])
    return [(x, y) for x, y in expand(keypoints, e)]

def face_polygon_left(keypoints, e=1):
    keypoints = np.array([(x, y) for x, y in keypoints[0:9]] +\
                         [(keypoints[i][0], keypoints[i][1]) for i in [57, 66, 62, 51, 33, 30, 29, 28, 27]] +\
                         [(keypoints[i][0], keypoints[i][1]) for i in [21, 20, 19, 18, 17]])
    return [(x, y) for x, y in expand(keypoints, e)]

def face_polygon_right(keypoints, e=1):
    keypoints = np.array([(x, y) for x, y in keypoints[8:17]] +\
                         [(keypoints[i][0], keypoints[i][1]) for i in [26, 25, 24, 23, 22]] +\
                         [(keypoints[i][0], keypoints[i][1]) for i in [27, 28, 29, 30, 33, 51, 62, 66, 57]])
    return [(x, y) for x, y in expand(keypoints, e)]

def left_eye_polygon(keypoints):
    return [(x, y) for x, y in expand(keypoints[36:41], 1.5)]

def right_eye_polygon(keypoints):
    return [(x, y) for x, y in expand(keypoints[42:47], 1.5)]

def left_eyebrow_polygon(keypoints):
    return [(x, y) for x, y in expand(keypoints[17:21], 2)]

def right_eyebrow_polygon(keypoints):
    return [(x, y) for x, y in expand(keypoints[22:26], 2)]

def up_mouth(keypoints):
    keypoints = np.array([(x, y) for x, y in keypoints[48:54]] + \
                [(keypoints[i][0], keypoints[i][1]) for i in [64, 63, 62, 61, 60]])
    return [(x, y) for x, y in expand(keypoints, 1)]

def down_mouth(keypoints):
    keypoints = np.array([(x, y) for x, y in keypoints[54:59]] + \
                [(keypoints[48][0], keypoints[48][1])] + \
                [(keypoints[60][0], keypoints[60][1])] + \
                [(keypoints[i][0], keypoints[i][1]) for i in [67, 66, 65, 64]])
    return [(x, y) for x, y in expand(keypoints, 1)]

def nose_l_part(keypoints):
    keypoints = np.array([(x, y) for x, y in keypoints[27:30]] + \
                [(keypoints[33][0], keypoints[33][1])] + \
                [(keypoints[32][0], keypoints[32][1])] + \
                [(keypoints[31][0], keypoints[31][1])])
    return [(x, y) for x, y in expand(keypoints, 1)]

def nose_r_part(keypoints):
    keypoints = np.array([(x, y) for x, y in keypoints[27:30]] + \
                [(keypoints[33][0], keypoints[33][1])] + \
                [(keypoints[34][0], keypoints[34][1])] + \
                [(keypoints[35][0], keypoints[35][1])])
    return [(x, y) for x, y in expand(keypoints, 1)]                         

def mask_face(image, pts, box, k=1):
    mask = Image.fromarray(np.zeros_like(image))
    draw = ImageDraw.Draw(mask)
    facepoly = face_polygon(pts)
    facepoly = [(a, b) for a, b in expand(facepoly,  k)]
    draw.polygon(facepoly, fill ="#ffffff") 
    mask = np.asarray(mask)[:, :, 0] > 127
    mask = mask.reshape(image.shape[0], image.shape[1], 1)
    return mask, image * mask + (127, 127, 127) * (1 - mask)

def cut(image, face, target_size, padding):
    w = face[2] - face[0] + 1
    h = face[3] - face[1] + 1
    w *= (1 + padding)
    h *= (1 + padding)
    w = int(round(w))
    h = int(round(h))
    cx = (face[0] + face[2]) // 2
    cy = (face[1] + face[3]) // 2
    s = max(w, h)
    cx -= min(cx - s//2, 0)
    cx += min(image.shape[1] - (cx + (s+1)//2), 0)
    cy -= min(cy - s//2, 0)
    cy += min(image.shape[0] - (cy + (s+1)//2), 0)
    s += 2 * min(min(cx - s//2, 0), min(image.shape[1] - (cx + (s+1)//2), 0),
                  min(cy - s//2, 0), min(image.shape[0] - (cy + (s+1)//2), 0))

    face = [cx - s//2, cy - s//2, cx + (s+1)//2, cy + (s+1)//2]
    assert face[0] >= 0 and face[1] >= 0 and face[2] <= image.shape[1] and face[3] <= image.shape[0]

    cut_face = image[face[1]:face[3], face[0]:face[2]]

    return cv2.resize(cut_face, (target_size, target_size)), face

def norm(image, mean=127.5, std=127.5):
    return (image.float() - mean) / std

def denorm(image, mean=127.5, std=127.5):
    return (image * std + mean).int()

def blurred_face_border(image, keypoints, box, s=0.8, e=0.97, st=17*2):
    pts = keypoints
    mask = Image.fromarray(np.zeros_like(image))
    draw = ImageDraw.Draw(mask)
    for f in np.linspace(e, s, st):
        c = min(max(int((1-(f-s)/(e-s))*255), 0), 255)
        draw.polygon(face_polygon(pts, e=f), fill=c)
        draw.polygon(face_polygon_left(pts, e=f), fill=c)
        draw.polygon(face_polygon_right(pts, e=f), fill=c)
    mask = np.asarray(mask)[:, :, :1]
    return mask.astype(np.float) / 255

class AttributeSwapper():
    def __init__(self,
                 encoder_path,
                 decoder_path,
                 generator_path,
                 encoder_params={
                     'num_images': 3,
                     'image_size': 64,
                     'n_outputs': 512,
                     'ch': 512
                },
                decoder_params={
                    'num_images': 3,
                    'image_size': 64,
                    'num_classes': 512,
                    'latent_dim_mul': 0,
                    'ch': 512
                },
                generator_params={
                    'num_images': 3,
                    'image_size': 128,
                    'num_classes': 512,
                    'ch': 512
                },
                device='cuda:0',
                ae_image_size=(64, 64),
                image_size=128,
                latent_dim=120,
                padding=0,
                noice_std=1,
                use_seamless_clone=True,
                use_gan = True
                ):

        self.noice_std = noice_std
        self.use_seamless_clone = use_seamless_clone
        self.use_gan = use_gan
        self.padding = padding
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.ae_image_size = ae_image_size
        self.device = device
        self.encoder = BigRegressor(**encoder_params).to(self.device).eval()
        self.decoder = BigGenerator(**decoder_params).to(self.device).eval()
        self.generator = BigGenerator(**generator_params).to(self.device).eval()
        self.encoder.load_state_dict(torch.load(encoder_path))
        self.decoder.load_state_dict(torch.load(decoder_path))
        self.generator.load_state_dict(torch.load(generator_path))
        self.gen_noice()

    def gen_noice(self):
        self.noice = torch.randn(1, self.latent_dim).to(self.device) * self.noice_std

    def get_image(self, img, keypoints):
        image = np.copy(img)
        box = (r(np.min(keypoints[:, 0])), r(np.min(keypoints[:, 1])), r(np.max(keypoints[:, 0])), r(np.max(keypoints[:, 1])))
        img, cut_face = cut(img, box, target_size=self.image_size, padding=self.padding)
        keypoints = recalc_keypoints(keypoints, box, self.image_size)
        mask = blurred_face_border(img, keypoints, box)
        _, img = mask_face(img, keypoints, box)
        with torch.no_grad():
            img_torch = torch.tensor(img).reshape(1, img.shape[0], img.shape[1], img.shape[2]).permute(0, 3, 1, 2).to(self.device)
            img_torch = norm(img_torch)
            original_size = img_torch.shape
            hidden = self.encoder(F.interpolate(img_torch, self.ae_image_size, mode='bilinear'))
            if self.use_gan:
                swapped = self.generator(x=self.noice, y=hidden)
            else:
                swapped = F.interpolate(self.decoder(y=hidden), self.image_size, mode='bilinear')
            swapped = denorm(F.interpolate(swapped, (original_size[2], original_size[3]), mode='bilinear'))
            swapped = swapped[0].permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)

            if self.use_seamless_clone:
                swapped_image_cut = image[int(cut_face[1]):int(cut_face[3]), int(cut_face[0]):int(cut_face[2])]
                swapped_image_cut = cv2.resize(swapped_image_cut, (swapped.shape[0], swapped.shape[1]))

                swapped = np.array(cv2.seamlessClone(swapped.astype(np.uint8),\
                                                     swapped_image_cut.astype(np.uint8),\
                                                     (np.ones_like(swapped) * mask * 255).astype(np.uint8),
                                                     (swapped.shape[0]//2, swapped.shape[1]//2),\
                                                     cv2.NORMAL_CLONE), dtype=np.uint8)

            swapped = cv2.resize(swapped, (int(cut_face[2]) - int(cut_face[0]), int(cut_face[3]) - int(cut_face[1])), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (int(cut_face[2]) - int(cut_face[0]), int(cut_face[3]) - int(cut_face[1])), interpolation=cv2.INTER_NEAREST)
            mask = mask.reshape(mask.shape[0], mask.shape[1], 1)

            swapped = swapped * mask +\
                      image[int(cut_face[1]):int(cut_face[3]), int(cut_face[0]):int(cut_face[2])] * (1 - mask)

            image[int(cut_face[1]):int(cut_face[3]), int(cut_face[0]):int(cut_face[2])] = swapped

        return image.astype(np.uint8)
