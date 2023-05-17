# https://github.com/kazuto1011/grad-cam-pytorch/blob/master/grad_cam.py
import torch
from torch.nn import functional as F

class _BaseWrapper(object):
    def __init__(self, model):
        super(_BaseWrapper, self).__init__()
        self.device = next(model.parameters()).device
        self.model = model
        self.handlers = []  # a set of hook function handlers

    def _encode_one_hot(self, ids):
        one_hot = torch.zeros_like(self.logits_vec).to(self.device)
        one_hot.scatter_(1, ids, 1.0)
        return one_hot

    def forward(self, image, get_prob=False):
        self.image_shape = image.shape[2:]
        self.logits = self.model(image, separate=True)

        self.logits_vec = gsp2d(self.logits, keepdims=True)[:, :, 0, 0]
        self.logits_vec = torch.sigmoid(self.logits_vec)

        if get_prob:
            return self.logits_vec
        else:
            return self.logits

    def backward(self, ids):
        """
        Class-specific backpropagation
        """
        self.model.zero_grad()
        self.logits_vec[:, ids].sum().backward(retain_graph=True)


    def generate(self):
        raise NotImplementedError

    def remove_hook(self):
        """
        Remove all the forward/backward hook functions
        """
        for handle in self.handlers:
            handle.remove()

class CAM(_BaseWrapper):
    """
    https://arxiv.org/pdf/1610.02391.pdf
    """

    def __init__(self, model, candidate_layers=None):
        super(CAM, self).__init__(model)
        self.fmap_pool = {}
        self.grad_pool = {}
        self.candidate_layers = candidate_layers  # list

        def save_fmaps(key):
            def forward_hook(module, input, output):
                self.fmap_pool[key] = output

            return forward_hook

        def save_grads(key):
            def backward_hook(module, grad_in, grad_out):
                self.grad_pool[key] = grad_out[0]

            return backward_hook

        # If any candidates are not specified, the hook is registered to all the layers.
        for name, module in self.model.named_modules():
            if self.candidate_layers is None or name in self.candidate_layers:
                self.handlers.append(module.register_forward_hook(save_fmaps(name)))
                self.handlers.append(module.register_backward_hook(save_grads(name)))

    def _find(self, pool, target_layer):
        if target_layer in pool.keys():
            return pool[target_layer]
        else:
            raise ValueError("Invalid layer name: {}".format(target_layer))

    def generate(self, target_layer, cls):
        with torch.no_grad():
            fmaps = self._find(self.fmap_pool, target_layer)
            weights = self.model.classifier.weight[cls]
            cam = torch.mul(fmaps, weights)
            cam = cam.sum(dim=1, keepdim=True)
            cam = F.relu(cam)

            return cam

class GeneratorCAM():
    def __init__(self, model, candidate_layers=None):
        self.model = model.eval()
        self.candidate_layers = candidate_layers

        self.cam = CAM(model, candidate_layers)

    def generate_all_cams(self, image, device, get_prob=False):
        img = image.clone().detach().to(device)
        n_classes = 20

        outputs = self.cam.forward(img, get_prob=get_prob)
        cam_all_classes = torch.zeros([n_classes, outputs.shape[2], outputs.shape[3]])

        with torch.no_grad():
            for i in range(n_classes):
                cam = self.cam.generate(target_layer=self.candidate_layers[0], cls=i)
                cam_all_classes[i] = cam[0, 0, :, :]

        return cam_all_classes


    def generate_cams_prob(self, image, device):
        img = image.clone().detach().to(device)
        n_classes = 20

        preds = self.cam.forward(img, get_prob=True)
        outputs = self.cam.forward(img, get_prob=False)
        cam_all_classes = torch.zeros([n_classes, outputs.shape[2], outputs.shape[3]])

        for i in range(n_classes):
            if preds[0, i] > 0.5:
                gcam = self.cam.generate(target_layer=self.candidate_layers[0], cls=i)
                cam_all_classes[i] = gcam[0, 0, :, :]

        return cam_all_classes, preds

    def get_cams_for_all(self, image, cls, device, get_prob=False, take_max=False):
        with torch.no_grad():
            cams_all_classes = self.generate_all_cams(image, device, get_prob=get_prob)
            class_cam = cams_all_classes[cls]

            other_classes = [i for i in range(20) if i != cls]
            other_cams = cams_all_classes[other_classes]
            if take_max:
                other_cams = torch.max(other_cams, dim=0)[0]
            else:
                other_cams = torch.relu(torch.sum(other_cams, dim=0))
            class_cam = class_cam / torch.max(class_cam)
            other_cams = other_cams / torch.max(other_cams)

        return class_cam, other_cams


    def get_cams_for_present(self, image, cls, other_cls, device, take_max=False):
        cams_all_classes = self.generate_all_cams(image, device)
        class_cam = cams_all_classes[cls]

        if other_cls == []:
            other_cams = torch.zeros_like(class_cam)
        else:
            if not take_max:
                other_cams = torch.relu(torch.sum(cams_all_classes[other_cls], dim=0))
            else:
                other_cams = torch.max(cams_all_classes[other_cls], dim=0)[0]
            other_cams = other_cams / torch.max(other_cams)
        class_cam = class_cam / torch.max(class_cam)
        return class_cam, other_cams


    def get_cams_for_predicted(self, image, cls, device, take_max=False):
        preds = self.cam.forward(image.to(device), get_prob=True)[0]
        other_cls = [i for i in range(20) if i != cls and preds[i] > 0.5]
        return self.get_cams_for_present(image, cls, other_cls, device, take_max=take_max)


    def get_cam(self, image, cls, device, get_prob=False):
        img = image.clone().detach().to(device)
        self.cam.forward(img, get_prob=get_prob)
        cam = self.cam.generate(target_layer=self.candidate_layers[0], cls=cls)
        return cam

    def get_cams(self, image, cls, device, pred_cams=True, take_max=True):
        if not pred_cams:
            return self.get_cams_for_all(image, cls, device=device, take_max=take_max)
        else:
            return self.get_cams_for_predicted(image, cls, device=device, take_max=take_max)


def gap2d_pos(x, keepdims=False):
    out = torch.sum(x.view(x.size(0), x.size(1), -1), -1) / (torch.sum(x>0)+1e-12)
    if keepdims:
        out = out.view(out.size(0), out.size(1), 1, 1)

    return out


def gsp2d(x, keepdims=False):
    out = torch.sum(x.view(x.size(0), x.size(1), -1), -1)
    if keepdims:
        out = out.view(out.size(0), out.size(1), 1, 1)

    return out


def gap2d(x, keepdims=False):
    out = torch.mean(x.view(x.size(0), x.size(1), -1), -1)
    if keepdims:
        out = out.view(out.size(0), out.size(1), 1, 1)
    return out

