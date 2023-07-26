# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
RVT PyTorch3D Renderer
"""
from math import ceil, floor
from typing import Optional, Tuple

import torch




# source: https://discuss.pytorch.org/t/batched-index-select/9115/6
def batched_index_select(inp, dim, index):
    """
    input: B x * x ... x *
    dim: 0 < scalar
    index: B x M
    """
    views = [inp.shape[0]] + [1 if i != dim else -1 for i in range(1, len(inp.shape))]
    expanse = list(inp.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(inp, dim, index)


def select_feat_from_hm(
    pt_cam: torch.Tensor, hm: torch.Tensor, pt_cam_wei: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor]:
    """
    :param pt_cam:
        continuous location of point coordinates from where value needs to be
        selected. it is of size [nc, npt, 2], locations in pytorch3d screen
        notations
    :param hm: size [nc, nw, h, w]
    :param pt_cam_wei:
        some predifined weight of size [nc, npt], it is used along with the
        distance weights
    :return:
        tuple with the first element being the wighted average for each point
        according to the hm values. the size is [nc, npt, nw]. the second and
        third elements are intermediate values to be used while chaching
    """
    nc, nw, h, w = hm.shape
    npt = pt_cam.shape[1]
    if pt_cam_wei is None:
        pt_cam_wei = torch.ones([nc, npt]).to(hm.device)

    # giving points outside the image zero weight
    pt_cam_wei[pt_cam[:, :, 0] < 0] = 0
    pt_cam_wei[pt_cam[:, :, 1] < 0] = 0
    pt_cam_wei[pt_cam[:, :, 0] > (w - 1)] = 0
    pt_cam_wei[pt_cam[:, :, 1] > (h - 1)] = 0

    pt_cam = pt_cam.unsqueeze(2).repeat([1, 1, 4, 1])
    # later used for calculating weight
    pt_cam_con = pt_cam.detach().clone()

    # getting discrete grid location of pts in the camera image space
    pt_cam[:, :, 0, 0] = torch.floor(pt_cam[:, :, 0, 0])
    pt_cam[:, :, 0, 1] = torch.floor(pt_cam[:, :, 0, 1])
    pt_cam[:, :, 1, 0] = torch.floor(pt_cam[:, :, 1, 0])
    pt_cam[:, :, 1, 1] = torch.ceil(pt_cam[:, :, 1, 1])
    pt_cam[:, :, 2, 0] = torch.ceil(pt_cam[:, :, 2, 0])
    pt_cam[:, :, 2, 1] = torch.floor(pt_cam[:, :, 2, 1])
    pt_cam[:, :, 3, 0] = torch.ceil(pt_cam[:, :, 3, 0])
    pt_cam[:, :, 3, 1] = torch.ceil(pt_cam[:, :, 3, 1])
    pt_cam = pt_cam.long()  # [nc, npt, 4, 2]
    # since we are taking modulo, points at the edge, i,e at h or w will be
    # mapped to 0. this will make their distance from the continous location
    # large and hence they won't matter. therefore we don't need an explicit
    # step to remove such points
    pt_cam[:, :, :, 0] = torch.fmod(pt_cam[:, :, :, 0], int(w))
    pt_cam[:, :, :, 1] = torch.fmod(pt_cam[:, :, :, 1], int(h))
    pt_cam[pt_cam < 0] = 0

    # getting normalized weight for each discrete location for pt
    # weight based on distance of point from the discrete location
    # [nc, npt, 4]
    pt_cam_dis = 1 / (torch.sqrt(torch.sum((pt_cam_con - pt_cam) ** 2, dim=-1)) + 1e-10)
    pt_cam_wei = pt_cam_wei.unsqueeze(-1) * pt_cam_dis
    _pt_cam_wei = torch.sum(pt_cam_wei, dim=-1, keepdim=True)
    _pt_cam_wei[_pt_cam_wei == 0.0] = 1
    # cached pt_cam_wei in select_feat_from_hm_cache
    pt_cam_wei = pt_cam_wei / _pt_cam_wei  # [nc, npt, 4]

    # transforming indices from 2D to 1D to use pytorch gather
    hm = hm.permute(0, 2, 3, 1).view(nc, h * w, nw)  # [nc, h * w, nw]
    pt_cam = pt_cam.view(nc, 4 * npt, 2)  # [nc, 4 * npt, 2]
    # cached pt_cam in select_feat_from_hm_cache
    pt_cam = (pt_cam[:, :, 1] * w) + pt_cam[:, :, 0]  # [nc, 4 * npt]
    # [nc, 4 * npt, nw]
    pt_cam_val = batched_index_select(hm, dim=1, index=pt_cam)
    # tranforming back each discrete location of point
    pt_cam_val = pt_cam_val.view(nc, npt, 4, nw)
    # summing weighted contribution of each discrete location of a point
    # [nc, npt, nw]
    pt_cam_val = torch.sum(pt_cam_val * pt_cam_wei.unsqueeze(-1), dim=2)
    return pt_cam_val, pt_cam, pt_cam_wei


def select_feat_from_hm_cache(
    pt_cam: torch.Tensor,
    hm: torch.Tensor,
    pt_cam_wei: torch.Tensor,
) -> torch.Tensor:
    """
    Cached version of select_feat_from_hm where we feed in directly the
    intermediate value of pt_cam and pt_cam_wei. Look into the original
    function to get the meaning of these values and return type. It could be
    used while inference if the location of the points remain the same.
    """

    nc, nw, h, w = hm.shape
    # transforming indices from 2D to 1D to use pytorch gather
    hm = hm.permute(0, 2, 3, 1).view(nc, h * w, nw)  # [nc, h * w, nw]
    # [nc, 4 * npt, nw]
    pt_cam_val = batched_index_select(hm, dim=1, index=pt_cam)
    # tranforming back each discrete location of point
    pt_cam_val = pt_cam_val.view(nc, -1, 4, nw)
    # summing weighted contribution of each discrete location of a point
    # [nc, npt, nw]
    pt_cam_val = torch.sum(pt_cam_val * pt_cam_wei.unsqueeze(-1), dim=2)
    return pt_cam_val




class BoxRenderer:
    """
    Can be used to render point clouds with fixed cameras and dynamic cameras.
    Code flow: Given the arguments we create a fixed set of cameras around the
    object. We can then render camera images (__call__()), project 3d
    points on the camera images (get_pt_loc_on_img()), project heatmap
    featues onto 3d points (get_feat_frm_hm_cube()) and get 3d point with the
    max heatmap value (get_max_3d_frm_hm_cube()).

    For each of the functions, we can either choose to use the fixed cameras
    which were defined by the argument or create dynamic cameras. The dynamic
    cameras are created by passing the dyn_cam_info argument (explained in
    _get_dyn_cam()).

    For the fixed camera, we optimize the code for projection of heatmap and
    max 3d calculation by caching the values in self._pts, self._fix_pts_cam
    and self._fix_pts_cam_wei.
    """

    def __init__(
        self,
        device,
        img_size,
        radius=0.012,
        points_per_pixel=5,
        compositor="norm",
        with_depth=False,
    ):
        """Rendering images form point clouds

        :param device:
        :param img_size:
        :param radius:
        :param points_per_pixel:
        :param compositor:
        """
        self.device = device
        self.img_size = img_size
        self.radius = radius
        self.points_per_pixel = points_per_pixel
        self.compositor = compositor
        self.with_depth = with_depth

        self.init()

    def init(self):
        h, w = self.img_size
        assert h == w



        # creates the fixed camera and renderer
        self._fix_cam = None
        self._fix_ren = None

        self.num_img = self.num_fix_cam

        self._pts = None
        self._fix_pts_cam = None
        self._fix_pts_cam_wei = None


    @torch.no_grad()
    def get_pt_loc_on_img(self, pt, fix_cam=True, dyn_cam_info=None):
        """
        returns the location of a point on the image of the cameras
        :param pt: torch.Tensor of shape (bs, np, 3)
        :param fix_cam: same as __call__
        :param dyn_cam_info: same as __call__
        :returns: the location of the pt on the image. this is different from the
            camera screen coordinate system in pytorch3d. the difference is that
            pytorch3d camera screen projects the point to [0, 0] to [H, W]; while the
            index on the img is from [0, 0] to [H-1, W-1]. We verified that
            the to transform from pytorch3d camera screen point to img we have to
            subtract (1/H, 1/W) from the pytorch3d camera screen coordinate.
        :return type: torch.Tensor of shape (bs, np, self.num_img, 2)
        """
        assert len(pt.shape) == 3
        assert pt.shape[-1] == 3
        bs, np = pt.shape[0:2]
        assert (dyn_cam_info is None) or (
            isinstance(dyn_cam_info, (list, tuple))
            and isinstance(dyn_cam_info[0], tuple)
        ), dyn_cam_info

        pt_img = []
        if fix_cam:
            fix_cam = self._get_fix_cam()
            # (num_cam, bs * np, 2)
            pt_scr = fix_cam.transform_points_screen(
                pt.view(-1, 3), image_size=self.img_size
            )[..., 0:2]
            if len(fix_cam) == 1:
                pt_scr = pt_scr.unsqueeze(0)

            pt_scr = torch.transpose(pt_scr, 0, 1)
            # transform from camera screen to image index
            h, w = self.img_size
            # (bs * np, num_img, 2)
            fix_pt_img = pt_scr - torch.tensor((1 / w, 1 / h)).to(pt_scr.device)
            fix_pt_img = fix_pt_img.view(bs, np, len(fix_cam), 2)
            pt_img.append(fix_pt_img)

        if not dyn_cam_info is None:
            assert pt.shape[0] == len(dyn_cam_info)
            dyn_pt_img = []
            for _pt, _dyn_cam_info in zip(pt, dyn_cam_info):
                dyn_cam = self._get_dyn_cam(_dyn_cam_info)
                # (num_cam, np, 2)
                _pt_scr = dyn_cam.transform_points_screen(
                    _pt, image_size=self.img_size
                )[..., 0:2]
                if len(dyn_cam) == 1:
                    _pt_scr = _pt_scr.unsqueeze(0)

                _pt_scr = torch.transpose(_pt_scr, 0, 1)
                # transform from camera screen to image index
                h, w = self.img_size
                # (np, num_img, 2)
                _dyn_pt_img = _pt_scr - torch.tensor((1 / w, 1 / h)).to(_pt_scr.device)
                dyn_pt_img.append(_dyn_pt_img.unsqueeze(0))

            # (bs, np, num_img, 2)
            dyn_pt_img = torch.cat(dyn_pt_img, 0)
            pt_img.append(dyn_pt_img)

        pt_img = torch.cat(pt_img, 2)

        return pt_img

    @torch.no_grad()
    def get_feat_frm_hm_cube(self, hm, fix_cam=True, dyn_cam_info=None):
        """
        :param hm: torch.Tensor of (1, num_img, h, w)
        :return: tupe of ((num_img, h^3, 1), (h^3, 3))
        """
        x, nc, h, w = hm.shape
        assert x == 1
        assert (dyn_cam_info is None) or (
            isinstance(dyn_cam_info, (list, tuple))
            and isinstance(dyn_cam_info[0], tuple)
        ), dyn_cam_info
        num_dyn_img = 0
        if not dyn_cam_info is None:
            num_dyn_img = dyn_cam_info[0][0].shape[0]
        assert nc == self.num_img
        assert self.img_size == (h, w)

        if self._pts is None:
            res = self.img_size[0]
            pts = torch.linspace(-1 + (1 / res), 1 - (1 / res), res).to(hm.device)
            pts = torch.cartesian_prod(pts, pts, pts)
            self._pts = pts

        pts_hm = []
        if fix_cam:
            if self._fix_pts_cam is None:
                # (np, nc, 2)
                pts_img = self.get_pt_loc_on_img(self._pts.unsqueeze(0)).squeeze(0)
                # (nc, np, bs)
                fix_pts_hm, pts_cam, pts_cam_wei = select_feat_from_hm(
                    pts_img.transpose(0, 1), hm.transpose(0, 1)[0 : self.num_fix_cam]
                )
                self._fix_pts_cam = pts_cam
                self._fix_pts_cam_wei = pts_cam_wei
            else:
                pts_cam = self._fix_pts_cam
                pts_cam_wei = self._fix_pts_cam_wei
                fix_pts_hm = select_feat_from_hm_cache(
                    pts_cam, hm.transpose(0, 1)[0 : self.num_fix_cam], pts_cam_wei
                )
            pts_hm.append(fix_pts_hm)

        if not dyn_cam_info is None:
            pts_img = self.get_pt_loc_on_img(
                self._pts.unsqueeze(0),
                fix_cam=False,
                dyn_cam_info=dyn_cam_info,
            ).squeeze(0)
            dyn_pts_hm, _, _ = select_feat_from_hm(
                pts_img.transpose(0, 1), hm.transpose(0, 1)[self.num_fix_cam :]
            )
            pts_hm.append(dyn_pts_hm)

        pts_hm = torch.cat(pts_hm, 0)
        return pts_hm, self._pts

    @torch.no_grad()
    def get_max_3d_frm_hm_cube(self, hm, fix_cam=True, dyn_cam_info=None):
        """
        given set of heat maps, return the 3d location of the point with the
            largest score, assumes the points are in a cube [-1, 1]. This function
            should be used  along with the render. For standalone version look for
            the other function with same name in the file.
        :param hm: (1, nc, h, w)
        :param fix_cam:
        :param dyn_cam_info:
        :return: (1, 3)
        """
        x, nc, h, w = hm.shape
        assert x == 1
        assert (dyn_cam_info is None) or (
            isinstance(dyn_cam_info, (list, tuple))
            and isinstance(dyn_cam_info[0], tuple)
        ), dyn_cam_info
        num_dyn_img = 0
        if not dyn_cam_info is None:
            num_dyn_img = dyn_cam_info[0][0].shape[0]
            assert len(dyn_cam_info) == 1
        assert nc == self.num_img
        assert self.img_size == (h, w)

        pts_hm, pts = self.get_feat_frm_hm_cube(hm, fix_cam, dyn_cam_info)
        # (bs, np, nc)
        pts_hm = pts_hm.permute(2, 1, 0)
        # (bs, np)
        pts_hm = torch.mean(pts_hm, -1)
        # (bs)
        ind_max_pts = torch.argmax(pts_hm, -1)
        return pts[ind_max_pts]

    def free_mem(self):
        """
        Could be used for freeing up the memory once a batch of testing is done
        """
        del self._pts
        del self._fix_pts_cam
        del self._fix_pts_cam_wei
        self._pts = None
        self._fix_pts_cam = None
        self._fix_pts_cam_wei = None
        with torch.cuda.device(self.device):
            torch.cuda.empty_cache()
