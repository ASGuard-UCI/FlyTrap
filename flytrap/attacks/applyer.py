import numpy as np
import torch
import torch.nn.functional as F

from ..builder import APPLYER


@APPLYER.register_module()
class PatchApplyer:
    """Apply patch to the image"""

    def __init__(self, rotate_mean=0, rotate_var=np.pi / 9, fix_orien=False, distribution='normal') -> None:
        """
        Apply the patch to the umbrella areas
        Args:
            rotate_mean (float): mean of the rotation angle
            rotate_var (float): variance of the rotation angle
            fix_orien (bool): whether to fix the orientation, set to True when testing
            distribution (str): distribution of the rotation angle, 'normal' or 'uniform'
        """
        self.rotate_mean = rotate_mean
        self.rotate_var = rotate_var
        self.fix_orien = fix_orien
        self.distribution = distribution

    def compute_transformation_matrix(self, src_coords, dst_coords):
        """Compute the affine transformation matrix.
        Args:
            src_coords (torch.Tensor): [4, 2] source coordinates
            dst_coords (torch.Tensor): [4, 2] target coordinates

        Returns:
            M (torch.Tensor): [2, 3] transformation matrix
        """
        device = src_coords.device
        A = []
        B = []
        for i in range(4):
            x, y = src_coords[i]
            u, v = dst_coords[i]
            A.append([x, y, 1, 0, 0, 0])
            A.append([0, 0, 0, x, y, 1])
            B.append([u])
            B.append([v])

        A = torch.tensor(A, dtype=torch.float32, device=device)  # [8, 6]
        B = torch.tensor(B, dtype=torch.float32, device=device)  # [8, 1]
        M = torch.lstsq(B, A).solution[:6]
        return M.reshape(2, 3)

    def _rotate(self, device):
        """Rotate the square patch

        Return:
            src_coords (torch.Tensor): [4, 2] rotated source coordinates
        """
        # constant orientation - gaussian orientation
        if not self.fix_orien:
            if self.distribution == 'normal':
                theta = np.random.normal(self.rotate_mean, self.rotate_var)
            elif self.distribution == 'uniform':
                theta = self.rotate_mean + np.random.uniform(-self.rotate_var, self.rotate_var)
            else:
                raise ValueError(f'Unknown distribution: {self.distribution}')
        else:
            Warning('fix_orien is deprecated, please use test_mode instead.')
            theta = self.rotate_mean
        radius = np.sqrt(2)
        src_coords = torch.tensor([
            [radius * np.cos(theta), radius * np.sin(theta)],
            [radius * np.cos(theta + np.pi / 2), radius * np.sin(theta + np.pi / 2)],
            [radius * np.cos(theta + np.pi), radius * np.sin(theta + np.pi)],
            [radius * np.cos(theta + 3 * np.pi / 2), radius * np.sin(theta + 3 * np.pi / 2)],
        ], dtype=torch.float32)

        return src_coords.to(device), theta

    # pylint: disable=unused-argument
    def __call__(self, img, patch, coords=None, test_mode=False):
        """Apply patch to target area, support batch processing
           Used for real-world digital experiments

        Args:
            img (torch.Tensor): [B, H, W, C] candidate image
            coords (torch.Tensor): [B, N, 2] coordinates of chessboard corners
            patch (torch.Tensor): [C, H, W] patch to be applied
            test_mode (bool): whether in test mode, if True, fix the orientation
        Returns:
            img (torch.Tensor): [B, C, H, W] image with patch
        """
        img = img.permute(0, 3, 1, 2)
        B, C, H, W = img.shape
        device = img.device
        transformed_images = []

        thetas = []
        for i in range(B):
            # random rotate patch orientation
            if test_mode:
                radius = np.sqrt(2)
                theta = self.rotate_mean
                src_coords = torch.tensor([
                    [radius * np.cos(theta), radius * np.sin(theta)],
                    [radius * np.cos(theta + np.pi / 2), radius * np.sin(theta + np.pi / 2)],
                    [radius * np.cos(theta + np.pi), radius * np.sin(theta + np.pi)],
                    [radius * np.cos(theta + 3 * np.pi / 2), radius * np.sin(theta + 3 * np.pi / 2)],
                ], dtype=torch.float32).to(device)
                oren_theta = 0
            else:
                src_coords, oren_theta = self._rotate(device)


            # compute the transformation matrix
            M = self.compute_transformation_matrix(coords[i], src_coords)
            theta = torch.cat([M, torch.tensor([[0, 0, 1]], dtype=torch.float32, device=device)], dim=0).unsqueeze(0)
            grid_transformed = F.affine_grid(theta[:, :2, :], [1, C, H, W], align_corners=False)

            # apply the patch to the image
            patch_transformed = F.grid_sample(patch.unsqueeze(0), grid_transformed, align_corners=False)
            patched_img = img[i].unsqueeze(0)
            mask = patch_transformed.sum(dim=1, keepdim=True) > 1e-3
            mask = mask.repeat(1, C, 1, 1)
            patched_img[mask] = patch_transformed[mask]

            transformed_images.append(patched_img.squeeze())
            thetas.append(oren_theta)

        new_img = torch.stack(transformed_images)
        return new_img.permute(0, 2, 3, 1), thetas
