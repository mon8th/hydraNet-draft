import torch


class FCOSTargetGenerator:
    """
    generates targets for FCOS object detection model
    - assigns ground truth boxes to feature map locations
    - computes regression (l, t, r, b) targets
    - center sampling
    """

    def __init__(
        self, object_sizes_of_interest, center_sample=True, radius=1.5
    ):  # 1.5 is default in original FCOS
        self.object_sizes_of_interest = object_sizes_of_interest
        self.center_sample = center_sample
        self.radius = radius

    def __call__(self, locations, targets, fpn_strides):
        batch_size = len(targets)
        labels_all = []
        reg_targets_all = []

        for i in range(batch_size):
            gt_boxes = targets[i]["boxes"]
            gt_labels = targets[i]["labels"]

            labels_per_image = []
            reg_targets_per_image = []

            for level, locations_per_level in enumerate(locations):
                size_range = self.object_sizes_of_interest[level]
                labels, reg_targets = self.compute_targets_single_level(
                    locations_per_level,
                    gt_boxes,
                    gt_labels,
                    size_range,
                    fpn_strides[level],
                )
                labels_per_image.append(labels)
                reg_targets_per_image.append(reg_targets)

            labels_all.append(torch.cat(labels_per_image, dim=0))
            reg_targets_all.append(torch.cat(reg_targets_per_image, dim=0))

        return torch.stack(labels_all), torch.stack(reg_targets_all)

    def compute_targets_single_level(
        self, locations, gt_boxes, gt_labels, size_range, stride
    ):
        """
        In FCOS, each feature-map location is assigned to one ground-truth box using min_area_inds (the smallest valid GT box covering that location).
        The class label is taken from that GT box, and the regression target (l,t,r,b) is computed from the same location to that GT box; unmatched locations are background.
        """
        num_locations = locations.shape[0]
        num_gt = gt_boxes.shape[0]

        if num_gt == 0:
            return torch.zeros((num_locations,), dtype=torch.int64), torch.zeros(
                num_locations, 4
            )

        xs, ys = (
            locations[:, 0],
            locations[:, 1],
        )  # xs is x-coordinates, ys is y-coordinates
        l = xs[:, None] - gt_boxes[None, :, 0]
        t = ys[:, None] - gt_boxes[None, :, 1]
        r = gt_boxes[None, :, 2] - xs[:, None]
        b = gt_boxes[None, :, 3] - ys[:, None]

        reg_targets = torch.stack([l, t, r, b], dim=2)

        in_boxes = (
            reg_targets.min(dim=2)[0] > 0
        )  # [0] is to get values from (values, indices)
        max_reg_targets_per_gt = reg_targets.max(dim=2)[
            0
        ]  # this gives the max of l,t,r,b for each location-gt pair
        in_size_range = (max_reg_targets_per_gt >= size_range[0]) & (
            max_reg_targets_per_gt <= size_range[1]
        )

        valid = in_boxes & in_size_range

        # smallest area
        areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
        areas = (
            areas[None, :].repeat(num_locations, 1).clone()
        )  # repeat for each location
        areas[~valid] = float("inf")

        min_area, min_area_inds = areas.min(dim=1)
        labels = gt_labels[min_area_inds]
        labels[min_area == float("inf")] = 0  # background

        reg_targets = reg_targets[torch.arange(num_locations), min_area_inds]
        reg_targets[min_area == float("inf")] = 0

        return labels, reg_targets

    def get_center_sample_region(self, gt_boxes, locations, stride, radius):
        cx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2
        cy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2

        center_radius = radius * stride
        x_min = cx - center_radius
        y_min = cy - center_radius
        x_max = cx + center_radius
        y_max = cy + center_radius

        xs, ys = locations[:, 0], locations[:, 1]
        l = xs[:, None] - x_min[None, :]
        t = ys[:, None] - y_min[None, :]
        r = x_max[None, :] - xs[:, None]
        b = y_max[None, :] - ys[:, None]

        center_region = torch.stack([l, t, r, b], dim=2)
        is_in_center = center_region.min(dim=2)[0] > 0

        return is_in_center