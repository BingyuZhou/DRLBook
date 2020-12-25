class PiecewiseScheduler:
    def __init__(self, keypoints, interpolation=None):
        """
        Piecewise scheduler with given keypoint [t, value]. The other queries are interpolated
        by specific interpolation method. The residule after the last keypoint will have constant value as the last keypoint.
        """
        indexes = [ptr[0] for ptr in keypoints]
        assert indexes == sorted(indexes)

        self.interpolation = (
            interpolation
            if interpolation
            else lambda start, end, alpha: start + (end - start) * alpha
        )

        self.keypoints = keypoints

    def value(self, t):
        for ptr_lhs, ptr_rhs in zip(self.keypoints[:-1], self.keypoints[1:]):
            if t >= ptr_lhs[0] and t <= ptr_rhs[0]:
                return self.interpolation(
                    ptr_lhs[1],
                    ptr_rhs[1],
                    float(t - ptr_lhs[0]) / (ptr_rhs[0] - ptr_lhs[0]),
                )

        return self.keypoints[-1][1]


class LinearScheduler:
    def __init__(self, schedule_timesteps, initial_p, final_p):
        """
    Linear interpolation between initial_p and final_p over scheduler timesteps.
    After these timesteps, final_p is used.
    
    :param schedule_timesteps: Number of timesteps for which probability linearly
    anneals from initial_p to final_p.
    :param initial_p: Initial probability
    :param final_p: Final probability
    """
        self.schedule_timesteps = schedule_timesteps
        self.initial_p = initial_p
        self.final_p = final_p

    def value(self, t):
        """ Get value(probability) at a certain timestep"""
        fraction = min(float(t) / self.schedule_timesteps, 1)
        return self.initial_p + fraction * (self.final_p - self.initial_p)

