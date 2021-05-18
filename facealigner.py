import numpy as np
from skimage import transform as sktransform

class FaceAligner():
    @staticmethod
    def kpts_68_to_5(pts):
        return np.array([[pts[36:42, 0].mean(), pts[36:42, 1].mean()],
                         [pts[42:48, 0].mean(), pts[42:48, 1].mean()],
                         [pts[30, 0], pts[30, 1]],
                         [pts[48, 0], pts[48, 1]],
                         [pts[54, 0], pts[54, 1]]])
    
    @staticmethod
    def get_v(pts):
        V = np.copy(pts).T.reshape(2, -1)
        V = np.concatenate((V, np.ones((1, V.shape[1]))), 0)
        return V
                
    def fit(self, from_pts, to_pts):
        if from_pts.shape[0] == 68:
            from_pts = self.kpts_68_to_5(from_pts)
        if to_pts.shape[0] == 68:
            to_pts = self.kpts_68_to_5(to_pts)
        
        assert from_pts.shape == (5, 2) and to_pts.shape == (5, 2)
            
        transform = sktransform.SimilarityTransform()
        transform.estimate(from_pts, to_pts)
        self.transform_ = transform
        
        return self
    
    def transform(self, image, shape=None):
        if shape is None:
            shape = self.shape
        assert self.transform_ is not None
        return sktransform.warp(image, self.transform_.inverse, output_shape=shape, preserve_range=True).astype(image.dtype)
            
    def transform_pts(self, pts):
        V = self.get_v(pts)
        V = self.transform_.params.dot(V)
        V = V.T[:, :2]
        return V
    
    def inverse_transform(self, image, shape=None):
        if shape is None:
            shape = self.shape
        assert self.transform_ is not None
        return sktransform.warp(image, self.transform_, output_shape=shape, preserve_range=True).astype(image.dtype)
    
    def __init__(self, shape=None):
        self.shape = shape
        self.transform_ = None
