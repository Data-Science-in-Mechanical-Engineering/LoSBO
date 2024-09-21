import torch

def n_sphere_to_cartesian(polar_coordinates, radius, center):
    '''
    function that transforms polar coordinates to cartesian coordinates
    Args:
        polar_coordinates: tensor of shape (no_points, 2) or (2)
        radius: radius of the sphere
        center: center of the sphere
    '''
    if len(polar_coordinates.shape) == 1:
        return center+radius*polar_coordinates.item()
    if len(polar_coordinates.shape) == 2:
        polar_coordinates = polar_coordinates.unsqueeze(1)
    angles = polar_coordinates[:,:, 1:].squeeze(1)*2*torch.pi
    r = polar_coordinates[:,:, 0]
    no_points = polar_coordinates.shape[0]    
    a = torch.concatenate((torch.ones(no_points,1)*2*torch.pi, angles),1)
    si = torch.sin(a)
    si[:,0] = 1
    sin_mask = torch.cumprod(si, -1)
    co = torch.cos(a)
    cos_mask = torch.roll(co, -1, 1)
    return r*radius*sin_mask*cos_mask+center



