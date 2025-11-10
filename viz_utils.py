import nibabel
import copy
import time
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from os.path import join
import mne
from preprocessing_utils import select_channels
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colormaps as cmaps
import imageio
import os
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.axes3d import Axes3D
from scipy.spatial.distance import cdist

def searchlight(loc, name, distance_threshold = 5, n_sub=2, n_repeat = 1, metrics = 'euclidean', weight_axis = None):
    """
    Function to launch a searchlight procedure. The idea is to supress channels that are not close enough to a channel from another subject. It helps generalize results.

    Parameters:
    ----------
    loc: ndarray
        Positions in 3D space (x,y,z), of shape (3, n_channels)
    name: ndarray
        Name of the channels, this needs to be in the format 'sujectID-nameofchannels' as to retrieve subjectID
    distance threshold : float
        Acceptable distance between to channel to consider them "close". Generally in millimeters but can change depending on data. 5-10mm is generally acceptable.
    n_sub: int
        Number of subjects within the same neighboorhood to consider. This number is the number of neighboors from different subjects than the considered channel i.e. if you want only 2 subjects in total, this number is equal to 1
    n_repeat: int
        Number of iterations. Since some channels might disappear, it is needed to reevalute the size of clusters. This is especially relevant for high n_sub.
    metrics: str
        The distance metric to use. If a string, the distance function can be ‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’, ‘correlation’, ‘cosine’, ‘dice’, ‘euclidean’, ‘hamming’, ‘jaccard’, ‘jensenshannon’, ‘kulczynski1’, ‘mahalanobis’, ‘matching’, ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘yule’.
    weight_axis: ndarray
        How to weight the different axes, when using metrics that use that argument (minkowski typically). None otherwise

    Returns:
    channels_keep: list
        List of kept channels by the end of the procedure.
    -------

    """
    temp_loc, temp_name = loc.copy(), name.copy()
    channels_ignore = []
    for i_repeat in range(n_repeat):
        dist_matrix = cdist(temp_loc.T,temp_loc.T, metrics, w = weight_axis)
        for channel_index in range(dist_matrix.shape[0]):
            subject_channel = temp_name[channel_index].split('-')[0]
            subject_neighbour_list = []
            for neighbour_index in range(dist_matrix.shape[0]):
                distance_neighbour = dist_matrix[channel_index,neighbour_index]
                subject_neighbour = temp_name[neighbour_index].split('-')[0]
                if subject_neighbour != subject_channel and not subject_neighbour in subject_neighbour_list and distance_neighbour < distance_threshold:
                    subject_neighbour_list.append(subject_neighbour)
            if len(subject_neighbour_list) < n_sub:
                channels_ignore.append(channel_index)
                temp_loc[:,channel_index] = [np.inf, np.inf, np.inf]
    channels_keep = [not i in (channels_ignore) for i in np.arange(loc.shape[1])]
    return channels_keep

def mask_distance(positions, vertices, max_dist):
    """
    Helper function to mask channels according to their distance to the surface

    Parameters:
    ----------
    positions: ndarray
        Positions in 3D space (x,y,z), of shape (3, n_channels)
    vertices: ndarray
        Positions of the vertices defined by nibabel.freesurfer.io.read_geometry
    max_dist : float
        maximum distance from surface to consider
    """
    mask = []
    for index_position, position in enumerate(positions.T):
        dist = np.sqrt(np.sum((position-vertices)**2, axis=1))
        min_dist = dist[np.argmin(dist)]
        if min_dist <= max_dist:
            mask.append(index_position)
    return mask

def place_surface(positions, vertices, dist_proj):
    """
    Helper function to stick channel coordinates to nearby surface

    Parameters:
    ----------
    positions: ndarray
        Positions in 3D space (x,y,z), of shape (3, n_channels)
    vertices: ndarray
        Positions of the vertices defined by nibabel.freesurfer.io.read_geometry
    dist_proj : float
        distance from which a point is projected onto the cortical surface
    """
    for index_position, position in enumerate(positions.T):
        dist = np.sqrt(np.sum((position-vertices)**2, axis=1))
        idx = np.argmin(dist)
        if dist[idx] <= dist_proj:
            closest_position = vertices[idx,:]
            positions[:,index_position] = closest_position
    return positions

def color_array(scores, color_type = 'cool'):
    """
    Helper function to create a colormap

    Parameters:
    ----------
    scores: ndarray
        Can be scores or categories: values to base colors on, of shape (n_channels,)
    color_type: str
        valid name for a matplotlib colormap
    """
    color_map = dict()
    for i, s in enumerate(scores):
        color_map[i] = cmaps[color_type](s/np.max(scores))
    return color_map


def brainscatter_static(scores, locations, channels,
                        min_scores = -1, dist_max = 5, dist_proj = 0, size_points = 10,
                        cmap = 'rainbow', alpha = 0.3, cortex_color=['w','grey'], background='white',
                        left_viewpoint = False, right_viewpoint = False, hems = ['lh','rh'],
                        surf_path = 'D:/DataSEEG_Sorciere/BIDS/freesurfer', path_save = 'D:/BrainPlot/temp/brain', 
                        close = True):
    """
    Function to plot points on a brain.
    
    Parameters:
    ----------
    scores: ndarray
        Can be scores or categories: values to base colors on, of shape (n_channels,)
    locations: ndarray
        Positions in 3D space (x,y,z), of shape (3, n_channels)
    channels: ndarray
        Names of channels, of shape (n_channels,)
    min_scores: float
        Minimal value of scores to be considered a relevant contact
    dist_max: float
        Maximum distance from surface to consider
    dist_proj: float
        Distance at which a point is projected on the surface.
        Default: 0, there are no projections
    size_points: float
        Size of points to plot
    cmap: string
        matplotlib colormap to use for plotting
    hems: list
        List of hemispheres to plot, can contain
            - 'lh': left hemisphere
            - 'rh': right hemisphere
            - 'all': whole brain
    surf_path: string
        Path to the freesurfer folder containing the necessary information to plot the 3D brain surface
    path_save: string
        Path to save snapshot of the brain according to the initial viewpoints
    close: bool
        Whether to close the interactive window after plotting the results. If False, you will be able to play around with the window
    
    """
    
    if not left_viewpoint:
        left_viewpoint = [(-280, 240, 20),
                    (-35, -30, 0),
                    (0.5, 1, 2)]
    if not right_viewpoint:
        right_viewpoint = [(280, 240, 20),
                        (35, -30, 0),
                        (-0.5, 1, 2)]
    
    scores[np.where(scores<min_scores)] = min_scores
    clim = [scores[scores > min_scores].min(), scores[scores > min_scores].max()]
    locs = locations.copy()
    s = scores.copy()

    #retrieve coordinates of surface vertices
    vertices = [None, None]
    vertices[0], _ = nibabel.freesurfer.read_geometry(join(surf_path + '/fsaverage/surf/','lh.pial'))
    vertices[1], _ = nibabel.freesurfer.read_geometry(join(surf_path + '/fsaverage/surf/','rh.pial'))
    vertices = np.concatenate(vertices, 0)

    # mask on maximum distance
    mask = mask_distance(locs, vertices, dist_max)
    s = s[mask]
    chans = channels[mask]
    locs = locs[:,mask]

    # projection on surface
    locs = place_surface(locs, vertices, dist_proj)

    #mask on treshold
    locs_empty = locs[:,s <= min_scores] #points without scalar
    locs = locs[:,s > min_scores] #points with scalar
    s = s[s > min_scores]



    for hem in hems:
        if hem == 'lh':
            hem_locs_empty = locs_empty[:,locs_empty[0,:]<0]
            index_interest = list(np.arange(locs.shape[1])[locs[0,:] < 0])
            hem_s = s[np.newaxis,index_interest]
            hem_chan = chans[index_interest]
            hem_locs = locs[:,index_interest]
            #hem_s, hem_chan, hem_locs = select_channels(s[np.newaxis, :], chans, locs, 
            #                                                        channel_select = ["'"], strict = True, exclude = False)
        elif hem == 'rh':
            hem_locs_empty = locs_empty[:,locs_empty[0,:]>0]
            #hem_s, hem_chan, hem_locs = select_channels(s[np.newaxis, :], chans, locs, 
            #                                                        channel_select = ["'"], strict = True, exclude = True)
            index_interest = list(np.arange(locs.shape[1])[locs[0,:] > 0])
            hem_s = s[np.newaxis,index_interest]
            hem_chan = chans[index_interest]
            hem_locs = locs[:,index_interest]

        elif hem == 'all':
            hem_locs_empty = locs_empty
            index_interest = list(np.arange(locs.shape[1]))
            hem_s = s[np.newaxis,index_interest]
            hem_chan = chans[index_interest]
            hem_locs = locs[:,index_interest]
        #plot brain
        if dist_proj == 0:
            proj = 'trans'
        else:
            proj = 'proj'

        if hem != 'all':
            brain = mne.viz.Brain(subjects_dir=surf_path, subject = 'fsaverage',
                                surf='pial', #'pial', 'inflated', 'white',...
                                hemi=hem,
                                background= background,
                                alpha = alpha,
                                cortex=cortex_color, #classic
                                offscreen=True,
                                theme=None)
        else:
            brain = mne.viz.Brain(subjects_dir=surf_path, subject = 'fsaverage',
                            surf='pial', #'pial', 'inflated', 'white',...
                            background= background,
                            alpha = alpha,
                            cortex=cortex_color, #classic
                            offscreen=True,
                            theme=None)

        #labels = ("ctx-lh-inferiortemporal", "ctx-lh-middletemporal", "ctx-lh-superiortemporal", "ctx-lh-temporalpole", "ctx-lh-insula", "ctx-lh-transversetemporal")
        #brain.add_volume_labels(aseg="aparc+aseg", labels=labels, alpha = 0.4)
        if hem_s.shape[0] > 0:
            brain.plotter.add_points(hem_locs.T, render_points_as_spheres=True, point_size = size_points, scalars = hem_s[0], cmap = cmap)  #lighting = False
            brain.plotter.update_scalar_bar_range(clim)

        if hem_locs_empty.shape[1] > 0:
            brain.plotter.add_points(hem_locs_empty.T, render_points_as_spheres=True, point_size = 3, color = 'black')

        if hem == 'lh':
            brain.plotter.camera_position = left_viewpoint

        elif hem == 'rh':
            brain.plotter.camera_position = right_viewpoint


        brain.save_image(path_save + hem + '.png')
        if close:
            brain.plotter.close()

    left_str = 'lh'
    right_str =  'rh'

    left_h = mpimg.imread(path_save + left_str + '.png')
    right_h = mpimg.imread(path_save + right_str + '.png')

    # Créer une figure et des sous-graphiques
    fig, axes = plt.subplots(1, 2, figsize=(20, 15))

    # Afficher la première image dans le premier sous-graphique
    axes[0].imshow(right_h)
    axes[0].set_title(right_str)

    # Afficher la deuxième image dans le deuxième sous-graphique
    axes[1].imshow(left_h)
    axes[1].set_title(left_str)

    fig.show()
    #plt.title(out_names[index])
    fig.savefig(path_save + '.png')

    return fig,axes



def create_matshow_gif(data, timesteps, output_file='matshow_animation.gif', 
                       labels = None, regressor_name = '',
                       cmap = 'viridis', duration = 1, loop = 5,
                       vmin = None, vmax = None,
                       ticks = False, title = False):
    """
    Create a GIF from a matshow changing over the specified number of timesteps.

    Parameters:
    data (ndarray): A 3D numpy array of shape (timesteps, rows, columns).
    timesteps (int): The number of timesteps to animate.
    output_file (str): The name of the output GIF file.
    """
    # Ensure the output directory exists
    os.makedirs('frames', exist_ok=True)
    
    if labels is None:
        labels = np.arange(data.shape[1])

    # Generate and save each frame
    for t in range(timesteps):
        plt.matshow(data[t], cmap=cmap, vmin = vmin, vmax = vmax)
        if ticks:
            plt.xticks(np.arange(len(labels)),labels,rotation = 90)
            plt.yticks(np.arange(len(labels)),labels)
        if not title:
            plt.title(regressor_name + ' ' + f'Timestep {t + 1}')
        else:
            plt.title(title[t])
        plt.colorbar()
        plt.savefig(f'frames/frame_{t:03d}.png')
        plt.close()
    
    # Read the saved frames and compile them into a GIF
    with imageio.get_writer(output_file, mode='I', duration=duration,loop=loop) as writer:
        for t in range(timesteps):
            image = imageio.imread(f'frames/frame_{t:03d}.png')
            writer.append_data(image)
    
    # Clean up the frames directory
    for t in range(timesteps):
        os.remove(f'frames/frame_{t:03d}.png')
    os.rmdir('frames')
    
    print(f'GIF saved as {output_file}')



def create_collection_gif(data, output_file='matshow_animation.gif', 
                    duration = 1, loop = 5, write_image = False):
    """
    Create a GIF from a collections of prebuilt images.

    Parameters:
    data (list): list of figures, they will be the frames of the gif
    timesteps (int): The number of timesteps to animate.
    output_file (str): The name of the output GIF file.
    """
    # Ensure the output directory exists
    os.makedirs('frames', exist_ok=True)
    

    # Generate and save each frame
    for t,fig in enumerate(data):
        if write_image:
            fig.write_image(file=f'frames/frame_{t:03d}.png', format='.png')
        else:
            fig.savefig(f'frames/frame_{t:03d}.png')
    
    # Read the saved frames and compile them into a GIF
    with imageio.get_writer(output_file, mode='I', duration=duration,loop=loop) as writer:
        for t in range(len(data)):
            image = imageio.imread(f'frames/frame_{t:03d}.png')
            writer.append_data(image)
    
    # Clean up the frames directory
    for t in range(len(data)):
        os.remove(f'frames/frame_{t:03d}.png')
    os.rmdir('frames')
    
    print(f'GIF saved as {output_file}')


class Arrow3D(FancyArrowPatch):

    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)
        
    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs) 
    
def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    '''Add an 3d arrow to an `Axes3D` instance.'''

    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)


setattr(Axes3D, 'arrow3D', _arrow3D)


def brainscatter_rotation(scores, locations, channels,
                        min_scores = -1, dist_max = 5, dist_proj = 0,
                        cmap = 'rainbow', 
                        left_viewpoint = False, right_viewpoint = False, hems = ['lh','rh'],
                        surf_path = 'D:/DataSEEG_Sorciere/BIDS/freesurfer', path_save = 'D:/BrainPlot/temp/brain', 
                        close = True, n_angles = 50):
    
    if not left_viewpoint:
        left_viewpoint = [(-400, 400, 20),
                    (-15, 15, 0),
                    (0, 0, 5)]


    if not right_viewpoint:
        right_viewpoint = [(400, 400, 20),
                    (15, 15, 0),
                    (0, 0, 5)]
        
    modulus = np.sqrt(left_viewpoint[0][0]**2 + left_viewpoint[0][1]**2)
    left_viewpoint_list = [left_viewpoint]
    right_viewpoint_list = [right_viewpoint]
    for i in range(n_angles):
        new_left_viewpoint = [(left_viewpoint_list[0][0][0] * np.cos(2*np.pi/n_angles*i), left_viewpoint_list[0][0][1] * -np.sin(2*np.pi/n_angles*i), left_viewpoint_list[0][0][2]), 
                        (left_viewpoint_list[0][1][0] * np.cos(2*np.pi/n_angles*i), left_viewpoint_list[0][1][1] * -np.sin(2*np.pi/n_angles*i), left_viewpoint_list[0][1][2]), 
                        left_viewpoint_list[0][2]]
        left_viewpoint_list.append(new_left_viewpoint)

        new_right_viewpoint = [(right_viewpoint_list[0][0][0] * np.cos(2*np.pi/n_angles*i), right_viewpoint_list[0][0][1] * -np.sin(2*np.pi/n_angles*i), right_viewpoint_list[0][0][2]), 
                        (right_viewpoint_list[0][1][0] * np.cos(2*np.pi/n_angles*i), right_viewpoint_list[0][1][1] * -np.sin(2*np.pi/n_angles*i), right_viewpoint_list[0][1][2]), 
                        right_viewpoint_list[0][2]]
        right_viewpoint_list.append(new_right_viewpoint)
    
    scores[np.where(scores<min_scores)] = min_scores
    clim = [scores[scores > min_scores].min(), scores[scores > min_scores].max()]
    locs = locations.copy()
    s = scores.copy()

    #retrieve coordinates of surface vertices
    vertices = [None, None]
    vertices[0], _ = nibabel.freesurfer.read_geometry(join(surf_path + '/fsaverage/surf/','lh.pial'))
    vertices[1], _ = nibabel.freesurfer.read_geometry(join(surf_path + '/fsaverage/surf/','rh.pial'))
    vertices = np.concatenate(vertices, 0)

    # mask on maximum distance
    mask = mask_distance(locs, vertices, dist_max)
    s = s[mask]
    chans = channels[mask]
    locs = locs[:,mask]

    # projection on surface
    locs = place_surface(locs, vertices, dist_proj)

    #mask on treshold
    locs_empty = locs[:,s <= min_scores] #points without scalar
    locs = locs[:,s > min_scores] #points with scalar
    s = s[s > min_scores]

    img_list = []
    for view_index, left_viewpoint, right_viewpoint in zip(np.arange(len(left_viewpoint_list) - 1),left_viewpoint_list[1:], right_viewpoint_list[1:]):
        for hem in hems:
            if hem == 'lh':
                hem_locs_empty = locs_empty[:,locs_empty[0,:]<0]
                index_interest = list(np.arange(locs.shape[1])[locs[0,:] < 0])
                hem_s = s[np.newaxis,index_interest]
                hem_chan = chans[index_interest]
                hem_locs = locs[:,index_interest]
                #hem_s, hem_chan, hem_locs = select_channels(s[np.newaxis, :], chans, locs, 
                #                                                        channel_select = ["'"], strict = True, exclude = False)
            elif hem == 'rh':
                hem_locs_empty = locs_empty[:,locs_empty[0,:]>0]
                #hem_s, hem_chan, hem_locs = select_channels(s[np.newaxis, :], chans, locs, 
                #                                                        channel_select = ["'"], strict = True, exclude = True)
                index_interest = list(np.arange(locs.shape[1])[locs[0,:] > 0])
                hem_s = s[np.newaxis,index_interest]
                hem_chan = chans[index_interest]
                hem_locs = locs[:,index_interest]

            #plot brain
            if dist_proj == 0:
                alpha = 0.3
                proj = 'trans'
            else:
                alpha = .3
                proj = 'proj'

            brain = mne.viz.Brain(subjects_dir='D:/DataSEEG_Sorciere/BIDS/freesurfer', subject = 'fsaverage',
                                surf='pial', #'pial', 'inflated', 'white',...
                                hemi= hem,
                                background='white',
                                alpha = 0.3,
                                cortex='classic',
                                offscreen=True,
                                theme=None)

            
            if hem_s.shape[0] > 0:
                brain.plotter.add_points(hem_locs.T, render_points_as_spheres=True, point_size = 10, scalars = hem_s[0], cmap = cmap)  #lighting = False
                brain.plotter.update_scalar_bar_range(clim)

            if hem_locs_empty.shape[1] > 0:
                brain.plotter.add_points(hem_locs_empty.T, render_points_as_spheres=True, point_size = 3, color = 'black')

            if hem == 'lh':
                brain.plotter.camera_position = left_viewpoint

            elif hem == 'rh':
                brain.plotter.camera_position = right_viewpoint

            brain.save_image(path_save + hem + str(view_index) +  '.png')
            brain.plotter.close()

        fig, ax = plt.subplots(1,2,figsize=(10,5))
        left_h = mpimg.imread(path_save + 'lh' + str(view_index) + '.png')
        right_h = mpimg.imread(path_save + 'rh' + str(view_index) + '.png')
        ax[1].imshow(left_h)
        ax[0].imshow(right_h)

        ax[0].spines[['top','bottom','right','left']].set_visible(False)
        ax[0].set_xticks([])
        ax[0].set_xticklabels([])
        ax[0].set_yticks([])
        ax[0].set_yticklabels([])

        ax[1].spines[['top','bottom','right','left']].set_visible(False)
        ax[1].set_xticks([])
        ax[1].set_xticklabels([])
        ax[1].set_yticks([])
        ax[1].set_yticklabels([])
        img_list.append(fig)
        plt.close()

    return img_list