import io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
mpl.use('Agg')
import imageio
from PIL import Image


def plt_fig_to_np(fig):
    buf = io.BytesIO()  # Create an in-memory binary stream
    fig.savefig(buf, format='png')  # Save the figure to the stream
    buf.seek(0)  # Reset the stream position to the beginning
    img = Image.open(buf)  # Read the image from the stream
    img = np.array(img)
    plt.close()
    return img


def create_2d_frame(matrix, measures, timestep):
    global_vmin = np.min(matrix)
    global_vmax = np.max(matrix)

    ratio = matrix.shape[0] / matrix.shape[1]
    plt.figure(figsize=(10 / ratio + 1.5, 10))  # Or any size you want
    plt.imshow(matrix, cmap='viridis', interpolation='nearest', vmin=global_vmin, vmax=global_vmax, origin='lower')
    # Create the colorbar and make it smaller, vertically
    cax = plt.gca().inset_axes([1.05, 0, 0.05, 1])
    cb = plt.colorbar(cax=cax)
    cb.set_label('Objective') # Label the colorbar
    plt.xlabel(measures[0])  # Set the x-axis label
    plt.ylabel(measures[1])  # Set the y-axis label
    title = plt.title('Timestep {}'.format(timestep), fontsize=15, loc='center', y=1.05, pad=10)

    buf = io.BytesIO()  # Create an in-memory binary stream
    plt.savefig(buf, format='png')  # Save the figure to the stream
    buf.seek(0)  # Reset the stream position to the beginning
    img = Image.open(buf)  # Read the image from the stream
    plt.close()
    # img = img.resize((1000, 1000), Image.ANTIALIAS)
    img = np.array(img)
    # Save image
    # imageio.imwrite('test_image.png', img)  # NOTE: this works!
    
    return img


def create_3d_frame(matrix, measures, timestep, do_rotate_3d=False, rotate_angle=0):
    global_vmin = np.min(matrix)
    global_vmax = np.max(matrix)

    max_dim = max(matrix.shape[i])
    padded_matrix = np.pad(matrix, ((0, max_dim - matrix.shape[0]), 
                                    (0, max_dim - matrix.shape[1]), 
                                    (0, max_dim - matrix.shape[2])))
    original_matrix = matrix
    original_shape = matrix.shape
    matrix = padded_matrix

    # Get dimensions of the matrix
    dim_x, dim_y, dim_z = matrix.shape

    # Create figure and axes with the same aspect ratio
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_axes([0, 0, 1, 1], projection='3d')
    
    # Normalize the values to a colormap
    norm = cm.colors.Normalize(vmin=global_vmin, vmax=global_vmax)
    # colormap = cm.hot
    colormap = mpl.colormaps['viridis']
    colors = colormap(norm(matrix))
    colors[..., -1] = 0.5
    
    # Make voxels transparent if value is 0
    filled = matrix > 0

    # Plot the voxels with no edges
    ax.voxels(filled, facecolors=colors, edgecolors=None, linewidth=0.5)

    # Add a colorbar
    mappable = cm.ScalarMappable(norm=norm, cmap=colormap)
    mappable.set_array([])
    # Get the current axes for the colorbar
    # cax = plt.gca().inset_axes([1, 0.1, 0.03, 0.8])  # [x, y, width, height]
    cax = plt.gca().inset_axes([1.05, 0.1, 0.03, 0.8])  # Increase the x value to move to the right
    cb = plt.colorbar(mappable, cax=cax)
    cb.set_label('Value')

    # Set limits of the axes
    ax.set_xlim([0, dim_x])
    ax.set_ylim([0, dim_y])
    ax.set_zlim([0, dim_z])

    # Set the title
    title = plt.title('Timestep {}'.format(timestep), fontsize=20, loc='center', y=0.9, pad=20)
    # To shift the title horizontally, you can manually adjust its position
    title.set_position((0.57, 0.9)) # adjust 0.55 for horizontal positioning
    # Optionally, you can adjust the overall layout to ensure everything fits
    plt.subplots_adjust(top=0.85)  # adjust top to create space for title

    # Set the axis labels
    ax.set_xlabel(measures[0])
    ax.set_ylabel(measures[1])
    ax.set_zlabel(measures[2])

    # Set ticks to not extend beyond unpadded matrix data
    ax.set_xticks(range(original_shape[0]))
    ax.set_yticks(range(original_shape[1]))
    ax.set_zticks(range(original_shape[2]))

    # Set the rotation angle for this matrix
    if do_rotate_3d:
        ax.view_init(elev=20, azim=rotate_angle)
    else: 
        ax.view_init(elev=20, azim=rotate_angle)

    buf = io.BytesIO()  # Create an in-memory binary stream
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)  # Save the figure to the stream
    buf.seek(0)  # Reset the stream position to the beginning
    # img = imageio.imread(buf)  # Read the image from the stream # NOTE: this doesn't work!
    img = Image.open(buf)
    img = np.array(img)
    plt.close()
    return img


def create_heatmap_frame(matrix, measures, timestep, do_rotate_3d=False):
    if len(matrix.shape) == 2:
        return create_2d_frame(matrix, measures, timestep)
    elif len(matrix.shape) == 3:
        return create_3d_frame(matrix, measures, timestep, do_rotate_3d)
    else:
        raise ValueError('Matrix must be 2D or 3D.')


def plot_trial_data(trial_latent_means, trial_latent_logvars, trial_events, trial_num, process_num, ensemble_size=None):
    # Extract and reshape data for the given trial and process
    means = np.array([trial_latent_means[trial_num][i][process_num] for i in range(len(trial_latent_means[trial_num]))])
    logvars = np.array([trial_latent_logvars[trial_num][i][process_num] for i in range(len(trial_latent_logvars[trial_num]))])

    # Check if means and logvars are 2D arrays (time x latent_dim)
    if means.ndim != 2 or logvars.ndim != 2:
        raise ValueError("Data for means and logvars must be 2D (time x latent_dim).")

    # Assuming means and logvars are now lists of arrays, one per timestep
    latent_dim = means[0].shape[0]
    events = trial_events[trial_num][process_num]

    # Create a figure with two subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # Time steps
    timesteps = np.arange(len(means))

    # Determine line styles for ensemble
    line_styles = ['-','--',':','-.'] if ensemble_size else ['-']
    ensemble_group_size = max(1, latent_dim // ensemble_size) if ensemble_size else latent_dim

    # Plot means and logvars, storing the line objects
    latent_lines = []
    for dim in range(latent_dim):
        line_style = line_styles[dim // ensemble_group_size % len(line_styles)]
        line, = axs[0].plot(timesteps, [mean[dim] for mean in means], label=f'LD {dim+1}', linestyle=line_style)
        axs[1].plot(timesteps, [logvar[dim] for logvar in logvars], linestyle=line_style)
        latent_lines.append(line)

    # Plotting vertical lines for events and creating legend entries
    event_lines = []  # To store the lines for the events
    for timestep, event in events.items():
        line = axs[0].axvline(x=timestep, linestyle='-', color='k', label=f'{event} at {timestep}')
        axs[1].axvline(x=timestep, linestyle='-', color='k')
        event_lines.append((line, event))

    axs[0].set_title(f'Trial {trial_num+1} Mean Values')
    axs[1].set_title(f'Trial {trial_num+1} Logvar Values')
    axs[0].set_xlabel('Timesteps')
    axs[1].set_xlabel('Timesteps')
    axs[0].set_ylabel('Mean Value')
    axs[1].set_ylabel('Logvar Value')

    # Creating separate legends
    axs[0].legend(handles=latent_lines, loc='upper right')
    axs[1].legend([line for line, label in event_lines], [label for line, label in event_lines], loc='upper right')
    
    # Adjust layout
    plt.tight_layout()

    # Save the figure
    # plt.savefig('test.png')

    # Convert the plot to an image array and return it
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close()

    return image

