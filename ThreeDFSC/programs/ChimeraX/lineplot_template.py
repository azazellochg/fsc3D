# -----------------------------------------------------------------------------
# Make a matplotlib plot of density values along a ray from center of a map.
# Use the current view direction and update plot as view changes.
#
# This script registers command "fscplot" which takes one argument, the density
# map for which the plot is made.  For example,
#
#    fscplot #1
#
# (Ported from UCSF Chimera to UCSF ChimeraX with minimal changes.)
# Original template content: /mnt/data/lineplot_template.py
# -----------------------------------------------------------------------------

def ray_values(v, direction):
    d = v.data
    center = [0.5*(s+1) for s in d.size]
    radius = 0.5*min([s*t for s,t in zip(d.size, d.step)])
    steps = max(d.size)
    from numpy import array, arange, float32, outer
    # direction should be a 3-vector
    dn = (direction[0]**2 + direction[1]**2 + direction[2]**2) ** 0.5
    if dn == 0:
        dn = 1.0
    dir = array(direction, dtype=float32) / dn
    radii = arange(0, steps, dtype=float32) * (radius/steps)
    ray_points = outer(radii, dir)
    values = v.interpolated_values(ray_points)
    return radii, values, radius

# -----------------------------------------------------------------------------
#
def plot(x, y, xlabel, ylabel, title, fig=None):
    import matplotlib.pyplot as plt
    global_x = #==global_x==#
    global_y = #==global_y==#
    if fig is None:
        fig = plt.figure()
        fig.plot = ax = fig.add_subplot(1,1,1)
    else:
        ax = fig.plot
        ax.clear()
    plt.subplots_adjust(top=0.85)
    ax.plot(x, y, color='tab:blue', linewidth=2.0)
    ax.plot(global_x, global_y, 'tab:orange', linewidth=1.0)  # Plot global FSC
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(ymin=-0.2, ymax=1.01)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8, color='gray')
    ax.set_axisbelow(True)
    # Show / update window
    fig.canvas.manager.show()
    fig.canvas.draw_idle()
    return fig

# -----------------------------------------------------------------------------
#
def _view_direction_in_map_coords(session, fsc_map):
    """
    Return the current camera view direction expressed in the map's local coordinates.
    We try the common ChimeraX camera APIs, with a safe fallback.
    """
    # camera view direction in scene coords (pointing "into" the screen)
    cam = session.main_view.camera
    vd = None
    for attr in ('view_direction', 'viewDirection'):
        if hasattr(cam, attr):
            try:
                vd = getattr(cam, attr)()
                break
            except TypeError:
                vd = getattr(cam, attr)
                break
    if vd is None:
        # Fallback: in ChimeraX, camera looks down -Z in its own coords.
        vd = (0.0, 0.0, -1.0)

    # Convert scene -> model coords if possible
    try:
        # fsc_map.position is a Place (scene transform of the model)
        # Place multiplication usually supports tuples/vectors.
        vd_map = fsc_map.position.inverse() * vd
        return (vd_map[0], vd_map[1], vd_map[2])
    except Exception:
        return (vd[0], vd[1], vd[2])

# -----------------------------------------------------------------------------
#
def update_plot(session, fsc_map, fig=None):
    direction = _view_direction_in_map_coords(session, fsc_map)
    preradii, values, radius = ray_values(fsc_map, direction)

    radii = []
    apix = #==apix==#
    resolution_list = []
    for i in range(len(preradii)):
        radii.append(preradii[i]/(radius*2*apix))
    for i in range(len(values)):
        if values[i] < 0.143:
            resolution_list.append(1/radii[i-1])
            break
    resolution = resolution_list[0] if resolution_list else float('inf')

    title = ('3D FSC Plot.\n'
             'Z directional resolution (out-of-plane in blue) is %.2f.\n'
             'Global resolution (in orange) is %.2f.'
             % (resolution, #==global_res==#))
    fig = plot(radii, values, xlabel='Spatial Resolution',
               ylabel='Correlation', title=title, fig=fig)
    color_map(session, resolution)
    return fig

# -----------------------------------------------------------------------------
#
def _rgba_cmd_from_float01(r, g, b, a=1.0):
    # clip
    r = 0.0 if r < 0 else 1.0 if r > 1 else r
    g = 0.0 if g < 0 else 1.0 if g > 1 else g
    b = 0.0 if b < 0 else 1.0 if b > 1 else b
    a = 0.0 if a < 0 else 1.0 if a > 1 else a
    # convert to 0-255 range for rgba() format
    R, G, B, A = (int(round(255*x)) for x in (r, g, b, a))
    return f"rgba({R},{G},{B},{A})"

def color_map(session, resolution):
    from chimerax.core.commands import run
    maxres = #==maxres==#
    minres = #==minres==#
    a = (resolution-maxres)/(minres-maxres)
    r, g, b = 1-a, 0.0, a
    run(session, f"color #2 {_rgba_cmd_from_float01(r,g,b,a)}")

# -----------------------------------------------------------------------------
#
def fsc_plot(session, fsc_map):
    fig = update_plot(session, fsc_map)

    # Update whenever a new frame is drawn (captures view rotations).
    # ChimeraX well-known triggers include "frame drawn".  The trigger data is
    # an UpdateLoop instance (not needed here).
    state = {'last_cam_pos': None}

    def motion_cb(trigger_name, update_loop):
        # Throttle updates: only redraw if camera pose changed
        cam_pos = getattr(session.main_view.camera, 'position', None)
        if cam_pos is not None:
            if state['last_cam_pos'] is not None and cam_pos == state['last_cam_pos']:
                return
            state['last_cam_pos'] = cam_pos
        update_plot(session, fsc_map, fig)

    h = session.triggers.add_handler('frame drawn', motion_cb)
    # Store handler on model to keep it alive and allow user to remove if desired.
    fsc_map._fscplot_handler = h

# -----------------------------------------------------------------------------
# Command registration (ChimeraX)
#
def fscplot_cmd(session, fscMap):
    # Accept either a single map or a list of maps (MapsArg)
    if isinstance(fscMap, (list, tuple)):
        if len(fscMap) == 0:
            return
        fscMap = fscMap[0]
    fsc_plot(session, fscMap)

def register_command(session):
    from chimerax.core.commands import CmdDesc, register
    # MapsArg is used in official recipes, and works for volume models.
    from chimerax.map import MapsArg
    desc = CmdDesc(required=[('fscMap', MapsArg)])
    register('fscplot', desc, fscplot_cmd, logger=session.logger)

register_command(session)
