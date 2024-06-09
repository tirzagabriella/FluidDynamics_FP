import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import colorchooser, messagebox


def get_barrier(shape, Nx, Ny):
    """ Define barrier shapes """
    X, Y = np.meshgrid(range(Nx), range(Ny))

    if shape == 'circle':
        barrier = (X - Nx/4)**2 + (Y - Ny/2)**2 < (Ny/4)**2
    elif shape == 'rectangle':
        barrier = (X > Nx/4) & (X < Nx/2) & (Y > Ny/4) & (Y < 3*Ny/4)
    elif shape == 'triangle':
        barrier = (X - Nx/4) + 2*(Y - Ny/2) < Ny/2
    elif shape == 'airplane':
        wing = (Y < Ny/2) & ((X - Nx/4)**2 + (Y - Ny/4)**2 < (Ny/8)**2)
        body = (X > Nx/4) & (X < 3*Nx/4) & (Y > Ny/4) & (Y < 3*Ny/4)
        barrier = wing | body
    elif shape == 'car':
        body = (X > Nx/4) & (X < 3*Nx/4) & (Y > Ny/4) & (Y < 3*Ny/4)
        wheel1 = ((X - Nx/4)**2 + (Y - Ny/4)**2 < (Ny/8)**2)
        wheel2 = ((X - Nx/4)**2 + (Y - 3*Ny/4)**2 < (Ny/8)**2)
        barrier = body | wheel1 | wheel2
    else:
        raise ValueError(f"Unknown shape: {shape}")

    return barrier


def run_simulation(shape, simulation_window, speed, fluid_colormap, barrier_color, fluid_type, lift_label, drag_label, stop_simulation):
    """ Lattice Boltzmann Simulation """
    global fig

    # Simulation parameters
    Nx = 400
    Ny = 100
    rho0 = 100
    Nt = 4000
    plotRealTime = True

    # Set tau based on fluid type
    fluid_types = {'water': 0.6, 'oil': 1.0, 'air': 0.2}
    tau = fluid_types.get(fluid_type, 0.6)

    if tau <= 0 or tau > 2:
        raise ValueError(
            "Relaxation time tau should be between 0 and 2 for stability.")

    # Lattice speeds / weights
    NL = 9
    idxs = np.arange(NL)
    cxs = np.array([0, 0, 1, 1, 1, 0, -1, -1, -1])
    cys = np.array([0, 1, 1, 0, -1, -1, -1, 0, 1])
    weights = np.array([4/9, 1/9, 1/36, 1/9, 1/36, 1 /
                       9, 1/36, 1/9, 1/36])  # sums to 1

    # Initial Conditions
    F = np.ones((Ny, Nx, NL)) * rho0 / NL
    np.random.seed(42)
    F += 0.01 * np.random.randn(Ny, Nx, NL)
    X, Y = np.meshgrid(range(Nx), range(Ny))
    F[:, :, 3] += 2 * (1 + 0.2 * np.cos(2 * np.pi * X / Nx * 4))
    rho = np.sum(F, 2)
    for i in idxs:
        F[:, :, i] *= rho0 / rho

    # Select barrier shape
    barrier = get_barrier(shape, Nx, Ny)

    # Convert hex color to RGB
    barrier_rgb = tuple(int(barrier_color[i:i+2], 16) / 255 for i in (1, 3, 5))

    # Prep figure
    fig = plt.figure(figsize=(4, 2), dpi=80)
    fig.canvas.manager.set_window_title('Lattice Simulation window figure')

    # Simulation Main Loop
    for it in range(Nt):
        if stop_simulation.get():
            break

        print(f"Step {it}/{Nt}")

        # Drift
        for i, cx, cy in zip(idxs, cxs, cys):
            F[:, :, i] = np.roll(F[:, :, i], cx, axis=1)
            F[:, :, i] = np.roll(F[:, :, i], cy, axis=0)

        # Set reflective boundaries
        bndryF = F[barrier, :]
        bndryF = bndryF[:, [0, 5, 6, 7, 8, 1, 2, 3, 4]]

        # Calculate fluid variables
        rho = np.sum(F, 2)
        rho[rho < np.finfo(float).eps] = np.finfo(
            float).eps  # Prevent density from going too low
        ux = np.sum(F * cxs, 2) / rho
        uy = np.sum(F * cys, 2) / rho

        # Debugging statements
        if np.any(np.isnan(rho)):
            print(f"NaN detected in rho at step {it}")
            rho[np.isnan(rho)] = rho0

        if np.any(np.isnan(ux)):
            print(f"NaN detected in ux at step {it}")
            ux[np.isnan(ux)] = 0  # Reset to zero velocity

        if np.any(np.isnan(uy)):
            print(f"NaN detected in uy at step {it}")
            uy[np.isnan(uy)] = 0  # Reset to zero velocity

        # Clamp velocities to prevent instability
        ux = np.clip(ux, -0.1, 0.1)
        uy = np.clip(uy, -0.1, 0.1)

        # Apply Collision
        Feq = np.zeros(F.shape)
        for i, cx, cy, w in zip(idxs, cxs, cys, weights):
            Feq[:, :, i] = rho * w * \
                (1 + 3 * (cx * ux + cy * uy) + 9 * (cx * ux + cy * uy)
                 ** 2 / 2 - 3 * (ux ** 2 + uy ** 2) / 2)

        F += -(1.0 / tau) * (F - Feq)

        # Ensure F stays within physical bounds
        F = np.clip(F, -1e5, 1e5)

        # Apply boundary
        F[barrier, :] = bndryF

        # plot in real time - color 1/2 particles blue, other half red
        if (plotRealTime and (it % 10) == 0) or (it == Nt - 1):
            plt.cla()
            ux[barrier] = 0
            uy[barrier] = 0
            vorticity = (np.roll(ux, -1, axis=0) - np.roll(ux, 1, axis=0)) - \
                        (np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1))
            vorticity[barrier] = np.nan
            vorticity = np.ma.array(vorticity, mask=barrier)
            plt.imshow(vorticity, cmap=fluid_colormap)
            plt.imshow(~barrier, cmap='gray', alpha=0.3)
            plt.clim(-.1, .1)
            ax = plt.gca()
            ax.invert_yaxis()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.set_aspect('equal')

            # Overlay the barrier with the chosen color
            barrier_overlay = np.zeros((Ny, Nx, 3))
            barrier_overlay[..., 0] = barrier_rgb[0] * barrier
            barrier_overlay[..., 1] = barrier_rgb[1] * barrier
            barrier_overlay[..., 2] = barrier_rgb[2] * barrier
            plt.imshow(barrier_overlay, alpha=0.7)

            plt.pause(0.1 / speed.get())

    if not stop_simulation.get():
        # Calculate lift and drag Forces
        lift_force = np.sum(2 * (F[:, :, 1] - F[:, :, 5]) * barrier)
        drag_force = np.sum(2 * (F[:, :, 3] - F[:, :, 7]) * barrier)

        # Update the labels with the calculated forces
        lift_label.config(text=f"Lift Force: {lift_force:.2f}")
        drag_label.config(text=f"Drag Force: {drag_force:.2f}")

        # Save figure
        plt.savefig('latticeboltzmann.png', dpi=240)
        plt.show()


def start_simulation(shape, speed, fluid_colormap, barrier_color, fluid_type):
    global simulation_window
    if simulation_window:
        simulation_window.destroy()

    root.withdraw()  # Hide the main GUI window

    # New Tk instance for the simulation window
    simulation_window = tk.Toplevel(root)
    simulation_window.title("Lattice Boltzmann Simulation")

    # Shared variable to signal the simulation to stop
    stop_simulation = tk.BooleanVar(value=False)

    # Labels to display lift and drag forces
    lift_label = tk.Label(simulation_window, text="Lift Force: Calculating...")
    lift_label.pack()
    drag_label = tk.Label(simulation_window, text="Drag Force: Calculating...")
    drag_label.pack()

    # Add Reset button to simulation window
    reset_button = tk.Button(simulation_window, text="Reset", command=lambda: reset_simulation(
        simulation_window, stop_simulation))
    reset_button.pack()

    # Run the simulation
    run_simulation(shape, simulation_window, speed, fluid_colormap,
                   barrier_color, fluid_type, lift_label, drag_label, stop_simulation)


def reset_simulation(simulation_window, stop_simulation):
    global fig  # Access the global fig variable
    stop_simulation.set(True)
    plt.close(fig)  # Close the Matplotlib figure
    simulation_window.destroy()
    root.deiconify()


def choose_color(title):
    color_code = colorchooser.askcolor(title=title)[1]
    return color_code


def choose_colormap(title):
    colormaps = sorted(m for m in plt.cm.datad if not m.endswith("_r"))
    colormap_window = tk.Toplevel(root)
    colormap_window.title(title)

    colormap_var = tk.StringVar(value=colormaps[0])

    for cmap in colormaps:
        tk.Radiobutton(colormap_window, text=cmap,
                       variable=colormap_var, value=cmap).pack(anchor='w')

    def select_colormap():
        colormap_window.destroy()

    tk.Button(colormap_window, text="Select", command=select_colormap).pack()

    root.wait_window(colormap_window)
    return colormap_var.get()


# GUI SET UP
root = tk.Tk()
root.title("Lattice Boltzmann Simulation")

tk.Label(root, text="Select Barrier Shape:").pack()

shapes = ["circle", "rectangle", "triangle", "airplane", "car"]

shape_var = tk.StringVar()
speed_var = tk.IntVar(value=5)
fluid_colormap_var = tk.StringVar(value='bwr')
barrier_color_var = tk.StringVar(value='#808080')  # Default to gray
fluid_type_var = tk.StringVar(value='water')  # Default to water

for shape in shapes:
    tk.Radiobutton(root, text=shape.capitalize(),
                   variable=shape_var, value=shape).pack()

tk.Label(root, text="Select Simulation Speed:").pack()
speed = tk.Scale(root, from_=1, to=10, orient=tk.HORIZONTAL,
                 length=200, variable=speed_var)
speed.pack()

tk.Button(root, text="Choose Barrier Color", command=lambda: barrier_color_var.set(
    choose_color("Select Barrier Color"))).pack()

tk.Label(root, text="Select Fluid Type:").pack()
fluid_types = ["water", "oil", "air"]
for fluid in fluid_types:
    tk.Radiobutton(root, text=fluid.capitalize(),
                   variable=fluid_type_var, value=fluid).pack()

simulation_window = None

tk.Button(root, text="Start Simulation", command=lambda: start_simulation(
    shape_var.get(), speed_var, fluid_colormap_var.get(), barrier_color_var.get(), fluid_type_var.get())).pack()

root.mainloop()