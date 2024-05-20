import matplotlib.pyplot as plt
import numpy as np

def create_cylinder(Nx, Ny, x_center, y_center, radius):
    """ Create a cylindrical boundary in a 2D grid. """
    X, Y = np.meshgrid(range(Nx), range(Ny))
    cylinder = (X - x_center) ** 2 + (Y - y_center) ** 2 < radius ** 2
    return cylinder

def main():
    """ Lattice Boltzmann Simulation """

    # User Inputs
    x_center = int(input("Enter the x-coordinate for the cylinder's center (0 to 799): "))
    y_center = int(input("Enter the y-coordinate for the cylinder's center (0 to 199): "))
    radius = int(input("Enter the cylinder's radius: "))
    initial_u_x = float(input("Enter the initial x-component of the fluid velocity: "))
    initial_u_y = float(input("Enter the initial y-component of the fluid velocity: "))
    
    # Add an input for the fluid type
    fluid_type = input("Enter the fluid type (e.g., 'water', 'oil', 'air'): ").lower()

    # Set properties based on the fluid type
    if fluid_type == 'water':
        rho0 = 1000  # density of water in kg/m^3
        default_tau = 0.6  # typical value for water
        colormap = 'Blues'
    elif fluid_type == 'oil':
        rho0 = 900  # density of oil in kg/m^3
        default_tau = 0.8  # higher tau for more viscous fluid
        colormap = 'Greens'
    elif fluid_type == 'air':
        rho0 = 1.2  # density of air in kg/m^3
        default_tau = 0.55  # lower tau for less viscous fluid
        colormap = 'Reds'
    else:
        print("Unknown fluid type, using default values.")
        rho0 = 100  # default density
        default_tau = 0.6  # default tau
        colormap = 'gray'

    # Allow user to optionally input a custom tau
    tau_input = input(f"Enter the relaxation time tau (default {default_tau}): ")
    tau = float(tau_input) if tau_input else default_tau

    # Simulation settings
    Nx, Ny = 800, 200  # increased grid resolution
    Nt = 4000  # total timesteps
    plot_in_real_time = True  # toggle for real-time plotting

    # Lattice velocities and weights
    num_lattice_directions = 9
    directions = np.arange(num_lattice_directions)
    vel_x = np.array([0, 0, 1, 1, 1, 0, -1, -1, -1])
    vel_y = np.array([0, 1, 1, 0, -1, -1, -1, 0, 1])
    weights = np.array([4/9, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36])  # must sum to 1

    # Initialize fluid density function
    F = np.ones((Ny, Nx, num_lattice_directions)) * rho0 / num_lattice_directions
    np.random.seed(42)
    F += 0.001 * np.random.randn(Ny, Nx, num_lattice_directions)  # Reduced noise magnitude
    X, Y = np.meshgrid(range(Nx), range(Ny))
    F[:, :, 3] += 2 * (1 + 0.2 * np.cos(2 * np.pi * X / Nx * 4))
    density = np.sum(F, axis=2)
    for i in directions:
        F[:, :, i] *= rho0 / density

    # Define cylinder boundary
    cylinder = create_cylinder(Nx, Ny, x_center, y_center, radius)

    # Initialize velocity field
    u_x = np.full((Ny, Nx), initial_u_x)
    u_y = np.full((Ny, Nx), initial_u_y)

    # Prepare plot
    fig = plt.figure(figsize=(12, 10), dpi=80)  # adjust figure size for larger grid

    # Main simulation loop
    for timestep in range(Nt):
        if timestep % 100 == 0:
            print(f'Timestep {timestep}/{Nt}')

        # Streaming step
        for i, vx, vy in zip(directions, vel_x, vel_y):
            F[:, :, i] = np.roll(F[:, :, i], vx, axis=1)
            F[:, :, i] = np.roll(F[:, :, i], vy, axis=0)

        # Reflective boundary conditions
        boundary_F = F[cylinder, :]
        boundary_F = boundary_F[:, [0, 5, 6, 7, 8, 1, 2, 3, 4]]

        # Calculate macroscopic variables
        density = np.sum(F, axis=2)
        density[density == 0] = np.finfo(float).eps  # Avoid division by zero
        u_x = np.sum(F * vel_x, axis=2) / density
        u_y = np.sum(F * vel_y, axis=2) / density

        # Collision step
        F_eq = np.zeros_like(F)
        for i, vx, vy, weight in zip(directions, vel_x, vel_y, weights):
            F_eq[:, :, i] = density * weight * (
                1 + 3 * (vx * u_x + vy * u_y) +
                9 * (vx * u_x + vy * u_y) ** 2 / 2 -
                3 * (u_x ** 2 + u_y ** 2) / 2
            )

        F += -(1.0 / tau) * (F - F_eq)

        # Apply boundary conditions
        F[cylinder, :] = boundary_F

        # Real-time plotting
        if (plot_in_real_time and (timestep % 50 == 0)) or (timestep == Nt - 1):  # Increased timestep for plotting
            plt.cla()
            u_x[cylinder] = 0
            u_y[cylinder] = 0
            vorticity = (np.roll(u_y, -1, axis=0) - np.roll(u_y, 1, axis=0)) - (np.roll(u_x, -1, axis=1) - np.roll(u_x, 1, axis=1))
            vorticity[cylinder] = np.nan
            plt.imshow(vorticity, cmap=colormap, origin='lower')
            plt.imshow(cylinder, cmap='gray', alpha=0.3, origin='lower')
            plt.clim(-0.1, 0.1)
            ax = plt.gca()
            ax.invert_yaxis()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.set_aspect('equal')
            plt.pause(0.001)

    # Calculate lift and drag forces
    lift_force = np.sum(2 * (F[:, :, 1] - F[:, :, 5]) * cylinder)
    drag_force = np.sum(2 * (F[:, :, 3] - F[:, :, 7]) * cylinder)
    print(f"Lift Force: {lift_force:.2f}")
    print(f"Drag Force: {drag_force:.2f}")

    # Save final plot
    plt.savefig('lattice_boltzmann_simulation.png', dpi=240)
    plt.show()

    return 0

if __name__ == "__main__":
    main()
