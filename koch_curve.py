import torch
import matplotlib.pyplot as plt
import math

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# Starting line
def make_starting_points():
    return torch.tensor([
        [0.0, 0.0],
        [1.0, 0.0]
    ], device=device)


def koch_step(points):

    # All segment starting points
    start = points[:-1]

    # All segment ending points
    end = points[1:]

    # Direction of every segment at once
    direction = end - start

    # Points at 1/3 and 2/3 for every segment
    p1 = start + direction / 3
    p3 = start + 2 * direction / 3

    # Middle third of every segment
    middle = direction / 3

    # Rotate every middle segment by 60 degrees
    angle = math.pi / 3

    # Create a rotation matrix for 60 degrees
    x = middle[:, 0]
    y = middle[:, 1]

    rotated = torch.stack([
        x * math.cos(angle) - y * math.sin(angle),
        x * math.sin(angle) + y * math.cos(angle)
    ], dim=1)

    # Peak of the triangular bump
    p2 = p1 + rotated

    # Put the new points together
    new_points = torch.stack([
        start,
        p1,
        p2,
        p3
    ], dim=1)

    # Reshape to a 2D array of points, flattening the segments into a single list of points
    new_points = new_points.reshape(-1, 2)

    # Add the final endpoint
    new_points = torch.cat([
        new_points,
        points[-1].unsqueeze(0) # Add the last point to the end of the new points
    ], dim=0)

    return new_points


# Number of iterations
points = make_starting_points()

iterations = 4

for i in range(iterations):
    points = koch_step(points)


# Move to CPU for Matplotlib
points_cpu = points.cpu().numpy()

plt.figure(figsize=(12, 4))

plt.plot(
    points_cpu[:, 0],
    points_cpu[:, 1],
    linewidth=1
)

plt.title(f"Koch Curve - Iteration {iterations}")
plt.axis("equal")
plt.axis("off")

plt.show()

# Number of iterations

points = make_starting_points()

iterations = 6

for i in range(iterations):
    points = koch_step(points)


# Move to CPU for Matplotlib
points_cpu = points.cpu().numpy()

plt.figure(figsize=(12, 4))

plt.plot(
    points_cpu[:, 0],
    points_cpu[:, 1],
    linewidth=1
)

plt.title(f"Koch Curve - Iteration {iterations}")
plt.axis("equal")
plt.axis("off")

plt.show()

# zoom in on the Koch curve by changing the figure size and axis limits

plt.figure(figsize=(12, 4))

plt.plot(
    points_cpu[:, 0],
    points_cpu[:, 1],
    linewidth=1
)

plt.xlim(0.11, 0.23)
plt.ylim(0.05, 0.17)

plt.title(f"Zoomed Koch Curve - Iteration {iterations}")
plt.axis("off")

plt.show()
