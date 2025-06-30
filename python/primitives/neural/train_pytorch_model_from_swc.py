import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import argparse
import json
import os
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

def load_swc_file(swc_path, max_points=None, aabb_min=None, aabb_max=None):
    """
    Load SWC file and return points and radii.
    SWC format: ID, type, x, y, z, radius, parent_ID
    """
    try:
        # Try to read as space/tab separated
        data = pd.read_csv(swc_path, sep=r'\s+', header=None, comment='#')
        if data.shape[1] < 6:
            raise ValueError("SWC file must have at least 6 columns")
        
        # Extract x, y, z coordinates (columns 2, 3, 4) and radius (column 5)
        points = data.iloc[:, [2, 3, 4]].values.astype(np.float32)
        radii = data.iloc[:, 5].values.astype(np.float32)
        
        print(f"Loaded {len(points)} points from SWC file (before filtering)")
        print(f"Original coordinate bounds: min={points.min(axis=0)}, max={points.max(axis=0)}")
        
        # Apply AABB filtering if specified
        if aabb_min is not None and aabb_max is not None:
            aabb_min = np.array(aabb_min, dtype=np.float32)
            aabb_max = np.array(aabb_max, dtype=np.float32)
            print(f"Applying AABB filter: min={aabb_min}, max={aabb_max}")
            
            # Filter points that are within the AABB
            mask = np.all((points >= aabb_min) & (points <= aabb_max), axis=1)
            
            if not np.any(mask):
                raise ValueError(f"No SWC points found within AABB bounds [{aabb_min}, {aabb_max}]")
            
            points = points[mask]
            radii = radii[mask]
            
            points_in_aabb = np.sum(mask)
            print(f"Points within AABB: {points_in_aabb}/{len(mask)} ({points_in_aabb/len(mask)*100:.1f}%)")
        
        # Limit number of points if specified (after AABB filtering)
        if max_points is not None and len(points) > max_points:
            print(f"Limiting SWC data to first {max_points} points (out of {len(points)} after AABB filtering)")
            points = points[:max_points]
            radii = radii[:max_points]
        
        print(f"Final dataset: {len(points)} points")
        print(f"Final coordinate bounds: min={points.min(axis=0)}, max={points.max(axis=0)}")
        print(f"Radius range: min={radii.min():.4f}, max={radii.max():.4f}")
        
        return points, radii
        
    except Exception as e:
        raise ValueError(f"Error loading SWC file: {e}")

def compute_sdf_spheres_batch(args):
    """
    Compute SDF for a batch of query points against a set of spheres.
    For each query point, find the minimum distance to any sphere surface.
    """
    sphere_centers, sphere_radii, batch_samples = args
    
    # Vectorized computation of distances
    diff = batch_samples[:, np.newaxis, :] - sphere_centers[np.newaxis, :, :]
    distances_to_centers = np.linalg.norm(diff, axis=2)  # (n_samples, n_spheres)
    
    # Compute SDF to each sphere surface: distance_to_center - radius
    sdf_to_spheres = distances_to_centers - sphere_radii[np.newaxis, :]  # (n_samples, n_spheres)
    
    # Find minimum SDF (closest sphere surface) for each sample
    min_sdf = np.min(sdf_to_spheres, axis=1)  # (n_samples,)
    
    return min_sdf

def sample_sdf_from_swc(swc_path, n_samples=10000, noise=0.01, batch_size=1000, n_cores=None, surface_sampling_ratio=0.6, max_swc_points=None, aabb_min=None, aabb_max=None):
    """
    Sample SDF from SWC file containing sphere data.
    Uses improved sampling strategy with more samples near surfaces.
    """
    # Load SWC data
    sphere_centers, sphere_radii = load_swc_file(swc_path, max_points=max_swc_points, aabb_min=aabb_min, aabb_max=aabb_max)
    
    # FIXED: Compute proper bounding box that considers individual sphere positions
    # For each sphere, its effective bounds are center ± radius
    sphere_mins = sphere_centers - sphere_radii[:, np.newaxis]
    sphere_maxs = sphere_centers + sphere_radii[:, np.newaxis]
    
    # Overall bounding box is the union of all sphere bounds
    min_coords = sphere_mins.min(axis=0)
    max_coords = sphere_maxs.max(axis=0)
    
    # Add small margin for better sampling
    margin = sphere_radii.max() * 0.2  # 20% of max radius
    min_coords -= margin
    max_coords += margin
    
    bounds = np.array([min_coords, max_coords])
    
    print(f"FIXED bounding box: min={min_coords}, max={max_coords}")
    print(f"Bounding box size: {max_coords - min_coords}")
    
    # IMPROVED: Mixed sampling strategy
    n_surface_samples = int(n_samples * surface_sampling_ratio)
    n_uniform_samples = n_samples - n_surface_samples
    
    print(f"Sampling strategy: {n_surface_samples} near surfaces, {n_uniform_samples} uniform")
    
    # 1. Surface-aware sampling: sample near spheres
    surface_samples = []
    if n_surface_samples > 0:
        samples_per_sphere = max(1, n_surface_samples // len(sphere_centers))
        for i, (center, radius) in enumerate(zip(sphere_centers, sphere_radii)):
            # Sample in a shell around each sphere (radius * 0.5 to radius * 2.0)
            n_this_sphere = samples_per_sphere
            if i < n_surface_samples % len(sphere_centers):
                n_this_sphere += 1
            
            # Random directions
            directions = np.random.randn(n_this_sphere, 3)
            directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
            
            # Random distances in shell around sphere
            shell_radii = radius * (0.5 + 1.5 * np.random.rand(n_this_sphere))
            sphere_samples = center + directions * shell_radii[:, np.newaxis]
            
            surface_samples.append(sphere_samples)
        
        surface_samples = np.vstack(surface_samples)[:n_surface_samples]
    
    # 2. Uniform sampling in bounding box
    uniform_samples = np.random.rand(n_uniform_samples, 3) * (max_coords - min_coords) + min_coords
    
    # Combine samples
    if n_surface_samples > 0:
        samples = np.vstack([surface_samples, uniform_samples])
    else:
        samples = uniform_samples
    
    # Shuffle the combined samples
    np.random.shuffle(samples)
    
    # Initialize arrays for results
    sdf = np.zeros(n_samples)
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    # Prepare batches for parallel processing
    batches = []
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_samples)
        batch_samples = samples[start_idx:end_idx]
        batches.append((sphere_centers, sphere_radii, batch_samples))
    
    # Use multiprocessing to compute SDF in parallel
    if n_cores is None:
        n_cores = max(1, mp.cpu_count() - 1)  # Leave one core free
    else:
        n_cores = min(n_cores, mp.cpu_count())
    print(f"Using {n_cores} CPU cores for SDF computation")
    
    with mp.Pool(n_cores) as pool:
        results = list(tqdm(
            pool.imap(compute_sdf_spheres_batch, batches),
            total=len(batches),
            desc="Computing SDF from spheres"
        ))
    
    # Combine results
    for i, result_sdf in enumerate(results):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_samples)
        sdf[start_idx:end_idx] = result_sdf
    
    # Add noise to samples
    samples += np.random.randn(*samples.shape) * noise
    
    # Print training data statistics
    inside_count = np.sum(sdf < 0)
    on_surface_count = np.sum(np.abs(sdf) < sphere_radii.min())
    outside_count = np.sum(sdf > 0)
    
    print(f"Training data distribution:")
    print(f"  Inside (SDF < 0): {inside_count} ({inside_count/n_samples*100:.1f}%)")
    print(f"  Near surface: {on_surface_count} ({on_surface_count/n_samples*100:.1f}%)")
    print(f"  Outside (SDF > 0): {outside_count} ({outside_count/n_samples*100:.1f}%)")
    print(f"  SDF range: [{sdf.min():.4f}, {sdf.max():.4f}]")
    
    return torch.from_numpy(samples).float(), torch.from_numpy(sdf).float().unsqueeze(1), bounds

# === Neural Network Definition ===
class SDFNet(nn.Module):
    def __init__(self, hidden_dim=128, n_layers=4, mesh_bounds=None):
        super().__init__()
        # Input is 3 dimensions: just position
        layers = [nn.Linear(3, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        # Output is 1 dimension: just SDF
        layers += [nn.Linear(hidden_dim, 1)]
        self.net = nn.Sequential(*layers)
        
        # Register original bounds as buffers so they're saved with the model
        self.register_buffer('mesh_bounds_min', torch.tensor(mesh_bounds[0], dtype=torch.float))
        self.register_buffer('mesh_bounds_max', torch.tensor(mesh_bounds[1], dtype=torch.float))
        
        # Calculate threshold as diagonal/256
        diagonal = torch.norm(torch.tensor(mesh_bounds[1] - mesh_bounds[0], dtype=torch.float))
        threshold = diagonal / 256
        print(f"Model threshold (diagonal/256): {threshold.item():.6f}")
        self.register_buffer('threshold', torch.tensor([threshold], dtype=torch.float))

    def forward(self, x):
        # Just predict SDF value
        return self.net(x)
    
    def get_mesh_bounds(self):
        return {
            'min': self.mesh_bounds_min.cpu().numpy(),
            'max': self.mesh_bounds_max.cpu().numpy()
        }
    
    def get_threshold(self):
        return self.threshold.item()

# === Training Function ===
def train_sdf_model(swc_path, epochs=500, lr=1e-4, batch_size=2048, n_layers=4, n_samples=10000, n_cores=None, num_workers=4, surface_sampling_ratio=0.6, max_swc_points=None, aabb_min=None, aabb_max=None):
    points, distances, bounds = sample_sdf_from_swc(swc_path, n_samples=n_samples, n_cores=n_cores, surface_sampling_ratio=surface_sampling_ratio, max_swc_points=max_swc_points, aabb_min=aabb_min, aabb_max=aabb_max)
    
    # SDF convention: negative inside, positive outside
    # Our computation already follows this convention
    
    batch_size = min(batch_size, 1024)
    dataset = torch.utils.data.TensorDataset(points, distances)
    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    model = SDFNet(n_layers=n_layers, mesh_bounds=bounds).cuda()
    model = torch.nn.DataParallel(model)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    sdf_loss_fn = nn.L1Loss()

    # Move data to GPU once
    points = points.cuda()
    distances = distances.cuda()

    for epoch in tqdm(range(epochs), desc="Training"):
        total_loss = 0
        for batch in loader:
            x, y = batch
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            
            # Enable automatic mixed precision
            with torch.amp.autocast('cuda'):
                pred_sdf = model(x)
                loss = sdf_loss_fn(pred_sdf, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    return model.module, bounds  # Unwrap from DataParallel

# === Main Execution ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train SDF model from SWC file')
    parser.add_argument('--swc_path', type=str, required=True, help='Path to the SWC file')
    parser.add_argument('--n_layers', type=int, default=4, help='Number of layers in the MLP')
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=2048, help='Batch size')
    parser.add_argument('--model_path', type=str, default='sdf_model_swc_fixed.pt', help='Path where to save the trained model')
    parser.add_argument('--n_samples', type=int, default=20000, help='Number of points to sample for training (increased default)')
    parser.add_argument('--n_cores', type=int, default=None, help='Number of CPU cores to use for SDF computation')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--surface_sampling_ratio', type=float, default=0.6, help='Fraction of samples to place near surfaces (0.0 to 1.0)')
    parser.add_argument('--max_swc_points', type=int, default=None, help='Maximum number of points to read from SWC file (default: read all)')
    parser.add_argument('--aabb_min', type=float, nargs=3, help='Minimum bounds for AABB filtering (x y z)')
    parser.add_argument('--aabb_max', type=float, nargs=3, help='Maximum bounds for AABB filtering (x y z)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.swc_path):
        raise FileNotFoundError(f"SWC file not found: {args.swc_path}")
    
    # Validate AABB arguments
    if (args.aabb_min is None) != (args.aabb_max is None):
        raise ValueError("Both --aabb_min and --aabb_max must be specified together, or neither")
    
    if args.aabb_min is not None and args.aabb_max is not None:
        if any(min_val >= max_val for min_val, max_val in zip(args.aabb_min, args.aabb_max)):
            raise ValueError("AABB min values must be less than max values")
    
    print("=== FIXED SWC Training Script ===")
    print("Key improvements:")
    print("1. Fixed bounding box computation")
    print("2. Surface-aware sampling strategy")
    print("3. Better training data distribution")
    print("4. Removed unused normal computation")
    if args.aabb_min is not None:
        print(f"5. AABB filtering: min={args.aabb_min}, max={args.aabb_max}")
    print()
    
    model, bounds = train_sdf_model(
        args.swc_path,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        n_layers=args.n_layers,
        n_samples=args.n_samples,
        n_cores=args.n_cores,
        num_workers=args.num_workers,
        surface_sampling_ratio=args.surface_sampling_ratio,
        max_swc_points=args.max_swc_points,
        aabb_min=args.aabb_min,
        aabb_max=args.aabb_max
    )
    
    # Move model to CPU for scripting
    model = model.cpu()
    model.eval()
    
    # Create example input for tracing (just position)
    example_position = torch.randn(1, 3)
    
    # Create TorchScript model
    traced_model = torch.jit.trace(model, example_position)
    
    # Save the traced model
    traced_model.save(args.model_path)
    
    print(f"FIXED model saved to {args.model_path}")
    print("Model metadata:")
    print(f"  n_layers: {args.n_layers}")
    print(f"  epochs: {args.epochs}")
    print(f"  learning_rate: {args.lr}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  bounds: min={bounds[0]}, max={bounds[1]}")
    print(f"  surface_sampling_ratio: {args.surface_sampling_ratio}")
    if args.max_swc_points is not None:
        print(f"  max_swc_points: {args.max_swc_points}") 