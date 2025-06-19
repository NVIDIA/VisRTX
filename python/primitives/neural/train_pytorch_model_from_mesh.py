import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import trimesh
import argparse
import json
import os
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

def compute_sdf_batch(args):
    mesh, batch_samples = args
    sdf = trimesh.proximity.signed_distance(mesh, batch_samples)
    # Get closest points and face indices
    closest_points, face_indices, _ = trimesh.proximity.closest_point(mesh, batch_samples)
    # Convert face indices to integers and get normals for the faces
    face_indices = face_indices.astype(np.int64)
    normals = mesh.face_normals[face_indices]
    return sdf, normals

# === Chargement du mesh et sampling des points + distances ===
def sample_sdf(mesh_path, n_samples=10000, noise=0.01, batch_size=1000, n_cores=None):
    # Load the scene
    scene = trimesh.load(mesh_path)
    
    # If it's a scene, combine all meshes
    if isinstance(scene, trimesh.Scene):
        # Get all mesh geometries from the scene
        meshes = []
        for geometry in scene.geometry.values():
            if isinstance(geometry, trimesh.Trimesh):
                meshes.append(geometry)
        
        if not meshes:
            raise ValueError("No valid meshes found in the scene")
            
        # Combine all meshes into one
        mesh = trimesh.util.concatenate(meshes)
    else:
        mesh = scene
    
    # Ensure mesh has face normals
    if not hasattr(mesh, 'face_normals') or len(mesh.face_normals) == 0:
        print("Computing face normals for the mesh...")
        mesh.fix_normals()
        mesh.compute_face_normals()
    
    bounds = mesh.bounds
    min_bound, max_bound = bounds
    samples = np.random.rand(n_samples, 3) * (max_bound - min_bound) + min_bound

    sdf = np.zeros(n_samples)
    normals = np.zeros((n_samples, 3))
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    # Prepare batches for parallel processing
    batches = []
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_samples)
        batch_samples = samples[start_idx:end_idx]
        batches.append((mesh, batch_samples))
    
    # Use multiprocessing to compute SDF in parallel
    if n_cores is None:
        n_cores = max(1, mp.cpu_count() - 1)  # Leave one core free
    else:
        n_cores = min(n_cores, mp.cpu_count())  # Don't exceed available cores
    print(f"Using {n_cores} CPU cores for SDF computation")
    
    with mp.Pool(n_cores) as pool:
        results = list(tqdm(
            pool.imap(compute_sdf_batch, batches),
            total=len(batches),
            desc="Computing SDF"
        ))
    
    # Combine results
    for i, (result_sdf, result_normals) in enumerate(results):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_samples)
        sdf[start_idx:end_idx] = result_sdf
        normals[start_idx:end_idx] = result_normals
    
    samples += np.random.randn(*samples.shape) * noise
    return torch.from_numpy(samples).float(), torch.from_numpy(sdf).float().unsqueeze(1), torch.from_numpy(normals).float(), bounds

# === Définition du MLP ===
class SDFNet(nn.Module):
    def __init__(self, hidden_dim=128, n_layers=4, mesh_bounds=None):
        super().__init__()
        # Input is now 3 dimensions: just position
        layers = [nn.Linear(3, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        # Output is now 1 dimension: just SDF
        layers += [nn.Linear(hidden_dim, 1)]
        self.net = nn.Sequential(*layers)
        
        # Register original mesh bounds as buffers so they're saved with the model
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

# === Entraînement ===
def train_sdf_model(mesh_path, epochs=500, lr=1e-4, batch_size=2048, n_layers=4, n_samples=10000, n_cores=None, num_workers=4):
    points, distances, _, mesh_bounds = sample_sdf(mesh_path, n_samples=n_samples, n_cores=n_cores)
    
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

    model = SDFNet(n_layers=n_layers, mesh_bounds=mesh_bounds).cuda()
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

    return model.module, mesh_bounds  # Unwrap from DataParallel

# === Exemple d'utilisation ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train SDF model')
    parser.add_argument('--mesh_path', type=str, required=True, help='Path to the mesh file')
    parser.add_argument('--n_layers', type=int, default=4, help='Number of layers in the MLP')
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=2048, help='Batch size')
    parser.add_argument('--model_path', type=str, default='sdf_model.pt', help='Path where to save the trained model')
    parser.add_argument('--n_samples', type=int, default=10000, help='Number of points to sample for training')
    parser.add_argument('--n_cores', type=int, default=None, help='Number of CPU cores to use for SDF computation (default: all cores - 1)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    
    args = parser.parse_args()
    
    model, bounds = train_sdf_model(
        args.mesh_path,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        n_layers=args.n_layers,
        n_samples=args.n_samples,
        n_cores=args.n_cores,
        num_workers=args.num_workers
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
    
    print(f"Model saved to {args.model_path}")
    print("Model metadata:")
    print(f"  n_layers: {args.n_layers}")
    print(f"  epochs: {args.epochs}")
    print(f"  learning_rate: {args.lr}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  bounds: min={bounds[0]}, max={bounds[1]}")
