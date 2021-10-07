import argparse
import glob
import os
import shutil
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from typing import Any

import numpy as np
import pymesh
import trimesh
from scipy.interpolate import RegularGridInterpolator


def normalize_mesh(input_path: str, output_path: str, padding: float = 0.1, verbose: bool = False):
    # meshlab_command = f"meshlabserver -i {input_path} -o {input_path.replace('obj', 'off')}"
    # os.system(meshlab_command)

    mesh = trimesh.load(input_path, force="mesh", process=False)

    if verbose:
        print("Mesh (not normalized):", mesh.bounds)

    total_size = (mesh.bounds[1] - mesh.bounds[0]).max()
    scale = total_size / (1 - padding)
    centers = (mesh.bounds[1] + mesh.bounds[0]) / 2

    mesh.apply_translation(-centers)
    mesh.apply_scale(1 / scale)

    if verbose:
        print("Mesh (normalized):", mesh.bounds)

    # pymesh.save_mesh_raw(output_path, mesh.vertices, mesh.faces)
    mesh.export(output_path, file_type="obj")


def sdf_from_mesh(input_path: str, resolution: int = 256, **kwargs):
    command = f"./isosurface/computeDistanceField {input_path} {resolution} {resolution} {resolution}"
    kwarg_list = ["n", "s", 'o', "m", "b", "c", "e", "d", "t", "w", "W", "g", "G", "r", "i", "v", "p"]

    if kwargs.get('w') and kwargs.get('W'):
        kwarg_list.remove('w')

    if kwargs.get('g') and kwargs.get('G'):
        kwarg_list.remove('g')

    for kwarg in kwarg_list:
        value = kwargs.get(kwarg)
        if value is not None:
            if isinstance(value, bool):
                if value:
                    command += f" -{kwarg}"
            else:
                command += f" -{kwarg} {value}"
    if kwargs.get("verbose"):
        print("Running command:", command)
    os.system(command)


def sample(sdf_path: str, args: Any):
    sdf_dict = load_sdf(sdf_path, args.res)
    sdf_values = sdf_dict["values"]
    sdf_bounds = sdf_dict["bounds"]
    rng = np.random.default_rng()

    x = np.linspace(sdf_bounds[0], sdf_bounds[3], num=args.res + 1).astype(np.float32)
    y = np.linspace(sdf_bounds[1], sdf_bounds[4], num=args.res + 1).astype(np.float32)
    z = np.linspace(sdf_bounds[2], sdf_bounds[5], num=args.res + 1).astype(np.float32)

    # 1. Uniform samples from voxel grid
    choice = rng.integers(sdf_values.size, size=args.num_points)
    x_ind = choice % (args.res + 1)
    y_ind = (choice // (args.res + 1)) % (args.res + 1)
    z_ind = choice // (args.res + 1) ** 2
    x_vals = x[x_ind]
    y_vals = y[y_ind]
    z_vals = z[z_ind]
    vals = sdf_values.flatten()[choice]
    uniform_grid_samples = np.vstack((x_vals, y_vals, z_vals, vals)).T
    print(uniform_grid_samples.shape)
    print(uniform_grid_samples)

    # 2. Equal (inside/outside) samples from voxel grid

    # 3. Surface/uniform samples from voxel grid
    # 3.1 Surface
    surface_condition = (sdf_values < (1 / args.res)) & (sdf_values > -(1 / args.res))
    surface_values = sdf_values[surface_condition]
    choice = rng.integers(surface_values.size, size=args.num_points // 2)
    x_ind = choice % (args.res + 1)
    y_ind = (choice // (args.res + 1)) % (args.res + 1)
    z_ind = choice // (args.res + 1) ** 2
    x_vals = x[x_ind]
    y_vals = y[y_ind]
    z_vals = z[z_ind]
    vals = surface_values.flatten()[choice]
    surface_samples = np.vstack((x_vals, y_vals, z_vals, vals)).T

    # 3.2 Uniform
    choice = rng.integers(sdf_values.size, size=args.num_points // 2)
    x_ind = choice % (args.res + 1)
    y_ind = (choice // (args.res + 1)) % (args.res + 1)
    z_ind = choice // (args.res + 1) ** 2
    x_vals = x[x_ind]
    y_vals = y[y_ind]
    z_vals = z[z_ind]
    vals = sdf_values.flatten()[choice]
    uniform_samples = np.vstack((x_vals, y_vals, z_vals, vals)).T

    # 3.3 Surface + Uniform
    surface_grid_samples = np.concatenate((surface_samples, uniform_samples))
    print("Surface grid samples:")
    print(surface_grid_samples.shape)
    print(surface_grid_samples)

    # 4. Uniform random samples in volume
    interpolator = RegularGridInterpolator((z, y, x), sdf_values)
    x = (sdf_bounds[3] - sdf_bounds[0]) * rng.random(args.num_points) + sdf_bounds[0]
    y = (sdf_bounds[1] - sdf_bounds[4]) * rng.random(args.num_points) + sdf_bounds[4]
    z = (sdf_bounds[2] - sdf_bounds[5]) * rng.random(args.num_points) + sdf_bounds[5]
    samples = np.vstack((x, y, z)).T
    uniform_random_samples = np.hstack((samples, np.expand_dims(interpolator(samples), axis=1)))
    print(uniform_random_samples.shape)
    print(uniform_random_samples)

    # 5. Equal (inside/outside) random samples in volume
    # 6. Surface/uniform random samples in volume


def load_sdf(sdf_path: str, resolution: int = 256):
    intsize = 4
    floatsize = 8
    sdf = {
        "bounds": [],
        "values": []
    }
    with open(sdf_path, "rb") as f:
        try:
            bytes = f.read()
            ress = np.frombuffer(bytes[:intsize * 3], dtype=np.int32)
            if -1 * ress[0] != resolution or ress[1] != resolution or ress[2] != resolution:
                raise Exception(sdf_path, "res not consistent with ", str(resolution))
            positions = np.frombuffer(bytes[intsize * 3:intsize * 3 + floatsize * 6], dtype=np.float64)
            # bottom left corner, x,y,z and top right corner, x, y, z
            sdf["bounds"] = [positions[0], positions[1], positions[2], positions[3], positions[4], positions[5]]
            sdf["bounds"] = np.float32(sdf["bounds"])
            sdf["values"] = np.frombuffer(bytes[intsize * 3 + floatsize * 6:], dtype=np.float32)
            sdf["values"] = np.reshape(sdf["values"], (resolution + 1, resolution + 1, resolution + 1))
        finally:
            f.close()
    return sdf


def get_mesh_path(mesh_path, args):
    filename = mesh_path.split('/')[-1].split('.')[0]
    extension = mesh_path.split('/')[-1].split('.')[-1]
    if args.shapenet:
        synthset = mesh_path.split('/')[-3] if args.version == 1 else mesh_path.split('/')[-4]
        uid = mesh_path.split('/')[-2] if args.version == 1 else mesh_path.split('/')[-3]
        dir_path = os.path.join(args.o, synthset, uid)
        os.makedirs(dir_path, exist_ok=True)
        return os.path.join(args.o, synthset, uid, f"{uid}.obj")
    return os.path.join(args.o, "normalized", filename + '.' + extension)


def get_sdf_path(mesh_path, args):
    if not args.shapenet:
        mesh_path = mesh_path.replace("normalized", "sdf")
    return mesh_path.replace("obj", "dist")


def get_sample_path(sdf_path, args):
    if not args.shapenet:
        sdf_path = sdf_path.replace("sdf", "sample")
    return sdf_path.replace("dist", "npz")


def run(mesh, args):
    mesh_path = get_mesh_path(mesh, args)
    if not os.path.isfile(mesh_path) or args.overwrite:
        if args.verbose:
            print("Saving to:", mesh_path)
        normalize_mesh(mesh, mesh_path, verbose=args.verbose)
    elif args.verbose:
        print("Skipping", mesh_path)

    sdf_path = get_sdf_path(mesh_path, args)
    if not os.path.isfile(sdf_path) or args.overwrite:
        if args.verbose:
            print("Saving to:", sdf_path)
        filename = sdf_path.split('/')[-1]
        kwargs = {'n': args.n,
                  's': args.s,
                  'o': filename,
                  'm': args.m,
                  'b': args.b,
                  'c': args.c,
                  'e': args.e,
                  'd': args.d,
                  't': args.t,
                  'w': args.w,
                  'W': args.W,
                  'g': args.g,
                  'G': args.G,
                  'r': args.r,
                  'i': args.i,
                  'v': args.v,
                  'p': args.p,
                  "verbose": args.verbose}
        sdf_from_mesh(mesh_path, resolution=args.res, **kwargs)
        try:
            shutil.move(os.path.join(os.getcwd(), filename), sdf_path)
        except (OSError, FileNotFoundError):
            print("File", mesh, "couldn't be processed. Skipping.")
    elif args.verbose:
        print("Skipping", sdf_path)

    sample_path = get_sample_path(sdf_path, args)
    if not os.path.isfile(sample_path) or args.overwrite:
        sample(sdf_path, args)


def main():
    parser = argparse.ArgumentParser(description="Computes SDFs from meshes.")
    parser.add_argument("meshes", nargs='+', type=str, help="List of meshes or glob pattern.")
    parser.add_argument("res", type=int, default=256, help="Voxel grid resolution.")
    parser.add_argument('-n', action="store_true", help="Compute only narrow band distance field.")
    parser.add_argument('-s', action="store_true", help="Compute signed distance field.")
    parser.add_argument('-o', type=str, required=True, help="Output path.")
    parser.add_argument('-m', type=int, help="Signed field computation mode.")
    parser.add_argument('-b', help="Specify scene bounding box.")
    parser.add_argument('-c', action="store_true", help="Force bounding box into a cube.")
    parser.add_argument('-e', type=float, help="Expansion ratio for box.")
    parser.add_argument('-d', type=int, help="Max octree depth to use.")
    parser.add_argument('-t', type=int, help="Max num triangles per octree cell.")
    parser.add_argument('-w', type=float, help="The band width for narrow band distance field.")
    parser.add_argument('-W', type=int, help="Band width represented as #grid sizes.")
    parser.add_argument('-g', type=float, help="Sigma value.")
    parser.add_argument('-G', type=int, help="Sigma value represented as #grid sizes.")
    parser.add_argument('-r', action="store_true", help="Do not subtract sigma when creating signed field.")
    parser.add_argument('-i', default=None, help="Precomputed unsigned field for creating signed field.")
    parser.add_argument('-v', action="store_true", help="Also compute voronoi diagram.")
    parser.add_argument('-p', action="store_true", help="also compute closest points.")
    parser.add_argument("--num_points", type=int, default=100000, help="Number of points to sample from the SDF grid.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
    parser.add_argument("--shapenet", action="store_true", help="Assumes ShapeNet file structure.")
    parser.add_argument("--version", type=int, default=1, help="ShapeNet version.")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output during execution.")

    args = parser.parse_args()

    os.makedirs(args.o, exist_ok=True)
    if not args.shapenet:
        os.makedirs(os.path.join(args.o, "normalized"), exist_ok=True)
        os.makedirs(os.path.join(args.o, "sdf"), exist_ok=True)
        os.makedirs(os.path.join(args.o, "samples"), exist_ok=True)

    if len(args.meshes) > 1:
        meshes = args.meshes
    else:
        meshes = glob.glob(args.meshes.pop())
    if args.verbose:
        print("Path(s) to mesh(es):", meshes)

    with Parallel(n_jobs=cpu_count()) as parallel:
        parallel(delayed(run)(mesh, args) for mesh in meshes)


if __name__ == "__main__":
    main()
