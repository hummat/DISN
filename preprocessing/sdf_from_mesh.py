import argparse
import glob
import logging
import os
import subprocess
import time
from multiprocessing import cpu_count
from typing import Any, List, Tuple, Union

import numpy as np
import trimesh
import tqdm
import pymesh
from joblib import Parallel, delayed
from matplotlib.cm import get_cmap
from scipy.interpolate import RegularGridInterpolator
# from create_point_sdf_grid import get_normalize_mesh


rng = np.random.default_rng()
logging.basicConfig(level=logging.CRITICAL)
logger = logging.getLogger(__name__)


def as_mesh(scene_or_mesh: Union[trimesh.Trimesh, trimesh.Scene]) -> trimesh.Trimesh:
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None
        else:
            mesh = trimesh.util.concatenate([trimesh.Trimesh(vertices=g.vertices, faces=g.faces) for g in scene_or_mesh.geometry.values()])
    else:
        assert(isinstance(scene_or_mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
    return mesh


def normalize_mesh(input_path: str, output_path: str, args: Any) -> None:
    path = input_path
    if not input_path.endswith(".off"):
        path = input_path.replace(".obj", ".off")
        command = f"meshlabserver -i {input_path} -o {path}"
        subprocess.run(command.split(' '), stdout=subprocess.DEVNULL)
        # os.system(command + " &> /dev/null")
    mesh = trimesh.load(path, process=False)

    total_size = (mesh.bounds[1] - mesh.bounds[0]).max()
    scale = total_size / (1 - args.padding)
    centers = (mesh.bounds[1] + mesh.bounds[0]) / 2

    mesh.apply_translation(-centers)
    mesh.apply_scale(1 / scale)

    # pymesh.save_mesh_raw(output_path, mesh.vertices, mesh.faces)
    mesh.export(output_path, file_type="obj")
    if not input_path.endswith(".off"):
        os.remove(path)


def sdf_from_mesh(mesh_path: str, resolution: int = 256, **kwargs: Any) -> None:
    command = f"./isosurface/computeDistanceField {mesh_path} {resolution} {resolution} {resolution}"
    kwarg_list = ['n', 's', 'o', 'm', 'b', 'c', 'e', 'd', 't', 'w', 'W', 'g', 'G', 'r', 'i', 'v', 'p']

    if kwargs.get('w') is not None and kwargs.get('W') is not None:
        kwarg_list.remove('w')

    if kwargs.get('g') is not None and kwargs.get('G') is not None:
        if kwargs.get('G') == 3:
            kwarg_list.remove('G')
        else:
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
        print("SDF command:", command)
    subprocess.run(command.split(' '), stdout=subprocess.DEVNULL)
    # os.system(command + " &> /dev/null")


def mesh_from_sdf(sdf_path: str, **kwargs: Any) -> None:
    command = f"./isosurface/computeMarchingCubes {sdf_path} {kwargs.get('o')}"

    value = kwargs.get('i')
    if value is not None:
        command += f" -i {value}"

    if kwargs.get('n'):
        command += " -n"

    if kwargs.get("verbose"):
        print("Marching cubes command:", command)
    subprocess.run(command.split(' '), stdout=subprocess.DEVNULL)
    # os.system(command + " &> /dev/null")


def uniform_grid_sampling(grid: np.ndarray,
                          bounds: Union[np.ndarray, List, Tuple],
                          num_points: int,
                          mask: Union[np.ndarray, List, Tuple] = None) -> np.ndarray:
    assert len(grid.shape) == 3, f"{grid.shape}"
    assert len(bounds) == 6, f"{bounds}"

    res_x, res_y, res_z = grid.shape
    x = np.linspace(bounds[0], bounds[3], num=res_x).astype(np.float32)
    y = np.linspace(bounds[1], bounds[4], num=res_y).astype(np.float32)
    z = np.linspace(bounds[2], bounds[5], num=res_z).astype(np.float32)

    if mask is None:
        choice = rng.integers(grid.size, size=num_points)
        x_ind = choice % res_x
        y_ind = (choice // res_y) % res_y
        z_ind = choice // res_z ** 2
        x_vals = x[x_ind]
        y_vals = y[y_ind]
        z_vals = z[z_ind]
        vals = grid.flatten()[choice]
    else:
        choice = rng.choice(np.argwhere(mask), size=num_points)
        x_vals = x[choice[:, 2]]
        y_vals = y[choice[:, 1]]
        z_vals = z[choice[:, 0]]
        vals = grid[choice[:, 0], choice[:, 1], choice[:, 2]]
    return np.vstack((x_vals, y_vals, z_vals, vals)).T


def uniform_random_sampling(grid: np.ndarray,
                            bounds: Union[np.ndarray, List, Tuple],
                            num_points: int = 0,
                            points: np.ndarray = None) -> np.ndarray:
    assert len(grid.shape) == 3
    assert len(bounds) == 6
    assert num_points > 0 or points is not None

    res_x, res_y, res_z = grid.shape
    x = np.linspace(bounds[0], bounds[3], num=res_x).astype(np.float32)
    y = np.linspace(bounds[1], bounds[4], num=res_y).astype(np.float32)
    z = np.linspace(bounds[2], bounds[5], num=res_z).astype(np.float32)

    interpolator = RegularGridInterpolator((z, y, x), grid)

    if points is None:
        x = (bounds[3] - bounds[0]) * rng.random(num_points) + bounds[0]
        y = (bounds[4] - bounds[1]) * rng.random(num_points) + bounds[1]
        z = (bounds[5] - bounds[2]) * rng.random(num_points) + bounds[2]
    else:
        x = np.clip(points[:, 0], bounds[0], bounds[3])
        y = np.clip(points[:, 1], bounds[1], bounds[4])
        z = np.clip(points[:, 2], bounds[2], bounds[5])
    points = np.vstack((z, y, x)).T
    return np.vstack((points[:, 2], points[:, 1], points[:, 0], interpolator(points))).T


def sample(sdf_path: str, mesh_path: str, args: Any) -> None:
    sdf_dict = load_sdf(sdf_path, args.res)
    sdf_values = sdf_dict["values"]
    sdf_bounds = sdf_dict["bounds"]
    mesh = trimesh.load(mesh_path, process=False)
    if args.visualize:
        mesh.show()
    assert mesh.is_watertight, f"Mesh {mesh_path} is not watertight. Skipping."

    # 1. Uniform samples from voxel grid
    uniform_grid_samples = uniform_grid_sampling(sdf_values, sdf_bounds, args.num_points)

    # 2. Equal (inside/outside) samples from voxel grid
    inside_mask = sdf_values <= 0
    outside_mask = sdf_values > 0
    inside_samples = uniform_grid_sampling(sdf_values, sdf_bounds, args.num_points // 2, inside_mask)
    outside_samples = uniform_grid_sampling(sdf_values, sdf_bounds, args.num_points // 2, outside_mask)
    equal_grid_samples = np.concatenate((inside_samples, outside_samples))

    # 3. Surface/uniform samples from voxel grid
    expansion_ration = args.e if args.e is not None else 1 + 2 * args.padding
    voxel_size = (1 - args.padding) * expansion_ration / args.res
    surface_mask = (sdf_values <= voxel_size) & (sdf_values >= -voxel_size)
    surface_samples = uniform_grid_sampling(sdf_values, sdf_bounds, args.num_points // 2, surface_mask)
    uniform_samples = uniform_grid_samples[:args.num_points // 2]
    surface_grid_samples = np.concatenate((surface_samples, uniform_samples))

    # 4. Uniform random samples in volume
    uniform_random_samples = uniform_random_sampling(sdf_values, sdf_bounds, args.num_points)

    # 5. Equal (inside/outside) random samples in volume
    inside_points = inside_samples[:, :3] + args.noise * rng.standard_normal((args.num_points // 2, 3))
    outside_points = outside_samples[:, :3] + args.noise * rng.standard_normal((args.num_points // 2, 3))
    inside_samples = uniform_random_sampling(sdf_values, sdf_bounds, points=inside_points)
    outside_samples = uniform_random_sampling(sdf_values, sdf_bounds, points=outside_points)
    equal_random_samples = np.concatenate((inside_samples, outside_samples))

    # 6. Surface/uniform random samples in volume
    surface_points = mesh.sample(max(100000, args.num_points))
    # trimesh.PointCloud(np.concatenate((surface_points, surface_samples[:, :3])),
    #                    np.concatenate((np.tile((1.0, 0.0, 0.0, 1.0), (100000, 1)),
    #                                    np.tile((0.0, 0.0, 1.0, 1.0), (50000, 1))))).show()
    noisy_points = surface_points[:args.num_points // 2] + args.noise * rng.standard_normal((args.num_points // 2, 3))
    surface_samples = uniform_random_sampling(sdf_values, sdf_bounds, points=noisy_points)
    surface_random_samples = np.concatenate((surface_samples, uniform_random_samples[:args.num_points // 2]))

    all_samples = [uniform_grid_samples,
                   equal_grid_samples,
                   surface_grid_samples,
                   uniform_random_samples,
                   equal_random_samples,
                   surface_random_samples]
    sample_names = ["uniform_grid",
                    "equal_grid",
                    "surface_grid",
                    "uniform_random",
                    "equal_random",
                    "surface_random"]

    for samples, name in zip(all_samples, sample_names):
        assert samples.shape[0] == args.num_points, f"Only {samples.shape[0]} {name} samples for {mesh_path}"
        assert samples.shape[1] == 4, f"{mesh_path} {name} {samples.shape[1]}"
        assert np.all((samples[:, 0] >= sdf_bounds[0]) & (samples[:, 0] <= sdf_bounds[3]))
        assert np.all((samples[:, 1] >= sdf_bounds[1]) & (samples[:, 1] <= sdf_bounds[4]))
        assert np.all((samples[:, 2] >= sdf_bounds[2]) & (samples[:, 2] <= sdf_bounds[5]))

    sample_path = get_sample_path(sdf_path, args)
    replace = sample_path.split('/')[-1]

    for samples, name in zip(all_samples, sample_names):
        np.save(sample_path.replace(replace, name), samples.astype(np.float32))

        if args.visualize:
            points = samples[:, :3]
            sdfs = samples[:, 3]

            reds = get_cmap("Reds")
            blues = get_cmap("Blues").reversed()
            inside = sdfs[sdfs < 0]
            outside = sdfs[sdfs > 0]
            inside_norm = (inside - inside.min()) / (inside.max() - inside.min())
            outside_norm = (outside - outside.min()) / (outside.max() - outside.min())
            inside = [reds(i) for i in inside_norm]
            outside = [blues(o) for o in outside_norm]

            colors = np.array([(0.0, 0.0, 0.0, 1.0) for _ in sdfs])
            colors[sdfs < 0] = inside
            colors[sdfs > 0] = outside

            trimesh.PointCloud(points, colors).show()

    np.save(sample_path.replace(replace, "surface"), surface_points[:100000].astype(np.float32))
    if args.visualize:
        trimesh.PointCloud(surface_points[:100000]).show()


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


def get_mesh_path(mesh_path: str, args: Any) -> str:
    filename = mesh_path.split('/')[-1].split('.')[0]
    if args.shapenet:
        synthset = mesh_path.split('/')[-3] if args.version == 1 else mesh_path.split('/')[-4]
        uid = mesh_path.split('/')[-2] if args.version == 1 else mesh_path.split('/')[-3]
        dir_path = os.path.join(args.o, synthset, uid)
        os.makedirs(dir_path, exist_ok=True)
        return os.path.join(args.o, synthset, uid, f"{uid}.obj")
    return os.path.join(args.o, "mesh", filename + ".obj")


def get_sdf_path(mesh_path: str, args: Any) -> str:
    if not args.shapenet:
        mesh_path = mesh_path.replace("/mesh/", "/sdf/")
    return mesh_path.replace(".obj", ".dist")


def get_sample_path(sdf_path, args):
    if not args.shapenet:
        sdf_path_list = sdf_path.replace("/sdf/", "/samples/").split('/')
        filename = sdf_path_list[-1].split('.')[0]
        sdf_path_list.insert(-1, filename)
        sdf_path = '/'.join(sdf_path_list)
        os.makedirs('/'.join(sdf_path_list[:-1]), exist_ok=True)
    return sdf_path.replace(".dist", ".npz")


def run(mesh: str, args: Any) -> None:
    start_run = time.time()
    mesh_path = get_mesh_path(mesh, args)
    sdf_path = get_sdf_path(mesh_path, args)
    sample_dir = '/'.join(get_sample_path(sdf_path, args).split('/')[:-1])
    sample_files = glob.glob(sample_dir + "/*.npy")
    if len(sample_files) == 7 and not args.overwrite:
        logger.warning(f"Sampling for mesh {mesh} done. Skipping.")
        return

    normalize_mesh(mesh, mesh_path, args)

    logger.debug(f"Saving SDF to: {sdf_path}")
    kwargs = {'n': args.n,
              's': False if args.u else True,
              'o': os.path.relpath(sdf_path),
              'm': args.m,
              'b': args.b,
              'c': args.c,
              'e': args.e if args.e is not None else 1 + 2 * args.padding,
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

    start = time.time()
    sdf_from_mesh(mesh_path, resolution=args.res, **kwargs)
    logger.debug(f"SDF time: {time.time() - start}")

    logger.debug(f"Saving mesh from marching cubes to: {mesh_path}")
    expansion_ration = args.e if args.e is not None else 1 + 2 * args.padding
    voxel_size = (1 - args.padding) * expansion_ration / args.res
    kwargs = {'o': os.path.relpath(mesh_path),
              'i': voxel_size,
              'n': args.n,
              "verbose": args.verbose}

    start = time.time()
    mesh_from_sdf(sdf_path, **kwargs)
    logger.debug(f"Marching cubes time: {time.time() - start}")

    start = time.time()
    try:
        sample(sdf_path, mesh_path, args)
    except AssertionError as e:
        logger.critical(e)
    logger.debug(f"Sample time: {time.time() - start}")

    if not args.sdf:
        os.remove(sdf_path)
    if not args.mesh:
        os.remove(mesh_path)

    # Remove unnecessary files
    dir_name = '/'.join(mesh_path.split('/')[:-1])
    test = os.listdir(dir_name)
    for item in test:
        if not item.endswith(".npy") and not item.endswith(".obj"):
            path = os.path.join(dir_name, item)
            logger.warning(f"Removing {path}")
            os.remove(path)

    logger.debug(f"Runtime: {time.time() - start_run}")


def main():
    start = time.time()
    parser = argparse.ArgumentParser(description="Computes SDFs from meshes.")
    parser.add_argument("meshes", nargs='+', type=str, help="List of meshes or glob pattern.")
    parser.add_argument("res", type=int, default=256, help="Voxel grid resolution.")
    parser.add_argument('-n', action="store_true", help="Compute only narrow band distance field.")
    parser.add_argument('-u', action="store_true", help="Compute unsigned distance field.")
    parser.add_argument('-o', type=str, required=True, help="Output path.")
    parser.add_argument('-m', type=int, default=2, help="Signed field computation mode.")
    parser.add_argument('-b', help="Specify scene bounding box.")
    parser.add_argument('-c', action="store_true", help="Force bounding box into a cube.")
    parser.add_argument('-e', type=float, help="Expansion ratio for box.")
    parser.add_argument('-d', type=int, help="Max octree depth to use.")
    parser.add_argument('-t', type=int, help="Max num triangles per octree cell.")
    parser.add_argument('-w', type=float, help="The band width for narrow band distance field.")
    parser.add_argument('-W', type=int, help="Band width represented as #grid sizes.")
    parser.add_argument('-g', type=float, help="Sigma value.")
    parser.add_argument('-G', type=int, default=3, help="Sigma value represented as #grid sizes.")
    parser.add_argument('-r', action="store_true", help="Do not subtract sigma when creating signed field.")
    parser.add_argument('-i', type=str, default=None, help="Precomputed unsigned field for creating signed field.")
    parser.add_argument('-v', action="store_true", help="Also compute voronoi diagram.")
    parser.add_argument('-p', action="store_true", help="also compute closest points.")
    parser.add_argument("--num_points", type=int, default=100000, help="Number of points to sample from the SDF grid.")
    parser.add_argument("--noise", type=float, default=0.01, help="Noise variance added to surface samples.")
    parser.add_argument("--padding", type=float, default=0.1, help="Padding applied when normalizing mesh.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
    parser.add_argument("--sdf", action="store_true", help="Store the computed SDF.")
    parser.add_argument("--mesh", action="store_true", help="Compute and store object mesh created from SDF.")
    parser.add_argument("--shapenet", action="store_true", help="Assumes ShapeNet file structure for in- and output.")
    parser.add_argument("--version", type=int, default=1, help="ShapeNet version.")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output during execution.")
    parser.add_argument("--visualize", action="store_true", help="Visualize SDF samples.")

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.WARNING)

    os.makedirs(args.o, exist_ok=True)
    if not args.shapenet:
        os.makedirs(os.path.join(args.o, "mesh"), exist_ok=True)
        os.makedirs(os.path.join(args.o, "sdf"), exist_ok=True)
        os.makedirs(os.path.join(args.o, "samples"), exist_ok=True)

    if len(args.meshes) > 1:
        meshes = args.meshes
    else:
        meshes = glob.glob(args.meshes.pop())
    if args.verbose:
        print("Path(s) to mesh(es):", meshes)

    if len(meshes) > 1:
        with Parallel(n_jobs=cpu_count()) as parallel:
            parallel(delayed(run)(mesh, args) for mesh in meshes)
    else:
        run(meshes[0], args)

    print("Time taken:", time.time() - start)


def mesh_test():
    synthset = "02876657"
    shapenet_v1_path = f"/home/matthias/Data2/datasets/shapenet/ShapeNetCore.v1/{synthset}/**/model.obj"
    shapenet_v2_path = f"/home/matthias/Data2/datasets/shapenet/ShapeNetCore.v2/{synthset}/**/models/model_normalized.obj"
    occnet_path = f"/home/matthias/Data/Ubuntu/git/occupancy_networks/data/ShapeNet.build/{synthset}/4_watertight_scaled/*.off"
    disn_path = f"/home/matthias/Data2/datasets/shapenet/disn/{synthset}/**/isosurf.obj"
    my_disn_path = f"/home/matthias/Data2/datasets/shapenet/matthias/disn/{synthset}/**/*.obj"
    disn_from_occnet_path = f"/home/matthias/Data2/datasets/shapenet/matthias/disn_from_occnet/mesh/*.obj"
    disn_from_manifold_path = f"/home/matthias/Data2/datasets/shapenet/matthias/disn_from_manifold/mesh/*.obj"
    manifold_path = f"/home/matthias/Data2/datasets/shapenet/matthias/manifold/{synthset}/*.obj"

    meshes = glob.glob(disn_from_manifold_path)
    results = list()
    for mesh in tqdm.tqdm(meshes):
        results.append(as_mesh(trimesh.load(mesh, process=False)).is_watertight)
    print(len(results), np.sum(results), np.sum(results) / len(results))

    # Results synthset 02876657:
    # ShapeNet v1: 498 11 0.02208835341365462
    # ShapeNet v2: 498 16 0.0321285140562249
    # ManifoldPlus: 498 498 1.0
    # OccNet: 498 498 1.0
    # DISN: 498 480 0.963855421686747
    # My DISN: 498 498 1.0
    # DISN from OccNet: 498 498 1.0
    # DISN from ManifoldPlus: 498 496 0.9959839357429718


def manifold_test():
    synthset = "02876657"
    shapenet_v1_path = f"/home/matthias/Data2/datasets/shapenet/ShapeNetCore.v1/{synthset}/**/model.obj"
    out = f"/home/matthias/Data2/datasets/shapenet/matthias/manifold/{synthset}"
    os.makedirs(out, exist_ok=True)
    meshes = glob.glob(shapenet_v1_path)
    with Parallel(n_jobs=cpu_count()) as parallel:
        parallel(delayed(run_manifold)(mesh, os.path.join(out, mesh.split('/')[-2] + ".obj"), 6) for mesh in meshes)


def run_manifold(input_path: str, output_path: str, depth: int = 6) -> None:
    command = f"./{os.path.relpath('/home/matthias/Data/Ubuntu/git/ManifoldPlus/build/manifold')} --input {input_path} --output {output_path} --depth {depth}"
    subprocess.run(command.split(' '))


if __name__ == "__main__":
    main()
