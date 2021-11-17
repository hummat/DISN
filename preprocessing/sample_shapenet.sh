synthsets=("02691156"
           "02828884"
           # "02876657"
           # "02880940"
           "02933112"
           # "02946921"
           "02958343"
           "03001627"
           "03211117"
           "03636649"
           "03691459"
           # "03797390"
           "04090263"
           "04256520"
           "04379243"
           "04401088"
           "04530566")

for synthset in "${synthsets[@]}"
do
  echo "Processing synthset $synthset ..."
  python preprocessing/sdf_from_mesh.py "/home/matthias/Data2/datasets/shapenet/ShapeNetCore.v1/$synthset/**/model.obj" 256 -o /home/matthias/Data2/datasets/shapenet/matthias/disn/core -r --shapenet --mesh --voxel
  mkdir -p /media/matthias/TRAVELER/shapenet_v1_disn/"$synthset"
  mv /home/matthias/Data2/datasets/shapenet/matthias/disn/"$synthset"/**/*.off /media/matthias/TRAVELER/shapenet_v1_disn/"$synthset"
done


