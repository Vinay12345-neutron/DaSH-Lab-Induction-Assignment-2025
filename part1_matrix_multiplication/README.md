
nvcc -O3 -arch=sm_86 -lcublas -o 2D_block_tiling_windows.exe 2D_block_tiling.cu


nsys profile --stats=true -t cuda,osrt,nvtx -o 2D_block_tiling_report ./2D_block_tiling.exe
nsys-ui 2D_block_tiling_report.nsys-rep
nsys stats naive_report.nsys-rep > naive_stats.txt

Before running Nsight:
export TMPDIR=/tmp
nsys profile --sample=cpu --trace=cuda,nvtx -o /mnt/d/nsys_tmp/2D_block_tiling_report ./2D_block_tiling.exe
nsys export /mnt/d/nsys_tmp/2D_block_tiling_report.nsys-rep --type sqlite --output /mnt/d/nsys_tmp/2D_block_tiling_report.sqlite

sudo apt-get install -y libcupti-dev
ls -lh /usr/lib/x86_64-linux-gnu/libcupti.so

export CUPTI_OVERRIDE_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/libcupti.so
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export TMPDIR=/tmp
nsys profile --trace=cuda,cublas,nvtx --sample=none \
  -o /mnt/d/nsys_tmp/2D_block_tiling_report ./2D_block_tiling.exe


nsys-ui /mnt/d/nsys_tmp/naive_report.nsys-rep


// for Windows adding below block of code
#ifndef uint
#define uint unsigned int
#endif
// To profile using nsight compute, add this line of code, then go to nsight compute, DONT START NEW ACTIVITY. Just in the powershell AS ADMIN write the following code. Basically, from POWERSHELL (not vs code powershell, regular powershell as administrator, you need to run the code to profile in nsight compute) Do all the work from powershell, WSL is running into issues with nsight compute. (I have fixed those issues)
nvcc -O3 -arch=sm_86 -lcublas -o 2D_block_tiling_windows.exe 2D_block_tiling.cu
 nsys profile --trace=cuda,cublas,nvtx --sample=none -o 2D_block_tiling_report 2D_block_tiling_windows.exe
 Then when the nsys-rep file is generated, OPEN the FILE in nsight compute, and go to the Kernels section. right click on kernel and click on profile kernel. DONT START ACTIVITY AND TRY TO LAUNCH IT. 

 nsys stats D:\Dash_assignment\DaSH-Lab-Induction-Assignment-2025\part1_matrix_multiplication\2D_block_tiling_report.nsys-rep

