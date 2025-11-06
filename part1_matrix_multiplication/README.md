nvcc -O3 -lcublas -o vectorize.exe vectorize.cu

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

## ⚙️ **Nsight Profiling Workflow Summary (WSL)**

### **1️⃣ Setup temporary profiling directory**

You created a folder on your Windows D: drive for all `.nsys-rep`, `.sqlite`, and `.txt` outputs:

```bash
mkdir -p /mnt/d/nsys_tmp
```

---

### **2️⃣ Set up environment variables for CUPTI and temp storage**

In your WSL terminal (VS Code terminal or Ubuntu), you exported:

```bash
export TMPDIR=/tmp
export CUPTI_OVERRIDE_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/libcupti.so
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
```

✅ These ensure:

* CUPTI (CUDA Profiling Tools Interface) is linked correctly
* Temporary report data is stored under `/tmp` (not C: drive)

---

### **3️⃣ Profile program using Nsight Systems CLI**

You profiled your optimized matrix multiplication executable (`2D_block_tiling.exe`) to collect **API-level profiling data**:

```bash
nsys profile --force-overwrite true \
  --trace=cuda,cublas,nvtx \
  --sample=none \
  -o /mnt/d/nsys_tmp/2D_block_tiling_report ./2D_block_tiling.exe
```

✅ This generated:

```
/mnt/d/nsys_tmp/2D_block_tiling_report.nsys-rep
```

which contains all CUDA API traces (`cudaMalloc`, `cudaMemcpy`, etc.).

---

### **4️⃣ Convert `.nsys-rep` to `.sqlite` for readable stats**

Then you exported the `.nsys-rep` to a SQLite database for analysis:

```bash
nsys export --type sqlite \
  --output /mnt/d/nsys_tmp/2D_block_tiling_report.sqlite \
  /mnt/d/nsys_tmp/2D_block_tiling_report.nsys-rep
```

✅ Output:

```
/mnt/d/nsys_tmp/2D_block_tiling_report.sqlite
```

---

### **5️⃣ Extract CUDA API statistics**

You attempted to pull CUDA API timings (e.g., `cudaMemcpy`, `cudaLaunchKernel`) using:

```bash
nsys stats --report cuda_api_sum /mnt/d/nsys_tmp/2D_block_tiling_report.sqlite > /mnt/d/nsys_tmp/api_stats.txt
```

If that report name wasn’t available, you listed all SQLite tables to see what existed:

```bash
sqlite3 /mnt/d/nsys_tmp/2D_block_tiling_report.sqlite ".tables"
```

✅ This gave you insight into available profiling data, even though WSL’s Nsight can’t show GPU kernel traces.

---

### **6️⃣ (Optional GUI Visualization)**

You opened the `.nsys-rep` timeline using:

```bash
nsys-ui /mnt/d/nsys_tmp/2D_block_tiling_report.nsys-rep
```

✅ You saw lanes:

```
CPU  |  Threads  |  CUDA API
```

(no “CUDA GPU Kernels” — expected under WSL).

---

### **7️⃣ GPU confirmation during run**

You verified your GPU was actually running by checking utilization:

```bash
nvidia-smi
```

Output confirmed:

```
GPU-Util: ~90%
Process name: /naive.exe
```

✅ Meaning your CUDA kernel executed correctly on GPU.

---

### **8️⃣ Nsight Compute (CLI alternative or GUI on Windows)**

Since `apt-get install nvidia-nsight-compute` didn’t work inside WSL,
you used (or can use) Nsight Compute from your **Windows CUDA Toolkit**:

```
"D:\CUDA\v13.0\Nsight Compute 2024.3.0\nv-nsight-cu-cli.exe" D:\Dash_assignment\DaSH-Lab-Induction-Assignment-2025\part1_matrix_multiplication\2D_block_tiling.exe > D:\nsys_tmp\ncu_output.txt
```

✅ This captures GPU-level metrics like:

* Achieved Occupancy
* DRAM Throughput (GB/s)
* FLOP Efficiency (%)
* SM / Warp Execution Efficiency

Or, equivalently, you can open the **Nsight Compute app (GUI)** on Windows and use:

* Application → `D:\Dash_assignment\...\2D_block_tiling.exe`
* Working Directory → same folder
* Start Profiling → View Summary tab

---

### **9️⃣ Output files created**

You now have these key profiling outputs:

| File                                              | Description                      |
| ------------------------------------------------- | -------------------------------- |
| `/mnt/d/nsys_tmp/2D_block_tiling_report.nsys-rep` | Nsight Systems raw report        |
| `/mnt/d/nsys_tmp/2D_block_tiling_report.sqlite`   | Exported database                |
| `/mnt/d/nsys_tmp/api_stats.txt`                   | CUDA API summary timings         |
| `/mnt/d/nsys_tmp/ncu_output.txt`                  | Nsight Compute GPU-level metrics |
| `/mnt/d/nsys_tmp/ncu_metrics.txt` *(optional)*    | Occupancy/throughput metrics     |

---

### **💡 Optional Advanced Metrics**

You can collect detailed GPU utilization metrics with:

```bash
nv-nsight-cu-cli --metrics achieved_occupancy,dram__throughput.avg.pct_of_peak_sustained_elapsed,flop_sp_efficiency ./2D_block_tiling.exe > /mnt/d/nsys_tmp/ncu_metrics.txt
```

---

### ✅ **Summary of All Key Commands (copy-paste ready)**

```bash
# Create output folder
mkdir -p /mnt/d/nsys_tmp

# Environment setup
export TMPDIR=/tmp
export CUPTI_OVERRIDE_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/libcupti.so
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# Run Nsight Systems profiling
nsys profile --force-overwrite true \
  --trace=cuda,cublas,nvtx \
  --sample=none \
  -o /mnt/d/nsys_tmp/2D_block_tiling_report ./2D_block_tiling.exe

# Export to SQLite
nsys export --type sqlite \
  --output /mnt/d/nsys_tmp/2D_block_tiling_report.sqlite \
  /mnt/d/nsys_tmp/2D_block_tiling_report.nsys-rep

# Extract CUDA API timing summary
nsys stats --report cuda_api_sum /mnt/d/nsys_tmp/2D_block_tiling_report.sqlite > /mnt/d/nsys_tmp/api_stats.txt

# Open GUI timeline (optional)
nsys-ui /mnt/d/nsys_tmp/2D_block_tiling_report.nsys-rep

# Check GPU utilization
nvidia-smi

# (Windows) Run Nsight Compute CLI
"D:\CUDA\v13.0\Nsight Compute 2024.3.0\nv-nsight-cu-cli.exe" \
 "D:\Dash_assignment\DaSH-Lab-Induction-Assignment-2025\part1_matrix_multiplication\2D_block_tiling.exe" \
 > D:\nsys_tmp\ncu_output.txt
```

---

### 🎯 Final Deliverables You Can Include in DaSH Report

* Screenshot of **Nsight Systems timeline (CUDA API view)**
* Screenshot of **Nsight Compute Summary tab**
* Table of metrics (Occupancy, Throughput, FLOP Efficiency)
* Explanation of **memory vs compute bottleneck**

---

## ⚙️ **Nsight Systems Profiling Workflow Summary**

All GPU profiling and reports were organized under the folder:
`D:\nsys_tmp` (mounted in WSL as `/mnt/d/nsys_tmp`).

---

### **1️⃣ Initial Profiling Attempts (Basic Stats Mode)**

To perform a quick capture and view using Nsight Systems GUI:

```bash
nsys profile --stats=true -t cuda,osrt,nvtx -o 2D_block_tiling_report ./2D_block_tiling.exe
nsys-ui 2D_block_tiling_report.nsys-rep
nsys stats naive_report.nsys-rep > naive_stats.txt
```

✅ This generated `.nsys-rep` and `.txt` summary files for initial performance verification.

---

### **2️⃣ Before Running Nsight: Setup Temporary Directory and Environment**

To prevent `/tmp` or `C:` drive space issues, a custom profiling directory was used:

```bash
export TMPDIR=/tmp
```

Profiling command with CPU sampling and CUDA/NVTX tracing enabled:

```bash
nsys profile --sample=cpu --trace=cuda,nvtx \
  -o /mnt/d/nsys_tmp/2D_block_tiling_report ./2D_block_tiling.exe
```

Export report to SQLite database for analysis:

```bash
nsys export /mnt/d/nsys_tmp/2D_block_tiling_report.nsys-rep \
  --type sqlite --output /mnt/d/nsys_tmp/2D_block_tiling_report.sqlite
```

---

### **3️⃣ Installing and Linking CUPTI (CUDA Profiling Interface)**

CUPTI is required for CUDA event tracing.
You installed and verified it with:

```bash
sudo apt-get install -y libcupti-dev
ls -lh /usr/lib/x86_64-linux-gnu/libcupti.so
```

This confirmed that the library existed at:

```
/usr/lib/x86_64-linux-gnu/libcupti.so
```

---

### **4️⃣ Setting Environment Variables for Nsight Profiling**

Before running Nsight Systems, you exported the following variables to ensure the correct library path and temp directory were used:

```bash
export CUPTI_OVERRIDE_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/libcupti.so
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export TMPDIR=/tmp
```

---

### **5️⃣ Full Nsight Systems Profiling (Final Working Command)**

You executed the main profiling run that successfully collected data from CUDA APIs and cuBLAS routines:

```bash
nsys profile --trace=cuda,cublas,nvtx --sample=none \
  -o /mnt/d/nsys_tmp/2D_block_tiling_report ./2D_block_tiling.exe
```

This created:

```
/mnt/d/nsys_tmp/2D_block_tiling_report.nsys-rep
```

---

### **6️⃣ GUI Visualization and Report Export**

You viewed the timeline and traces using the Nsight Systems UI:

```bash
nsys-ui /mnt/d/nsys_tmp/naive_report.nsys-rep
```

✅ Nsight UI displayed:

```
CPU  | Threads  | CUDA API
```

Lanes (expected under WSL).
Even though “CUDA HW” didn’t appear, GPU utilization was confirmed through `nvidia-smi` (90%+ usage).

---

### **7️⃣ Confirmed Output Directory**

All profiling reports (`.nsys-rep`, `.sqlite`, `.txt`) were directed to and stored under:

```
D:\nsys_tmp
```

(WSL path: `/mnt/d/nsys_tmp`)

This prevented space issues on the Linux root filesystem and C: drive.

---

## ✅ **Summary Table of Your Nsight Commands**

| Step | Command                                                   | Purpose                    |
| ---- | --------------------------------------------------------- | -------------------------- |
| 1    | `nsys profile --stats=true -t cuda,osrt,nvtx ...`         | Quick summary run          |
| 2    | `export TMPDIR=/tmp`                                      | Set temp directory         |
| 3    | `nsys profile --sample=cpu --trace=cuda,nvtx ...`         | CPU + CUDA API tracing     |
| 4    | `nsys export ... --type sqlite ...`                       | Convert to SQLite database |
| 5    | `sudo apt-get install -y libcupti-dev`                    | Install CUPTI              |
| 6    | `export CUPTI_OVERRIDE_LIBRARY_PATH=...`                  | Enable CUDA event tracing  |
| 7    | `nsys profile --trace=cuda,cublas,nvtx --sample=none ...` | Final detailed profiling   |
| 8    | `nsys-ui /mnt/d/nsys_tmp/naive_report.nsys-rep`           | View report in GUI         |

---

### ✅ **Outcome**

* Nsight Systems profiling worked successfully under WSL.
* All reports saved to D:/nsys_tmp.
* CUPTI correctly loaded (`libcupti.so.13.1`).
* CUDA API calls traced (no GPU HW lane due to WSL limitation).
* `nvidia-smi` confirmed GPU was actively executing kernels.





